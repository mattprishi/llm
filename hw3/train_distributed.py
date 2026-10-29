#!/usr/bin/env python3
"""Full SFT on two GPUs: torchrun --standalone --nproc_per_node=2 train_distributed.py."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.distributed as dist
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer


def drop_nulls(value):
    if isinstance(value, dict):
        return {key: drop_nulls(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [drop_nulls(item) for item in value]
    return value


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def to_dataset(rows):
    dataset = Dataset.from_list([
        {"messages": row["messages"], "tools": row["tools"]} for row in rows
    ])
    # Arrow adds null fields while unifying the tool schemas; remove them on read.
    return dataset.with_transform(lambda batch: {
        key: [drop_nulls(item) for item in items] for key, items in batch.items()
    })


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        help="Flat JSON: model/tokenizer paths and SFTConfig overrides")
    parser.add_argument("--model-name")
    parser.add_argument("--model-revision")
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--output-dir")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-length", type=int)
    args = parser.parse_args()
    # The notebook may materialize this worker under outputs/ before torchrun.
    root = Path.cwd()
    if not (root / "data/train.jsonl").is_file():
        root = Path(__file__).resolve().parent
    os.chdir(root)
    settings = json.loads(args.config.read_text()) if args.config else {}
    for name in ("model_name", "model_revision", "tokenizer_path", "output_dir", "seed", "max_length"):
        value = getattr(args, name)
        if value is not None:
            settings[name] = value

    model_name = settings.pop("model_name", "Qwen/Qwen2.5-0.5B-Instruct")
    model_revision = settings.pop("model_revision", None)
    tokenizer_path = settings.pop("tokenizer_path", "outputs/tokenizer_for_training")
    seed = settings.pop("seed", 42)
    max_length = settings.pop("max_length", 2048)
    output_dir = Path(settings.pop("output_dir", "outputs/full_sft"))
    train_path = Path(settings.pop("train_path", "data/train.jsonl"))
    validation_path = Path(settings.pop("validation_path", "data/validation.jsonl"))

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    assert world_size == 2, f"Expected two DDP ranks, got {world_size}"
    assert torch.cuda.is_available() and torch.cuda.device_count() == 2
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.set_num_threads(8)
    set_seed(seed)
    dist.init_process_group(backend="nccl", device_id=device)
    rank = dist.get_rank()
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        assert "{% generation %}" in tokenizer.chat_template, "Missing assistant training mask"
        model = AutoModelForCausalLM.from_pretrained(
            model_name, revision=model_revision,
            dtype=torch.bfloat16, attn_implementation="sdpa",
        ).to(device)
        config_values = {
            "output_dir": str(output_dir), "num_train_epochs": 3,
            "per_device_train_batch_size": 4, "per_device_eval_batch_size": 4,
            "gradient_accumulation_steps": 1, "learning_rate": 5e-5,
            "lr_scheduler_type": "linear", "warmup_ratio": 0.05,
            "max_length": max_length, "assistant_only_loss": True,
            "packing": False, "bf16": True, "tf32": True,
            "gradient_checkpointing": False, "optim": "adamw_torch_fused",
            "eval_strategy": "epoch", "save_strategy": "no",
            "logging_steps": 10, "logging_first_step": True,
            "report_to": "none", "seed": seed, "data_seed": seed,
            "dataloader_num_workers": 0, "prediction_loss_only": True,
            "local_rank": local_rank, "ddp_find_unused_parameters": False,
        }
        config_values.update(settings)
        config = SFTConfig(**config_values)
        assert config.per_device_train_batch_size == 4
        assert config.gradient_accumulation_steps == 1
        trainer = SFTTrainer(
            model=model, args=config,
            train_dataset=to_dataset(read_jsonl(train_path)),
            eval_dataset=to_dataset(read_jsonl(validation_path)),
            processing_class=tokenizer,
        )
        assert trainer.args.n_gpu == 1, "Each DDP process must use one GPU"
        assert trainer.accelerator.num_processes == 2
        total_params = sum(parameter.numel() for parameter in model.parameters())
        trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        assert total_params == trainable_params, "Full fine-tuning requires all parameters"

        features = [trainer.train_dataset[index] for index in range(8)]
        batch = trainer.data_collator(features)
        supervised_tokens = 0
        for index, feature in enumerate(features):
            mask = torch.tensor(feature["assistant_masks"], dtype=torch.bool)
            labels = batch["labels"][index, :len(mask)]
            ids = batch["input_ids"][index, :len(mask)]
            assert mask.any(), "Example has no supervised assistant tokens"
            assert torch.all(labels[~mask] == -100)
            assert torch.equal(labels[mask], ids[mask])
            supervised_tokens += int(mask.sum())
        del batch, features
        metadata = {
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_name": model_name, "model_revision": model.config._commit_hash,
            "tokenizer_path": str(tokenizer_path), "seed": seed,
            "parallelism": "DistributedDataParallel", "world_size": world_size,
            "global_batch_size": config.per_device_train_batch_size * world_size * config.gradient_accumulation_steps,
            "total_parameters": total_params, "trainable_params": trainable_params,
            "trainable_parameters": trainable_params,
            "collator_labels_verified": True, "collator_supervised_tokens": supervised_tokens,
            "train_examples": len(trainer.train_dataset),
            "validation_examples": len(trainer.eval_dataset),
            "config": config.to_dict(),
            "packages": {name: importlib.metadata.version(name) for name in
                         ["torch", "transformers", "trl", "datasets", "accelerate"]},
            "data_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in [train_path, validation_path]},
        }
        print(f"rank={rank}, local_rank={local_rank}, gpu={torch.cuda.get_device_name(local_rank)}, "
              f"trainable={trainable_params:,}, global_batch={metadata['global_batch_size']}, "
              "collator_labels=verified", flush=True)
        if trainer.is_world_process_zero():
            write_json(output_dir / "training_preflight.json", metadata)

        dist.barrier()
        torch.cuda.reset_peak_memory_stats(local_rank)
        torch.cuda.synchronize(local_rank)
        started = time.monotonic()
        result = trainer.train()
        torch.cuda.synchronize(local_rank)
        training_seconds = time.monotonic() - started
        trainer.save_model()
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(output_dir)

        local_stats = {
            "rank": rank, "local_rank": local_rank, "hostname": platform.node(),
            "gpu": torch.cuda.get_device_name(local_rank),
            "training_seconds": training_seconds,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(local_rank) / 2**30,
            "global_step": trainer.state.global_step,
        }
        rank_stats = [None] * world_size
        dist.all_gather_object(rank_stats, local_stats)
        assert all(stat["peak_allocated_gib"] > 0.1 for stat in rank_stats)
        if trainer.is_world_process_zero():
            metadata.update({
                "status": "completed", "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "training_seconds": max(stat["training_seconds"] for stat in rank_stats),
                "peak_allocated_gib": [stat["peak_allocated_gib"] for stat in rank_stats],
                "rank_stats": rank_stats, "global_step": trainer.state.global_step,
                "metrics": result.metrics, "training_metrics": result.metrics,
                "log_history": trainer.state.log_history,
            })
            write_json(output_dir / "training_state.json", metadata)
            print(f"COMPLETED: steps={trainer.state.global_step}, "
                  f"seconds={metadata['training_seconds']:.2f}, "
                  f"peak_GiB={metadata['peak_allocated_gib']}", flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
