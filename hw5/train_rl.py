"""Single-device GRPO-family LoRA training for the ToolACE function-call task."""

from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
from statistics import mean, median
import time

BASE_MODEL = "Qwen/Qwen3.5-2B"
BASE_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
EXPERIMENTS = {
    "rl1": {"reward": "strict", "loss": "grpo"},
    "rl2": {"reward": "shaped", "loss": "grpo"},
    "rl3": {"reward": "shaped", "loss": "dapo"},
}
# Target only the language tower, including both full and linear attention.
LORA_TARGETS = r"model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|in_proj_qkv|out_proj)"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=EXPERIMENTS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--eval-every", type=int)
    parser.add_argument("--val-limit", type=int, default=48)
    parser.add_argument("--val-batch", type=int, default=8)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--max-completion", type=int, default=192)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vllm", action="store_true")
    cli = parser.parse_args()
    for key in ("steps", "val_limit", "val_batch", "batch", "max_completion"):
        if getattr(cli, key) < 1:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if cli.generations < 2 or cli.batch % cli.generations:
        parser.error("--batch must be a multiple of --generations, which must be at least 2")
    if cli.eval_every is not None and cli.eval_every < 1:
        parser.error("--eval-every must be positive")
    return cli


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")


def read_rows(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"Empty dataset: {path}")
    return rows


def select_validation(rows, limit):
    """Freeze a deterministic slice with coverage of each available example kind."""
    by_kind = defaultdict(list)
    for row in rows:
        key = "no_call" if not row["expected"] else (
            "parallel" if len(row["expected"]) > 1 else "single"
        )
        by_kind[key].append(row)
    for group in by_kind.values():
        group.sort(key=lambda row: sha256(row["id"].encode()).hexdigest())
    selected = []
    for index in range(max(map(len, by_kind.values()))):
        for group in by_kind.values():
            if index < len(group):
                selected.append(group[index])
                if len(selected) == limit:
                    return selected
    return selected


def main():
    cli = parse_args()
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForImageTextToText, AutoTokenizer, GenerationConfig, TrainerCallback, set_seed
    from trl import GRPOConfig, GRPOTrainer

    from rewards import score_completion

    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one assigned GPU via CUDA_VISIBLE_DEVICES or Docker --gpus.")
    start = time.monotonic()
    set_seed(cli.seed)
    cli.output.mkdir(parents=True, exist_ok=False)
    train_path, val_path = cli.data / "train.jsonl", cli.data / "val.jsonl"
    train_rows = read_rows(train_path)
    all_val_rows = read_rows(val_path)
    val_rows = select_validation(all_val_rows, cli.val_limit)
    train_ids = {row["dialogue_id"] for row in train_rows}
    val_ids = {row["dialogue_id"] for row in all_val_rows}
    if train_ids & val_ids:
        raise ValueError("ToolACE train and validation share dialogue IDs")
    train_set = Dataset.from_list([
        {
            "id": row["id"],
            "prompt": row["prompt"],
            "expected": json.dumps(row["expected"], ensure_ascii=False),
            "tools": json.dumps(row["tools"], ensure_ascii=False),
        }
        for row in train_rows
    ])
    config = EXPERIMENTS[cli.experiment]
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION, padding_side="left")
    model_source = BASE_MODEL
    if cli.vllm:
        from huggingface_hub import snapshot_download

        # TRL's colocated engine opens model.name_or_path without a revision argument.
        model_source = snapshot_download(
            BASE_MODEL, revision=BASE_REVISION,
            allow_patterns=["*.safetensors", "*.json", "*.jinja", "merges.txt", "tokenizer.model"],
        )
    model = AutoModelForImageTextToText.from_pretrained(
        model_source, revision=BASE_REVISION, dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    configured_eos = model.generation_config.eos_token_id
    if isinstance(configured_eos, int):
        configured_eos = [configured_eos]
    eos_ids = set(configured_eos or []) | {tokenizer.eos_token_id}
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0, bias="none", target_modules=LORA_TARGETS, task_type="CAUSAL_LM"
    )
    args = GRPOConfig(
        output_dir=str(cli.output),
        max_steps=cli.steps,
        per_device_train_batch_size=cli.batch,
        gradient_accumulation_steps=1,
        num_generations=cli.generations,
        max_completion_length=cli.max_completion,
        learning_rate=cli.lr,
        lr_scheduler_type="constant",
        warmup_steps=0,
        bf16=True,
        tf32=True,
        beta=0.0,
        epsilon=0.2,
        epsilon_high=0.2,
        temperature=0.85,
        top_p=1.0,
        top_k=0,
        scale_rewards="group",
        num_iterations=1,
        loss_type=config["loss"],
        gradient_checkpointing=False,
        max_grad_norm=1.0,
        remove_unused_columns=False,
        mask_truncated_completions=False,
        use_vllm=cli.vllm,
        vllm_mode="colocate",
        vllm_max_model_length=4096,
        vllm_gpu_memory_utilization=0.35,
        save_strategy="no",
        logging_steps=1,
        report_to=[],
        seed=cli.seed,
        data_seed=cli.seed,
        disable_tqdm=True,
    )
    versions = {p: version(p) for p in ("trl", "transformers", "torch", "peft", "datasets", "accelerate", "numpy", "scipy")}
    for p in ("flash-linear-attention", "causal-conv1d", "vllm"):
        try:
            versions[p] = version(p)
        except PackageNotFoundError:
            versions[p] = None
    metadata = {
        "experiment": cli.experiment,
        "reward": config["reward"],
        "model": BASE_MODEL,
        "revision": BASE_REVISION,
        "runtime": versions,
        "gpu": torch.cuda.get_device_name(0),
        "data_sha256": {p.name: sha256(p.read_bytes()).hexdigest() for p in (train_path, val_path)},
        "train_count": len(train_set),
        "validation_ids": [row["id"] for row in val_rows],
        "arguments": vars(cli),
        "trainer": args.to_dict(),
        "lora": lora.to_dict(),
    }
    write_json(cli.output / "config.json", metadata)

    def calculate_score(ids, expected, tools):
        trimmed = list(ids)
        while trimmed and trimmed[-1] == tokenizer.pad_token_id and trimmed[-1] not in eos_ids:
            trimmed.pop()
        truncated = not trimmed or trimmed[-1] not in eos_ids
        text = tokenizer.decode(trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        return score_completion(text, expected, tools, truncated=truncated), text

    def toolace_reward(expected, tools, completion_ids, log_metric, **kwargs):
        scored = [
            calculate_score(ids, gold, schema)
            for ids, gold, schema in zip(completion_ids, expected, tools, strict=True)
        ]
        log_metric("toolace/strict", mean(score.strict for score, _ in scored))
        log_metric("toolace/shaped", mean(score.shaped for score, _ in scored))
        values = [getattr(score, config["reward"]) for score, _ in scored]
        if kwargs["trainer_state"].global_step == 0:
            with (cli.output / "first_rollouts.jsonl").open("a") as stream:
                for row_id, (score, raw), reward in zip(kwargs["id"], scored, values, strict=True):
                    stream.write(json.dumps({
                        "id": row_id, "completion": raw, "strict": score.strict,
                        "shaped": score.shaped, "reward": reward, "reason": score.reason, "calls": score.calls,
                    }, ensure_ascii=False) + "\n")
        return values

    class RunLog(TrainerCallback):
        def __init__(self):
            self.validation = []
            self.step_times = []
            self.best = None
            self.step_start = None
            self.nonzero_grad_steps = 0

        def on_log(self, args, state, control, logs=None, **kwargs):
            grad_norm = (logs or {}).get("grad_norm")
            self.nonzero_grad_steps += int(grad_norm is not None and math.isfinite(grad_norm) and grad_norm > 0)
            with (cli.output / "history.jsonl").open("a") as stream:
                stream.write(json.dumps({"step": state.global_step, **(logs or {})}) + "\n")

        def on_step_begin(self, args, state, control, **kwargs):
            self.step_start = time.monotonic()

        def on_step_end(self, args, state, control, model=None, **kwargs):
            if self.step_start is not None:
                torch.cuda.synchronize()
                self.step_times.append(time.monotonic() - self.step_start)
            step = state.global_step
            interval = cli.eval_every or max(1, cli.steps // 2)
            if step % interval == 0 or step == cli.steps:
                self.validate(model, step)

        @torch.inference_mode()
        def validate(self, wrapped_model, step):
            begin = time.monotonic()
            wrapped_model.eval()
            validation_config = GenerationConfig(
                do_sample=False,
                max_new_tokens=cli.max_completion,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=sorted(eos_ids),
                use_cache=True,
            )
            observations = []
            for begin_row in range(0, len(val_rows), cli.val_batch):
                batch = val_rows[begin_row : begin_row + cli.val_batch]
                model_inputs = tokenizer(
                    [row["prompt"] for row in batch], add_special_tokens=False,
                    return_tensors="pt", padding=True, return_token_type_ids=False,
                ).to(wrapped_model.device)
                generated = wrapped_model.generate(**model_inputs, generation_config=validation_config)
                generated = generated[:, model_inputs["input_ids"].size(1) :].tolist()
                for row, ids in zip(batch, generated, strict=True):
                    score, raw = calculate_score(ids, row["expected"], row["tools"])
                    observations.append({
                        "id": row["id"], "kind": row.get("kind"), "completion": raw,
                        "strict": score.strict, "shaped": score.shaped, "reason": score.reason,
                    })
            wrapped_model.train()
            strict_score = mean(row["strict"] for row in observations)
            summary = {
                "step": step,
                "strict": strict_score,
                "shaped": mean(row["shaped"] for row in observations),
                "n": len(observations),
                "seconds": time.monotonic() - begin,
            }
            self.validation.append(summary)
            write_json(cli.output / f"validation-{step}.json", {"summary": summary, "rows": observations})
            # Equal scores prefer the later scheduled checkpoint.
            if self.best is None or strict_score >= self.best["strict"]:
                wrapped_model.save_pretrained(cli.output / "best")
                self.best = summary
            print(f"ToolACE validation: {summary}", flush=True)

    logger = RunLog()
    trainer = GRPOTrainer(
        model=model,
        args=args,
        processing_class=tokenizer,
        reward_funcs=toolace_reward,
        train_dataset=train_set,
        peft_config=lora,
        callbacks=[logger],
    )
    trainer.model.print_trainable_parameters()
    train_result = trainer.train()
    if not logger.best:
        raise RuntimeError("No validation checkpoint was selected")
    if not math.isfinite(train_result.metrics["train_loss"]):
        raise RuntimeError("Training produced a nonfinite loss")
    write_json(cli.output / "summary.json", {
        "best": logger.best,
        "validation": logger.validation,
        "train": train_result.metrics,
        "nonzero_grad_steps": logger.nonzero_grad_steps,
        "step_seconds": logger.step_times,
        "steady_median_step_seconds": median(logger.step_times[1:] or logger.step_times),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "elapsed_seconds": time.monotonic() - start,
    })


if __name__ == "__main__":
    main()
