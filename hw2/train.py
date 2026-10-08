import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

import torch
import transformers
import trl
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerBase,
    Qwen3Config,
    Qwen3ForCausalLM,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    set_seed,
)
from transformers.utils import ModelOutput
from trl import SFTConfig, SFTTrainer
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling


DATA: Path = Path("data")
DEFAULT_TRAIN_SECONDS: float = 300
DEFAULT_ATTENTION_IMPLEMENTATION: str = "sdpa"

MODEL_CONFIG: dict[str, Any] = dict(
    hidden_size=2048,
    num_hidden_layers=12,
    num_attention_heads=16,
    num_key_value_heads=8,
    intermediate_size=8192,
    head_dim=128,
    use_cache=False,
)

TRAINING_CONFIG: dict[str, Any] = dict(
    output_dir="results/baseline",
    # Настройки для экспериментов.
    per_device_train_batch_size=32,
    gradient_accumulation_steps=2,
    gradient_checkpointing=False,
    torch_compile=False,
    use_liger_kernel=False,
    activation_offloading=False,
    packing=False,
    padding_free=False,
    # Общие параметры обучения и замера.
    max_steps=1_000_000,
    learning_rate=3e-4,
    lr_scheduler_type="constant_with_warmup",
    warmup_steps=200,
    weight_decay=0.01,
    optim="adamw_torch_fused",
    bf16=True,
    tf32=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    packing_strategy="bfd",
    max_length=512,
    completion_only_loss=False,
    loss_type="nll",
    dataloader_num_workers=2,
    dataloader_pin_memory=True,
    logging_steps=10,
    logging_nan_inf_filter=False,
    report_to="none",
    save_strategy="no",
    eval_strategy="no",
    seed=42,
    data_seed=42,
    disable_tqdm=True,
)


# Only the optimization knobs named in the assignment are changed between runs.
# The model, data, split, max length, precision, optimizer, LR, scheduler, seed,
# and metric calculation remain identical to the baseline template.
EXPERIMENTS: dict[str, dict[str, Any]] = {
    "baseline": {},
    "batch64": {
        "per_device_train_batch_size": 64,
        "gradient_accumulation_steps": 1,
    },
    "checkpointing": {"gradient_checkpointing": True},
    "flash_attention_2": {"attention_implementation": "flash_attention_2"},
    "torch_compile": {"torch_compile": True},
    "liger": {"use_liger_kernel": True},
    "activation_offloading": {"activation_offloading": True},
    "fa2_padding_free": {
        "attention_implementation": "flash_attention_2",
        "padding_free": True,
    },
    "fa2_packing": {
        "attention_implementation": "flash_attention_2",
        "packing": True,
    },
    "batch64_fa2_packing": {
        "attention_implementation": "flash_attention_2",
        "per_device_train_batch_size": 64,
        "gradient_accumulation_steps": 1,
        "packing": True,
    },
    "batch64_fa2_packing_liger": {
        "attention_implementation": "flash_attention_2",
        "per_device_train_batch_size": 64,
        "gradient_accumulation_steps": 1,
        "packing": True,
        "use_liger_kernel": True,
    },
}

FIXED_TRAINING_KEYS = {
    "max_steps",
    "learning_rate",
    "lr_scheduler_type",
    "warmup_steps",
    "weight_decay",
    "optim",
    "bf16",
    "tf32",
    "max_length",
    "completion_only_loss",
    "loss_type",
    "seed",
    "data_seed",
}


def resolve_run() -> tuple[str, str, float, str, dict[str, Any]]:
    experiment = os.environ.get("EXPERIMENT", "baseline")
    if experiment not in EXPERIMENTS:
        choices = ", ".join(EXPERIMENTS)
        raise ValueError(f"Unknown EXPERIMENT={experiment!r}; choose one of: {choices}")

    run_name = os.environ.get("RUN_NAME", experiment)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_name):
        raise ValueError("RUN_NAME may contain only letters, digits, '.', '_' and '-'")

    seconds = float(os.environ.get("TRAIN_SECONDS", str(DEFAULT_TRAIN_SECONDS)))
    if seconds <= 0:
        raise ValueError("TRAIN_SECONDS must be positive")

    overrides = EXPERIMENTS[experiment]
    forbidden = FIXED_TRAINING_KEYS.intersection(overrides)
    if forbidden:
        raise ValueError(f"Experiment changes fixed settings: {sorted(forbidden)}")

    attention = overrides.get(
        "attention_implementation", DEFAULT_ATTENTION_IMPLEMENTATION
    )
    config = dict(TRAINING_CONFIG)
    config.update(
        {key: value for key, value in overrides.items() if key != "attention_implementation"}
    )
    results_root = Path(os.environ.get("RESULTS_ROOT", "results"))
    config["output_dir"] = str(results_root / run_name)
    return experiment, run_name, seconds, attention, config


def prepare_data() -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        "ai-forever/rugpt3small_based_on_gpt2",
        revision="a9307e696cd3c5b7f953ff4cb19d76a4d81821d5",
    )
    tokenizer.pad_token = tokenizer.eos_token
    source = load_dataset(
        "wikimedia/wikipedia",
        "20231101.ru",
        split="train",
        streaming=True,
        revision="b04c8d1ceb2f5cd4588862100d08de323dccfbaa",
    )
    rows: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    for index, article in enumerate(source.take(50_500)):
        split = "validation" if index < 500 else "train"
        paragraphs = [
            p.strip() for p in article["text"].splitlines() if len(p.strip()) >= 80
        ][:8]
        if paragraphs:
            encoded = tokenizer(
                paragraphs,
                add_special_tokens=False,
                truncation=True,
                max_length=511,
                return_attention_mask=False,
            )["input_ids"]
            for paragraph_index, tokens in enumerate(encoded):
                if len(tokens) >= 32:
                    rows[split].append(
                        dict(
                            input_ids=tokens + [tokenizer.eos_token_id],
                            article_id=article["id"],
                            paragraph_index=paragraph_index,
                        )
                    )
        if (index + 1) % 5000 == 0:
            print(f"Подготовлено {index + 1}/50500 статей", flush=True)
    DatasetDict(
        {name: Dataset.from_list(items) for name, items in rows.items()}
    ).save_to_disk(str(DATA / "dataset"))
    tokenizer.save_pretrained(DATA / "tokenizer")


class CausalCollator(DataCollatorForLanguageModeling):
    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch = super().torch_call(examples)
        batch["labels"][:, 0] = -100
        return batch


class BenchmarkTrainer(SFTTrainer):
    target_tokens: torch.Tensor

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, ModelOutput]:
        return Trainer.compute_loss(
            self,
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )

    def training_step(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        num_items_in_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.target_tokens += inputs["labels"].ne(-100).sum()
        return super().training_step(model, inputs, num_items_in_batch)

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        for key in ("loss", "grad_norm", "train_loss"):
            if key in logs and not math.isfinite(logs[key]):
                raise FloatingPointError(f"Значение {key} не валидное")
        logs.pop("train_samples_per_second", None)
        logs.pop("train_steps_per_second", None)
        super().log(logs, start_time)


class Benchmark(TrainerCallback):
    start: float
    previous_end: float
    previous_tokens: int

    def __init__(self, trainer: BenchmarkTrainer, seconds: float) -> None:
        self.trainer, self.seconds = trainer, seconds
        self.steps: list[dict[str, float | int | bool]] = []

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.start = self.previous_end = time.perf_counter()
        self.previous_tokens = 0

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        torch.cuda.synchronize()
        now = time.perf_counter()
        tokens = self.trainer.target_tokens.item()
        record: dict[str, float | int | bool] = dict(
            step=state.global_step,
            elapsed_seconds=now - self.start,
            step_seconds=now - self.previous_end,
            target_tokens=tokens - self.previous_tokens,
            warmup=state.global_step <= 10,
        )
        self.steps.append(record)
        if state.global_step % 10 == 0:
            print(json.dumps(record), flush=True)
        self.previous_end, self.previous_tokens = now, tokens
        if now - self.start >= self.seconds:
            control.should_training_stop = True

    def summary(self) -> dict[str, float | int | None]:
        measured = self.steps[10:]
        duration = sum(step["step_seconds"] for step in measured)
        tokens = sum(step["target_tokens"] for step in measured)
        elapsed = self.previous_end - self.start
        return dict(
            train_seconds=elapsed,
            optimizer_steps=len(self.steps),
            target_tokens=self.previous_tokens,
            tokens_per_second_including_warmup=self.previous_tokens / elapsed,
            warmup_steps=10,
            measured_steps=len(measured),
            measured_seconds=duration,
            tokens_per_second=tokens / duration if measured else None,
            seconds_per_optimizer_step=duration / len(measured) if measured else None,
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
        )


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, float | int]:
    loader = DataLoader(
        dataset.select_columns(["input_ids"]),
        batch_size=8,
        collate_fn=CausalCollator(tokenizer.pad_token_id, completion_only_loss=False),
    )
    model.eval()
    loss_sum, tokens = 0.0, 0
    for batch in loader:
        batch = {key: value.cuda() for key, value in batch.items()}
        count = batch["labels"].ne(-100).sum().item()
        loss_sum += model(**batch).loss.item() * count
        tokens += count
    loss = loss_sum / tokens

    return dict(validation_loss=loss, validation_target_tokens=tokens)


def runtime_metadata(model: torch.nn.Module) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(0)
    return {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "transformers_version": transformers.__version__,
        "trl_version": trl.__version__,
        "gpu_name": properties.name,
        "gpu_total_memory_gib": properties.total_memory / 2**30,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
    }


def main() -> None:
    experiment, run_name, train_seconds, attention, config_dict = resolve_run()
    training_args = SFTConfig(**config_dict)

    output = Path(training_args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    if not (DATA / "dataset").exists():
        prepare_data()
    tokenizer = AutoTokenizer.from_pretrained(DATA / "tokenizer")
    dataset = load_from_disk(str(DATA / "dataset"))

    set_seed(training_args.seed)
    config = Qwen3Config(
        vocab_size=tokenizer.vocab_size,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        **MODEL_CONFIG,
    )
    model = Qwen3ForCausalLM._from_config(
        config,
        attn_implementation=attention,
        torch_dtype=torch.bfloat16,
    )

    print(
        json.dumps(
            {
                "experiment": experiment,
                "run_name": run_name,
                "train_seconds": train_seconds,
                "attention_implementation": attention,
                "training_config": config_dict,
            },
            indent=2,
        ),
        flush=True,
    )

    trainer = BenchmarkTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        processing_class=tokenizer,
        data_collator=(
            None
            if training_args.packing or training_args.padding_free
            else CausalCollator(tokenizer.pad_token_id)
        ),
    )
    trainer.target_tokens = torch.zeros((), dtype=torch.int64, device="cuda")
    benchmark = Benchmark(trainer, train_seconds)
    trainer.add_callback(benchmark)

    trainer.train()
    summary = benchmark.summary()
    summary.update(
        evaluate(trainer.model, dataset["validation"].select(range(512)), tokenizer)
    )
    summary.update(
        {
            "experiment": experiment,
            "run_name": run_name,
            "attention_implementation": attention,
            "effective_batch_size": (
                training_args.per_device_train_batch_size
                * training_args.gradient_accumulation_steps
            ),
            "effective_padding_free": bool(
                training_args.padding_free
                or (
                    training_args.packing
                    and training_args.packing_strategy == "bfd"
                )
            ),
            "training_config": config_dict,
            "model_config": MODEL_CONFIG,
            "train_examples": len(dataset["train"]),
            "validation_examples_evaluated": 512,
            "runtime": runtime_metadata(trainer.model),
        }
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
