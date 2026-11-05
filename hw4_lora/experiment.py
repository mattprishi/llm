"""Train and evaluate one Banking77 configuration."""

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SEED = 42
TRAIN_BATCH = 8
GEN_BATCH = 16
LORA = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "v_proj"],
}
CONFIGS = {
    "base": None,
    "full": None,
    "lora": LORA,
    "rank": dict(LORA, r=32),
    "alpha": dict(LORA, lora_alpha=64),
    "dropout": dict(LORA, lora_dropout=0.0),
    "targets": dict(LORA, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]),
}


def write_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def tokenize(tokenizer, rows, instruction):
    encoded = []
    for row in rows:
        messages = [{
            "role": "user",
            "content": instruction + "\n\nCustomer message:\n" + row["question"].strip(),
        }]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
        )
        full = tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": row["solution"]}],
            tokenize=True, add_generation_prompt=False,
        )
        if full[:len(prompt)] != prompt:
            raise ValueError(f"Prompt is not an exact token prefix for {row['id']}")
        try:
            end = full.index(tokenizer.eos_token_id, len(prompt))
        except ValueError as exc:
            raise ValueError(f"Missing assistant EOS for {row['id']}") from exc
        if end == len(prompt):
            raise ValueError(f"Empty assistant target for {row['id']}")
        encoded.append((prompt, full[:end + 1]))
    return encoded


def batches(rows, size):
    for start in range(0, len(rows), size):
        yield rows[start:start + size]


def collate(rows, pad_id):
    import torch

    width = max(len(full) for _, full in rows)
    width = (width + 7) // 8 * 8
    input_ids, masks, labels = [], [], []
    for prompt, full in rows:
        padding = width - len(full)
        input_ids.append(full + [pad_id] * padding)
        masks.append([1] * len(full) + [0] * padding)
        labels.append([-100] * len(prompt) + full[len(prompt):] + [-100] * padding)
    return {
        "input_ids": torch.tensor(input_ids, device="cuda"),
        "attention_mask": torch.tensor(masks, device="cuda"),
        "labels": torch.tensor(labels, device="cuda"),
    }


def train(model, rows, pad_id, lr, pilot_steps):
    import torch
    from transformers import get_linear_schedule_with_warmup

    steps = math.ceil(len(rows) / TRAIN_BATCH)
    order = random.Random(SEED).sample(range(len(rows)), len(rows))
    ordered = [rows[i] for i in order]
    params = [param for param in model.parameters() if param.requires_grad]
    model.train()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.01, fused=True)
    scheduler = get_linear_schedule_with_warmup(optimizer, max(1, steps // 20), steps)
    history, durations = [], []
    for step, examples in enumerate(batches(ordered, TRAIN_BATCH), start=1):
        step_start = time.perf_counter()
        batch = collate(examples, pad_id)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(**batch, use_cache=False).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        scheduler.step()
        loss_value = float(loss.detach())
        if not math.isfinite(loss_value):
            raise RuntimeError(f"Non-finite loss at step {step}: {loss_value}")
        durations.append(time.perf_counter() - step_start)
        history.append({"step": step, "loss": loss_value, "next_lr": scheduler.get_last_lr()[0]})
        if step == 1 or step % 10 == 0 or step == steps or step == pilot_steps:
            print(f"train step {step}/{steps}: loss={loss_value:.4f}", flush=True)
        if pilot_steps == step:
            break

    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    result = dict(
        steps=len(history), planned_steps=steps, train_seconds=seconds,
        steady_seconds_per_step=sum(durations[3:] or durations) / len(durations[3:] or durations),
        peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
    )
    optimizer.zero_grad(set_to_none=True)
    return result, history


def generate_predictions(model, tokenizer, encoded, rows, max_new_tokens):
    import torch

    pad = tokenizer.pad_token_id
    eos = model.generation_config.eos_token_id
    stop_ids = {eos} if isinstance(eos, int) else set(eos or [])
    count = 0
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for examples, batch_rows in zip(batches(encoded, GEN_BATCH), batches(rows, GEN_BATCH)):
            prompts = [prompt for prompt, _ in examples]
            width = max(map(len, prompts))
            ids = [[pad] * (width - len(prompt)) + prompt for prompt in prompts]
            masks = [[0] * (width - len(prompt)) + [1] * len(prompt) for prompt in prompts]
            output = model.generate(
                input_ids=torch.tensor(ids, device="cuda"),
                attention_mask=torch.tensor(masks, device="cuda"),
                do_sample=False, num_beams=1, use_cache=True,
                max_new_tokens=max_new_tokens, pad_token_id=pad,
            )[:, width:].tolist()
            for row, generated in zip(batch_rows, output):
                text = tokenizer.decode(generated, skip_special_tokens=True).strip()
                capped = len(generated) == max_new_tokens and stop_ids.isdisjoint(generated)
                yield row, text, capped
            count += len(batch_rows)
            if count % 64 == 0 or count == len(rows):
                print(f"generated {count}/{len(rows)}", flush=True)


def run(args):
    import torch
    from importlib.metadata import version
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise SystemExit("Expose one BF16 GPU with CUDA_VISIBLE_DEVICES=1.")
    raw = args.data.read_bytes()
    data = json.loads(raw)
    if not data["train"] or not data["test"]:
        raise ValueError("Train and test must be nonempty")
    steps = math.ceil(len(data["train"]) / TRAIN_BATCH)
    if not 0 <= args.pilot_steps <= steps:
        raise SystemExit(f"--pilot-steps must be in [0, {steps}].")
    pilot = args.pilot_steps > 0
    test_rows = data["test"][:32] if pilot else data["test"]
    set_seed(SEED)
    revision = data["model_revision"]
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=revision)
    from banking import DATASET, MAX_LABEL_TOKENS, evaluate_labels, label_prompt

    if data["dataset"] != DATASET:
        raise ValueError("Expected a Banking77 split")
    instruction = label_prompt(data["labels"])
    test_encoded = tokenize(tokenizer, test_rows, instruction)
    train_encoded = tokenize(tokenizer, data["train"], instruction) if args.name != "base" else []
    out = args.output / args.name
    out.mkdir(parents=True, exist_ok=False)
    lora = CONFIGS[args.name]
    lr = None if args.name == "base" else (2e-5 if args.name == "full" else 2e-4)
    config = dict(
        run=args.name, pilot=pilot, pilot_steps=args.pilot_steps, seed=SEED,
        model=MODEL, model_revision=revision, dataset=data["dataset"],
        dataset_revision=data["dataset_revision"], data_sha256=hashlib.sha256(raw).hexdigest(),
        lora=lora, lr=lr, epochs=1 if train_encoded else 0, train_batch=TRAIN_BATCH,
        generation_batch=GEN_BATCH, max_new_tokens=MAX_LABEL_TOKENS,
        generation="greedy", optimizer="fused AdamW", weight_decay=0.01, max_grad_norm=1.0,
        schedule="linear, 5% warmup rounded down", attention="sdpa", autocast="bf16",
        train_max_tokens=max((len(full) for _, full in train_encoded), default=None),
        test_max_tokens=max(len(full) for _, full in test_encoded),
        weight_dtype="float32", cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        gpu=torch.cuda.get_device_name(), cuda=torch.version.cuda,
        versions={name: version(name) for name in ("torch", "transformers", "peft",
                                                   "accelerate", "huggingface-hub")},
    )
    write_json(out / "config.json", config)

    # Match storage dtype across methods. Forward uses BF16 autocast.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=revision, dtype=torch.float32, attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to("cuda")
    if lora is not None:
        model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", bias="none", **lora))
    if args.name == "base":
        model.requires_grad_(False)

    summary = dict(
        config,
        total_parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
    if train_encoded:
        stats, history = train(model, train_encoded, tokenizer.pad_token_id, lr, args.pilot_steps)
        summary.update(stats)
        write_json(out / "history.json", history)
    torch.cuda.empty_cache()
    metrics, predictions = evaluate_labels(model, tokenizer, test_encoded, test_rows, data["labels"])
    summary.update(metrics)
    if args.save_model and train_encoded and not pilot:
        model.save_pretrained(out / "model")
        tokenizer.save_pretrained(out / "model")
    write_json(out / "predictions.json", predictions)
    write_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", choices=CONFIGS)
    parser.add_argument("--data", type=Path, default=Path("data/banking77.json"))
    parser.add_argument("--output", type=Path, default=Path("runs"))
    parser.add_argument("--pilot-steps", type=int, default=0)
    parser.add_argument("--save-model", action="store_true")
    run(parser.parse_args())



if __name__ == "__main__":
    main()
