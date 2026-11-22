#!/usr/bin/env python3
"""Run HW7 notebook cells in separate single-GPU processes on the VM."""

import argparse
import ast
import json
import math
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile


COMMON = ("imports", "config", "load-dataset", "parse-gold", "cfb89075",
          "model-config", "model-functions", "metrics")
STUDENT_FILES = (
    "student_sft", "student_trials.csv", "student_128_dev.json",
    "student_128_dev_predictions.jsonl", "student_128_high_lr_dev.json",
    "student_128_high_lr_dev_predictions.jsonl", "student_1024_dev.json",
    "student_1024_dev_predictions.jsonl", "student_sft_dev.json",
    "student_sft_dev_predictions.jsonl",
)
DEV_METRICS = ("quality", "intent_accuracy", "slot_micro_f1", "frame_exact_match", "valid_json_rate")


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    with NamedTemporaryFile(mode="w", dir=path.parent, prefix=f".{path.name}.",
                            delete=False, encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")
    Path(file.name).replace(path)


def assigns(node, variable):
    return isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id == variable for target in node.targets
    )


class NotebookRun:
    def __init__(self, args, artifacts):
        self.args = args
        self.artifacts = artifacts
        self.artifacts.mkdir(parents=True, exist_ok=True)
        self.cells = {
            cell["id"]: "".join(cell["source"])
            for cell in read_json(args.notebook)["cells"] if cell["cell_type"] == "code"
        }
        self.namespace = {"__name__": "__main__", "ARTIFACTS": artifacts}
        self.started = time.monotonic()
        self.status_path = artifacts / f"run-{args.stage}.json"
        self.status = {"stage": args.stage, "started_at": now(), "status": "running", "cells": []}

    def update(self, **values):
        self.status.update(values, elapsed_seconds=round(time.monotonic() - self.started, 2))
        torch = self.namespace.get("torch")
        if torch is not None and torch.cuda.is_initialized():
            self.status["peak_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
            self.status["peak_reserved_gib"] = round(torch.cuda.max_memory_reserved() / 2**30, 3)
        write_json(self.status_path, self.status)
        if self.args.stage == "student" and self.artifacts != self.args.artifacts:
            write_json(self.args.artifacts / self.status_path.name, self.status)

    def execute(self, cell_id, include=None):
        filename = f"{self.args.notebook.name}:{cell_id}"
        source = self.cells[cell_id]
        tree = ast.parse(source, filename=filename)
        if include is not None:
            tree.body = [node for node in tree.body if include(node)]
        started = time.monotonic()
        print(f"{now()} START {cell_id}", flush=True)
        self.update(active_cell=cell_id)
        exec(compile(tree, filename, "exec"), self.namespace)
        elapsed = round(time.monotonic() - started, 2)
        self.status["cells"].append({"id": cell_id, "elapsed_seconds": elapsed})
        self.update(active_cell=None)
        print(f"{now()} FINISH {cell_id} {elapsed:.2f}s", flush=True)

    def prepare(self):
        for cell_id in COMMON:
            if cell_id == "config":
                self.execute(cell_id, lambda node: not assigns(node, "ARTIFACTS"))
            else:
                self.execute(cell_id)
        ns = self.namespace
        torch = ns["torch"]
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Set CUDA_VISIBLE_DEVICES to one GPU for this stage")
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
        self.update(gpu=torch.cuda.get_device_name(0))
        write_json(self.artifacts / "label_schema.json", {
            "intents": ns["INTENTS"], "slot_types": ns["SLOT_TYPES"],
        })
        if self.args.stage in {"smoke", "teacher-smoke"}:
            rows = ns["dev_data"]
            ns["dev_data"] = rows.select(range(min(self.args.dev_samples, len(rows))))

    def tokenizer(self):
        ns = self.namespace
        ns["tokenizer"] = ns["AutoTokenizer"].from_pretrained(
            ns["BASE_MODEL_ID"], cache_dir=self.args.base_cache,
        )


def checkpoint_ready(path):
    from safetensors import SafetensorError, safe_open

    try:
        if not isinstance(read_json(path / "config.json"), dict):
            return False
        index = path / "model.safetensors.index.json"
        if index.is_file():
            weight_map = read_json(index)["weight_map"]
            if not isinstance(weight_map, dict) or not weight_map:
                return False
            shards = set(weight_map.values())
        else:
            shards = {"model.safetensors"}
        for shard in shards:
            with safe_open(path / shard, framework="np") as file:
                if not file.keys():
                    return False
    except (OSError, ValueError, TypeError, KeyError, SafetensorError):
        return False
    return True


def load_records(path, expected_ids, value_key):
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        ids = [row["id"] for row in rows]
        values_valid = all(isinstance(row[value_key], str) for row in rows)
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise RuntimeError(f"Invalid or partial JSONL: {path}") from error
    if ids != list(expected_ids) or not values_valid:
        raise RuntimeError(f"Row IDs do not match the complete planned sample: {path}")
    return rows


def load_dev_result(artifacts, run, ids):
    metrics_path = artifacts / f"{run}_dev.json"
    predictions_path = artifacts / f"{run}_dev_predictions.jsonl"
    try:
        metrics = read_json(metrics_path)
        valid = metrics["n"] == len(ids) and all(0.0 <= metrics[name] <= 1.0 for name in DEV_METRICS)
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise RuntimeError(f"Invalid or partial metrics: {metrics_path}") from error
    if not valid:
        raise RuntimeError(f"Metrics do not match the planned dev sample: {metrics_path}")
    rows = load_records(predictions_path, ids, "prediction")
    return metrics, [row["prediction"] for row in rows]


def baseline(run):
    ns = run.namespace
    path = run.args.baseline or run.artifacts / "baseline"
    path.parent.mkdir(parents=True, exist_ok=True)
    ns["BASELINE_PATH"] = path
    metrics_path = run.artifacts / "baseline_dev.json"
    predictions_path = run.artifacts / "baseline_dev_predictions.jsonl"

    if checkpoint_ready(path):
        run.tokenizer()
        if metrics_path.is_file() and predictions_path.is_file():
            ns["baseline_metrics"], ns["baseline_predictions"] = load_dev_result(
                run.artifacts, "baseline", ns["dev_data"]["id"],
            )
            ns["results"] = [{
                "run": "baseline", "path": str(path), "method": "dense",
                "calibration": None, **ns["baseline_metrics"],
            }]
            return
        ns["baseline_model"] = ns["AutoModelForCausalLM"].from_pretrained(
            path, torch_dtype=ns["torch"].bfloat16, device_map=ns["DEVICE"],
        ).eval()
    else:
        run.execute("load-baseline", lambda node: not assigns(node, "BASELINE_PATH"))
    run.execute("evaluate-baseline")
    ns["save_dev"]("baseline", ns["baseline_metrics"], ns["baseline_predictions"])


def validate_quant_checkpoint(run, name):
    ns = run.namespace
    path = run.artifacts / name
    if not checkpoint_ready(path):
        raise RuntimeError(f"Quantized checkpoint is incomplete: {name}")
    config = read_json(path / "config.json").get("quantization_config", {})
    if config.get("quantization_status") != "compressed":
        raise RuntimeError(f"Checkpoint weights are not saved compressed: {name}")
    if ns["checkpoint_gib"](path) >= ns["checkpoint_gib"](ns["BASELINE_PATH"]):
        raise RuntimeError(f"Quantized weights are not smaller than BF16: {name}")
    load_dev_result(run.artifacts, name, ns["dev_data"]["id"])


def quantization(run, *, smoke=False):
    ns = run.namespace
    baseline(run)
    if smoke:
        ns["CAL_SAMPLES"] = run.args.cal_samples

    cell_id = "a8d5351b"
    def quant_setup(node):
        return not isinstance(node, ast.For) and not (
            smoke and assigns(node, "CAL_SAMPLES")
        ) and not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                   and isinstance(node.value.func, ast.Name) and node.value.func.id == "display")

    run.execute(cell_id, quant_setup)
    configurations = ns["runs"]
    if run.args.quant_run is not None:
        configurations = [row for row in configurations if row[0] == run.args.quant_run]
        if not configurations:
            raise ValueError(f"Unknown quant configuration {run.args.quant_run!r}")
    if smoke:
        configurations = configurations[:1]
        gptq = ns["GPTQModifier"]

        def first_block_gptq(**options):
            options["targets"] = [r"re:^model\.layers\.0\..*_proj$"]
            return gptq(**options)

        ns["GPTQModifier"] = first_block_gptq

    table_path = run.artifacts / "quantization_dev.csv"
    if table_path.is_file():
        existing = ns["pd"].read_csv(table_path)
        ns["results"].extend(existing.loc[existing["run"] != "baseline"].to_dict("records"))
    completed = {row["run"] for row in ns["results"]}
    for configuration in configurations:
        name = configuration[0]
        if name in completed:
            validate_quant_checkpoint(run, name)
            print(f"{now()} SKIP complete {name}", flush=True)
            continue
        ns["runs"] = [configuration]
        print(f"{now()} QUANT {name}", flush=True)
        run.update(active_configuration=name)
        path = run.artifacts / name
        if checkpoint_ready(path):
            print(f"{now()} REUSE saved weights {name}", flush=True)
            ns["record_quant_result"](*configuration, path)
        else:
            run.execute(cell_id, lambda node: isinstance(node, ast.For))
        validate_quant_checkpoint(run, name)
    run.update(active_configuration=None)


def teacher(run, *, smoke=False):
    ns = run.namespace
    run.tokenizer()
    run.execute("b2ad65a4")
    if run.args.base_cache is not None:
        ns["student_vocab_size"] = ns["AutoConfig"].from_pretrained(
            ns["STUDENT_MODEL_ID"], cache_dir=run.args.base_cache,
        ).vocab_size
    run.execute("6c3b2c0d", lambda node: not (
        run.args.base_cache is not None and assigns(node, "student_vocab_size")
    ))

    baseline_metrics = run.args.artifacts / "baseline_dev.json"
    if baseline_metrics.is_file():
        ns["baseline_metrics"] = read_json(baseline_metrics)
    if (run.artifacts / "teacher_dev.json").is_file() and (
        run.artifacts / "teacher_dev_predictions.jsonl"
    ).is_file():
        load_dev_result(run.artifacts, "teacher", ns["dev_data"]["id"])
    else:
        run.execute("e96d949e", lambda node: "baseline_metrics" in ns or not (
            isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name) and node.value.func.id == "display"
        ))
    if smoke:
        return

    extra_path = run.artifacts / "teacher_answers_extra"
    if (extra_path / "answers.jsonl").is_file():
        run.execute("65248c62", lambda node: not assigns(node, "extra_teacher_answers"))
        load_records(extra_path / "answers.jsonl", ns["extra_rows"]["id"], "teacher_answer")
        ns["extra_teacher_answers"] = extra_path
    else:
        run.execute("65248c62")
    pilot_path = run.artifacts / "teacher_answers"
    if not (pilot_path / "answers.jsonl").is_file():
        run.execute("1ac579de")
    else:
        run.execute("1ac579de", lambda node: assigns(node, "distill_rows"))
        load_records(pilot_path / "answers.jsonl", ns["distill_rows"]["id"], "teacher_answer")
        del ns["teacher"]
        ns["clean_memory"]()


def prepare_student(run, *, smoke=False):
    ns = run.namespace
    run.tokenizer()
    ns["STUDENT_MODEL_ID"] = ns["BASE_MODEL_ID"]
    if smoke:
        fixture = run.artifacts / "synthetic_sft_fixture"
        fixture.mkdir(exist_ok=True)
        rows = ns["train_data"].select(range(8))
        with (fixture / "answers.jsonl").open("w", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps({
                    "teacher_answer": row["gold"],
                    "prompt_ids": ns["encode_prompt"](ns["tokenizer"], row["text"]),
                    "answer_ids": ns["tokenizer"].encode(row["gold"], add_special_tokens=False),
                }) + "\n")
        ns["TEACHER_ANSWERS"] = fixture
    else:
        ns["TEACHER_ANSWERS"] = run.args.artifacts / "teacher_answers"
        ns["extra_teacher_answers"] = run.args.artifacts / "teacher_answers_extra"
        ns["baseline_metrics"] = read_json(run.args.artifacts / "baseline_dev.json")
        ns["teacher_metrics"] = read_json(run.args.artifacts / "teacher_dev.json")
    run.execute("b0fa8cc9")


def student(run, *, smoke=False):
    ns = run.namespace
    prepare_student(run, smoke=smoke)
    run.execute("73d6086d")
    if smoke:
        trainer = ns["sft"]
        trainer.args.max_steps = 1
        trainer.args.warmup_ratio = 0.0
        head = trainer.model.get_output_embeddings().weight
        before = head.detach().view(-1)[:65536].cpu().clone()
        started = time.monotonic()
        result = trainer.train()
        if trainer.state.global_step != 1:
            raise RuntimeError("Smoke test did not complete one SFT optimizer step")
        if not math.isfinite(result.training_loss):
            raise RuntimeError("Smoke SFT loss is not finite")
        after = head.detach().view(-1)[:65536].cpu()
        if ns["torch"].equal(before, after):
            raise RuntimeError("Smoke SFT did not update the sampled LM-head parameters")
        run.update(smoke_optimizer_step_seconds=round(time.monotonic() - started, 2),
                   smoke_loss=result.training_loss, smoke_parameters_changed=True)
        return

    run.execute("30e7767b")
    ns["save_dev"]("student_128", ns["sft_metrics"], ns["_"])
    run.execute("a9d3342d")

    if run.artifacts != run.args.artifacts:
        for name in STUDENT_FILES:
            source = run.artifacts / name
            destination = run.args.artifacts / name
            temporary = destination.with_name(destination.name + ".publishing")
            if source.is_dir():
                shutil.copytree(source, temporary)
            else:
                shutil.copy2(source, temporary)
            temporary.replace(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, default=Path("homework_compression_student.ipynb"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts_compression_hw"))
    parser.add_argument("--baseline", type=Path, help="Dense checkpoint directory shared between stages")
    parser.add_argument("--base-cache", type=Path, help="8B tokenizer/config cache separate from teacher weights")
    parser.add_argument("--student-work", type=Path, help="Temporary student artifact directory on a roomy filesystem")
    parser.add_argument("--dev-samples", type=int, default=8, help="Dev rows in smoke stages")
    parser.add_argument("--cal-samples", type=int, default=4, help="Calibration rows in smoke")
    parser.add_argument("--quant-run", help="Run only the named quantization configuration")
    parser.add_argument("stage", choices=("smoke", "teacher-smoke", "quant", "teacher", "student"))
    args = parser.parse_args()
    if min(args.dev_samples, args.cal_samples) <= 0:
        parser.error("Smoke sample counts must be positive")

    args.artifacts = args.artifacts.resolve()
    args.artifacts.mkdir(parents=True, exist_ok=True)
    artifacts = args.artifacts
    if args.stage in {"smoke", "teacher-smoke"}:
        artifacts /= args.stage
    elif args.stage == "student" and args.student_work is not None:
        artifacts = args.student_work.resolve()
    run = NotebookRun(args, artifacts)
    run.update()
    try:
        run.prepare()
        if args.stage == "smoke":
            quantization(run, smoke=True)
            student(run, smoke=True)
        elif args.stage == "teacher-smoke":
            teacher(run, smoke=True)
        elif args.stage == "quant":
            quantization(run)
        elif args.stage == "teacher":
            teacher(run)
        else:
            student(run)
    except (Exception, KeyboardInterrupt) as error:
        run.update(status="failed", finished_at=now(), error=f"{type(error).__name__}: {error}")
        raise
    else:
        run.update(status="completed", finished_at=now())


if __name__ == "__main__":
    main()
