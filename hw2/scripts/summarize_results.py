#!/usr/bin/env python3
"""Aggregate successful training summaries into CSV and Markdown tables."""

import csv
import json
from pathlib import Path
from typing import Any


RESULTS = Path("results")
COLUMNS = [
    "run_name",
    "experiment",
    "train_seconds",
    "batch_size",
    "gradient_accumulation",
    "effective_batch_size",
    "attention",
    "packing",
    "padding_free",
    "checkpointing",
    "compile",
    "liger",
    "activation_offloading",
    "tokens_per_second",
    "speedup_vs_baseline",
    "peak_allocated_gib",
    "peak_reserved_gib",
    "validation_loss",
    "optimizer_steps",
]


def load_rows() -> list[dict[str, Any]]:
    summaries = []
    for path in sorted(RESULTS.glob("*/summary.json")):
        summary = json.loads(path.read_text())
        config = summary["training_config"]
        summaries.append(
            {
                "run_name": summary["run_name"],
                "experiment": summary["experiment"],
                "train_seconds": summary["train_seconds"],
                "batch_size": config["per_device_train_batch_size"],
                "gradient_accumulation": config["gradient_accumulation_steps"],
                "effective_batch_size": summary["effective_batch_size"],
                "attention": summary["attention_implementation"],
                "packing": config["packing"],
                "padding_free": summary["effective_padding_free"],
                "checkpointing": config["gradient_checkpointing"],
                "compile": config["torch_compile"],
                "liger": config["use_liger_kernel"],
                "activation_offloading": config["activation_offloading"],
                "tokens_per_second": summary["tokens_per_second"],
                "peak_allocated_gib": summary["peak_allocated_gib"],
                "peak_reserved_gib": summary["peak_reserved_gib"],
                "validation_loss": summary["validation_loss"],
                "optimizer_steps": summary["optimizer_steps"],
            }
        )

    baseline = next(
        (row for row in summaries if row["run_name"] == "baseline_300s"), None
    )
    if baseline is None:
        baseline = next(
            (row for row in summaries if row["experiment"] == "baseline"), None
        )
    baseline_speed = baseline["tokens_per_second"] if baseline else None
    for row in summaries:
        row["speedup_vs_baseline"] = (
            row["tokens_per_second"] / baseline_speed if baseline_speed else None
        )
    return summaries


def display(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    rows = load_rows()
    RESULTS.mkdir(exist_ok=True)
    with (RESULTS / "experiments.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    header = "| " + " | ".join(COLUMNS) + " |"
    divider = "| " + " | ".join("---" for _ in COLUMNS) + " |"
    lines = [header, divider]
    lines.extend(
        "| " + " | ".join(display(row[column]) for column in COLUMNS) + " |"
        for row in rows
    )
    (RESULTS / "experiments.md").write_text("\n".join(lines) + "\n")
    print(f"Collected {len(rows)} successful runs in {RESULTS / 'experiments.md'}")


if __name__ == "__main__":
    main()
