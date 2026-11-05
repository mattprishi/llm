"""Check a completed suite and write the benchmark table and manual review."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from banking import DATASET, score_labels
from experiment import CONFIGS, TRAIN_BATCH, write_json


def read_json(path):
    return json.loads(path.read_bytes())


def summarize(data_path, runs):
    raw = data_path.read_bytes()
    data = json.loads(raw)
    digest = hashlib.sha256(raw).hexdigest()
    test = {row["id"]: row for row in data["test"]}
    steps = math.ceil(len(data["train"]) / TRAIN_BATCH)
    if data["dataset"] != DATASET:
        raise ValueError("Expected a Banking77 split")
    allowed = {label.casefold(): label for label in data["labels"]}
    metrics, predictions = {}, {}
    for name, lora in CONFIGS.items():
        directory = runs / name
        summary = read_json(directory / "summary.json")
        rows = read_json(directory / "predictions.json")
        if (summary["run"] != name or summary["pilot"] or summary["lora"] != lora
                or summary["data_sha256"] != digest or summary["test_examples"] != len(test)
                or [row["id"] for row in rows] != list(test)
                or summary["correct"] != sum(row["correct"] for row in rows)
                or summary.get("steps", 0) != (0 if name == "base" else steps)):
            raise ValueError(f"Inconsistent final results in {directory}")
        for row in rows:
            gold = test[row["id"]]["gold"]
            predicted = allowed.get(row["prediction"].strip().casefold())
            if (row["gold_label"], row["predicted_label"], row["correct"]) != (
                gold, predicted, predicted == gold,
            ):
                raise ValueError(f"Inconsistent label scoring in {directory}: {row['id']}")
        for key, value in score_labels(rows, data["labels"]).items():
            matches = (math.isclose(summary[key], value, abs_tol=1e-12, rel_tol=0)
                       if key == "macro_f1" else summary[key] == value)
            if not matches:
                raise ValueError(f"Inconsistent {key} in {directory}")
        metrics[name] = summary
        predictions[name] = {row["id"]: row for row in rows}

    lines = [
        "# HW4. Результаты",
        "",
        f"Train — {len(data['train'])}, test — {len(test)}, seed — {data['seed']}.",
        "",
        "| Модель | Верно | Accuracy | Macro-F1 | Извлечено | Достигли лимита "
        "| Обучаемые параметры | Train, с | Allocated, GiB | Reserved, GiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in metrics.items():
        costs = "— | — | —" if name == "base" else (
            f"{row['train_seconds']:.1f} | {row['peak_allocated_gib']:.2f} | "
            f"{row['peak_reserved_gib']:.2f}"
        )
        secondary = f"{row['macro_f1']:.2%}"
        lines.append(
            f"| {name} | {row['correct']}/{len(test)} | {row['accuracy']:.1%} | "
            f"{secondary} | {row['parsed']} | {row['capped']} | "
            f"{row['trainable_parameters']:,} | {costs} |"
        )

    review = ["# Ручная оценка", "", "Сверяем интент с запросом клиента. "
              "Порядок задач зафиксирован до обучения."]
    for row_id in data["manual_ids"]:
        row = test[row_id]
        review.extend(["", f"## {row_id}", "", row["question"], "",
                       "**Эталон**", "", "```text", row["solution"], "```"])
        for name in ("base", "full", "lora"):
            response = predictions[name][row_id]["prediction"]
            review.extend(["", f"**{name}**", "", "```text", response, "```",
                           "", "Ручная оценка — …"])

    write_json(runs / "metrics.json", metrics)
    (runs / "benchmark.md").write_text("\n".join(lines) + "\n")
    (runs / "manual_review.md").write_text("\n".join(review) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/banking77.json"))
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    args = parser.parse_args()
    summarize(args.data, args.runs)
