"""Prepare the Banking77 split and score generated intent labels."""

import argparse
import csv
import hashlib
import io
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from urllib.request import urlopen

from experiment import MODEL, SEED, generate_predictions, write_json

DATASET = "PolyAI/banking77"
SOURCE = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/"
CHECKSUMS = {
    "train": "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b",
    "test": "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d",
}
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
MAX_LABEL_TOKENS = 24


def label_prompt(labels):
    return (
        "Classify the banking customer message into exactly one of the allowed intents. "
        "Return only the exact intent label, with underscores and without explanation.\n"
        "Allowed intent labels:\n" + "\n".join(labels)
    )


def prepare(path, source_dir=None):
    if path.exists():
        raise SystemExit(f"{path} already exists")
    normalized = set()
    splits = {}
    classes = None
    for split, per_class in (("train", 30), ("test", 10)):
        filename = f"{split}.csv"
        if source_dir:
            raw = (source_dir / filename).read_bytes()
        else:
            with urlopen(SOURCE + filename, timeout=30) as response:
                raw = response.read()
        if hashlib.sha256(raw).hexdigest() != CHECKSUMS[split]:
            raise ValueError(f"Unexpected SHA-256 for {filename}")

        groups = defaultdict(list)
        for index, row in enumerate(csv.DictReader(io.StringIO(raw.decode("utf-8")))):
            question = row["text"].strip()
            key = " ".join(question.casefold().split())
            if key not in normalized:
                normalized.add(key)
                label = row["category"]
                groups[label].append(dict(id=f"{split}:{index}", question=question,
                                          solution=label, gold=label))
        if classes is None:
            classes = sorted(groups, key=str.casefold)
        if len(classes) != 77 or set(groups) != set(classes):
            raise ValueError(f"Unexpected labels in {split}")

        rng = random.Random(SEED)
        rows = [row for label in classes for row in rng.sample(groups[label], per_class)]
        rng.shuffle(rows)
        splits[split] = rows

    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, dict(
        dataset=DATASET, dataset_revision=CHECKSUMS, subset="stratified 30/10 per intent",
        model=MODEL, model_revision=MODEL_REVISION, seed=SEED, labels=classes,
        manual_ids=[row["id"] for row in splits["test"][:8]], **splits,
    ))
    print(f"Prepared {path}: 2310 train, 770 test", flush=True)


def score_labels(predictions, labels):
    gold = Counter(row["gold_label"] for row in predictions)
    predicted = Counter(row["predicted_label"] for row in predictions)
    correct_by_label = Counter(row["gold_label"] for row in predictions if row["correct"])
    f1 = [
        2 * correct_by_label[label] / (gold[label] + predicted[label])
        for label in labels if gold[label] + predicted[label]
    ]
    correct = sum(correct_by_label.values())
    return dict(
        test_examples=len(predictions), correct=correct, accuracy=correct / len(predictions),
        macro_f1=math.fsum(f1) / len(f1), parsed=len(predictions) - predicted[None],
        capped=sum(row["hit_generation_cap"] for row in predictions),
    )


def evaluate_labels(model, tokenizer, encoded, rows, labels):
    model.eval()
    start = time.perf_counter()
    by_name = {label.casefold(): label for label in labels}
    predictions = []
    for row, text, capped in generate_predictions(model, tokenizer, encoded, rows, MAX_LABEL_TOKENS):
        predicted = by_name.get(text.strip().casefold())
        predictions.append(dict(
            id=row["id"], prediction=text, predicted_label=predicted, gold_label=row["gold"],
            correct=predicted == row["gold"], hit_generation_cap=capped,
        ))
    summary = score_labels(predictions, labels)
    summary["eval_seconds"] = time.perf_counter() - start
    return summary, predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/banking77.json"))
    parser.add_argument("--source-dir", type=Path, help="Use locally downloaded official CSVs")
    args = parser.parse_args()
    prepare(args.data, args.source_dir)
