"""Recompute official BFCL metrics and paired changes from saved predictions."""
import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean

GROUPS = {
    "simple": ("simple",),
    "multiple": ("multiple", "live_multiple"),
    "parallel": ("parallel",),
    "parallel_multiple": ("parallel_multiple",),
    "irrelevance": ("irrelevance", "live_irrelevance"),
}
BENCHMARK_SHA256 = "b257ffe2c539897123706ed2211f9ad0e68b8f8fbad7b523ba5a3fcca2a457b2"
BASE_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
DECODING = {"temperature": 0, "max_new_tokens": 1024, "enable_thinking": False, "seed": 42}


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def inspect_run(path, examples):
    metrics = json.loads((path / "metrics.json").read_text())
    predictions = read_jsonl(path / "predictions.jsonl")
    require(metrics["benchmark_sha256"] == BENCHMARK_SHA256, "benchmark hash")
    require(metrics["base_revision"] == BASE_REVISION, "model revision")
    require(metrics["decoding"] == DECODING, "decoding")
    require(len(predictions) == len(examples), "prediction count")
    flags = {}
    prompt_lengths = []
    correct, total, errors, with_call = Counter(), Counter(), Counter(), Counter()
    for expected, prediction in zip(examples, predictions):
        case_id, category = expected["id"], expected["category"]
        require((prediction["id"], prediction["category"]) == (case_id, category), f"row mismatch: {case_id}")
        n_tokens = prediction["prompt_tokens"]
        require(type(n_tokens) is int and 0 < n_tokens <= 3072, f"prompt token budget: {case_id}")
        prompt_lengths.append(n_tokens)
        ok = prediction["check"]["valid"]
        require(type(ok) is bool, f"nonboolean checker result: {case_id}")
        flags[case_id] = ok
        correct[category] += ok
        total[category] += 1
        with_call["irrelevance" if "irrelevance" in category else "call_required"] += bool(prediction["calls"])
        if not ok:
            if prediction["finish_reason"] == "length":
                kind = "truncated"
            elif not prediction["response"].strip():
                kind = "empty"
            elif "irrelevance" in category and prediction["calls"]:
                kind = "unnecessary_call"
            elif "irrelevance" not in category and not prediction["calls"]:
                kind = "no_parsed_call"
            else:
                kind = "other_checker_error"
            errors[kind] += 1
    category_metrics = {c: {"correct": correct[c], "total": n, "accuracy": correct[c] / n} for c, n in total.items()}
    group_accuracy = {g: mean(category_metrics[c]["accuracy"] for c in cats) for g, cats in GROUPS.items()}
    overall = mean(group_accuracy.values())
    require(metrics["categories"] == category_metrics, "category metrics")
    require(metrics["groups"] == group_accuracy, "group metrics")
    require(math.isclose(metrics["overall"], overall, abs_tol=1e-12), "overall metric")
    return {
        "model": metrics["model"], "overall": overall, "groups": group_accuracy,
        "categories": category_metrics, "errors": dict(errors),
        "with_parsed_call": dict(with_call), "micro_correct": sum(correct.values()),
        "elapsed_seconds": metrics["elapsed_seconds"], "versions": metrics["versions"],
        "prompt_lengths_sha256": hashlib.sha256(json.dumps(prompt_lengths).encode()).hexdigest(),
    }, flags


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    benchmark = args.root / "bfcl.jsonl"
    require(hashlib.sha256(benchmark.read_bytes()).hexdigest() == BENCHMARK_SHA256, "original bfcl.jsonl required")
    examples = read_jsonl(benchmark)
    require(len(examples) == len({row["id"] for row in examples}) == 350, "benchmark IDs")
    summary, flags, unavailable = {}, {}, {}
    for name in ("base", "rl1", "rl2", "rl3"):
        try:
            run, run_flags = inspect_run(args.root / "results" / name, examples)
            require(run["model"] == "Qwen/Qwen3.5-2B" if name == "base" else Path(run["model"]).name == name, "unexpected checkpoint name")
            summary[name], flags[name] = run, run_flags
        except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError) as exc:
            unavailable[name] = str(exc)
    if "base" in flags:
        for name in tuple(summary):
            if name == "base":
                continue
            if any(summary[name][k] != summary["base"][k] for k in ("versions", "prompt_lengths_sha256")):
                unavailable[name] = "package versions or prompt lengths differ from baseline"
                del summary[name]
                continue
            changed = {}
            for g, cats in GROUPS.items():
                ids = [r["id"] for r in examples if r["category"] in cats]
                changed[g] = {
                    "improved": sum(not flags["base"][i] and flags[name][i] for i in ids),
                    "regressed": sum(flags["base"][i] and not flags[name][i] for i in ids),
                }
            summary[name]["paired_vs_base"] = changed
            summary[name]["delta_pp"] = 100 * (summary[name]["overall"] - summary["base"]["overall"])
    print(json.dumps({"validated": summary, "unavailable": unavailable}, ensure_ascii=False, indent=2))
    if unavailable and args.require_complete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
