"""Summarize AIPerf 0.13.0 runs and plot the requested latency curves."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


def stat(summary, metric, field="avg", seconds=False):
    entry = summary.get(metric, {})
    value = entry.get(field)
    if value is None:
        return None
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"Invalid {metric}.{field}: {value}")
    if seconds:
        return value * {"ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1}[entry["unit"]]
    return value


def read_run(manifest):
    folder = manifest.parent
    row = {"run_id": folder.name, **json.loads(manifest.read_text())}
    if not (folder / "complete").exists():
        return {**row, "status": "incomplete", "note": "Run did not finish cleanly"}

    try:
        profile = folder / "profile"
        summary = json.loads((profile / "profile_export_aiperf.json").read_text())
        with (profile / "profile_export.jsonl").open() as records:
            entries = [json.loads(line) for line in records if line.strip()]
        entries = [entry for entry in entries if entry["metadata"]["benchmark_phase"] == "profiling"]
        starts = [entry["metadata"]["request_start_ns"] for entry in entries]
        sent = len(entries)
        span = (max(starts) - min(starts)) / 1e9 if sent > 1 else 0
        errors = sum(bool(entry.get("error") or entry["metadata"].get("was_cancelled")) for entry in entries)
        row.update(
            sent=sent,
            successful=sent - errors,
            errors=errors,
            send_span_s=span,
            offered_rpm=60 * (sent - 1) / span if span else None,
            request_throughput_rps=stat(summary, "request_throughput"),
            mean_e2e_s=stat(summary, "request_latency", seconds=True),
            p90_e2e_s=stat(summary, "request_latency", "p90", seconds=True),
            mean_ttft_s=stat(summary, "time_to_first_token", seconds=True),
            mean_itl_s=stat(summary, "inter_token_latency", seconds=True),
            mean_output_tokens=stat(summary, "output_token_count"),
            p90_output_tokens=stat(summary, "output_token_count", "p90"),
            mean_input_tokens=stat(summary, "input_sequence_length"),
        )
        expected = row["expected_requests"]
        rpm = row["rpm"]
        duration = row["duration_s"]
        count_tolerance = max(2, 0.02 * expected) if duration else 0
        if not span or abs(row["offered_rpm"] - rpm) > 0.02 * rpm:
            raise ValueError("Actual send rate differs from the target by more than 2%")
        if abs(sent - expected) > count_tolerance:
            raise ValueError(f"Sent {sent} requests, expected {expected}")
        if duration and span + 60 / rpm < duration - 1:
            raise ValueError("Send window did not cover the requested 300 seconds")
        reported_success = stat(summary, "request_count")
        if not (errors == sent and reported_success is None) and reported_success != sent - errors:
            raise ValueError("Request count disagrees between the aggregate and per-request exports")
        if not errors and (row["mean_e2e_s"] is None or row["p90_e2e_s"] is None):
            raise ValueError("Missing E2E latency stats")
        row["status"] = "request_errors" if errors else "ok"
        if duration:
            row["sla_pass"] = not errors and row["p90_e2e_s"] < 15
    except (OSError, ValueError, KeyError, TypeError) as exc:
        row.update(status="invalid", note=str(exc))
    return row


def tune_summary(rows, root):
    groups = defaultdict(list)
    for row in rows:
        if row["status"] == "ok" and row["mode"] == "benchmark" and row["rpm"] == 120:
            groups[row["config"], row["spec"]].append(row)
    baseline = groups.get(("baseline", "off"))
    if not baseline:
        return
    baseline_mean = mean(row["mean_e2e_s"] for row in baseline)
    scores = []
    for (config, spec), runs in groups.items():
        latency = mean(row["mean_e2e_s"] for row in runs)
        for row in runs:
            row["speedup_vs_baseline"] = baseline_mean / row["mean_e2e_s"]
        scores.append(dict(config=config, spec=spec, runs=len(runs), mean_e2e_s=latency,
                           median_run_p90_s=median(row["p90_e2e_s"] for row in runs),
                           speedup_vs_baseline=baseline_mean / latency))
    write_csv(root / "tuning_120rpm.csv", sorted(scores, key=lambda score: score["mean_e2e_s"]))


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def p90_by_rate(rows):
    values = defaultdict(list)
    for row in rows:
        values[row["rpm"]].append(row["p90_e2e_s"])
    return {rpm: median(results) for rpm, results in values.items()}


def plot_spec(ax, rows, config, method):
    matched_config = [row for row in rows if row["mode"] == "benchmark"
                      and row["config"] == config and row["spec"] in {"off", method}]
    clean = [row for row in matched_config if row["status"] == "ok"]
    off = p90_by_rate(row for row in clean if row["spec"] == "off")
    on = p90_by_rate(row for row in clean if row["spec"] == method)
    rates = sorted(off.keys() & on.keys())
    if not rates:
        raise SystemExit("No paired successful runs for the requested speculative curve.")
    ax.plot(sorted(off), [off[rpm] for rpm in sorted(off)], "o-", label="Speculation off")
    ax.plot(sorted(on), [on[rpm] for rpm in sorted(on)], "o-", label=method)
    errors = [row for row in matched_config if row["status"] == "request_errors"
              and row.get("p90_e2e_s") is not None]
    if errors:
        ax.scatter([row["rpm"] for row in errors], [row["p90_e2e_s"] for row in errors],
                   marker="^", facecolors="none", edgecolors="tab:red",
                   label="P90 of successful requests only")
        for row in errors:
            ax.annotate(f"{row['errors']} errors", (row["rpm"], row["p90_e2e_s"]),
                        xytext=(-7, -16), textcoords="offset points", ha="right")
    ax.set_title(f"Speculative decoding under load ({config})")
    worse = next((rpm for rpm in rates if on[rpm] > off[rpm]), None)
    print(f"Paired rates: {rates} RPM. First tested rate with higher P90 for {method}: {worse or 'none'}")
    if worse is not None and worse != rates[0]:
        print(f"Transition bracket: {rates[rates.index(worse) - 1]} to {worse} RPM")


def plot_sla(ax, rows, config, method):
    runs = [row for row in rows if row["mode"] == "load" and row["config"] == config
            and row["spec"] == method and "sla_pass" in row]
    if not runs:
        raise SystemExit("No valid 300-second runs for the requested SLA curve.")
    groups = defaultdict(list)
    for row in runs:
        groups[row["rpm"]].append(row)
    passing = [rpm for rpm, at_rate in groups.items() if all(row["sla_pass"] for row in at_rate)]
    best = max(passing, default=None)
    failed = min((rpm for rpm in groups if rpm not in passing and (best is None or rpm > best)), default=None)
    styles = [
        (False, True, "o", "SLA pass"),
        (False, False, "x", "P90 ≥ 15 s"),
        (True, False, "^", "Request errors"),
    ]
    for has_errors, passed, marker, label in styles:
        points = [
            row for row in runs if row["p90_e2e_s"] is not None
            and bool(row["errors"]) == has_errors and row["sla_pass"] == passed
        ]
        if points:
            ax.scatter([row["rpm"] for row in points], [row["p90_e2e_s"] for row in points],
                       marker=marker, label=label)
    ax.axhline(15, color="grey", linestyle="--", label="15 s limit")
    for rpm, label, linestyle in [(best, "Highest tested pass", "-"), (failed, "Next tested miss", ":")]:
        if rpm is not None:
            ax.axvline(rpm, color="grey", linestyle=linestyle, alpha=0.65, label=f"{label}: {rpm} RPM")
    if best is not None and failed is not None:
        zoom = ax.inset_axes([0.42, 0.53, 0.38, 0.39])
        zoom.axhline(15, color="grey", linestyle="--")
        for rpm, marker, color in [(best, "o", "tab:blue"), (failed, "x", "tab:orange")]:
            latencies = [row["p90_e2e_s"] for row in groups[rpm] if row["p90_e2e_s"] is not None]
            zoom.scatter([rpm] * len(latencies), latencies, marker=marker, color=color)
        padding = (failed - best) * 0.6
        zoom.set(xlim=(best - padding, failed + padding), xticks=[best, failed])
        zoom.set_title("Around the SLA boundary", fontsize=10)
        zoom.tick_params(labelsize=8)
        zoom.grid(alpha=0.2)
    ax.set_title(f"300-second SLA test ({config}, {method})")
    print(f"Highest tested passing rate: {best} RPM. Next tested miss: {failed} RPM.")


def save_plot(root, kind, choice, plot, rows):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    config, method = choice
    fig, ax = plt.subplots(figsize=(8, 4.75), layout="constrained")
    plot(ax, rows, config, method)
    ax.set(xlabel="Target load (requests/minute)", ylabel="P90 end-to-end latency (s)")
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right" if kind == "sla" else "best")
    fig.savefig(root / f"{kind}-{config}-{method}.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--spec", nargs=2, metavar=("CONFIG", "METHOD"))
    parser.add_argument("--sla", nargs=2, metavar=("CONFIG", "METHOD"))
    args = parser.parse_args()
    rows = [read_run(path) for path in sorted(args.results.glob("*/run.json"))]
    if not rows:
        parser.error("results directory has no run manifests")
    tune_summary(rows, args.results)
    write_csv(args.results / "runs.csv", rows)
    if args.spec:
        save_plot(args.results, "spec", args.spec, plot_spec, rows)
    if args.sla:
        save_plot(args.results, "sla", args.sla, plot_sla, rows)


if __name__ == "__main__":
    main()
