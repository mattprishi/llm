#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

mode="$1"
rpm="$2"
config="$3"
spec="$4"
run_id="${5:-${mode}-${config}-${spec}-${rpm}rpm-$(date -u +%Y%m%dT%H%M%SZ)}"
if [[ ! "$rpm" =~ ^[1-9][0-9]*$ || ! "$run_id" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]]; then
    echo "Use a positive integer RPM and a run ID made of letters, digits, dots, underscores or hyphens." >&2
    exit 2
fi
server_command="$(bash serve.sh --print "$config" "$spec")"

data=data/requests.jsonl
if [[ ! -f "$data" && -f _setup/Homework/data/requests.jsonl ]]; then
    data=_setup/Homework/data/requests.jsonl
fi
if [[ ! -f "$data" ]]; then
    echo "Place the original course file at data/requests.jsonl." >&2
    exit 2
fi
expected_sha=67c48379e5ae2608f7ca08650fd346facdc01f766a3e8eecea6459d7cec00ebc
data_sha="$(python3 -c 'import hashlib, pathlib, sys; print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())' "$data")"
if [[ "$data_sha" != "$expected_sha" ]]; then
    echo "Dataset checksum differs from the original requests.jsonl." >&2
    exit 2
fi

case "$mode" in
    benchmark)
        duration=0
        requests=640
        limit=(--request-count "$requests")
        ;;
    screen)
        duration=0
        requests=160
        limit=(--request-count "$requests")
        ;;
    load)
        duration=300
        requests=$((rpm * duration / 60))
        limit=(--benchmark-duration "$duration" --benchmark-grace-period inf)
        ;;
    *) echo "Unknown benchmark mode: $mode" >&2; exit 2 ;;
esac

results="${RESULTS_DIR:-results}"
output="$results/$run_id"
mkdir -p "$results"
mkdir "$output"
printf '{"mode":"%s","rpm":%d,"config":"%s","spec":"%s","duration_s":%d,"expected_requests":%d,"data_sha256":"%s"}\n' \
    "$mode" "$rpm" "$config" "$spec" "$duration" "$requests" "$data_sha" > "$output/run.json"
printf '%s\n' "$server_command" > "$output/server-command.txt"

url="http://127.0.0.1:${PORT:-8000}"
rps="$(LC_ALL=C awk -v rpm="$rpm" 'BEGIN { printf "%.12g", rpm / 60 }')"
cmd=(aiperf profile --url "$url" --model Qwen/Qwen3-8B
    --endpoint-type chat --custom-dataset-type raw_payload
    --input-file "$data" --dataset-sampling-strategy sequential
    --warmup-request-count 40 --warmup-concurrency 8
    "${limit[@]}" --request-rate "$rps" --arrival-pattern constant
    --streaming --request-timeout-seconds 120
    --export-outputs-json --output-artifact-dir "$output/profile" --ui simple)
if [[ -n "${MODEL_REVISION:-}" ]]; then
    cmd+=(--tokenizer-revision "$MODEL_REVISION")
fi
printf '%q ' "${cmd[@]}" > "$output/benchmark-command.txt"
printf '\n' >> "$output/benchmark-command.txt"

reset_cache() {
    curl -fsS --max-time 15 -X POST "$url/reset_prefix_cache" >/dev/null
}

reset_cache
trap 'reset_cache || echo "Final cache reset failed for $run_id" >&2' EXIT
"${cmd[@]}" 2>&1 | tee "$output/aiperf.log"
trap - EXIT
reset_cache
touch "$output/complete"
