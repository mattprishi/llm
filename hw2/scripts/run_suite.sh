#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs results

run_experiment() {
    local experiment="$1"
    local requested_name="$2"
    local seconds="$3"
    local run_name="${requested_name}"

    if [[ -f "results/${run_name}/summary.json" ]]; then
        echo "SKIP: ${run_name} already has summary.json"
        return 0
    fi

    # Preserve failed runs for the report and choose a fresh output directory.
    local retry=2
    while [[ -e "results/${run_name}" ]]; do
        run_name="${requested_name}_retry${retry}"
        retry=$((retry + 1))
    done

    echo "=== ${run_name}: config=${experiment}, seconds=${seconds} ==="
    EXPERIMENT="${experiment}" RUN_NAME="${run_name}" TRAIN_SECONDS="${seconds}" \
        python3 train.py 2>&1 | tee "logs/${run_name}.log"
    local status="${PIPESTATUS[0]}"
    if [[ "${status}" -ne 0 ]]; then
        echo "FAILED: ${run_name} (exit ${status})" | tee -a "logs/failures.log"
        return "${status}"
    fi
}

# The required five-minute baseline.
run_experiment baseline baseline_300s 300 || exit 1

# The first suite already established two machine-specific negative results:
# batch64 OOMs, while torch.compile and Liger cannot load Triton kernels with
# driver 470. Keep those logs, but do not waste the remaining GPU window by
# repeating them. The runs below still cover five mechanisms and combinations.
screening_configs=(
    batch48
    batch16_accum4
    checkpointing
    flash_attention_2
    activation_offloading
    fa2_padding_free
    fa2_packing
    batch48_fa2
    batch48_fa2_packing
)
for experiment in "${screening_configs[@]}"; do
    run_experiment "${experiment}" "screen_${experiment}_75s" 75 || true
done

python3 scripts/summarize_results.py
mapfile -t finalists < <(python3 scripts/select_top.py)
if [[ "${#finalists[@]}" -lt 2 ]]; then
    echo "Fewer than two screening configurations succeeded; inspect logs/failures.log." >&2
    exit 1
fi

# Rerun the two fastest candidates from fresh initialization for five minutes.
for index in 0 1; do
    experiment="${finalists[${index}]}"
    run_experiment "${experiment}" "final_$((index + 1))_${experiment}_300s" 300 || true
done

python3 scripts/summarize_results.py
