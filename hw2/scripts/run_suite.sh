#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs results

run_experiment() {
    local experiment="$1"
    local run_name="$2"
    local seconds="$3"

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

# Ten short runs cover five individual mechanisms and several combinations.
screening_configs=(
    batch64
    checkpointing
    flash_attention_2
    torch_compile
    liger
    activation_offloading
    fa2_padding_free
    fa2_packing
    batch64_fa2_packing
    batch64_fa2_packing_liger
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
