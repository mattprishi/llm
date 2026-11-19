#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Each triplet is MODE RPM RUN_ID. Keep one server alive for the batch.
if (( $# < 5 || ($# - 2) % 3 != 0 )); then
    echo "Usage: bash run_vm.sh CONFIG SPEC MODE RPM RUN_ID [MODE RPM RUN_ID ...]" >&2
    exit 2
fi
config="$1"
spec="$2"
batch="$5"
shift 2
if [[ ! "$batch" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]]; then
    echo "Invalid batch ID: $batch" >&2
    exit 2
fi

: "${HW6_GPU_UUID:?Set HW6_GPU_UUID to the UUID of the reserved GPU}"
: "${HW6_IMAGE:?Set HW6_IMAGE to the locally built HW6 image}"
gpu_index="${HW6_GPU_INDEX:-1}"
if [[ ! "$gpu_index" =~ ^(0|[1-9][0-9]*)$ ]]; then
    echo "HW6_GPU_INDEX must be a nonnegative integer." >&2
    exit 2
fi
model_revision="$(<model-revision)"
MODEL_REVISION="$model_revision" bash serve.sh --print "$config" "$spec"
root="$(pwd -P)"
mkdir -p logs results hf-cache .homework-cache
exec 9>"logs/gpu-${gpu_index}.lock"
flock -n 9 || { echo "Another HW6 batch is running in this workspace." >&2; exit 75; }

check_gpu() {
    local actual_uuid memory compute_apps
    actual_uuid="$(nvidia-smi -i "$gpu_index" --query-gpu=uuid --format=csv,noheader)"
    memory="$(nvidia-smi -i "$HW6_GPU_UUID" --query-gpu=memory.used --format=csv,noheader,nounits)"
    compute_apps="$(nvidia-smi -i "$HW6_GPU_UUID" --query-compute-apps=pid --format=csv,noheader)"
    if [[ "$actual_uuid" != "$HW6_GPU_UUID" || -n "$compute_apps" ]] || (( memory > 512 )); then
        echo "GPU $gpu_index is unavailable (UUID $actual_uuid, memory used ${memory} MiB)." >&2
        return 75
    fi
}
check_gpu
sleep 2
check_gpu
nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.used,memory.total,utilization.gpu --format=csv \
    > "logs/$batch-gpus-before.csv"

docker=(docker)
if ! docker info >/dev/null 2>&1; then
    docker=(sudo -n docker)
fi
container_id=""
cleanup() {
    local status="$?"
    if [[ -n "$container_id" ]]; then
        "${docker[@]}" stop --time 10 "$container_id" >/dev/null || status=1
        "${docker[@]}" logs "$container_id" > "logs/$batch-server.log" 2>&1 || status=1
        "${docker[@]}" rm "$container_id" >/dev/null || status=1
        nvidia-smi -i "$HW6_GPU_UUID" --query-gpu=memory.used,utilization.gpu --format=csv \
            > "logs/$batch-gpu-after.csv" || true
    fi
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
container_id="$("${docker[@]}" run --detach --name "hw6-${batch}-$$" \
    --gpus "device=$HW6_GPU_UUID" --shm-size 2g --workdir /hw6 \
    --mount "type=bind,src=$root,dst=/hw6,readonly" \
    --mount "type=bind,src=$root/results,dst=/hw6/results" \
    --mount "type=bind,src=$root/hf-cache,dst=/hw6/hf-cache" \
    --mount "type=bind,src=$root/.homework-cache,dst=/hw6/.homework-cache" \
    -e HF_HOME=/hw6/hf-cache -e VLLM_CACHE_ROOT=/hw6/.homework-cache/vllm \
    -e TRITON_CACHE_DIR=/hw6/.homework-cache/triton \
    -e TORCHINDUCTOR_CACHE_DIR=/hw6/.homework-cache/inductor \
    -e VLLM_ENABLE_CUDA_COMPATIBILITY="${HW6_CUDA_COMPAT:-0}" \
    -e MODEL_REVISION="$model_revision" \
    "$HW6_IMAGE" bash serve.sh "$config" "$spec")"
printf '%s\n' "$container_id" > "logs/$batch-container-id"
echo "Started $container_id for $config $spec."

ready=false
for (( i = 0; i < 240; i++ )); do
    if [[ "$("${docker[@]}" inspect --format '{{.State.Running}}' "$container_id")" != true ]]; then
        echo "vLLM container exited before becoming ready." >&2
        exit 1
    fi
    if "${docker[@]}" exec "$container_id" curl -fsS --max-time 3 http://127.0.0.1:8000/health >/dev/null 2>&1; then
        ready=true
        break
    fi
    sleep 5
done
if ! "$ready"; then
    echo "vLLM health check timed out." >&2
    exit 1
fi
"${docker[@]}" exec "$container_id" curl -fsS http://127.0.0.1:8000/v1/models > "logs/$batch-models.json"

status=0
while (( $# )); do
    mode="$1"
    rpm="$2"
    run_id="$3"
    shift 3
    echo "$(date -u +'%Y-%m-%dT%H:%M:%SZ') $mode $rpm $config $spec $run_id"
    case "$mode" in
        benchmark) "${docker[@]}" exec "$container_id" bash benchmark.sh "$config" "$spec" "$rpm" "$run_id" || status=1 ;;
        load) "${docker[@]}" exec "$container_id" bash load_test.sh "$rpm" "$config" "$spec" "$run_id" || status=1 ;;
        screen) "${docker[@]}" exec "$container_id" bash profile.sh screen "$rpm" "$config" "$spec" "$run_id" || status=1 ;;
        *) echo "Unknown profile mode: $mode" >&2; exit 2 ;;
    esac
done
exit "$status"
