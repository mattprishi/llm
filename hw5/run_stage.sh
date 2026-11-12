#!/usr/bin/env bash
# Run inside the course image, with only the assigned GPU visible.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export PYTHONUNBUFFERED=1
stage=${1:-}
case "$stage" in
  preflight)
    python3 - <<'PY'
from importlib.metadata import version
import torch
for pkg in ("torch", "vllm", "transformers", "trl", "peft", "bfcl-eval"):
    print(pkg, version(pkg))
assert torch.cuda.device_count() == 1, "Expose only the allocated physical GPU"
a = torch.ones((128, 128), dtype=torch.bfloat16, device="cuda")
assert (a @ a)[0, 0].item() == 128
print("CUDA BF16:", torch.cuda.get_device_name(0), torch.version.cuda)
PY
    ;;
  data)
    if [[ -e data/train.jsonl || -e data/val.jsonl ]]; then
      echo 'Existing data split; preserve or move it before generating another.' >&2
      exit 1
    fi
    shift
    python3 prepare_data.py --output data --max-prompt-tokens 1536 --val-size 128 "$@"
    ;;
  base)
    test ! -e results/base
    python3 evaluate.py Qwen/Qwen3.5-2B --output results/base
    ;;
  pilot)
    test -s results/base/metrics.json
    test ! -e runs/pilot
    python3 train_rl.py rl1 --steps "${2:-6}" --eval-every "${2:-6}" --val-limit 12 --max-completion "${3:-256}" --output runs/pilot
    python3 export_model.py runs/pilot/best checkpoints/pilot
    ;;
  rl1|rl2|rl3)
    steps=${2:?Specify the shared step budget based on the pilot}
    interval=${3:-$steps}
    test -s results/base/metrics.json
    test ! -e "runs/$stage"
    test ! -e "checkpoints/$stage"
    test ! -e "results/$stage"
    python3 train_rl.py "$stage" --steps "$steps" --eval-every "$interval" --val-limit "${TRAIN_VAL_LIMIT:-36}" --max-completion "${4:-256}" --output "runs/$stage"
    python3 export_model.py "runs/$stage/best" "checkpoints/$stage"
    python3 evaluate.py "checkpoints/$stage" --output "results/$stage"
    ;;
  *)
    echo 'Usage: bash run_stage.sh {preflight|data|base|pilot [steps] [max_completion]|rl1/rl2/rl3 steps [eval_every] [max_completion]}' >&2
    exit 2
    ;;
esac
