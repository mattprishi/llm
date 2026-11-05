#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if command -v uv >/dev/null; then
    [[ -x .venv/bin/python ]] || uv venv --python "$(command -v python3)" .venv
    installer=(uv pip install --python .venv/bin/python)
else
    [[ -x .venv/bin/python ]] || python3 -m venv .venv
    installer=(.venv/bin/python -m pip install)
fi
"${installer[@]}" --index-url https://download.pytorch.org/whl/cu126 'torch==2.8.0+cu126'
"${installer[@]}" -r requirements.txt

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" .venv/bin/python - <<'PY'
import peft
import torch
import transformers

assert torch.cuda.device_count() == 1
assert torch.cuda.is_bf16_supported()
x = torch.ones((128, 128), device="cuda", dtype=torch.bfloat16)
assert (x @ x)[0, 0].item() == 128
print(torch.cuda.get_device_name(), torch.__version__, transformers.__version__, peft.__version__)
PY
