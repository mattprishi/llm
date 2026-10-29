#!/usr/bin/env bash
# Run on the homework GPU server: bash setup_a100.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv-hw3"
PYTHON_BIN="${HW3_PYTHON:-python3}"
VENV_PYTHON="${VENV_DIR}/bin/python"
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0,1

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    printf 'Python executable not found: %s\n' "${PYTHON_BIN}" >&2
    exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
    printf 'nvidia-smi is unavailable; run this script on the GPU server.\n' >&2
    exit 1
fi
nvidia-smi

if [[ ! -x "${VENV_PYTHON}" ]] || ! "${VENV_PYTHON}" -m pip --version >/dev/null 2>&1; then
    if "${PYTHON_BIN}" -c 'import ensurepip' >/dev/null 2>&1; then
        if [[ ! -x "${VENV_PYTHON}" ]]; then
            "${PYTHON_BIN}" -m venv "${VENV_DIR}"
        else
            "${VENV_PYTHON}" -m ensurepip --upgrade
        fi
    else
        UV_BIN="$(command -v uv || true)"
        if [[ -z "${UV_BIN}" && -x "${HOME}/.local/bin/uv" ]]; then
            UV_BIN="${HOME}/.local/bin/uv"
        fi
        if [[ -z "${UV_BIN}" ]]; then
            printf 'Python has no ensurepip, and uv was not found in PATH or ~/.local/bin.\n' >&2
            exit 1
        fi
        "${UV_BIN}" venv --seed --allow-existing --python "${PYTHON_BIN}" "${VENV_DIR}"
    fi
fi
"${VENV_PYTHON}" -m pip --version

# Both installations target the isolated venv. Existing matching packages are reused.
"${VENV_PYTHON}" -m pip --disable-pip-version-check install \
    'torch==2.8.0+cu126' --index-url https://download.pytorch.org/whl/cu126
"${VENV_PYTHON}" -m pip --disable-pip-version-check install \
    -r "${SCRIPT_DIR}/requirements.txt"
"${VENV_PYTHON}" -m pip check

"${VENV_PYTHON}" - <<'PY'
import importlib.metadata
import os
import sys

import torch

print(f"Python: {sys.version.split()[0]} ({sys.executable})")
for package in (
    "torch", "transformers", "trl", "datasets", "accelerate", "nbclient", "nbformat"
):
    print(f"{package}: {importlib.metadata.version(package)}")
print(f"torch CUDA runtime: {torch.version.cuda}")
print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")

assert torch.__version__ == "2.8.0+cu126", torch.__version__
assert torch.version.cuda == "12.6", torch.version.cuda
assert torch.cuda.is_available(), "CUDA initialization failed"
assert torch.cuda.device_count() == 2, (
    f"Expected both GPUs; found {torch.cuda.device_count()} visible devices"
)

for index in range(2):
    with torch.cuda.device(index):
        properties = torch.cuda.get_device_properties(index)
        assert torch.cuda.is_bf16_supported(including_emulation=False), (
            f"GPU {index} lacks native BF16 support"
        )
        values = torch.ones((128, 128), dtype=torch.bfloat16, device=f"cuda:{index}")
        result = values @ values
        torch.cuda.synchronize(index)
        assert result.device.index == index
        assert torch.isfinite(result).all().item(), f"GPU {index}: non-finite output"
        assert (result == 128).all().item(), f"GPU {index}: incorrect CUDA matmul"
        print(
            f"GPU {index}: {properties.name}; "
            f"{properties.total_memory / 2**30:.1f} GiB; "
            f"compute capability {properties.major}.{properties.minor}; "
            "BF16 CUDA matmul passed"
        )

print("Environment ready; both GPUs passed real BF16 CUDA computation.")
PY

printf '\nActivate with: source "%s/bin/activate"\n' "${VENV_DIR}"
printf 'Execute homework with: cd "%s" && CUDA_VISIBLE_DEVICES=0,1 .venv-hw3/bin/python execute_notebook.py\n' "${SCRIPT_DIR}"
