#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${DFINE_VENV:-${ROOT_DIR}/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
CHECK_ONLY=false

if [[ "${1:-}" == "--check-only" ]]; then
  CHECK_ONLY=true
elif [[ $# -ne 0 ]]; then
  echo "usage: $0 [--check-only]" >&2
  exit 2
fi

uname -a
command -v nvidia-smi >/dev/null
nvidia-smi

if [[ -x "${VENV}/bin/python" ]]; then
  # shellcheck disable=SC1091
  source "${VENV}/bin/activate"
elif [[ "${CHECK_ONLY}" == false ]]; then
  "${PYTHON_BIN}" -c \
    'import sys; assert sys.version_info[:2] == (3, 11), "Python 3.11 is required"'
  "${PYTHON_BIN}" -m venv "${VENV}"
  # shellcheck disable=SC1091
  source "${VENV}/bin/activate"
fi

python --version
if [[ "${CHECK_ONLY}" == false ]]; then
  python -m pip install --upgrade pip
  python -m pip install \
    torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu124
  python -m pip install -r "${ROOT_DIR}/requirements.txt"
fi

python -m pip check
python - <<'PY'
import torch
import torchvision
import yaml
import faster_coco_eval

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("cuDNN:", torch.backends.cudnn.version())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
print("GPU:", torch.cuda.get_device_name(0))
PY
