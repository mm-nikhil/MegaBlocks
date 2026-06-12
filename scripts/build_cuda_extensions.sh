#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${CUDA_HOME:-}" ]]; then
  if [[ -x .cuda/extract/usr/local/cuda-12.6/bin/nvcc ]]; then
    export CUDA_HOME="$PWD/.cuda/extract/usr/local/cuda-12.6"
  elif [[ -d /usr/local/cuda-12.6 ]]; then
    export CUDA_HOME=/usr/local/cuda-12.6
  elif [[ -d /usr/local/cuda ]]; then
    export CUDA_HOME=/usr/local/cuda
  else
    echo "CUDA_HOME is unset and no /usr/local/cuda[-12.6] directory was found." >&2
    exit 1
  fi
fi

export PATH="$CUDA_HOME/bin:$PATH"

if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc was not found on PATH. Install a CUDA toolkit before running this script." >&2
  exit 1
fi

if [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
  TORCH_CUDA_ARCH_LIST="$(python - <<'PY'
import torch
major, minor = torch.cuda.get_device_capability(0)
print(f"{major}.{minor}")
PY
)"
  export TORCH_CUDA_ARCH_LIST
fi

echo "CUDA_HOME=$CUDA_HOME"
echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"

python -m pip install --no-build-isolation --no-deps -e third_party/grouped_gemm
python -m pip install --no-build-isolation --no-deps -e third_party/megablocks

python - <<'PY'
import torch
import megablocks_ops
import grouped_gemm
print("CUDA extensions OK")
PY
