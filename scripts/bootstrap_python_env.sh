#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install "setuptools<79.0.0" wheel ninja
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install "numpy>=1.21.5,<2.1.0" "packaging>=21.3.0,<24.2"
python -m pip install "matplotlib>=3.8,<3.10"
python -m pip install jax==0.4.35 jaxlib==0.4.35 pytest
python -m pip install optax==0.1.9
python -m pip install --no-deps flax==0.8.4
python -m pip install msgpack PyYAML rich absl-py
python -m pip install --no-build-isolation stanford-stk==0.7.1

python - <<'PY'
import torch
print("python env OK")
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0))
PY
