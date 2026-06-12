# Setup

These steps are meant to be portable across NVIDIA GPU servers. They separate
system CUDA requirements from the Python environment.

## System Requirements

- NVIDIA GPU with a working driver.
- Python 3.10 or newer.
- CUDA toolkit with `nvcc`.
- Build tools: `gcc`, `g++`, `make`.

Check the GPU:

```bash
nvidia-smi
```

Check for `nvcc`:

```bash
which nvcc
nvcc --version
```

If `nvcc` is missing on an Ubuntu host with NVIDIA CUDA apt repos configured:

```bash
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-6
```

For the current Torch wheel, CUDA 12.6 is the tested target.

If sudo is unavailable, a workspace-local compiler overlay can also work:

```bash
scripts/install_cuda_overlay.sh
export CUDA_HOME="$PWD/.cuda/extract/usr/local/cuda-12.6"
export PATH="$CUDA_HOME/bin:$PATH"
```

Equivalent manual commands:

```bash
mkdir -p .cuda/debs .cuda/extract
cd .cuda/debs
apt download cuda-nvcc-12-6 cuda-cudart-dev-12-6 cuda-cudart-12-6 \
  cuda-driver-dev-12-6 cuda-cccl-12-6 cuda-nvvm-12-6 cuda-crt-12-6
cd ../..
for deb in .cuda/debs/*.deb; do dpkg-deb -x "$deb" .cuda/extract; done
ln -s targets/x86_64-linux/include .cuda/extract/usr/local/cuda-12.6/include
ln -s targets/x86_64-linux/lib .cuda/extract/usr/local/cuda-12.6/lib64
export CUDA_HOME="$PWD/.cuda/extract/usr/local/cuda-12.6"
export PATH="$CUDA_HOME/bin:$PATH"
```

## Python Environment

From the repo root:

```bash
scripts/bootstrap_python_env.sh
source .venv/bin/activate
```

Equivalent manual commands:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "setuptools<79.0.0" wheel ninja
```

Install PyTorch CUDA 12.6:

```bash
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu126
```

Install common dependencies:

```bash
python -m pip install "numpy>=1.21.5,<2.1.0" "packaging>=21.3.0,<24.2"
python -m pip install jax==0.4.35 jaxlib==0.4.35 pytest
python -m pip install --no-deps flax==0.8.4
python -m pip install msgpack PyYAML rich absl-py
python -m pip install --no-build-isolation stanford-stk==0.7.1
```

Build grouped GEMM and MegaBlocks after `nvcc` is available:

```bash
export CUDA_HOME=/usr/local/cuda-12.6
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=8.6

scripts/build_cuda_extensions.sh
```

Use the GPU's compute capability for `TORCH_CUDA_ARCH_LIST`. For an RTX 3080 this
is `8.6`.

Verify imports:

```bash
python - <<'PY'
import torch
import megablocks_ops
import grouped_gemm
print(torch.__version__, torch.cuda.get_device_name(0))
print("MegaBlocks kernels import OK")
PY
```

## Notes

- `pip install nvidia-cuda-nvcc-cu12` is not a substitute for a full CUDA toolkit;
  it does not provide the `nvcc` compiler executable needed by these extensions.
- If `CUDA_HOME` is unset, MegaBlocks may install as Python code, but its routing
  ops will not run.
