# Reporting Results

Each timing report should include enough metadata to reproduce the number.

## Required Fields

- Date and host.
- GPU model, driver, and CUDA toolkit version.
- Python version.
- Torch version and Torch CUDA runtime.
- MegaBlocks commit.
- grouped_gemm commit or package version.
- Nano-MoE-JAX commit.
- Command line used.
- Backend: `reference`, `megablocks/moe`, or `megablocks/dmoe`.
- Shape: batch, sequence length, token count, `d_model`, `d_ff`.
- Experts: `n_experts`, `top_k`.
- Dtype.
- Warmup iterations and measured iterations.
- Mean forward time in milliseconds.
- Whether output was checked against the PyTorch reference.
- Output-check metrics: max absolute error, mean absolute error, max relative
  error, and max absolute reference value.
- Whether expert biases were zeroed, matched, or intentionally mismatched.

## Minimal Result Template

```text
date:
host:
gpu:
driver:
cuda_toolkit:
python:
torch:
megablocks_commit:
grouped_gemm_commit:
nano_moe_jax_commit:
command:
backend:
shape:
experts:
dtype:
warmup:
iters:
mean_forward_ms:
output_check:
bias_semantics:
notes:
```

## Raw Sweep Capture

Use JSONL for raw benchmark capture:

```bash
scripts/run_smoke_matrix.sh
```

or for one run:

```bash
.venv/bin/python src/profiling/profile_moe_layer.py \
  --backend megablocks \
  --megablocks-layer moe \
  --zero-expert-biases \
  --check-output \
  --dtype float32 \
  --jsonl-out results/raw/my_run.jsonl
```

Raw files under `results/raw/` are ignored by git. Promote only reviewed
summaries into `results/`.

## Transparent Verification

Attach or paste the output of:

```bash
.venv/bin/python src/profiling/check_nano_moe_port.py
.venv/bin/python - <<'PY'
import torch
print("torch", torch.__version__, "torch_cuda", torch.version.cuda)
print("gpu", torch.cuda.get_device_name(0))
print("capability", torch.cuda.get_device_capability(0))
PY
git submodule status
```
