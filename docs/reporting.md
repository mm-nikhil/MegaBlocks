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
- Weight source: `nano_jax_init` or `synthetic`.
- Timing scope: `megablocks_core` or `adapter_boundary`.
- Warmup iterations and measured iterations.
- Mean forward time in milliseconds.
- Trial count and forward-time standard deviation.
- Tokens per second.
- Peak allocated/reserved memory after warmup.
- Approximate active-expert and backend-estimated TFLOP/s.
- Tokens per expert min/max for MegaBlocks runs.
- Whether output was checked against the PyTorch reference.
- Output-check metrics: max absolute error, mean absolute error, max relative
  error, and max absolute reference value.
- Aux-loss metrics: reference aux loss, actual aux loss, and absolute difference.
- Router-check metrics: positional index mismatch, expert-set mismatch, and gate
  differences.
- Outlier diagnosis.
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
weight_source:
timing_scope:
warmup:
iters:
mean_forward_ms:
std_forward_ms:
tokens_per_second:
active_expert_tflops_per_second:
backend_estimated_tflops_per_second:
peak_memory_allocated_bytes:
peak_memory_allocated_delta_bytes:
tokens_per_expert_max:
output_check:
aux_loss_check:
router_check:
outlier_diagnosis:
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
  --use-expert-biases \
  --check-output \
  --dtype float32 \
  --jsonl-out results/raw/my_run.jsonl
```

Raw files under `results/raw/` are ignored by git. Promote only reviewed
summaries into `results/`.

For a configurable sweep:

```bash
.venv/bin/python src/profiling/sweep_moe_layer.py --help
```

For a compact comparison table:

```bash
.venv/bin/python src/profiling/summarize_moe_sweep.py results/raw/sweep.jsonl
```

## Transparent Verification

Attach or paste the output of:

```bash
.venv/bin/python src/profiling/check_nano_moe_port.py
.venv/bin/python src/profiling/verify_moe_layer.py
.venv/bin/python - <<'PY'
import torch
print("torch", torch.__version__, "torch_cuda", torch.version.cuda)
print("gpu", torch.cuda.get_device_name(0))
print("capability", torch.cuda.get_device_capability(0))
PY
git submodule status
```
