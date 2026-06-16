# Findings

This is the current handoff summary for profiling the Nano-MoE-JAX MoE layer
with MegaBlocks.

## Current Artifacts

- Result: [results/nano-token-sweep/report.md](results/nano-token-sweep/report.md)
- Setup: [docs/setup.md](docs/setup.md)
- Usage: [docs/usage.md](docs/usage.md)
- Architecture: [docs/architecture.md](docs/architecture.md)
- Terminology: [docs/terminology.md](docs/terminology.md)
- Reporting fields: [docs/reporting.md](docs/reporting.md)

## Goal

Measure one Nano-MoE-JAX MoE layer on GPU using MegaBlocks while preserving the
Nano-MoE-JAX layer semantics.

Current boundary:

```text
input:    (batch_size, seq_len, d_model)
output:   (batch_size, seq_len, d_model)
aux_loss: scalar Nano-style load-balancing loss
```

The full Transformer, attention, embeddings, tokenizer, optimizer, and training
loop are outside scope.

## Nano MoE Semantics

For input `x`:

```text
router_logits = x @ router_kernel
router_probs  = softmax(router_logits)
top_values, top_indices = top_k(router_logits, top_k)
gates = softmax(top_values)

expert_i(x) = GELU(x @ w1_i + b1_i) @ w2_i + b2_i
output = weighted sum of selected expert outputs
```

Auxiliary loss:

```text
n_experts * sum_i(top1_token_fraction_i * mean_router_probability_i)
```

Default Nano-shaped profiling config:

```text
d_model=128
d_ff=512
n_experts=4
top_k=2
seq_len=128
tokens=batch_size * seq_len
```

The token sweep changes `batch_size`, so `tokens` increases while the MoE layer
shape stays fixed.

## Components

```text
Nano-MoE-JAX source layer
  -> PyTorch NanoMoE reference
  -> Nano-compatible MegaBlocks adapter
  -> MegaBlocks CUDA MoE path
```

JAX:

- Source of truth for semantics.
- Used by `check_nano_moe_port.py`.
- Used to reproduce Nano-JAX initialization for profiling weights.
- Not part of the current timing comparison.

PyTorch reference:

- Implemented in [src/profiling/nano_moe_torch.py](src/profiling/nano_moe_torch.py).
- Mirrors Nano-MoE-JAX directly.
- Can run on CPU or CUDA.
- Computes all experts before gathering selected top-k outputs.
- Used as correctness reference, not as an optimized MoE backend.

MegaBlocks adapter:

- Implemented in [src/profiling/profile_moe_layer.py](src/profiling/profile_moe_layer.py).
- Used only for MegaBlocks runs.
- Copies Nano-compatible weights into a MegaBlocks layer.
- Computes Nano-compatible routing and gates.
- Calls MegaBlocks dispatch / expert / combine.
- Converts layout between Nano `(B, T, D)` and MegaBlocks `(T, B, D)`.
- Computes Nano-compatible aux loss for checking/reporting.

## Why The Adapter Exists

We cannot pass the PyTorch reference directly to MegaBlocks. The reference is a
literal dense PyTorch implementation of Nano-MoE-JAX math; it does not call
MegaBlocks layers or kernels.

MegaBlocks expects its own PyTorch modules, CUDA extensions, tensor layout, router
objects, expert MLP layout, and routing inputs. Also, stock MegaBlocks behavior is
not identical to Nano-MoE-JAX in every detail:

- Nano takes top-k over raw router logits.
- Stock MegaBlocks takes top-k after router softmax.
- Nano experts include Dense biases `b1` and `b2`.
- MegaBlocks stock expert MLPs are bias-free.

The adapter is the translation layer that keeps Nano semantics while using the
MegaBlocks execution path.

Key locations:

- Reference math: `nano_moe_forward()` in [src/profiling/nano_moe_torch.py](src/profiling/nano_moe_torch.py)
- MegaBlocks layer construction: `build_megablocks_layer()` in [src/profiling/profile_moe_layer.py](src/profiling/profile_moe_layer.py)
- Nano-style routing for MegaBlocks: `megablocks_prepare_routing()`
- MegaBlocks timed path: `megablocks_expert_dispatch()`
- Full adapter forward: `megablocks_forward()`

## MegaBlocks

MegaBlocks is a PyTorch/CUDA library for efficient Mixture-of-Experts training
and inference paths. Its core layers are:

```text
MoE   = standard MegaBlocks mixture-of-experts layer
dMoE  = dropless MoE path using block-sparse/grouped expert computation
```

MegaBlocks is not a general runtime that automatically optimizes an arbitrary
JAX or PyTorch model. In this project, it is the CUDA MoE backend/library we call
from a PyTorch adapter.

MegaBlocks consumes PyTorch CUDA tensors and compiled CUDA extensions such as
`megablocks_ops`; grouped dMoE also uses `grouped_gemm`.

## MoE And dMoE

Standard MegaBlocks MoE:

```text
route tokens -> gather/pad by expert -> expert MLP -> scatter/combine
```

MegaBlocks dMoE:

```text
route tokens -> group routed assignments -> grouped/block-sparse expert compute -> scatter/combine
```

dMoE is closer to the main reason MegaBlocks exists, especially for dropless MoE
and grouped GEMM paths. In this checkout, grouped dMoE is BF16-only. The current
curated result reports standard `megablocks_moe`, not final dMoE performance.

## Verification

Verification is two-stage:

```text
Nano-MoE-JAX -> PyTorch reference
PyTorch reference -> MegaBlocks adapter
```

Commands:

```bash
.venv/bin/python src/profiling/check_nano_moe_port.py
.venv/bin/python src/profiling/verify_moe_layer.py
```

Checks include output, router indices, gates, aux loss, and router expert-set
mismatches.

## Timing

Default MegaBlocks timing scope:

```text
megablocks_core = MegaBlocks dispatch / expert / combine
```

Nano-compatible routing and layout preparation are outside this core timing
scope. Correctness is still checked against the full adapter output after timing.

Reference timing uses the dense PyTorch reference on the selected device:

```text
reference/cpu
reference/cuda
```

MegaBlocks timing uses CUDA:

```text
megablocks_moe/cuda
```

## Metrics

Performance metrics:

```text
mean_forward_ms
std_forward_ms
peak_memory_allocated_delta_bytes
tokens_per_second
active_expert_tflops_per_second
backend_estimated_tflops_per_second
tokens_per_expert_min/max/mean/std
```

Correctness metrics:

```text
max_abs_vs_reference
mean_abs_vs_reference
max_rel_vs_reference
aux_loss_abs_diff
router_index_mismatch_count
router_expert_set_mismatch_count
router_gate_max_abs
outlier_diagnosis
```

`reference` means the PyTorch NanoMoE reference. `max_abs_vs_reference` is the
largest elementwise output difference between the MegaBlocks adapter output and
the reference output.

`tokens_per_expert_*` describes how routing load is distributed across experts.
This matters because imbalance and padding affect MegaBlocks work.

## Current Result

Curated result: [results/nano-token-sweep/report.md](results/nano-token-sweep/report.md)

Fixed layer:

```text
d_model=128, d_ff=512, n_experts=4, top_k=2, seq_len=128, dtype=float32
```

Observed on RTX 3080:

| tokens | reference/cuda ms | megablocks_moe/cuda ms | reference/cpu ms | MegaBlocks speedup vs CUDA ref |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 0.3339 | 1.0376 | 1.9342 | 0.32x |
| 4096 | 0.6533 | 1.0397 | 26.0400 | 0.63x |
| 8192 | 1.1926 | 1.0382 | 50.6691 | 1.15x |
| 16384 | 2.2562 | 2.0832 | 104.9297 | 1.08x |

All MegaBlocks rows in this result passed output checks with `max_abs=0` and
zero router expert-set mismatches.

## Interpretation

CPU reference is much slower than CUDA reference for this layer.

For small token counts, dense PyTorch reference on CUDA is faster than MegaBlocks
on CUDA because MegaBlocks has dispatch/expert/combine overhead.

As `tokens = batch_size * seq_len` grows, that overhead is amortized. In the
current RTX 3080 result, MegaBlocks starts to beat the CUDA dense reference at
`tokens=8192`.

The broader expectation is that larger batches, larger expert workloads, and
newer GPUs should be more favorable to MegaBlocks. The current measured result
establishes the token-count crossover for this Nano-sized layer on RTX 3080; it
does not by itself prove performance for larger models or Hopper GPUs.

## Optional JAX Timing

Adding JAX timing is technically simple for this one-layer benchmark because
`check_nano_moe_port.py` already constructs and runs the JAX `MoELayer`.

It should be reported separately as:

```text
jax/cpu or jax/cuda framework timing
```

It should not replace the PyTorch reference for MegaBlocks comparison, because it
would mix framework/runtime differences with backend differences. It is useful as
a sanity/reference point, not as the primary denominator for MegaBlocks speedup.

## Current Claims

Established:

- PyTorch NanoMoE reference matches Nano-MoE-JAX smoke checks.
- MegaBlocks standard MoE adapter matches the PyTorch reference for the curated
  FP32 Nano token sweep.
- CPU reference is slower than CUDA reference for this layer.
- On RTX 3080, standard MegaBlocks MoE crosses the CUDA dense reference around
  `tokens=8192` for this exact Nano MoE shape.

Not established:

- Full-model Nano-MoE-JAX performance.
- Trained checkpoint/router-distribution performance.
- Hopper-generation performance.
- Final dMoE performance claims.
