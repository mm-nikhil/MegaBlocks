# Architecture Summary

## Goal

Measure GPU execution time for the Nano-MoE-JAX MoE-layer semantics using
MegaBlocks where the semantics can be matched.

This repo does not run JAX inside MegaBlocks. JAX is the source-of-truth
implementation. PyTorch is the bridge to MegaBlocks.

## Boundary

The correctness boundary is one MoE layer:

```text
input:  hidden states, shape (batch, seq_len, d_model)
output: hidden states, shape (batch, seq_len, d_model)
extra:  scalar auxiliary load-balancing loss
```

Inside the boundary:

```text
Nano layout adapter -> router logits -> top-k expert ids -> gates -> expert FFNs -> weighted combine -> Nano layout adapter
```

The full Transformer block, attention layer, token embeddings, optimizer, and
training loop are outside the current scope.

## Reference

`reference` means the PyTorch implementation in `src/profiling/nano_moe_torch.py`.
It is written to match `third_party/Nano-MoE-JAX/nano_moe/layers.py`.

Correctness is checked against real JAX execution before MegaBlocks timings are
interpreted.

Profiling weights default to `--weight-source nano_jax_init`, which initializes a
real Nano-MoE-JAX `MoELayer` with Nano's Flax initializers and converts those
parameters into PyTorch. Use `--weight-source synthetic` only for stress tests
with simple `N(0, 0.02)` synthetic weights.

## MegaBlocks Mapping

Matched:

- Router weight layout.
- Top-k expert ids from raw logits, matching Nano-MoE-JAX.
- Gate normalization over selected logits.
- Auxiliary load-balancing loss.
- Expert weight layout.
- GELU approximation.
- Input/output layout conversion between Nano `(B, T, D)` and MegaBlocks `(T, B, D)`.

The adapter computes Nano-compatible routing itself, then feeds the selected
expert weights and indices into MegaBlocks' dispatch/expert/combine path. This is
intentional: stock MegaBlocks takes top-k after the full softmax, while
Nano-MoE-JAX takes top-k on raw logits. Those are equivalent in exact arithmetic
but can differ for low-precision near-ties.

Not matched by stock MegaBlocks:

- Nano-MoE-JAX expert Dense layers have biases.
- MegaBlocks expert MLPs are bias-free.

The profiling adapter supports Nano expert biases for the standard MegaBlocks
MoE path and for BF16 grouped dMoE:

```text
--megablocks-layer moe --use-expert-biases
--megablocks-layer dmoe --dtype bfloat16 --use-expert-biases
```

This keeps MegaBlocks sorting, gathering, scattering, and combining, but replaces
the stock bias-free expert MLP with a bias-aware adapter matching Nano-MoE-JAX
when nonzero expert biases are present:

```text
x @ w1 + b1 -> GELU -> x @ w2 + b2
```

For standard `moe`, the adapter uses a bias-aware batched MLP. For grouped
`dmoe`, the adapter uses grouped GEMMs and adds `b1`/`b2` in the grouped routed
layout. The current grouped GEMM extension accepts BF16 inputs only, so dMoE
FP16/FP32 rows are unsupported in this checkout.

When the actual expert biases are zero, as they are for Nano's default
initializers, the stock MegaBlocks expert MLP is kept because it is already exact
for that bias term.

## GPU Execution

When running `profile_moe_layer.py --device cuda`, tensors are created on the
CUDA device and timing uses CUDA events. MegaBlocks runs through compiled CUDA
extensions:

- `megablocks_ops`
- `grouped_gemm_backend`

If those extensions are missing, the MegaBlocks backend refuses to run.

## Current Working Set

Trusted correctness smoke:

```text
backend=megablocks
megablocks_layer=moe
dtype=float32,float16
weight_source=nano_jax_init
use_expert_biases=true
```

Also validated for zero-bias Nano-initialized weights:

```text
backend=megablocks, megablocks_layer=dmoe, dtype=bfloat16
```

Also validated for synthetic nonzero expert biases:

```text
backend=megablocks, megablocks_layer=dmoe, dtype=bfloat16, use_expert_biases=true
```

BF16 rows can still show small numeric/expert-path differences against the dense
PyTorch reference on some shapes. These are recorded in the sweep output and
should be reported separately from routing correctness.

Unsupported as a model dtype:

```text
int32
```

`int32` is used for routing indices, not for neural-network activations and GEMMs.
An integer activation benchmark would require a quantized/int8 design, not this
NanoMoE/MegaBlocks mapping.

## Metrics

Timing metric:

```text
mean_forward_ms
std_forward_ms
min_forward_ms
max_forward_ms
```

These are forward-pass wall times measured with CUDA events after warmup.

For MegaBlocks, default timing scope is `megablocks_core`: Nano-compatible routing
is prepared outside the timed region, then the timed callable runs MegaBlocks'
dispatch/expert/combine path. This excludes the adapter's `(B, T, D)` layout
conversion, router logits/top-k/gates, and auxiliary-loss bookkeeping.

Use `--timing-scope adapter_boundary` only when intentionally timing the full
Nano-compatible adapter boundary. Router diagnostics and expert-count histograms
are always collected once after timing so they do not dominate nano-scale timings.

Throughput and resource metrics:

```text
tokens_per_second
peak_memory_allocated_bytes
peak_memory_allocated_delta_bytes
peak_memory_reserved_bytes
active_expert_tflops_per_second
backend_estimated_tflops_per_second
tokens_per_expert_min
tokens_per_expert_max
```

FLOP metrics are estimates. For the dense PyTorch reference, estimated backend
FLOPs include all experts because the reference computes all expert outputs before
selecting top-k. For MegaBlocks MoE, estimated backend FLOPs account for routed
expert rows and standard-MoE padding. These estimates do not include softmax,
top-k, layout copies, or auxiliary-loss bookkeeping.

Output-check metrics:

```text
max_abs_vs_reference
mean_abs_vs_reference
max_rel_vs_reference
max_abs_reference
aux_loss_reference
aux_loss_actual
aux_loss_abs_diff
router_indices_equal
router_index_mismatch_count
router_expert_set_mismatch_count
router_gate_max_abs
router_gate_mean_abs
router_gate_aligned_max_abs
output_outlier_token_count
max_abs_on_router_match_tokens
max_abs_on_router_mismatch_tokens
outlier_diagnosis
```

Here `reference` is the PyTorch NanoMoE reference. `max` means the largest
elementwise absolute difference across the output tensor.

Router index mismatch is recorded positionally, and expert-set mismatch is
recorded per token. Expert-set mismatch is what we use to identify actual
router-choice flips, since a top-k order swap alone does not change the weighted
expert set.

`outlier_diagnosis` separates likely router-choice issues from numeric/expert-path
issues:

- `within_threshold`: max output difference is below threshold.
- `router_choice_flip_at_max_error`: the token with largest error routed to a
  different expert.
- `router_choice_flips_elsewhere`: some router choices differ, but not at the max
  error token.
- `numeric_or_expert_path`: router choices match, so the error is likely numeric
  precision or expert computation.

## Benchmarking Model

Record a JSONL row for every benchmark run. A sweep is still one MoE layer; the
count is the number of benchmark configurations, not the number of model layers.

Default sweep mode is focused: vary one axis at a time around a NanoMoE baseline,
then compare `reference` vs `megablocks_moe`.

Full Cartesian grid mode is available with `--preset grid`, but it is noisy and
should be used only when we already know which axes are worth expanding.

Sweep axes:

- backend: reference, MegaBlocks MoE, MegaBlocks dMoE
- dtype: float32, float16, bfloat16 where supported
- weight source: Nano-JAX initialized weights by default, synthetic weights for stress tests
- tokens: batch size times sequence length
- model dimensions: `d_model`, `d_ff`
- MoE parameters: `n_experts`, `top_k`

Treat a timing as publishable only when the relevant output-check status is
understood and recorded.
