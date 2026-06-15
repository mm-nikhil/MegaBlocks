# Architecture Summary

## Goal

Measure GPU execution time for the Nano-MoE-JAX MoE-layer semantics using
MegaBlocks where the semantics can be matched.

This repo does not run JAX inside MegaBlocks. JAX is the source-of-truth
implementation. PyTorch is the bridge to MegaBlocks.

## Boundary

The profiled boundary is one MoE layer:

```text
input:  hidden states, shape (batch, seq_len, d_model)
output: hidden states, shape (batch, seq_len, d_model)
extra:  scalar auxiliary load-balancing loss
```

Inside the boundary:

```text
router logits -> top-k expert ids -> gates -> expert FFNs -> weighted combine
```

The full Transformer block, attention layer, token embeddings, optimizer, and
training loop are outside the current timing boundary.

## Reference

`reference` means the PyTorch implementation in `src/profiling/nano_moe_torch.py`.
It is written to match `third_party/Nano-MoE-JAX/nano_moe/layers.py`.

Correctness is checked against real JAX execution before MegaBlocks timings are
interpreted.

## MegaBlocks Mapping

Matched:

- Router weight layout.
- Top-k expert ids.
- Gate normalization.
- Expert weight layout.
- GELU approximation.
- Input/output layout conversion between Nano `(B, T, D)` and MegaBlocks `(T, B, D)`.

Not matched by stock MegaBlocks:

- Nano-MoE-JAX expert Dense layers have biases.
- MegaBlocks expert MLPs are bias-free.

Therefore current MegaBlocks timings use `--zero-expert-biases`. These timings
represent a biasless NanoMoE MoE variant, not the exact default JAX layer.

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
dtype=float32
zero_expert_biases=true
```

Runs but not final-equivalence:

```text
backend=megablocks, megablocks_layer=moe,  dtype=float16
backend=megablocks, megablocks_layer=dmoe, dtype=bfloat16
```

Those lower-precision runs execute on GPU, but current output checks show large
max-error outliers even when mean error is small.

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
```

This is average forward-pass wall time measured with CUDA events after warmup.

Output-check metrics:

```text
max_abs_vs_reference
mean_abs_vs_reference
max_rel_vs_reference
max_abs_reference
```

Here `reference` is the PyTorch NanoMoE reference. `max` means the largest
elementwise absolute difference across the output tensor.

## Performance Mapping Plan

Record a JSONL row for every benchmark run. Sweep:

- backend: reference, MegaBlocks MoE, MegaBlocks dMoE
- dtype: float32, float16, bfloat16 where supported
- tokens: batch size times sequence length
- model dimensions: `d_model`, `d_ff`
- MoE parameters: `n_experts`, `top_k`

Treat a timing as publishable only when the relevant output-check status is
understood and recorded.
