# Terminology And Semantics

## NanoMoE MoE Boundary

For input `x` with shape `(batch, seq_len, d_model)`:

```text
router_logits = x @ router_kernel
router_probs = softmax(router_logits)
top_values, top_indices = top_k(router_logits, top_k)
gates = softmax(top_values)

expert_i(x) = gelu(x @ w1_i + b1_i, approximate=tanh) @ w2_i + b2_i
output = weighted sum of selected expert outputs
```

Auxiliary loss:

```text
n_experts * sum_i(top1_token_fraction_i * mean_router_probability_i)
```

## Matching MegaBlocks

The current adapter matches NanoMoE semantics first, then calls MegaBlocks'
dispatch/expert/combine path.

Matches:

- Top-k expert indices are computed from raw logits, matching NanoMoE.
- Gates are `softmax(top_values)`, matching NanoMoE.
- Auxiliary loss uses NanoMoE's top-1 load-balancing formula.
- GELU matches with `torch.nn.functional.gelu(..., approximate="tanh")`.
- Tensor layout maps from Nano `(B, T, D)` to MegaBlocks `(T, B, D)`.
- Standard `moe` supports nonzero Nano expert biases through the local
  bias-aware expert MLP adapter.

Remaining limits:

- Grouped `dmoe` still does not support nonzero Nano expert biases.
- Default profiling timing uses `megablocks_core`, so Nano-compatible routing is
  prepared outside the timed region and the timed region is MegaBlocks
  dispatch/expert/combine.

## Interpretation Rule

Only report a run as an exact Nano-MoE-JAX MoE implementation if:

1. The PyTorch reference check passes.
2. Expert bias semantics are matched or the tested weights have zero expert biases.
3. Router indices, gates, output, and auxiliary loss match the PyTorch reference.
4. Dropout is disabled for deterministic forward timing.
