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
- Grouped `dmoe` supports nonzero Nano expert biases through the local BF16
  bias-aware grouped MLP adapter.

Remaining limits:

- Grouped `dmoe` FP16/FP32 rows are unsupported by the current grouped GEMM
  extension.
- BF16 rows may need a dtype-specific numeric tolerance for expert-path
  differences.
- `moe_layer` timing includes Nano-compatible routing plus the MegaBlocks expert
  block. `expert_path` timing starts after routing has been prepared and measures
  MegaBlocks dispatch/sort/binning, gather, expert MLP, and weighted
  scatter/combine.

## Benchmark Path

The benchmark uses one MoE execution convention so the result does not mix model
semantics:

- Router projection runs with NanoJAX checkpoint weights.
- Full row-wise softmax is computed for router probabilities and auxiliary
  load-balancing loss.
- Top-k expert selection is computed over raw router logits.
- Gate weights are a row-wise softmax over the selected top-k logits.
- MegaBlocks receives the selected expert indices and gates and performs sparse
  dispatch, expert MLP compute, and weighted scatter/combine.

This keeps the model behavior fixed across the PyTorch reference, MegaBlocks
MoE, and MegaBlocks dMoE lines. The comparison is therefore about implementation
performance, not about changing the router definition.

## Interpretation Rule

Only report a run as an exact Nano-MoE-JAX MoE implementation if:

1. The PyTorch reference check passes.
2. Expert bias semantics are matched or the tested weights have zero expert biases.
3. Router indices, gates, output, and auxiliary loss match the PyTorch reference.
4. Dropout is disabled for deterministic forward timing.
