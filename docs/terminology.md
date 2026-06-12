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

Matches:

- Top-k expert indices match because softmax is monotonic in logits.
- Gates match when MegaBlocks uses `moe_normalize_expert_weights=1`.
- GELU matches with `torch.nn.functional.gelu(..., approximate="tanh")`.
- Tensor layout maps from Nano `(B, T, D)` to MegaBlocks `(T, B, D)`.

Mismatches:

- Nano experts have per-expert Dense biases. MegaBlocks expert MLPs are bias-free.
- Nano auxiliary loss uses only top-1 assignment. MegaBlocks load balancing counts
  all top-k assignments and scales differently.

## Interpretation Rule

Only report a run as an exact Nano-MoE-JAX MoE implementation if:

1. The PyTorch reference check passes.
2. Expert bias semantics are matched or the tested weights have zero expert biases.
3. Router gate normalization is configured to match NanoMoE.
4. Dropout is disabled for deterministic forward timing.
