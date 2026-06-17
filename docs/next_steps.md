# Next Steps

This document tracks what is done and what should happen next. Detailed
MegaBlocks source notes live in `docs/moe_megablocks_deep_dive.md`; metrics and
results live in `docs/metrics_and_results.md`.

## Done

`nano_moe_jax` exact adapter:

```text
PyTorch reference path
MegaBlocks standard moe path
MegaBlocks grouped dMoE path
correctness checks for exact-adapter runs
token-capacity dashboard
phase-profile dashboard
standard-moe high-N binned-gather grid-limit finding
```

`olmoe_1b_7b_0924` Level 1 shape/expert simulation:

```text
synthetic dense GLU PyTorch CUDA reference
standard megablocks_moe with local GLU wrapper
grouped megablocks_dmoe GLU path
memory preflight before large allocations
token-capacity dashboard with caveat in the plot
phase-profile dashboard with caveat in the plot
reference_dense_glu memory-preflight finding
```

Documentation cleanup:

```text
MegaBlocks implementation details centralized in docs/moe_megablocks_deep_dive.md
metrics/results centralized in docs/metrics_and_results.md
old Findings.md removed
READMEs simplified to point at the specific docs
```

MegaBlocks source walkthrough:

```text
documented Arguments contract and profiler adapter mapping
documented stock router and Nano/Level 1 adapter-router semantics
documented standard moe padded binned path and high-N binned-grid limitation
documented grouped dMoE path, grouped MLP/GLU, and BF16 grouped_gemm limitation
documented shared expert behavior for DeepSeek-shaped Level 1
```

## Next

### 1. Shape-Level Behavior Sweep

Goal:

```text
Isolate "bigger MoE layer shape" from "more token rows" by holding N fixed and
sweeping the MoE shape axes that determine per-row work.
```

Fixed input-row count:

```text
Use one safe N first, likely 8192 or 16384.
Keep B/T consistent with the existing runner, but interpret the run by N = B*T.
Run larger N only after this fixed-N shape comparison is clear.
```

Shape axes:

```text
width sweep: D and H increase together in a small number of readable points
expert-count sweep: E increases while D/H/K are fixed
top-k sweep: K increases while D/H/E are fixed
expert-type comparison: FFN/MLP vs GLU/SwiGLU for the same D/H/E/K
catalog points: nano_moe_jax and olmoe_1b_7b_0924
```

Backends:

```text
reference_dense_ffn or reference_dense_glu where memory allows
megablocks_moe
megablocks_dmoe
```

Dashboard 1, config-only shape summary:

```text
routed_params vs shape
active_expert_flops_per_input_row vs shape
router_flops_per_input_row vs shape
dense_reference_rows / sparse_rows ratio vs shape
```

Dashboard 2, fixed-N performance:

```text
mean_forward_ms vs active_expert_flops_per_input_row
ms_per_input_row vs active_expert_flops_per_input_row
active_expert_tflops_per_second vs active_expert_flops_per_input_row
padding_factor vs active_expert_flops_per_input_row
```

Questions this chunk should answer:

```text
Does MegaBlocks amortize dispatch better as useful expert compute per row grows?
Where does standard moe padding become visible?
Where does grouped dMoE pull away from standard moe?
At what shapes does the dense reference become memory/time infeasible?
Is Nano slow-looking because N is small, or because D/H/E/K make each row cheap?
```

### 2. DeepSeek Level 1 Preflight

After the shape-level sweep:

```text
Run deepseek_v3_moe_layer through memory preflight.
Record whether full DeepSeek-shaped Level 1 fits on this GPU.
If rejected, record that as the result instead of forcing an OOM.
Do not label scaled or sharded follow-ups as "DeepSeek-V3 on MegaBlocks."
```

### 3. Router-Semantic Adapters

Later work:

```text
OLMoE router adapter
DeepSeek-V3 router adapter
```

This is required before claiming exact OLMoE or DeepSeek model semantics.
