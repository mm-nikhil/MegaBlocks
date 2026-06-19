# Model Token-Capacity Sweep

model_shape_name: `nano_moe_jax`
simulation_level: `exact_adapter`
weight_source: `trained_nano_checkpoint`
requested_dtype: `float32`

checkpoint_dir: `results/trained_nano_moe_checkpoint`
checkpoint_block_index: `0`

correctness_gate: `True`
row_check_output: `False`
row_check_outlier_abs_threshold: `0.001`

NanoJAX correctness is checked once before the sweep and recorded in
`verification_summary.json` and `verification_summary.md`. Performance
rows do not run dense reference checks unless `--row-check-output` is set.

Max successful `N` by backend:

- `megablocks_dmoe [bfloat16] (BF16-only dMoE)`: `524288`
- `megablocks_moe [float32]`: `65536`

This sweep varies `N = B*T`, the number of input-token hidden rows at one MoE layer.
It is not generated output tokens per second.

Primary graph:

- `graphs_moe_layer_ops.png`
- disjoint replay timings for the logical MoE-layer blocks.

Use this graph to explain where the MoE-layer replay spends time.
Use `mean_forward_ms` in `summary.csv` for authoritative production
latency.

MoE-layer op diagnostics:

This result includes `graphs_moe_layer_ops.png` and `moe_op_*` columns.
Those fields are disjoint diagnostic replays of logical MoE blocks:
input layout, router projection matmul, full row-wise router softmax,
top-k expert selection, row-wise selected-gate softmax, aux/router
bookkeeping, expert block, and output layout.
The expert block is MegaBlocks dispatch/sort/binning, gather, expert
MLP compute, and weighted scatter/combine. Gate multiply and reduce
back to token rows are folded into weighted scatter/combine.
The component sum and whole replay are reported for sanity checking,
but the authoritative layer latency remains `mean_forward_ms`.

Dtype policy:

The run-level NanoJAX dtype defaults to the model-shape catalog dtype.
For `megablocks_dmoe`, the local grouped GEMM extension is BF16-only,
so dMoE rows may use `dtype=bfloat16` with
`dtype_policy=dmoe_bf16_only_local_grouped_gemm` even when the requested
NanoJAX dtype is FP32.


Backend success, failure, and unsupported status is recorded in `backend_status.md`.

Shape:

```json
{
  "activation": "gelu_tanh",
  "dtype": "float32",
  "expert_intermediate_size": 512,
  "expert_type": "ffn",
  "hidden_size": 128,
  "max_position_embeddings": 128,
  "notes": "This is the current exact adapter target. Nano computes all routed experts in the reference path, while MegaBlocks computes selected experts.",
  "num_experts_per_token": 2,
  "num_hidden_layers": 4,
  "num_routed_experts": 4,
  "num_shared_experts": 0,
  "router_score_function": "softmax_logits_for_probs_topk_over_logits",
  "shared_expert_intermediate_size": 0,
  "simulation_level": "exact_adapter",
  "source": "third_party/Nano-MoE-JAX/nano_moe/config.py",
  "source_verified_on": "2026-06-17"
}
```

Failures:

- N=131072 backend=megablocks_moe: returncode=1 reason=error: Triton Error [CUDA]: invalid argument
- N=262144 backend=megablocks_moe: returncode=1 reason=error: Triton Error [CUDA]: invalid argument
- N=524288 backend=megablocks_moe: returncode=1 reason=error: Triton Error [CUDA]: invalid argument
- N=1048576 backend=megablocks_moe: returncode=1 reason=error: Memory preflight rejected this run before allocation. estimated=9470216908 base_estimated=7014975488 allowed=8903983104 free=9893314560 total=10351935488 fraction=0.9. Use a smaller N or --skip-memory-preflight if you intentionally want to test the limit.
- N=1048576 backend=megablocks_dmoe: returncode=1 reason=error: CUDA out of memory. Tried to allocate 2.00 GiB. GPU 0 has a total capacity of 9.64 GiB of which 1.80 GiB is free. Process 1855005 has 10.85 MiB memory in use. Including non-PyTorch memory, this process has 7.62 GiB memory in use. Of the allocated memory 5.62 GiB is allocated by PyTorch, and 1.74 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
