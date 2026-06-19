# Hardware MoE Profile

model_shape_name: `nano_moe_jax`
timing_scope: `moe_layer`
weight_source: `trained_nano_checkpoint`
requested_dtype: `float32`
effective_dtypes: `bfloat16, float32`
correctness_gate: `True`

Metric:

`clock_compute_utilization = W / (t * f * P * R)`

- `W`: active useful selected-expert FLOPs.
- `t`: measured MoE runtime from `profile_moe_layer.py`.
- `f`: SM clock in cycles/second.
- `P`: SM count / PE count.
- `R`: assumed peak FLOPs per SM cycle.

This is clock-derived compute-slot utilization, not observed GPU idle cycles,
SM active cycles, hardware occupancy, or measured memory stalls.
`clock_equivalent_unused_sms` is an algebraic compute-slot equivalent,
not a measurement of physically idle SMs.

GPU and denominator:

- name: `NVIDIA GeForce RTX 3080`
- compute capability: `8.6`
- SM count used: `68` (torch_cuda_properties)
- current SM clock MHz reported: `225.0`
- max SM clock MHz reported: `2115.0`
- SM clock MHz used: `2115.0` (max_sm_clock)
- peak FLOPs per SM-cycle: `256.0`
- peak TFLOP/s: `36.81792`

For the default RTX 3080 FP32 presentation run, the denominator is the
configured CUDA-core roof:

`max_sm_clock * SM_count * peak_flops_per_sm_cycle`

With the local defaults this is `2115 MHz * 68 * 256 = 36.82 TFLOP/s`.
If BF16 dMoE rows appear in the same run, they are still normalized by
this configured denominator unless `--peak-flops-per-sm-cycle` is changed;
do not interpret that as a BF16 tensor-core roof.

Per-op timing diagnostics:

The optional `moe_op_*` fields are disjoint diagnostic replays of logical
MoE blocks. They are separate from the clock-derived metric. Their
component sum is useful for explanation, but the authoritative whole-layer
latency remains `mean_forward_ms`. The expert-block timing is the whole
MegaBlocks dispatch/sort/binning, gather, expert MLP, and weighted
scatter/combine call; lower-level gather/MLP/scatter diagnostics belong
to `--phase-profile`.

The full presentation timing boundary is `moe_layer`:
router projection, full row-wise router softmax, top-k, row-wise
selected-logit softmax/gating, expert block, weighted scatter/combine,
and output layout back to Nano `[N x D]`.

Dtype policy:

The run-level NanoJAX dtype defaults to the model-shape catalog dtype.
For `megablocks_dmoe`, the local grouped GEMM extension is BF16-only,
so dMoE rows may use `dtype=bfloat16` with
`dtype_policy=dmoe_bf16_only_local_grouped_gemm` even when the requested
NanoJAX dtype is FP32. The hardware denominator remains the configured
`peak_flops_per_sm_cycle`; mixed-dtype rows should be interpreted with
that assumption visible.

NanoJAX correctness is checked once before the sweep and recorded in
`verification_summary.json` and `verification_summary.md`. Large hardware
rows do not run dense reference checks row-by-row.

Failures:

- N=131072 backend=megablocks_moe: Profiler failed for timing_megablocks_moe_N131072.jsonl: error: Triton Error [CUDA]: invalid argument
- N=262144 backend=megablocks_moe: Profiler failed for timing_megablocks_moe_N262144.jsonl: error: Triton Error [CUDA]: invalid argument
- N=524288 backend=megablocks_moe: Profiler failed for timing_megablocks_moe_N524288.jsonl: error: Triton Error [CUDA]: invalid argument
- N=1048576 backend=megablocks_moe: Profiler failed for timing_megablocks_moe_N1048576.jsonl: error: Memory preflight rejected this run before allocation. estimated=9470216908 base_estimated=7014975488 allowed=8903983104 free=9893314560 total=10351935488 fraction=0.9. Use a smaller N or --skip-memory-preflight if you intentionally want to test the limit.
