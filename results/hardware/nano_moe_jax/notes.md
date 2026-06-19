# Hardware MoE Profile

model_shape_name: `nano_moe_jax`
timing_scope: `adapter_boundary`
weight_source: `trained_nano_checkpoint`
dtype: `bfloat16`

Metric:

`clock_compute_utilization = W / (t * f * P * R)`

- `W`: active useful selected-expert FLOPs.
- `t`: measured MoE runtime from `profile_moe_layer.py`.
- `f`: SM clock in cycles/second.
- `P`: SM count / PE count.
- `R`: assumed peak FLOPs per SM cycle.

This is clock-derived compute utilization, not observed GPU idle cycles,
SM active cycles, hardware occupancy, or measured memory stalls.

GPU and denominator:

- name: `NVIDIA GeForce RTX 3080`
- compute capability: `8.6`
- SM count used: `68` (torch_cuda_properties)
- current SM clock MHz reported: `225.0`
- max SM clock MHz reported: `2115.0`
- SM clock MHz used: `2115.0` (max_sm_clock)
- peak FLOPs per SM-cycle: `256.0`
- peak TFLOP/s: `36.81792`

Per-op timing diagnostics:

The optional `moe_op_*` fields are independent diagram-level replays.
They are separate from the clock-derived metric. The expert block timing
is the whole MegaBlocks expert dispatch/compute/combine call. The
`moe_op_gate_multiply_combine_ms` field times the weighted scatter/combine
subset, so it is not additive with the expert-block timing.

The full presentation timing boundary is the adapter boundary:
router projection, top-k, selected-logit softmax/gating, expert block,
gate multiply, and combine back to Nano `[N x D]` layout.
