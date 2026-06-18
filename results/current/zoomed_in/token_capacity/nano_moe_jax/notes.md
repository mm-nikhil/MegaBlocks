# Model Token-Capacity Sweep

model_shape_name: `nano_moe_jax`
simulation_level: `exact_adapter`
weight_source: `nano_jax_init`
dtype: `bfloat16`

check_output: `False`
outlier_abs_threshold: `0.001`

## Bias / Scope Note

This zoomed-in run is the no-bias initialized Nano baseline. `weight_source=nano_jax_init` has zero expert Dense biases (`expert_bias_max_abs=0.0` in `summary.csv`), so the MegaBlocks lines use the stock bias-free expert path rather than the trained Nano bias adapter. This is a performance-focused no-bias run; output correctness checks were skipped.

Max successful `N` by backend:

- `megablocks_dmoe`: `16384`
- `megablocks_moe`: `16384`
- `reference_dense_ffn`: `16384`

This sweep varies `N = B*T`, the number of input-token hidden rows at one MoE layer.
It is not generated output tokens per second.

The first-cut dashboard shows:

- `mean_forward_ms`: average timed forward call for the selected timing scope.
- `ms_per_input_token`: `mean_forward_ms / N`.
- `active_expert_tflops_per_second`: useful active expert math normalized by runtime.
- `padding_factor`: backend expert rows divided by routed token-expert pairs.

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
