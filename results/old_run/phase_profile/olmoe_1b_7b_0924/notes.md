# Phase Profile

model_shape_name: `olmoe_1b_7b_0924`
simulation_level: `level1_shape_expert`

## Simulation Caveat

Caveat: synthetic shape/expert simulation; not exact checkpoint/router semantics.

This means the run uses the catalog's MoE layer geometry and expert type
such as `D`, `H`, `E`, `K`, GLU/SwiGLU, activation, and dtype, with
synthetic weights. It does not load the real model checkpoint and does
not implement the model-specific router exactly.

For OLMoE-shaped runs, `reference_dense_glu` is a PyTorch CUDA dense
all-expert GLU baseline for this synthetic shape. It is useful as a
dense-vs-sparse comparison point, but it is not exact OLMoE execution.

Max successful `N` by backend:

- `megablocks_dmoe`: `32768`
- `megablocks_moe`: `32768`

This sweep varies `N = B*T`, the number of input-token hidden rows at one MoE layer.
It is not generated output tokens per second.

The dashboard shows:

- phase timings for MegaBlocks routing metadata, gather, expert MLP, and scatter.

Phase timings are independent diagnostic replays of MegaBlocks operations.
Use them to explain bottlenecks, not as an exact additive breakdown of
`mean_forward_ms`.

Backend success, failure, and unsupported status is recorded in `backend_status.md`.

Shape:

```json
{
  "activation": "silu",
  "dtype": "bfloat16",
  "expert_intermediate_size": 1024,
  "expert_type": "glu",
  "hidden_size": 2048,
  "max_position_embeddings": 4096,
  "norm_topk_prob": false,
  "notes": "MegaBlocks can approximate this MoE-layer geometry with mlp_type=glu, E=64, K=8, D=2048, H=1024. Exact semantic matching needs an OLMoE-compatible PyTorch reference and router behavior.",
  "num_experts_per_token": 8,
  "num_hidden_layers": 16,
  "num_routed_experts": 64,
  "num_shared_experts": 0,
  "router_aux_loss_coef": 0.01,
  "router_score_function": "softmax_topk",
  "shared_expert_intermediate_size": 0,
  "simulation_level": "level1_shape_expert",
  "source": "https://huggingface.co/allenai/OLMoE-1B-7B-0924/raw/main/config.json",
  "source_verified_on": "2026-06-17"
}
```
