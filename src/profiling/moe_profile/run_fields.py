"""Shared output-field groups for profiling run summaries."""

from __future__ import annotations


# Presentation-facing semantic fields emitted by ``profile_moe_layer.py``.
# These make each retained run self-describing: the numbers should say which
# router convention, gate convention, activation, and combine convention they
# correspond to, without needing to reverse-engineer the adapter code.
MOE_SEMANTIC_FIELDS = (
    "routing_semantics",
    "softmax_location",
    "topk_location",
    "gate_weight_semantics",
    "gate_multiply_location",
    "expert_mlp_semantics",
    "expert_path_semantics",
)


# Logical MoE-layer op timings emitted by ``--moe-op-profile``. The component
# fields are a disjoint replay-level breakdown, but the authoritative whole-layer
# latency remains ``mean_forward_ms`` from the main timed forward call.
MOE_OP_FIELDS = (
    "moe_op_profile",
    "moe_op_profile_scope",
    "moe_op_profile_warmup",
    "moe_op_profile_iters",
    "moe_op_path",
    "moe_op_expert_capacity",
    "moe_op_input_layout_to_megablocks_ms",
    "moe_op_router_projection_matmul_ms",
    "moe_op_router_full_softmax_ms",
    "moe_op_topk_selection_ms",
    "moe_op_selected_softmax_gating_ms",
    "moe_op_router_aux_loss_ms",
    "moe_op_expert_path_dispatch_compute_combine_ms",
    "moe_op_expert_block_dispatch_compute_combine_ms",
    "moe_op_output_layout_to_nano_ms",
    "moe_op_disjoint_replay_sum_ms",
    "moe_op_whole_moe_layer_replay_ms",
    "moe_op_replay_sum_minus_whole_ms",
    "moe_op_main_replay_sum_ms",
    "moe_op_profile_note",
)
