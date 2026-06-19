"""Shared output-field groups for profiling run summaries."""

from __future__ import annotations


# Logical MoE-layer op timings emitted by ``--moe-op-profile``. These are
# independent diagnostic replays, not an additive decomposition of forward time.
MOE_OP_FIELDS = (
    "moe_op_profile",
    "moe_op_profile_scope",
    "moe_op_profile_warmup",
    "moe_op_profile_iters",
    "moe_op_path",
    "moe_op_expert_capacity",
    "moe_op_router_projection_matmul_ms",
    "moe_op_topk_selection_ms",
    "moe_op_selected_softmax_gating_ms",
    "moe_op_expert_block_dispatch_compute_combine_ms",
    "moe_op_gate_multiply_combine_ms",
    "moe_op_output_layout_to_nano_ms",
    "moe_op_main_replay_sum_ms",
    "moe_op_profile_note",
)

