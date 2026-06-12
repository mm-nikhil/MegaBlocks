"""PyTorch reference implementation of Nano-MoE-JAX's MoE layer.

This module is intentionally small and literal. It mirrors
``Nano-MoE-JAX/nano_moe/layers.py`` at the MoE boundary:

* router logits use a bias-free dense projection
* expert indices come from top-k over raw logits
* gates are a softmax over the selected top-k logits
* auxiliary loss uses only the top-1 assignment, Switch-style
* every expert has two biased dense layers with tanh-approx GELU

The implementation computes all experts, just like Nano-MoE-JAX. It is a
correctness reference, not an optimized backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class NanoMoEWeights:
    """Torch tensors for one NanoMoE MoE layer.

    Shapes:
        router_kernel: (d_model, n_experts)
        w1: (n_experts, d_model, d_ff)
        b1: (n_experts, d_ff)
        w2: (n_experts, d_ff, d_model)
        b2: (n_experts, d_model)
    """

    router_kernel: torch.Tensor
    w1: torch.Tensor
    b1: torch.Tensor
    w2: torch.Tensor
    b2: torch.Tensor

    @property
    def n_experts(self) -> int:
        return int(self.w1.shape[0])

    @property
    def d_model(self) -> int:
        return int(self.w1.shape[1])

    @property
    def d_ff(self) -> int:
        return int(self.w1.shape[2])

    def to(
        self,
        *,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "NanoMoEWeights":
        return NanoMoEWeights(
            router_kernel=self.router_kernel.to(device=device, dtype=dtype),
            w1=self.w1.to(device=device, dtype=dtype),
            b1=self.b1.to(device=device, dtype=dtype),
            w2=self.w2.to(device=device, dtype=dtype),
            b2=self.b2.to(device=device, dtype=dtype),
        )

    def max_abs_bias(self) -> float:
        return float(torch.maximum(self.b1.abs().max(), self.b2.abs().max()).item())


@dataclass(frozen=True)
class NanoMoEForward:
    output: torch.Tensor
    aux_loss: torch.Tensor
    router_logits: torch.Tensor
    router_probs: torch.Tensor
    gates: torch.Tensor
    indices: torch.Tensor


def _as_torch(
    value,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.as_tensor(np.array(value, copy=True), device=device, dtype=dtype).contiguous()


def from_flax_moe_params(
    params: Mapping,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> NanoMoEWeights:
    """Convert a Flax ``MoELayer`` params tree into ``NanoMoEWeights``."""

    if "params" in params:
        params = params["params"]

    router_kernel = _as_torch(
        params["Router_0"]["Dense_0"]["kernel"],
        device=device,
        dtype=dtype,
    )

    expert_ids = sorted(
        int(key.split("_", 1)[1])
        for key in params.keys()
        if key.startswith("expert_")
    )
    if not expert_ids:
        raise ValueError("No expert_N entries found in Flax MoELayer params.")

    w1 = []
    b1 = []
    w2 = []
    b2 = []
    for expert_id in expert_ids:
        expert = params[f"expert_{expert_id}"]
        w1.append(_as_torch(expert["Dense_0"]["kernel"], device=device, dtype=dtype))
        b1.append(_as_torch(expert["Dense_0"]["bias"], device=device, dtype=dtype))
        w2.append(_as_torch(expert["Dense_1"]["kernel"], device=device, dtype=dtype))
        b2.append(_as_torch(expert["Dense_1"]["bias"], device=device, dtype=dtype))

    return NanoMoEWeights(
        router_kernel=router_kernel,
        w1=torch.stack(w1, dim=0),
        b1=torch.stack(b1, dim=0),
        w2=torch.stack(w2, dim=0),
        b2=torch.stack(b2, dim=0),
    )


def nano_moe_forward(
    x: torch.Tensor,
    weights: NanoMoEWeights,
    *,
    top_k: int,
    deterministic: bool = True,
    dropout_p: float = 0.0,
) -> NanoMoEForward:
    """Run the exact Nano-MoE-JAX MoE-layer semantics in PyTorch.

    Args:
        x: Input tensor with shape ``(batch, seq_len, d_model)``.
        weights: Converted Flax MoE parameters.
        top_k: Number of active experts per token.
        deterministic: Matches Flax dropout convention.
        dropout_p: Dropout probability on the combined MoE output.
    """

    if x.ndim != 3:
        raise ValueError(f"Expected x to have shape (batch, seq, hidden), got {tuple(x.shape)}.")
    if top_k < 1 or top_k > weights.n_experts:
        raise ValueError(f"top_k={top_k} must be in [1, {weights.n_experts}].")
    if x.shape[-1] != weights.d_model:
        raise ValueError(f"Input hidden size {x.shape[-1]} does not match weights {weights.d_model}.")

    router_logits = torch.matmul(x, weights.router_kernel)
    router_probs = torch.softmax(router_logits, dim=-1)

    top_k_values, top_k_indices = torch.topk(router_logits, top_k, dim=-1)
    gates = torch.softmax(top_k_values, dim=-1)

    top1 = top_k_indices[..., 0]
    dispatch_mask = F.one_hot(top1, num_classes=weights.n_experts).to(router_probs.dtype)
    token_fraction = dispatch_mask.mean(dim=(0, 1))
    prob_mean = router_probs.mean(dim=(0, 1))
    aux_loss = weights.n_experts * torch.sum(token_fraction * prob_mean)

    hidden = torch.einsum("btd,edf->ebtf", x, weights.w1)
    hidden = hidden + weights.b1[:, None, None, :]
    hidden = F.gelu(hidden, approximate="tanh")
    expert_outputs = torch.einsum("ebtf,efd->ebtd", hidden, weights.w2)
    expert_outputs = expert_outputs + weights.b2[:, None, None, :]

    batch_size, seq_len, _ = x.shape
    batch_idx = torch.arange(batch_size, device=x.device)[:, None, None]
    seq_idx = torch.arange(seq_len, device=x.device)[None, :, None]
    selected = expert_outputs[top_k_indices, batch_idx, seq_idx, :]
    output = torch.sum(gates[..., None] * selected, dim=2)

    if not deterministic and dropout_p > 0:
        output = F.dropout(output, p=dropout_p, training=True)

    return NanoMoEForward(
        output=output,
        aux_loss=aux_loss,
        router_logits=router_logits,
        router_probs=router_probs,
        gates=gates,
        indices=top_k_indices,
    )
