"""Check that the PyTorch NanoMoE MoE port matches Nano-MoE-JAX.

Example:
    python src/profiling/check_nano_moe_port.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import torch

from nano_moe_torch import from_flax_moe_params, nano_moe_forward


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nano-jax-dir", type=Path, default=Path("third_party/Nano-MoE-JAX"))
    parser.add_argument(
        "--preset",
        choices=("smoke", "single"),
        default="smoke",
        help="smoke checks a small suite of shapes/top-k values; single checks only the explicit args.",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--d-ff", type=int, default=64)
    parser.add_argument("--n-experts", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--atol", type=float, default=2e-5)
    parser.add_argument("--rtol", type=float, default=2e-5)
    parser.add_argument(
        "--tie-diagnostic",
        action="store_true",
        help="Print a known exact-tie router ordering diagnostic without making it a pass/fail check.",
    )
    return parser.parse_args()


def max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def smoke_cases(args: argparse.Namespace) -> list[dict[str, int]]:
    base = {
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "d_model": args.d_model,
        "d_ff": args.d_ff,
        "n_experts": args.n_experts,
        "seed": args.seed,
    }
    cases = []
    seen = set()
    for top_k in (1, args.top_k, args.n_experts):
        if top_k < 1 or top_k > args.n_experts:
            continue
        case = {**base, "top_k": top_k}
        key = tuple(case.items())
        if key not in seen:
            seen.add(key)
            cases.append(case)

    alt = {
        "batch_size": max(1, args.batch_size + 1),
        "seq_len": max(1, args.seq_len - 3),
        "d_model": max(8, args.d_model // 2),
        "d_ff": max(16, args.d_ff // 2),
        "n_experts": args.n_experts,
        "top_k": min(max(1, args.top_k), args.n_experts),
        "seed": args.seed + 1,
    }
    key = tuple(alt.items())
    if key not in seen:
        cases.append(alt)
    return cases


def single_case(args: argparse.Namespace) -> list[dict[str, int]]:
    return [{
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "d_model": args.d_model,
        "d_ff": args.d_ff,
        "n_experts": args.n_experts,
        "top_k": args.top_k,
        "seed": args.seed,
    }]


def run_case(case: dict[str, int], *, atol: float, rtol: float) -> dict[str, Any]:
    from nano_moe.config import NanoMoEConfig
    from nano_moe.layers import MoELayer, Router

    config = NanoMoEConfig(
        d_model=case["d_model"],
        d_ff=case["d_ff"],
        n_experts=case["n_experts"],
        top_k=case["top_k"],
        dropout_rate=0.0,
    )
    rng = jax.random.PRNGKey(case["seed"])
    x = jax.random.normal(rng, (case["batch_size"], case["seq_len"], case["d_model"]))

    layer = MoELayer(config=config)
    params = layer.init(rng, x, deterministic=True)["params"]
    jax_out, jax_aux = layer.apply({"params": params}, x, deterministic=True)

    router = Router(n_experts=case["n_experts"], top_k=case["top_k"])
    router_params = {"Router_0": params["Router_0"]}
    jax_gates, jax_indices, _ = router.apply(
        {"params": router_params["Router_0"]},
        x,
    )

    torch_weights = from_flax_moe_params(params)
    torch_x = torch.as_tensor(np.array(x, copy=True), dtype=torch.float32)
    torch_out = nano_moe_forward(
        torch_x,
        torch_weights,
        top_k=case["top_k"],
        deterministic=True,
        dropout_p=0.0,
    )

    out_np = torch_out.output.detach().cpu().numpy()
    aux_np = np.asarray(torch_out.aux_loss.detach().cpu())
    gates_np = torch_out.gates.detach().cpu().numpy()
    indices_np = torch_out.indices.detach().cpu().numpy()

    checks = {
        "output": np.allclose(out_np, np.asarray(jax_out), atol=atol, rtol=rtol),
        "aux_loss": np.allclose(aux_np, np.asarray(jax_aux), atol=atol, rtol=rtol),
        "gates": np.allclose(gates_np, np.asarray(jax_gates), atol=atol, rtol=rtol),
        "indices": np.array_equal(indices_np, np.asarray(jax_indices)),
    }
    return {
        **case,
        "max_abs_output": max_abs(out_np, np.asarray(jax_out)),
        "max_abs_gates": max_abs(gates_np, np.asarray(jax_gates)),
        "abs_aux_loss": abs(float(aux_np) - float(jax_aux)),
        "indices_equal": checks["indices"],
        "failed": [name for name, ok in checks.items() if not ok],
    }


def print_tie_diagnostic() -> None:
    from nano_moe.layers import Router

    x = jnp.ones((1, 1, 4), dtype=jnp.float32)
    router = Router(n_experts=4, top_k=2)
    params = router.init(jax.random.PRNGKey(0), x)["params"]
    params["Dense_0"]["kernel"] = jnp.zeros_like(params["Dense_0"]["kernel"])
    _, jax_indices, _ = router.apply({"params": params}, x)
    _, torch_indices = torch.topk(torch.zeros(1, 1, 4), 2, dim=-1)
    print("tie_diagnostic: exact equal logits have framework-specific top-k ordering")
    print(f"tie_diagnostic_jax_indices: {np.asarray(jax_indices).reshape(-1).tolist()}")
    print(f"tie_diagnostic_torch_indices: {torch_indices.numpy().reshape(-1).tolist()}")


def main() -> None:
    args = parse_args()
    nano_dir = args.nano_jax_dir.resolve()
    if not nano_dir.exists():
        raise SystemExit(f"Nano-MoE-JAX checkout not found: {nano_dir}")
    sys.path.insert(0, str(nano_dir))

    cases = smoke_cases(args) if args.preset == "smoke" else single_case(args)
    print("NanoMoE MoE JAX vs PyTorch reference")
    print(f"preset: {args.preset} cases={len(cases)}")

    results = [run_case(case, atol=args.atol, rtol=args.rtol) for case in cases]
    for index, result in enumerate(results, start=1):
        print(
            f"[{index}/{len(results)}] "
            f"batch={result['batch_size']} seq={result['seq_len']} d_model={result['d_model']} "
            f"d_ff={result['d_ff']} experts={result['n_experts']} top_k={result['top_k']} "
            f"seed={result['seed']} max_abs(output)={result['max_abs_output']:.6g} "
            f"max_abs(gates)={result['max_abs_gates']:.6g} "
            f"abs(aux_loss)={result['abs_aux_loss']:.6g} "
            f"indices_equal={result['indices_equal']}"
        )

    if args.tie_diagnostic:
        print_tie_diagnostic()

    failed = [result for result in results if result["failed"]]
    if failed:
        details = "; ".join(
            f"top_k={result['top_k']} seed={result['seed']}: {','.join(result['failed'])}"
            for result in failed
        )
        raise SystemExit(f"FAILED checks: {details}")

    print(f"all {len(results)} checks passed")


if __name__ == "__main__":
    main()
