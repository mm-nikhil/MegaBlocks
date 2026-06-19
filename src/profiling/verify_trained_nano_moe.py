"""Train, save, and verify Nano-MoE-JAX trained MoE weights.

This script is intentionally about verification, not model quality. It can train
a Nano-MoE-JAX language model once, save the full trained Flax parameter tree,
then reuse that checkpoint for repeated MoE-layer verification runs.

Verification extracts a trained MoE layer, computes the real hidden-state input
to that MoE layer for a batch of token data, then compares:

1. Original Nano-MoE-JAX MoE execution.
2. The PyTorch NanoMoE reference.
3. The MegaBlocks adapter.

The default config keeps the Nano MoE-layer shape from the project catalog
(``D=128, H=512, E=4, K=2``) but uses shorter sequences and a few training steps
so the CPU-only JAX install in this environment can complete the run quickly.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from argparse import Namespace
from dataclasses import asdict
from pathlib import Path
from typing import Any

# The current workspace has torch CUDA, but not CUDA-enabled jaxlib. Keep JAX on
# CPU so it does not try to initialize a GPU backend it cannot use.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax import linen as nn
from flax import serialization

from nano_moe_torch import from_flax_moe_params, nano_moe_forward
from profile_moe_layer import (
    build_megablocks_layer,
    compare_moe_outputs,
    megablocks_forward,
    measure_forward,
    parse_dtype,
)


BUILTIN_TEXT = """
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles.
""".strip()


class NanoPrefixToBlockMoEInput(nn.Module):
    """Run NanoMoE from token ids up to one block's MoE input.

    The module structure intentionally matches the start of
    ``nano_moe.model.NanoMoE`` and ``nano_moe.layers.TransformerBlock`` so it can
    apply the trained full-model parameter tree directly.
    """

    config: Any
    block_index: int = 0

    @nn.compact
    def __call__(self, token_ids: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        from nano_moe.layers import TransformerBlock

        cfg = self.config
        _, seq_len = token_ids.shape
        x = nn.Embed(
            num_embeddings=cfg.vocab_size,
            features=cfg.d_model,
            embedding_init=nn.initializers.normal(stddev=0.02),
        )(token_ids)
        pos_emb = self.param(
            "pos_emb",
            nn.initializers.normal(stddev=0.02),
            (1, cfg.block_size, cfg.d_model),
        )
        x = x + pos_emb[:, :seq_len, :]
        x = nn.Dropout(rate=cfg.dropout_rate)(x, deterministic=deterministic)

        for block_id in range(self.block_index):
            x, _ = TransformerBlock(config=cfg, name=f"block_{block_id}")(
                x,
                deterministic=deterministic,
            )

        return NanoBlockMoEInput(config=cfg, name=f"block_{self.block_index}")(
            x,
            deterministic=deterministic,
        )


class NanoBlockMoEInput(nn.Module):
    """Run one Nano transformer block up to the normalized MoE input."""

    config: Any

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        from nano_moe.layers import MultiHeadAttention

        residual = x
        x = nn.LayerNorm()(x)
        x = MultiHeadAttention(config=self.config)(x, deterministic=deterministic)
        x = x + residual
        return nn.LayerNorm()(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("train_and_verify", "train", "verify_saved"),
        default="train_and_verify",
        help=(
            "train saves a reusable checkpoint; verify_saved consumes an existing "
            "checkpoint without training; train_and_verify does both."
        ),
    )
    parser.add_argument("--nano-jax-dir", type=Path, default=Path("third_party/Nano-MoE-JAX"))
    parser.add_argument("--data-mode", choices=("builtin", "tiny_shakespeare"), default="builtin")
    parser.add_argument("--train-steps", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=0, help="Print training loss every N steps when training.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--block-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--verify-batch-size", type=int, default=2)
    parser.add_argument("--verify-seq-len", type=int, default=16)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--n-experts", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--megablocks-layer", choices=("moe", "dmoe"), default="moe")
    parser.add_argument("--outlier-abs-threshold", type=float, default=1e-3)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("results/trained_nano_moe_checkpoint"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("results/trained_nano_moe_verification"))
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--jsonl-out", type=Path)
    parser.add_argument("--sweep-batch-sizes", default="", help="Comma-separated verify batch sizes for scaling runs.")
    parser.add_argument("--timing-warmup", type=int, default=0)
    parser.add_argument("--timing-iters", type=int, default=0)
    parser.add_argument("--timing-trials", type=int, default=1)
    parser.add_argument("--save-tensors", action="store_true")
    return parser.parse_args()


def parse_csv_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def load_builtin_text_data() -> tuple[np.ndarray, np.ndarray, int]:
    text = (BUILTIN_TEXT + "\n") * 256
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    data = np.asarray([stoi[ch] for ch in text], dtype=np.int32)
    split = int(len(data) * 0.9)
    return data[:split], data[split:], len(chars)


def load_training_data(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, int]:
    if args.data_mode == "builtin":
        return load_builtin_text_data()

    from nano_moe.utils import load_text_data

    train_data, val_data, vocab_size, _, _ = load_text_data()
    return train_data, val_data, vocab_size


def make_config(args: argparse.Namespace, *, vocab_size: int):
    from nano_moe.config import NanoMoEConfig

    return NanoMoEConfig(
        vocab_size=vocab_size,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_model=args.d_model,
        d_ff=args.d_ff,
        n_experts=args.n_experts,
        top_k=args.top_k,
        block_size=args.block_size,
        dropout_rate=0.1,
        aux_loss_coeff=0.01,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        max_iters=args.train_steps,
        eval_interval=max(1, args.train_steps),
        eval_iters=1,
    )


def config_from_dict(config_data: dict[str, object]):
    from nano_moe.config import NanoMoEConfig

    fields = NanoMoEConfig.__dataclass_fields__
    values = {key: config_data[key] for key in fields if key in config_data}
    return NanoMoEConfig(**values)


def train_briefly(args: argparse.Namespace, config, train_data: np.ndarray):
    from nano_moe.train import create_train_state, train_step
    from nano_moe.utils import get_batch

    rng = jax.random.PRNGKey(args.seed)
    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng, config)
    losses: list[float] = []
    aux_losses: list[float] = []

    for step in range(1, args.train_steps + 1):
        rng, batch_rng = jax.random.split(rng)
        x, y = get_batch(train_data, config.batch_size, config.block_size, batch_rng)
        state, metrics = train_step(state, x, y, config)
        losses.append(float(metrics["loss"]))
        aux_losses.append(float(metrics["aux_loss"]))
        if args.log_every and (step == 1 or step % args.log_every == 0 or step == args.train_steps):
            print(
                f"train_step {step}/{args.train_steps} "
                f"loss={losses[-1]:.6g} aux={aux_losses[-1]:.6g}",
                flush=True,
            )

    return state, rng, losses, aux_losses


def params_template(config):
    from nano_moe.model import NanoMoE

    dummy = jnp.ones((1, config.block_size), dtype=jnp.int32)
    rng = jax.random.PRNGKey(0)
    return NanoMoE(config=config).init({"params": rng, "dropout": rng}, dummy, deterministic=True)["params"]


def save_checkpoint(
    checkpoint_dir: Path,
    *,
    config,
    params,
    data_mode: str,
    train_steps: int,
    seed: int,
    losses: list[float],
    aux_losses: list[float],
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format": 1,
        "config": asdict(config),
        "data_mode": data_mode,
        "train_steps": train_steps,
        "seed": seed,
        "initial_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "initial_aux_loss": aux_losses[0] if aux_losses else None,
        "final_aux_loss": aux_losses[-1] if aux_losses else None,
        "losses": losses,
        "aux_losses": aux_losses,
    }
    (checkpoint_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (checkpoint_dir / "params.msgpack").write_bytes(serialization.to_bytes(params))


def load_checkpoint(checkpoint_dir: Path):
    metadata_path = checkpoint_dir / "metadata.json"
    params_path = checkpoint_dir / "params.msgpack"
    if not metadata_path.exists() or not params_path.exists():
        raise SystemExit(
            f"Checkpoint not found in {checkpoint_dir}. Expected metadata.json and params.msgpack.",
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    config = config_from_dict(metadata["config"])
    params = serialization.from_bytes(params_template(config), params_path.read_bytes())
    losses = [float(value) for value in metadata.get("losses", [])]
    aux_losses = [float(value) for value in metadata.get("aux_losses", [])]
    return config, params, metadata, losses, aux_losses


def extract_moe_params(params, block_index: int):
    return params[f"block_{block_index}"]["MoELayer_0"]


def max_abs_np(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a.astype(np.float32) - b.astype(np.float32))))


def mean_abs_np(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))


def exact_equal_np(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.array_equal(a, b))


def tensor_compare(prefix: str, expected: np.ndarray, actual: np.ndarray) -> dict[str, object]:
    return {
        f"{prefix}_exact_equal": exact_equal_np(expected, actual),
        f"{prefix}_max_abs": max_abs_np(expected, actual),
        f"{prefix}_mean_abs": mean_abs_np(expected, actual),
    }


def save_tensors(args: argparse.Namespace, tensors: dict[str, np.ndarray]) -> None:
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    for name, value in tensors.items():
        np.save(args.artifact_dir / f"{name}.npy", value)


def verify_params(
    args: argparse.Namespace,
    *,
    config,
    params,
    rng,
    train_data: np.ndarray,
    val_data: np.ndarray,
    data_mode: str,
    losses: list[float],
    aux_losses: list[float],
    checkpoint_metadata: dict[str, object] | None,
) -> dict[str, object]:
    from nano_moe.layers import MoELayer
    from nano_moe.utils import get_batch

    device = torch.device(args.device)
    dtype = parse_dtype(args.dtype)

    rng, verify_rng = jax.random.split(rng)
    verify_tokens, _ = get_batch(
        val_data if len(val_data) > args.verify_seq_len + 1 else train_data,
        args.verify_batch_size,
        args.verify_seq_len,
        verify_rng,
    )

    prefix = NanoPrefixToBlockMoEInput(config=config, block_index=args.block_index)
    moe_input = prefix.apply({"params": params}, verify_tokens, deterministic=True)
    moe_params = extract_moe_params(params, args.block_index)

    jax_layer = MoELayer(config=config)
    jax_output, jax_aux = jax_layer.apply({"params": moe_params}, moe_input, deterministic=True)

    weights = from_flax_moe_params(moe_params, device=device, dtype=dtype)
    torch_input = torch.as_tensor(np.array(moe_input, copy=True), device=device, dtype=dtype).contiguous()
    with torch.inference_mode():
        torch_reference = nano_moe_forward(
            torch_input,
            weights,
            top_k=config.top_k,
            deterministic=True,
            dropout_p=0.0,
        )

    model_shape = {
        "simulation_level": "trained_nano_moe_jax",
        "expert_type": "ffn",
        "activation": "gelu_tanh",
        "num_shared_experts": 0,
        "shared_expert_intermediate_size": 0,
    }
    mb_args = build_case_args(args, config=config)
    layer = build_megablocks_layer(mb_args, weights, model_shape, dtype, device)
    with torch.inference_mode():
        mb_forward = megablocks_forward(
            layer,
            torch_input,
            batch_size=args.verify_batch_size,
            seq_len=args.verify_seq_len,
            n_experts=config.n_experts,
            top_k=config.top_k,
            collect_diagnostics=True,
        )
        mb_metrics = compare_moe_outputs(
            reference_for_checks=torch_reference,
            actual_forward=mb_forward,
            outlier_abs_threshold=args.outlier_abs_threshold,
        )

    jax_output_np = np.asarray(jax_output)
    torch_output_np = torch_reference.output.detach().cpu().numpy()
    mb_output_np = mb_forward.output.detach().cpu().numpy()
    timing_metrics: dict[str, object] = {}
    if args.timing_iters > 0:
        with torch.inference_mode():
            reference_timing = measure_forward(
                lambda: nano_moe_forward(
                    torch_input,
                    weights,
                    top_k=config.top_k,
                    deterministic=True,
                    dropout_p=0.0,
                ).output,
                warmup=args.timing_warmup,
                iters=args.timing_iters,
                trials=args.timing_trials,
                device=device,
            )
            megablocks_timing = measure_forward(
                lambda: megablocks_forward(
                    layer,
                    torch_input,
                    batch_size=args.verify_batch_size,
                    seq_len=args.verify_seq_len,
                    n_experts=config.n_experts,
                    top_k=config.top_k,
                ).output,
                warmup=args.timing_warmup,
                iters=args.timing_iters,
                trials=args.timing_trials,
                device=device,
            )
        tokens = args.verify_batch_size * args.verify_seq_len
        active_expert_flops = tokens * 4 * config.top_k * config.d_model * config.d_ff
        ref_ms = float(reference_timing["mean_forward_ms"])
        mb_ms = float(megablocks_timing["mean_forward_ms"])
        timing_metrics = {
            "timing_scope": "moe_layer",
            "timing_warmup": args.timing_warmup,
            "timing_iters": args.timing_iters,
            "timing_trials": args.timing_trials,
            "reference_mean_forward_ms": ref_ms,
            "reference_std_forward_ms": float(reference_timing["std_forward_ms"]),
            "megablocks_mean_forward_ms": mb_ms,
            "megablocks_std_forward_ms": float(megablocks_timing["std_forward_ms"]),
            "megablocks_speedup_vs_reference": ref_ms / mb_ms if mb_ms else 0.0,
            "reference_active_expert_tflops_per_second": (
                active_expert_flops / (ref_ms / 1000.0) / 1e12 if ref_ms else 0.0
            ),
            "megablocks_active_expert_tflops_per_second": (
                active_expert_flops / (mb_ms / 1000.0) / 1e12 if mb_ms else 0.0
            ),
        }
    record: dict[str, object] = {
        "mode": args.mode,
        "checkpoint_dir": str(args.checkpoint_dir),
        "data_mode": data_mode,
        "jax_platform": str(jax.default_backend()),
        "torch_device": str(device),
        "dtype": args.dtype,
        "megablocks_layer": args.megablocks_layer,
        "train_steps": int(checkpoint_metadata.get("train_steps", len(losses))) if checkpoint_metadata else args.train_steps,
        "initial_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "initial_aux_loss": aux_losses[0] if aux_losses else None,
        "final_aux_loss": aux_losses[-1] if aux_losses else None,
        "block_index": args.block_index,
        "batch_size": args.verify_batch_size,
        "seq_len": args.verify_seq_len,
        "tokens": args.verify_batch_size * args.verify_seq_len,
        "d_model": config.d_model,
        "d_ff": config.d_ff,
        "n_experts": config.n_experts,
        "top_k": config.top_k,
        "trained_expert_bias_max_abs": weights.max_abs_bias(),
        "jax_aux_loss": float(jax_aux),
        "torch_aux_loss": float(torch_reference.aux_loss.detach().cpu()),
        "megablocks_aux_loss": float(mb_forward.aux_loss.detach().cpu()),
        **tensor_compare("jax_vs_torch_output", jax_output_np, torch_output_np),
        **tensor_compare("jax_vs_megablocks_output", jax_output_np, mb_output_np),
        **mb_metrics,
        **timing_metrics,
    }
    record["torch_vs_megablocks_output_exact_equal"] = bool(
        torch.equal(torch_reference.output.detach().cpu(), mb_forward.output.detach().cpu()),
    )

    if args.save_tensors:
        save_tensors(
            args,
            {
                "moe_input": np.asarray(moe_input),
                "jax_output": jax_output_np,
                "torch_reference_output": torch_output_np,
                "megablocks_output": mb_output_np,
                "torch_router_indices": torch_reference.indices.detach().cpu().numpy(),
                "megablocks_router_indices": mb_forward.indices.detach().cpu().numpy(),
                "torch_router_gates": torch_reference.gates.detach().cpu().numpy(),
                "megablocks_router_gates": mb_forward.gates.detach().cpu().numpy(),
            },
        )

    return record


def build_case_args(args: argparse.Namespace, *, config) -> Namespace:
    return Namespace(
        megablocks_layer=args.megablocks_layer,
        d_model=config.d_model,
        d_ff=config.d_ff,
        n_experts=config.n_experts,
        top_k=config.top_k,
        seed=args.seed,
        use_expert_biases=True,
        zero_expert_biases=False,
        allow_bias_mismatch=False,
    )


def print_record(record: dict[str, object], out_path: Path) -> None:
    print("trained Nano-MoE-JAX MoE verification")
    print(f"mode: {record['mode']}")
    print(f"checkpoint_dir: {record['checkpoint_dir']}")
    if record["initial_loss"] is not None and record["final_loss"] is not None:
        print(f"loss: {record['initial_loss']:.6g} -> {record['final_loss']:.6g}")
    print(f"trained_expert_bias_max_abs: {record['trained_expert_bias_max_abs']:.6g}")
    print(
        "jax_vs_torch_output: "
        f"exact={record['jax_vs_torch_output_exact_equal']} "
        f"max_abs={record['jax_vs_torch_output_max_abs']:.6g} "
        f"mean_abs={record['jax_vs_torch_output_mean_abs']:.6g}",
    )
    print(
        "torch_vs_megablocks_output: "
        f"exact={record['torch_vs_megablocks_output_exact_equal']} "
        f"max_abs={record['max_abs_vs_reference']:.6g} "
        f"mean_abs={record['mean_abs_vs_reference']:.6g}",
    )
    print(f"router_expert_set_mismatches: {record['router_expert_set_mismatch_count']}")
    print(f"correctness_passed: {record['correctness_passed']}")
    if "reference_mean_forward_ms" in record:
        print(
            "timing: "
            f"reference_ms={record['reference_mean_forward_ms']:.6g} "
            f"megablocks_ms={record['megablocks_mean_forward_ms']:.6g} "
            f"speedup={record['megablocks_speedup_vs_reference']:.3g}",
        )
    print(f"wrote_json: {out_path}")


def main() -> None:
    args = parse_args()
    nano_dir = args.nano_jax_dir.resolve()
    if not nano_dir.exists():
        raise SystemExit(f"Nano-MoE-JAX checkout not found: {nano_dir}")
    sys.path.insert(0, str(nano_dir))

    if args.block_index < 0 or args.block_index >= args.n_layers:
        raise SystemExit("--block-index must be in [0, n_layers).")
    if args.mode != "verify_saved" and args.verify_seq_len > args.block_size:
        raise SystemExit("--verify-seq-len must be <= --block-size.")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false.")

    checkpoint_metadata = None
    if args.mode == "verify_saved":
        config, params, checkpoint_metadata, losses, aux_losses = load_checkpoint(args.checkpoint_dir)
        effective_data_mode = str(checkpoint_metadata.get("data_mode", args.data_mode))
        train_data, val_data, _ = load_training_data(Namespace(data_mode=effective_data_mode))
        rng = jax.random.PRNGKey(args.seed)
    else:
        train_data, val_data, vocab_size = load_training_data(args)
        config = make_config(args, vocab_size=vocab_size)
        state, rng, losses, aux_losses = train_briefly(args, config, train_data)
        params = state.params
        effective_data_mode = args.data_mode
        save_checkpoint(
            args.checkpoint_dir,
            config=config,
            params=params,
            data_mode=effective_data_mode,
            train_steps=args.train_steps,
            seed=args.seed,
            losses=losses,
            aux_losses=aux_losses,
        )
        if args.mode == "train":
            print("trained Nano-MoE-JAX checkpoint saved")
            print(f"checkpoint_dir: {args.checkpoint_dir}")
            if losses:
                print(f"loss: {losses[0]:.6g} -> {losses[-1]:.6g}")
            return

    if args.block_index < 0 or args.block_index >= config.n_layers:
        raise SystemExit("--block-index must be in [0, checkpoint n_layers).")
    if args.verify_seq_len > config.block_size:
        raise SystemExit("--verify-seq-len must be <= checkpoint block_size.")

    sweep_batch_sizes = parse_csv_ints(args.sweep_batch_sizes)
    if sweep_batch_sizes:
        if args.jsonl_out is None:
            args.jsonl_out = args.artifact_dir / "scaling_verify.jsonl"
        args.jsonl_out.parent.mkdir(parents=True, exist_ok=True)
        if args.jsonl_out.exists():
            args.jsonl_out.unlink()
        for batch_size in sweep_batch_sizes:
            case_args = Namespace(**vars(args))
            case_args.verify_batch_size = batch_size
            case_args.save_tensors = False
            record = verify_params(
                case_args,
                config=config,
                params=params,
                rng=jax.random.fold_in(rng, batch_size),
                train_data=train_data,
                val_data=val_data,
                data_mode=effective_data_mode,
                losses=losses,
                aux_losses=aux_losses,
                checkpoint_metadata=checkpoint_metadata,
            )
            with args.jsonl_out.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            print(
                f"batch={batch_size} tokens={record['tokens']} "
                f"correct={record['correctness_passed']} "
                f"max_abs={record['max_abs_vs_reference']:.6g} "
                f"router_mismatch={record['router_expert_set_mismatch_count']} "
                f"ref_ms={record.get('reference_mean_forward_ms', 0):.6g} "
                f"mb_ms={record.get('megablocks_mean_forward_ms', 0):.6g}",
                flush=True,
            )
        print(f"wrote_jsonl: {args.jsonl_out}")
        return

    record = verify_params(
        args,
        config=config,
        params=params,
        rng=rng,
        train_data=train_data,
        val_data=val_data,
        data_mode=effective_data_mode,
        losses=losses,
        aux_losses=aux_losses,
        checkpoint_metadata=checkpoint_metadata,
    )

    out_path = args.json_out or (args.artifact_dir / "verification.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print_record(record, out_path)


if __name__ == "__main__":
    main()
