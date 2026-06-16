# MegaBlocks NanoMoE Profiling

This repo profiles MegaBlocks MoE execution against the MoE layer semantics used by
Nano-MoE-JAX.

Semantic checks come before timing interpretation. The PyTorch reference in
`src/profiling` matches Nano-MoE-JAX at the MoE boundary, then the MegaBlocks
adapter is checked against that reference.

## Layout

- `src/profiling/`: correctness and profiling scripts.
- `docs/`: setup, usage, terminology, and result reporting.
- `third_party/megablocks/`: MegaBlocks submodule.
- `third_party/Nano-MoE-JAX/`: Nano-MoE-JAX submodule.
- `third_party/grouped_gemm/`: grouped GEMM submodule for build/debug context.

## Quick Start

```bash
git submodule update --init --recursive
scripts/bootstrap_python_env.sh
```

Then follow `docs/setup.md`. On a GPU server, the non-negotiable dependency is
`nvcc`. It can come from a system CUDA toolkit or the workspace-local CUDA overlay
described in the setup docs.

## Current Status

The PyTorch NanoMoE reference matches Nano-MoE-JAX for output, router gates, expert
indices, and auxiliary loss. `grouped_gemm` and `megablocks_ops` build successfully
using the local CUDA 12.6 overlay on the RTX 3080 host.

Current validated MegaBlocks correctness smoke uses Nano-JAX initialized weights,
the full `(B, T, D)` adapter boundary, Nano-style aux loss, and paired reference
rows:

- `megablocks/moe`, float32 and float16: exact agreement with the PyTorch Nano
  reference on the smoke shape.
- `megablocks/dmoe`, bfloat16, zero expert biases: exact agreement with the
  PyTorch Nano reference on the smoke shape.
- `megablocks/dmoe`, bfloat16, synthetic nonzero expert biases: exact agreement
  with the PyTorch Nano reference through the bias-aware grouped adapter.

Still under investigation:

- FP16/FP32 grouped `dmoe`; the current grouped GEMM extension requires BF16.
- Broader plotted sweeps over larger shapes and checkpoint-like activation/weight
  distributions.

Start with:

```bash
source .venv/bin/activate
python src/profiling/check_nano_moe_port.py
python src/profiling/verify_moe_layer.py
```

See:

- `Findings.md`
- `docs/setup.md`
- `docs/usage.md`
- `docs/architecture.md`
- `docs/terminology.md`
- `docs/reporting.md`
