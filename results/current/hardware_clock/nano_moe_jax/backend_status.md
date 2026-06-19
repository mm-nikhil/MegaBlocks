# Backend Status

Requested backends:

- `megablocks_moe`
- `megablocks_dmoe`

Max successful `N`:

- `megablocks_dmoe [bfloat16] (BF16-only dMoE)`: `1048576`
- `megablocks_moe [float32]`: `65536`

Failures:

- N=131072 backend=megablocks_moe: Profiler failed for timing_megablocks_moe_N131072.jsonl: error: Triton Error [CUDA]: invalid argument
- N=262144 backend=megablocks_moe: Profiler failed for timing_megablocks_moe_N262144.jsonl: error: Triton Error [CUDA]: invalid argument
- N=524288 backend=megablocks_moe: Profiler failed for timing_megablocks_moe_N524288.jsonl: error: Triton Error [CUDA]: invalid argument
- N=1048576 backend=megablocks_moe: Profiler failed for timing_megablocks_moe_N1048576.jsonl: error: Memory preflight rejected this run before allocation. estimated=9470216908 base_estimated=7014975488 allowed=8903983104 free=9893314560 total=10351935488 fraction=0.9. Use a smaller N or --skip-memory-preflight if you intentionally want to test the limit.
