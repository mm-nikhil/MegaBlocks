# Backend Status

Requested backends:

- `megablocks_moe`
- `megablocks_dmoe`

Max successful `N`:

- `megablocks_dmoe`: `1048576`
- `megablocks_moe`: `65536`

Failures:

- N=131072 backend=megablocks_moe: Profiler failed for timing_megablocks_moe_N131072.jsonl: error: Triton Error [CUDA]: invalid argument
- N=262144 backend=megablocks_moe: Profiler failed for timing_megablocks_moe_N262144.jsonl: error: Triton Error [CUDA]: invalid argument
- N=524288 backend=megablocks_moe: Profiler failed for timing_megablocks_moe_N524288.jsonl: error: Triton Error [CUDA]: invalid argument
- N=1048576 backend=megablocks_moe: Profiler failed for timing_megablocks_moe_N1048576.jsonl: error: Triton Error [CUDA]: invalid argument
