# Backend Status

Requested backend families:

- `reference`
- `megablocks_moe`
- `megablocks_dmoe`

Successful plotted backend variants:

- `megablocks_dmoe`: max successful `N=49152`
- `megablocks_moe`: max successful `N=49152`
- `reference_dense_glu`: max successful `N=4096`

Failures / unsupported rows:

- N=8192 backend=reference: returncode=1 reason=error: Memory preflight rejected this run before allocation. estimated=10558616371 base_estimated=7821197312 allowed=8858507673 free=9842786304 total=10351935488 fraction=0.9. Use a smaller N or --skip-memory-preflight if you intentionally want to test the limit.
- N=16384 backend=reference: returncode=1 reason=error: Memory preflight rejected this run before allocation. estimated=19304054784 base_estimated=14299299840 allowed=8858507673 free=9842786304 total=10351935488 fraction=0.9. Use a smaller N or --skip-memory-preflight if you intentionally want to test the limit.
- N=32768 backend=reference: returncode=1 reason=error: Memory preflight rejected this run before allocation. estimated=36794931609 base_estimated=27255504896 allowed=8858507673 free=9842786304 total=10351935488 fraction=0.9. Use a smaller N or --skip-memory-preflight if you intentionally want to test the limit.
- N=49152 backend=reference: returncode=1 reason=error: Memory preflight rejected this run before allocation. estimated=54285808435 base_estimated=40211709952 allowed=8858507673 free=9842786304 total=10351935488 fraction=0.9. Use a smaller N or --skip-memory-preflight if you intentionally want to test the limit.
- N=65536 backend=reference: returncode=1 reason=error: Memory preflight rejected this run before allocation. estimated=71776685260 base_estimated=53167915008 allowed=8858507673 free=9842786304 total=10351935488 fraction=0.9. Use a smaller N or --skip-memory-preflight if you intentionally want to test the limit.
- N=65536 backend=megablocks_moe: returncode=1 reason=error: Memory preflight rejected this run before allocation. estimated=10895523840 base_estimated=8070758400 allowed=8858507673 free=9842786304 total=10351935488 fraction=0.9. Use a smaller N or --skip-memory-preflight if you intentionally want to test the limit.
- N=65536 backend=megablocks_dmoe: returncode=1 reason=error: Memory preflight rejected this run before allocation. estimated=10895523840 base_estimated=8070758400 allowed=8858507673 free=9842786304 total=10351935488 fraction=0.9. Use a smaller N or --skip-memory-preflight if you intentionally want to test the limit.
