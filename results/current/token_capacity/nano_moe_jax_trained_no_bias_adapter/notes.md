# Trained NanoJAX No-Bias-Adapter Control

This is an intentional performance control, not a correctness-equivalent NanoJAX run.

- weight_source: `trained_nano_checkpoint`
- dtype: `bfloat16`
- MegaBlocks bias mode: `--allow-bias-mismatch` without `--use-expert-biases`
- Purpose: isolate whether the trained run changed because nonzero expert biases selected the bias-aware adapter path.

Failures:

- N=131072 backend=megablocks_moe: returncode=1 reason=error: Triton Error [CUDA]: invalid argument
- N=262144 backend=megablocks_moe: returncode=1 reason=error: Triton Error [CUDA]: invalid argument
