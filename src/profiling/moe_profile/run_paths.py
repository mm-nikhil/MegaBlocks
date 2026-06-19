"""Result-directory safety checks shared by profiling runners."""

from __future__ import annotations

from pathlib import Path


# These markers are fine in /tmp, but not under the project results tree. The
# point is to keep checked project results meaningful: real run roots should be
# named by purpose, model, or date, not by smoke/test/debug intent.
SCRATCH_RESULT_MARKERS = ("smoke", "test", "tmp", "debug", "scratch")


def validate_result_root(
    result_root: Path,
    *,
    allow_current_results: bool,
    allow_results_smoke: bool,
    workspace_root: Path | None = None,
) -> None:
    """Prevent accidental pollution of curated project result directories.

    Runners are free to write quick checks under ``/tmp``. Under the repository's
    ``results/`` tree, however, we keep two guardrails:

    * ``results/current`` is review/promote-only unless explicitly overridden.
    * smoke/debug/test roots are rejected unless explicitly overridden.
    """

    root = (workspace_root or Path.cwd()).resolve()
    results_root = (root / "results").resolve()
    resolved = result_root.resolve()
    try:
        relative = resolved.relative_to(results_root)
    except ValueError:
        return

    parts = tuple(part.lower() for part in relative.parts)
    if parts and parts[0] == "current" and not allow_current_results:
        raise SystemExit(
            "Refusing to write directly under results/current. Write to a reviewed "
            "run root first, then promote manually after inspection. Use "
            "--allow-current-results only if that is intentional.",
        )

    if not allow_results_smoke:
        scratch_part = next(
            (
                part
                for part in parts
                if any(marker in part for marker in SCRATCH_RESULT_MARKERS)
            ),
            None,
        )
        if scratch_part is not None:
            raise SystemExit(
                f"Refusing to write scratch/smoke output under results/: {result_root}. "
                "Use /tmp for smoke checks, or pass --allow-results-smoke if this "
                "is intentionally a retained result.",
            )
