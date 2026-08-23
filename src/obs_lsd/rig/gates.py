"""Hard gates a candidate must pass before it is worth judging."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from obs_lsd.rig import RUNS_DIR

MS_FACTOR = 1.15  # per-generation budget, against the parent
MS_BASELINE_FACTOR = 1.5  # absolute ceiling, against the baseline
NO_HEAD_SLACK = 0.01
JUMP_FACTOR = 2.0
JUMP_FLOOR = 5  # a handful of jumps is noise, even against a very calm baseline
JUMP_THRESHOLD_FACTOR = 3.0


def candidate_jumps(cand: dict, base: dict) -> int:
    """Count the candidate's halo-delta jumps against the *baseline's* median.

    `frame_stats` counts them against the candidate's own median, which makes a
    calmer halo look jumpier than a busy one. When the render has handed us the
    per-frame deltas, recompute against a fixed threshold instead."""
    deltas = cand.get("deltas")
    med = base.get("delta_median", 0.0)
    if not deltas or not med:
        return int(cand["jump_count"])
    return sum(1 for d in deltas if d > JUMP_THRESHOLD_FACTOR * med)


def gate(
    cand: dict, base: dict, checks_ok: bool = True, parent: dict | None = None
) -> tuple[bool, list[str]]:
    why = []
    if not checks_ok:
        why.append("tests/lint failed")
        return False, why
    ref = parent if parent and "ms_total_mean" in parent else base
    if cand["ms_total_mean"] > MS_FACTOR * ref["ms_total_mean"]:
        why.append(
            f"ms/frame {cand['ms_total_mean']:.1f} > {MS_FACTOR}x parent {ref['ms_total_mean']:.1f}"
        )
    if cand["ms_total_mean"] > MS_BASELINE_FACTOR * base["ms_total_mean"]:
        why.append(
            f"ms/frame {cand['ms_total_mean']:.1f} > {MS_BASELINE_FACTOR}x baseline {base['ms_total_mean']:.1f}"
        )
    if cand["no_head_frac"] > base["no_head_frac"] + NO_HEAD_SLACK:
        why.append(
            f"no-head frac {cand['no_head_frac']:.3f} > baseline {base['no_head_frac']:.3f}+{NO_HEAD_SLACK}"
        )
    jumps = candidate_jumps(cand, base)
    limit = max(JUMP_FACTOR * base["jump_count"], JUMP_FLOOR)
    if jumps > limit:
        why.append(
            f"jump count {jumps} > max({JUMP_FACTOR}x baseline {base['jump_count']}, {JUMP_FLOOR})"
        )
    return not why, why


def run_checks(cwd: Path, timeout_s: int = 600) -> tuple[bool, str]:
    cmd = "uv run pytest -q && uv run ruff check . && uv run ruff format --check ."
    try:
        r = subprocess.run(
            ["bash", "-lc", cmd],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return False, "tests timed out"
    return r.returncode == 0, (r.stdout + r.stderr)[-2000:]


def baseline_path() -> Path:
    return RUNS_DIR / "baseline.json"


def write_baseline(run_dir: Path, stats: dict, score: dict) -> None:
    baseline_path().parent.mkdir(parents=True, exist_ok=True)
    baseline_path().write_text(
        json.dumps({"run_dir": str(run_dir), "stats": stats, "score": score}, indent=2)
    )


def load_baseline() -> dict:
    return json.loads(baseline_path().read_text())
