# Eval Rig and Evolutionary Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `obs-lsd-rig` replays a clip through obs-lsd, computes stability metrics, builds judge images, scores the result with a vision model over OpenRouter against Danny's rubric, gates candidates, and runs an evolutionary loop where `claude -p` agents mutate the halo code on branches and the best survive.

**Architecture:** A `src/obs_lsd/rig/` package of small pure modules (metrics, images, gates, selection) plus three side-effecting ones (render: runs the existing Pipeline; judge: OpenRouter HTTP; evolve: git worktrees + subprocess `claude -p`). A click group `obs-lsd-rig` exposes `render judge pairwise calibrate baseline evolve prune`. State lives in gitignored `runs/` (per-run dirs, `ledger.json`, `baseline.json`).

**Tech Stack:** Python 3.12, numpy, opencv, click (existing); new: `httpx`. Git worktrees; `claude` CLI 2.1.x (`-p`, `--output-format json`, `--dangerously-skip-permissions`; no `--max-turns`). `pass` for the OpenRouter key.

**Spec:** `docs/superpowers/specs/2026-08-22-eval-rig-evolution-design.md`

## Global Constraints

- Judge model default `anthropic/claude-opus-5`; `--judge-model` overrides. Key: `OPENROUTER_API_KEY` env else `pass show openrouter/api-key`; never printed or logged.
- Scores: 3 samples, median per category (seamless, distracting, ornate, each 0–30), total = sum of medians (/90).
- Gates: tests+ruff green; `ms_total_mean ≤ 1.15 × baseline`; `no_head_frac ≤ baseline + 0.01`; `jump_count ≤ 2 × baseline`.
- Population 3; branches `evo/g<gen>-<k>`; worktrees in `.worktrees/`; agents never run on `main`; human merges champions.
- Render default output 1280x720, replay on file clock (`FrameSource(..., realtime=False)`), `Pipeline` segmenter injectable (`Pipeline(cfg, segmenter=...)`).
- `runs/`, `clips/`, `.worktrees/` gitignored. Rubric = `loop.md` copied verbatim into `src/obs_lsd/rig/prompts/rubric.md`.
- Existing style: ruff line-length 100, `from __future__ import annotations`, tests in `tests/`, `uv run` for everything, `\rm` in Makefiles. Commits: imperative subject, why-body.
- mediapipe `<1.0`; no new heavy deps beyond httpx.

---

### Task 1: Rig scaffold (package, CLI group, prompts, deps, Makefile, gitignore)

**Files:**
- Create: `src/obs_lsd/rig/__init__.py`, `src/obs_lsd/rig/cli.py`, `src/obs_lsd/rig/prompts/rubric.md`, `src/obs_lsd/rig/prompts/judge.md`, `src/obs_lsd/rig/prompts/operators.md`, `src/obs_lsd/rig/prompts/mutate.md`, `tests/test_rig_cli.py`
- Modify: `pyproject.toml` (httpx dep, `obs-lsd-rig` script, package data), `.gitignore`, `Makefile`

**Interfaces:**
- Produces: `obs_lsd.rig.PROMPTS_DIR: Path`, `obs_lsd.rig.load_prompt(name) -> str`, click group `obs_lsd.rig.cli:rig` with subcommands (stubs now, filled in later tasks): `render judge pairwise calibrate baseline evolve prune`.

- [ ] **Step 1: Failing test**

`tests/test_rig_cli.py`:
```python
from click.testing import CliRunner

from obs_lsd.rig import load_prompt
from obs_lsd.rig.cli import rig


def test_rig_group_lists_commands():
    r = CliRunner().invoke(rig, ["--help"])
    assert r.exit_code == 0
    for cmd in ("render", "judge", "pairwise", "calibrate", "baseline", "evolve", "prune"):
        assert cmd in r.output


def test_prompts_are_packaged():
    rubric = load_prompt("rubric.md")
    assert "Seamless" in rubric and "Ornate" in rubric and "30 points" in rubric
    assert "{operator}" in load_prompt("mutate.md")
```

- [ ] **Step 2: Run, expect ImportError**

`uv run pytest tests/test_rig_cli.py -q`

- [ ] **Step 3: Implement**

`src/obs_lsd/rig/__init__.py`:
```python
"""Evaluation rig: replay clips, score with a vision judge, evolve the halo code."""

from __future__ import annotations

from pathlib import Path

PROMPTS_DIR = Path(__file__).parent / "prompts"
RUNS_DIR = Path("runs")


def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text()
```

`src/obs_lsd/rig/cli.py` (stubs; later tasks replace the bodies):
```python
"""obs-lsd-rig: evaluation and evolution commands."""

from __future__ import annotations

import click


@click.group()
def rig():
    """Replay, score and evolve obs-lsd."""


@rig.command()
def render():
    """Replay a clip and write a run directory."""
    raise click.ClickException("not implemented yet")


@rig.command()
def judge():
    """Score a run directory with the vision judge."""
    raise click.ClickException("not implemented yet")


@rig.command()
def pairwise():
    """A/B two run directories."""
    raise click.ClickException("not implemented yet")


@rig.command()
def calibrate():
    """Compare judge models against human picks."""
    raise click.ClickException("not implemented yet")


@rig.command()
def baseline():
    """Render + score main and save runs/baseline.json."""
    raise click.ClickException("not implemented yet")


@rig.command()
def evolve():
    """Run generations of the evolutionary loop."""
    raise click.ClickException("not implemented yet")


@rig.command()
def prune():
    """Remove worktrees of non-survivors."""
    raise click.ClickException("not implemented yet")
```

`src/obs_lsd/rig/prompts/rubric.md`: `\cp loop.md src/obs_lsd/rig/prompts/rubric.md` (verbatim).

`src/obs_lsd/rig/prompts/judge.md`:
```
You are judging the output of a video filter that cuts a person's head out of a
webcam feed, floats it on black, and surrounds it with an animated rainbow halo.
You are shown: (1) a contact sheet of 8 output frames spread across a 30 s clip,
(2) three full-resolution centre crops, (3) a motion strip of 8 consecutive
frames 100 ms apart (for judging jumpiness), and (4) computed motion metrics.

Score strictly against the rubric below. Use the whole range; a competent but
ordinary effect is about 15/30 in each category. Be specific in `notes` about
what you see (name frames/crops), and give ONE concrete `top_fix` an engineer
could implement next.

Respond with JSON only, no prose outside it:
{"seamless": <0-30>, "distracting": <0-30>, "ornate": <0-30>,
 "notes": "<2-5 sentences>", "top_fix": "<one sentence>"}

Computed metrics for this run:
{metrics_table}

Rubric:
{rubric}
```

`src/obs_lsd/rig/prompts/operators.md` (parsed by evolve: sections `## <name>`):
```
## fix_top_complaint
The judge's single most important complaint about the parent is quoted below as
TOP_FIX. Implement a fix for exactly that. Do not start other changes.

## new_technique
Add ONE genuinely new visual technique to the halo/compose stage that is not in
the techniques log. Candidates (pick one not yet tried, or invent): painterly
brush-stroke texture on the rays, a second slower outer ring of detail, gold-leaf
style fine filigree near the rim, subtle parallax/3D shading of rays, soft
directional lighting shared by head and halo, iconographic geometric ornament
(Byzantine/medieval), colour grading so halo and skin share a palette.

## tune
Do not add techniques. Adjust existing parameters and defaults in HaloParams /
compose / segmenter feather to improve the rubric scores, guided by the notes.

## simplify
Remove or tone down one element that reads as a cheap special-effects trick or
adds distraction without beauty. Less, but better.

## ornate
Push the effect toward hand-painted, Byzantine/woodcut richness: more detail
where the eye rests (near the head), restraint further out. Keep it slow.

## crossover
You are given the diffs of two sibling variants (A and B) relative to their
common ancestor. Combine the best ideas of both into one coherent change.
```

`src/obs_lsd/rig/prompts/mutate.md`:
```
You are improving obs-lsd, a Python webcam filter (glowing floating head with a
rainbow halo). You are working in an isolated git worktree on branch {branch}.
Read CLAUDE.md first. The code you may change: src/obs_lsd/halo.py,
src/obs_lsd/compose.py, src/obs_lsd/segment.py, src/obs_lsd/track.py,
src/obs_lsd/pipeline.py and their tests. Do NOT touch src/obs_lsd/rig/.

## Your operator for this round: {operator}
{operator_text}

## How the parent scored (0-30 each, /90 total)
seamless={seamless} distracting={distracting} ornate={ornate} total={total}
Judge notes: {notes}
TOP_FIX: {top_fix}
Metrics: {metrics_table}

## Techniques already tried in this lineage (do not repeat them)
{techniques}

## Rubric the judge uses
{rubric}

## Hard rules
- Keep `uv run pytest -q` and `uv run ruff check . && uv run ruff format --check .` green.
- Keep per-frame cost roughly flat: `make smoke` must still pass; do not add
  dependencies.
- Make ONE coherent change. Add or adjust tests for it.
- Append a line to techniques.md at the repo root: `- g{generation}: <what you did, one line>`.
- Commit everything with a short imperative subject and a body saying why.
- When done, print one line starting with `RESULT:` summarising the change.
```

`pyproject.toml` changes:
```toml
dependencies = [ ..., "httpx>=0.27" ]
[project.scripts]
obs-lsd = "obs_lsd.cli:cli"
obs-lsd-rig = "obs_lsd.rig.cli:rig"
```
(uv_build includes non-py files under the package by default; verify `load_prompt` works from an installed wheel by `uv pip install -e .` then the test.)

`.gitignore` append: `runs/`, `.worktrees/`.

`Makefile` append:
```makefile
baseline: models ## Render + judge main on CLIP, save runs/baseline.json (CLIP=clips/x.mov)
	$(UV) run obs-lsd-rig baseline $(CLIP)

eval: models ## Render + judge the current tree on CLIP
	$(UV) run obs-lsd-rig render $(CLIP) --out runs/$$(git rev-parse --abbrev-ref HEAD | tr / -)-$$(git rev-parse --short HEAD) && \
	$(UV) run obs-lsd-rig judge runs/$$(git rev-parse --abbrev-ref HEAD | tr / -)-$$(git rev-parse --short HEAD)

judge-smoke: ## One real judge call against the latest run dir
	$(UV) run obs-lsd-rig judge $$(ls -dt runs/*/ | head -1) --samples 1

evolve: models ## Run GENS generations of the loop on CLIP
	$(UV) run obs-lsd-rig evolve $(CLIP) --gens $(GENS)
```

- [ ] **Step 4: `uv pip install -e ".[dev]"`, run tests, lint. Commit**

```bash
git add -A && git commit -m "Scaffold obs-lsd-rig: CLI group, prompts, httpx dep"
```

---

### Task 2: Metrics (pure)

**Files:**
- Create: `src/obs_lsd/rig/metrics.py`, `tests/test_rig_metrics.py`

**Interfaces:**
- Produces:
  ```python
  def halo_delta(prev_bgr: np.ndarray, cur_bgr: np.ndarray, head_mask: np.ndarray) -> float
      # mean |cur-prev| over pixels where head_mask<0.5, in 0..1 (uint8/255); 0.0 if no such pixels
  def frame_stats(boxes: list[HeadBox | None], timings: list[dict], deltas: list[float]) -> dict
      # keys: frames, no_head, no_head_frac, delta_median, delta_p95, jump_count, jump_frac,
      #       centre_jump_p95, height_std, ms_total_mean, ms_segment_mean, ms_halo_mean, ms_compose_mean
  def metrics_table(stats: dict) -> str   # markdown-ish "key: value" lines for prompts
  ```

- [ ] **Step 1: Failing tests**

`tests/test_rig_metrics.py`:
```python
import numpy as np

from obs_lsd.rig.metrics import frame_stats, halo_delta, metrics_table
from obs_lsd.track import HeadBox


def test_halo_delta_ignores_head_pixels():
    a = np.zeros((4, 4, 3), np.uint8)
    b = a.copy()
    b[0, 0] = 255  # outside head
    b[2, 2] = 255  # inside head
    mask = np.zeros((4, 4), np.float32)
    mask[2, 2] = 1.0
    assert abs(halo_delta(a, b, mask) - (255 / 15) / 255) < 1e-6
    assert halo_delta(a, b, np.ones((4, 4), np.float32)) == 0.0


def test_frame_stats_counts_jumps_and_misses():
    boxes = [HeadBox(100, 100, 50), HeadBox(101, 100, 50), None, HeadBox(110, 100, 52)]
    timings = [{"total": 10.0, "segment": 5.0, "halo": 3.0, "compose": 1.0}] * 4
    deltas = [0.01, 0.01, 0.01, 0.10]  # last one is > 3x median -> one jump
    s = frame_stats(boxes, timings, deltas)
    assert s["frames"] == 4 and s["no_head"] == 1 and abs(s["no_head_frac"] - 0.25) < 1e-9
    assert s["jump_count"] == 1 and abs(s["jump_frac"] - 0.25) < 1e-9
    assert abs(s["delta_median"] - 0.01) < 1e-9
    assert s["centre_jump_p95"] > 1.0  # 101->110 jump dominates (None skipped)
    assert s["ms_total_mean"] == 10.0 and s["ms_segment_mean"] == 5.0


def test_frame_stats_empty_and_table():
    s = frame_stats([], [], [])
    assert s["frames"] == 0 and s["no_head_frac"] == 0.0 and s["jump_count"] == 0
    t = metrics_table(s)
    assert "frames: 0" in t and "ms_total_mean" in t
```

- [ ] **Step 2: Run, expect ImportError**

- [ ] **Step 3: Implement**

```python
"""Per-frame stability numbers for a replayed run. Pure functions."""

from __future__ import annotations

import numpy as np

from obs_lsd.track import HeadBox


def halo_delta(prev_bgr: np.ndarray, cur_bgr: np.ndarray, head_mask: np.ndarray) -> float:
    outside = head_mask < 0.5
    if not outside.any():
        return 0.0
    d = np.abs(cur_bgr.astype(np.int16) - prev_bgr.astype(np.int16)).mean(axis=2)
    return float(d[outside].mean() / 255.0)


def frame_stats(boxes: list[HeadBox | None], timings: list[dict], deltas: list[float]) -> dict:
    n = len(boxes)
    misses = sum(1 for b in boxes if b is None)
    d = np.asarray(deltas, dtype=float)
    med = float(np.median(d)) if d.size else 0.0
    p95 = float(np.percentile(d, 95)) if d.size else 0.0
    jumps = int((d > 3.0 * med).sum()) if d.size and med > 0 else 0
    pts = [(b.cx, b.cy) for b in boxes if b is not None]
    if len(pts) >= 2:
        p = np.asarray(pts)
        step = np.hypot(*np.diff(p, axis=0).T)
        centre_p95 = float(np.percentile(step, 95))
    else:
        centre_p95 = 0.0
    heights = [b.height for b in boxes if b is not None]
    height_std = float(np.std(heights)) if heights else 0.0

    def mean_ms(key: str) -> float:
        vals = [t[key] for t in timings if key in t]
        return float(np.mean(vals)) if vals else 0.0

    return {
        "frames": n,
        "no_head": misses,
        "no_head_frac": misses / n if n else 0.0,
        "delta_median": med,
        "delta_p95": p95,
        "jump_count": jumps,
        "jump_frac": jumps / n if n else 0.0,
        "centre_jump_p95": centre_p95,
        "height_std": height_std,
        "ms_total_mean": mean_ms("total"),
        "ms_segment_mean": mean_ms("segment"),
        "ms_halo_mean": mean_ms("halo"),
        "ms_compose_mean": mean_ms("compose"),
    }


def metrics_table(stats: dict) -> str:
    return "\n".join(
        f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in stats.items()
    )
```

- [ ] **Step 4: Tests pass, lint, commit** — `git commit -m "Add rig metrics: halo delta, jump counts, timing means"`

---

### Task 3: Judge images (pure)

**Files:**
- Create: `src/obs_lsd/rig/images.py`, `tests/test_rig_images.py`

**Interfaces:**
- Produces:
  ```python
  def contact_sheet(frames: list[np.ndarray], times: list[float], cols: int = 4, width: int = 480) -> np.ndarray
  def crops(frames: list[np.ndarray], size: int = 560) -> np.ndarray   # hstack of centre crops, labelled
  def motion_strip(frames: list[np.ndarray], width: int = 320) -> np.ndarray  # hstack, labelled 0..n-1
  def sample_indices(n_frames: int, k: int) -> list[int]   # k indices spread evenly, excluding the very first/last 2%
  ```
  All return BGR uint8 arrays; callers write PNGs.

- [ ] **Step 1: Failing tests**

```python
import numpy as np

from obs_lsd.rig.images import contact_sheet, crops, motion_strip, sample_indices


def _frame(v):
    f = np.zeros((90, 160, 3), np.uint8)
    f[:] = v
    return f


def test_sample_indices_spread_and_inside():
    idx = sample_indices(1000, 8)
    assert len(idx) == 8 and idx[0] >= 20 and idx[-1] <= 979
    assert all(b > a for a, b in zip(idx, idx[1:], strict=True))
    assert sample_indices(5, 8) == [0, 1, 2, 3, 4, 4, 4, 4] or len(sample_indices(5, 8)) == 8


def test_contact_sheet_shape():
    sheet = contact_sheet([_frame(i * 30) for i in range(8)], [i * 4.0 for i in range(8)], cols=4, width=160)
    assert sheet.shape[0] == 2 * 90 and sheet.shape[1] == 4 * 160 and sheet.dtype == np.uint8


def test_crops_and_motion_strip_shapes():
    big = [np.zeros((720, 1280, 3), np.uint8) for _ in range(3)]
    c = crops(big, size=200)
    assert c.shape == (200, 600, 3)
    s = motion_strip([_frame(i) for i in range(8)], width=80)
    assert s.shape[1] == 8 * 80 and s.shape[2] == 3
```

- [ ] **Step 2: Run, expect ImportError**

- [ ] **Step 3: Implement**

```python
"""Images the judge looks at: contact sheet, full-res crops, motion strip. Pure."""

from __future__ import annotations

import cv2
import numpy as np


def _label(img: np.ndarray, text: str) -> np.ndarray:
    cv2.putText(img, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img


def _fit_width(img: np.ndarray, width: int) -> np.ndarray:
    h, w = img.shape[:2]
    return cv2.resize(img, (width, max(1, round(h * width / w))), interpolation=cv2.INTER_AREA)


def sample_indices(n_frames: int, k: int) -> list[int]:
    if n_frames <= 0:
        return [0] * k
    lo, hi = int(n_frames * 0.02), max(int(n_frames * 0.98) - 1, 0)
    return [int(round(lo + (hi - lo) * i / max(k - 1, 1))) for i in range(k)]


def contact_sheet(
    frames: list[np.ndarray], times: list[float], cols: int = 4, width: int = 480
) -> np.ndarray:
    tiles = [_label(_fit_width(f.copy(), width), f"t={t:.1f}s") for f, t in zip(frames, times, strict=True)]
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i : i + cols]) for i in range(0, len(tiles), cols)]
    return np.vstack(rows)


def crops(frames: list[np.ndarray], size: int = 560) -> np.ndarray:
    out = []
    for i, f in enumerate(frames):
        h, w = f.shape[:2]
        cy, cx = h // 2, w // 2
        half = min(size // 2, h // 2, w // 2)
        c = f[cy - half : cy + half, cx - half : cx + half]
        if c.shape[0] != size:
            c = cv2.resize(c, (size, size), interpolation=cv2.INTER_AREA)
        out.append(_label(c.copy(), f"crop {i + 1}"))
    return np.hstack(out)


def motion_strip(frames: list[np.ndarray], width: int = 320) -> np.ndarray:
    return np.hstack([_label(_fit_width(f.copy(), width), f"+{i * 100}ms") for i, f in enumerate(frames)])
```

- [ ] **Step 4: Tests pass, lint, commit** — `git commit -m "Add rig judge images: contact sheet, crops, motion strip"`

---

### Task 4: Render a run directory

**Files:**
- Create: `src/obs_lsd/rig/render.py`, `tests/test_rig_render.py`
- Modify: `src/obs_lsd/rig/cli.py` (`render` command)

**Interfaces:**
- Consumes: `obs_lsd.capture.FrameSource(spec, w, h, fps, realtime=False)` (`.file_fps`), `obs_lsd.pipeline.Pipeline(cfg, segmenter=None)` (`.step(frame,t)`, `.timings`, `._last_head` is private — use the segmenter result instead: see below), `obs_lsd.output.VideoFileSink`, Task 2/3 functions.
- Produces:
  ```python
  @dataclass
  class RunResult: run_dir: Path; stats: dict; n_frames: int
  def render(clip: str, out_dir: Path, cfg: PipelineConfig | None = None, segmenter=None,
             sheet_frames: int = 8) -> RunResult
  ```
  Writes `out.mp4`, `metrics.json`, `sheet.png`, `crops.png`, `motion.png` into `out_dir`.

Implementation note: `Pipeline.step` does not expose the per-frame `HeadMask`/`HeadBox`, and the halo delta needs the output-space mask. Add two read-only attributes to `Pipeline` in this task (small, reviewed change): `self.last_box: HeadBox | None` (the box passed to the tracker this step, None if missing) and `self.last_mask_out: np.ndarray | None` (the warped mask `m` before gain). Set them in `step()`; they are None before the first step.

- [ ] **Step 1: Failing test** (uses the FakeSeg pattern from tests/test_pipeline.py; no models)

`tests/test_rig_render.py`:
```python
import json

import cv2
import numpy as np

from obs_lsd.rig.render import render
from obs_lsd.segment import HeadMask
from obs_lsd.track import HeadBox


class FakeSeg:
    def mask(self, frame, ts):
        h, w = frame.shape[:2]
        m = np.zeros((h, w), np.float32)
        cv2.ellipse(m, (w // 2, h // 2), (w // 8, h // 5), 0, 0, 360, 1.0, -1)
        return HeadMask(mask=m, box=HeadBox(cx=w / 2, cy=h / 2, height=2 * h / 5))

    def close(self):
        pass


def _clip(path, n=30, fps=10.0, size=(160, 90)):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(n):
        f = np.full((size[1], size[0], 3), 40, np.uint8)
        f[:, :, 1] = min(255, 60 + i * 5)
        w.write(f)
    w.release()


def test_render_writes_run_dir(tmp_path):
    from obs_lsd.pipeline import PipelineConfig

    clip = tmp_path / "c.mp4"
    _clip(clip)
    out = tmp_path / "run"
    res = render(str(clip), out, cfg=PipelineConfig(width=320, height=180), segmenter=FakeSeg())
    assert res.n_frames == 30 and res.run_dir == out
    for name in ("out.mp4", "metrics.json", "sheet.png", "crops.png", "motion.png"):
        assert (out / name).exists(), name
    stats = json.loads((out / "metrics.json").read_text())
    assert stats["frames"] == 30 and stats["no_head"] == 0 and stats["ms_total_mean"] > 0
    sheet = cv2.imread(str(out / "sheet.png"))
    assert sheet.shape[1] == 4 * 480
    cap = cv2.VideoCapture(str(out / "out.mp4"))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 30
```

- [ ] **Step 2: Run, expect ImportError**

- [ ] **Step 3: Implement** `src/obs_lsd/rig/render.py`:

```python
"""Replay a clip through the pipeline and write a run directory for judging."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from obs_lsd.capture import FrameSource
from obs_lsd.output import VideoFileSink
from obs_lsd.pipeline import Pipeline, PipelineConfig
from obs_lsd.rig.images import contact_sheet, crops, motion_strip, sample_indices
from obs_lsd.rig.metrics import frame_stats, halo_delta


@dataclass
class RunResult:
    run_dir: Path
    stats: dict
    n_frames: int


def render(
    clip: str,
    out_dir: Path,
    cfg: PipelineConfig | None = None,
    segmenter=None,
    sheet_frames: int = 8,
) -> RunResult:
    cfg = cfg or PipelineConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src = FrameSource(clip, cfg.width, cfg.height, 30.0, realtime=False)
    fps = src.file_fps or 30.0
    pipe = Pipeline(cfg, segmenter=segmenter)
    sink = VideoFileSink(str(out_dir / "out.mp4"), cfg.width, cfg.height, fps)
    boxes, timings, deltas, times, kept = [], [], [], [], {}
    prev = None
    # which frame indices to keep in memory for the images
    try:
        n_total = int(cv2.VideoCapture(clip).get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        sheet_idx = set(sample_indices(n_total, sheet_frames))
        crop_idx = {int(n_total * f) for f in (0.25, 0.5, 0.75)}
        step100 = max(1, round(fps / 10))
        mid = n_total // 2
        motion_idx = {mid + i * step100 for i in range(8)}
        want = sheet_idx | crop_idx | motion_idx
        i = 0
        for frame, t in src:
            out = pipe.step(frame, t)
            sink.write(out)
            boxes.append(pipe.last_box)
            timings.append(dict(pipe.timings))
            if prev is not None and pipe.last_mask_out is not None:
                deltas.append(halo_delta(prev, out, pipe.last_mask_out))
            if i in want:
                kept[i] = out.copy()
                times.append(t)
            prev = out
            i += 1
    finally:
        sink.close()
        pipe.close()
        src.close()
    stats = frame_stats(boxes, timings, deltas)
    (out_dir / "metrics.json").write_text(json.dumps(stats, indent=2))
    if kept:
        s_keys = sorted(k for k in sheet_idx if k in kept)[:sheet_frames]
        cv2.imwrite(str(out_dir / "sheet.png"), contact_sheet([kept[k] for k in s_keys], [k / fps for k in s_keys]))
        c_keys = sorted(k for k in crop_idx if k in kept)
        cv2.imwrite(str(out_dir / "crops.png"), crops([kept[k] for k in c_keys]))
        m_keys = sorted(k for k in motion_idx if k in kept)
        cv2.imwrite(str(out_dir / "motion.png"), motion_strip([kept[k] for k in m_keys]))
    return RunResult(run_dir=out_dir, stats=stats, n_frames=i)
```

Pipeline change (`src/obs_lsd/pipeline.py`): in `__init__` add `self.last_box = None; self.last_mask_out = None`; in `step()` after `tr = self.tracker.update(...)` set `self.last_box = head.box if head is not None else None`; after `m = cv2.warpAffine(src_mask, ...)` set `self.last_mask_out = m`. Add to `tests/test_pipeline.py`:
```python
def test_step_exposes_last_box_and_mask():
    cfg = PipelineConfig(width=320, height=180, halo=HaloParams(spill_gain=0.0))
    p = Pipeline(cfg, segmenter=FakeSeg())
    assert p.last_box is None and p.last_mask_out is None
    p.step(np.zeros((240, 320, 3), np.uint8), 0.0)
    assert p.last_box.height == 80 and p.last_mask_out.shape == (180, 320)
```

CLI `render` command in `src/obs_lsd/rig/cli.py`:
```python
@rig.command()
@click.argument("clip", type=click.Path(exists=True))
@click.option("--out", "out_dir", required=True, type=click.Path())
@click.option("--width", default=1280, show_default=True)
@click.option("--height", default=720, show_default=True)
def render(clip, out_dir, width, height):
    """Replay a clip and write a run directory."""
    from obs_lsd.pipeline import PipelineConfig
    from obs_lsd.rig.render import render as _render

    res = _render(clip, Path(out_dir), cfg=PipelineConfig(width=width, height=height))
    click.echo(f"{res.run_dir}: {res.n_frames} frames, ms/frame {res.stats['ms_total_mean']:.1f}")
```
(add `from pathlib import Path` at top of cli.py.)

- [ ] **Step 4: Tests pass (whole suite), lint. Try it for real:** `uv run obs-lsd-rig render clips/danny-01.mov --out runs/main-$(git rev-parse --short HEAD)` and look at `sheet.png` with the Read tool (head centred, halo present). Commit: `git commit -m "Add rig render: replay a clip into a scored run directory"`

---

### Task 5: Judge (OpenRouter)

**Files:**
- Create: `src/obs_lsd/rig/judge.py`, `tests/test_rig_judge.py`
- Modify: `src/obs_lsd/rig/cli.py` (`judge`, `pairwise`, `calibrate` commands)

**Interfaces:**
- Produces:
  ```python
  DEFAULT_MODEL = "anthropic/claude-opus-5"
  def resolve_api_key(env: dict | None = None, pass_cmd=subprocess.run) -> str   # env OPENROUTER_API_KEY else `pass show openrouter/api-key`
  @dataclass
  class Score: seamless: float; distracting: float; ornate: float; total: float; notes: str; top_fix: str; samples: list[dict]; model: str; cost_usd: float
      def to_json(self) -> dict
  class OpenRouterJudge:
      def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None, transport: httpx.BaseTransport | None = None, rng: random.Random | None = None)
      def score(self, run_dir: Path, samples: int = 3) -> Score        # writes run_dir/score.json
      def pairwise(self, run_a: Path, run_b: Path, votes: int = 3) -> dict  # {"prefer": "a"|"b"|"tie", "votes": [...], "cost_usd": float}
      def calibrate(self, pairs: list[dict], models: list[str]) -> dict  # {model: agreement_frac}
      cost_usd: float  # cumulative
  def parse_score_json(text: str) -> dict   # strict json, else first {...} block via regex; raises ValueError
  ```

- [ ] **Step 1: Failing tests** (httpx.MockTransport; no network)

`tests/test_rig_judge.py`:
```python
import json
import random
from pathlib import Path

import cv2
import httpx
import numpy as np
import pytest

from obs_lsd.rig.judge import OpenRouterJudge, parse_score_json, resolve_api_key


def _run_dir(tmp_path, name="r"):
    d = tmp_path / name
    d.mkdir()
    img = np.zeros((20, 20, 3), np.uint8)
    for n in ("sheet.png", "crops.png", "motion.png"):
        cv2.imwrite(str(d / n), img)
    (d / "metrics.json").write_text(json.dumps({"frames": 10, "jump_count": 1}))
    return d


def _transport(replies):
    calls = []

    def handler(request: httpx.Request):
        calls.append(json.loads(request.content))
        body = replies[len(calls) - 1] if len(calls) - 1 < len(replies) else replies[-1]
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": body}}],
                  "usage": {"prompt_tokens": 1000, "completion_tokens": 100}},
        )

    return httpx.MockTransport(handler), calls


def test_parse_score_json_strict_and_fallback():
    assert parse_score_json('{"seamless": 10, "distracting": 20, "ornate": 5, "notes": "n", "top_fix": "f"}')["ornate"] == 5
    assert parse_score_json('Sure! ```json\n{"seamless": 1, "distracting": 2, "ornate": 3, "notes": "", "top_fix": ""}\n```')["seamless"] == 1
    with pytest.raises(ValueError):
        parse_score_json("no json here")


def test_score_takes_median_and_writes_file(tmp_path):
    d = _run_dir(tmp_path)
    replies = [
        json.dumps({"seamless": 10, "distracting": 20, "ornate": 12, "notes": "a", "top_fix": "x"}),
        json.dumps({"seamless": 14, "distracting": 18, "ornate": 30, "notes": "b", "top_fix": "y"}),
        json.dumps({"seamless": 12, "distracting": 25, "ornate": 6, "notes": "c", "top_fix": "z"}),
    ]
    tr, calls = _transport(replies)
    j = OpenRouterJudge(model="m", api_key="k", transport=tr)
    s = j.score(d, samples=3)
    assert (s.seamless, s.distracting, s.ornate, s.total) == (12, 20, 12, 44)
    assert s.notes == "c" and s.top_fix == "z"  # from the sample closest to the median total
    assert len(calls) == 3 and calls[0]["model"] == "m"
    # images and metrics made it into the request
    content = calls[0]["messages"][0]["content"]
    assert sum(1 for c in content if c["type"] == "image_url") == 3
    assert any("jump_count" in c.get("text", "") for c in content)
    assert json.loads((d / "score.json").read_text())["total"] == 44
    assert j.cost_usd > 0


def test_pairwise_randomises_sides_and_counts_votes(tmp_path):
    a, b = _run_dir(tmp_path, "a"), _run_dir(tmp_path, "b")
    # the model always says "A"; with randomised sides the real winner alternates
    tr, calls = _transport(['{"prefer": "A", "why": "..."}'])
    j = OpenRouterJudge(model="m", api_key="k", transport=tr, rng=random.Random(0))
    res = j.pairwise(a, b, votes=4)
    assert res["prefer"] in ("a", "b", "tie") and len(res["votes"]) == 4
    assert {v["left"] for v in res["votes"]} == {"a", "b"}  # both orders were used


def test_resolve_api_key_env_then_pass():
    assert resolve_api_key({"OPENROUTER_API_KEY": "e"}) == "e"

    class R:
        returncode = 0
        stdout = "frompass\n"

    assert resolve_api_key({}, pass_cmd=lambda *a, **k: R()) == "frompass"
```

- [ ] **Step 2: Run, expect ImportError**

- [ ] **Step 3: Implement** `src/obs_lsd/rig/judge.py`:

```python
"""Vision-model judge over OpenRouter: absolute rubric scores and pairwise A/B."""

from __future__ import annotations

import base64
import json
import random
import re
import statistics
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx

from obs_lsd.rig import load_prompt
from obs_lsd.rig.metrics import metrics_table

DEFAULT_MODEL = "anthropic/claude-opus-5"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
IMAGES = ("sheet.png", "crops.png", "motion.png")
# rough $/M tokens for the cost print; exact billing is OpenRouter's business
PRICE = {"anthropic/claude-opus-5": (5.0, 25.0), "anthropic/claude-sonnet-5": (2.0, 10.0),
         "google/gemini-3.1-pro-preview": (2.0, 12.0), "openai/gpt-5.5": (5.0, 30.0)}


def resolve_api_key(env: dict | None = None, pass_cmd=subprocess.run) -> str:
    import os

    env = os.environ if env is None else env
    if env.get("OPENROUTER_API_KEY"):
        return env["OPENROUTER_API_KEY"]
    r = pass_cmd(["pass", "show", "openrouter/api-key"], capture_output=True, text=True)
    if r.returncode != 0 or not r.stdout.strip():
        raise RuntimeError("No OPENROUTER_API_KEY and `pass show openrouter/api-key` failed")
    return r.stdout.strip().splitlines()[0]


def parse_score_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise ValueError(f"no JSON in judge reply: {text[:200]!r}") from None
        return json.loads(m.group(0))


@dataclass
class Score:
    seamless: float
    distracting: float
    ornate: float
    total: float
    notes: str
    top_fix: str
    samples: list[dict] = field(default_factory=list)
    model: str = DEFAULT_MODEL
    cost_usd: float = 0.0

    def to_json(self) -> dict:
        return asdict(self)


def _img_part(path: Path) -> dict:
    b64 = base64.b64encode(path.read_bytes()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}


class OpenRouterJudge:
    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None,
                 transport: httpx.BaseTransport | None = None, rng: random.Random | None = None):
        self.model = model
        self.api_key = api_key or resolve_api_key()
        self.client = httpx.Client(transport=transport, timeout=120.0)
        self.rng = rng or random.Random()
        self.cost_usd = 0.0

    def _chat(self, content: list[dict]) -> str:
        body = {"model": self.model, "messages": [{"role": "user", "content": content}]}
        r = self.client.post(API_URL, json=body,
                             headers={"Authorization": f"Bearer {self.api_key}",
                                      "HTTP-Referer": "https://github.com/dob/obs-lsd",
                                      "X-Title": "obs-lsd rig"})
        r.raise_for_status()
        data = r.json()
        u = data.get("usage") or {}
        pin, pout = PRICE.get(self.model, (5.0, 25.0))
        self.cost_usd += u.get("prompt_tokens", 0) / 1e6 * pin + u.get("completion_tokens", 0) / 1e6 * pout
        return data["choices"][0]["message"]["content"]

    def _run_content(self, run_dir: Path, preamble: str) -> list[dict]:
        parts = [{"type": "text", "text": preamble}]
        for name in IMAGES:
            parts.append({"type": "text", "text": f"Image: {name}"})
            parts.append(_img_part(run_dir / name))
        return parts

    def score(self, run_dir: Path, samples: int = 3) -> Score:
        run_dir = Path(run_dir)
        stats = json.loads((run_dir / "metrics.json").read_text())
        prompt = load_prompt("judge.md").format(metrics_table=metrics_table(stats), rubric=load_prompt("rubric.md"))
        results = []
        for _ in range(samples):
            for attempt in range(2):
                try:
                    results.append(parse_score_json(self._chat(self._run_content(run_dir, prompt))))
                    break
                except (ValueError, httpx.HTTPError):
                    if attempt == 1:
                        pass
        if len(results) < 2:
            raise RuntimeError(f"judge failed: only {len(results)} usable samples")
        med = {k: float(statistics.median(float(r[k]) for r in results)) for k in ("seamless", "distracting", "ornate")}
        total = sum(med.values())
        best = min(results, key=lambda r: abs(sum(float(r[k]) for k in med) - total))
        s = Score(**med, total=total, notes=str(best.get("notes", "")), top_fix=str(best.get("top_fix", "")),
                  samples=results, model=self.model, cost_usd=self.cost_usd)
        (run_dir / "score.json").write_text(json.dumps(s.to_json(), indent=2))
        return s

    def pairwise(self, run_a: Path, run_b: Path, votes: int = 3) -> dict:
        run_a, run_b = Path(run_a), Path(run_b)
        rubric = load_prompt("rubric.md")
        tally = {"a": 0, "b": 0}
        out = []
        for _ in range(votes):
            left, right = (run_a, run_b) if self.rng.random() < 0.5 else (run_b, run_a)
            parts = [{"type": "text", "text": "Compare two outputs of the same filter on the same clip, A then B, "
                      "against the rubric. Reply JSON only: {\"prefer\": \"A\"|\"B\"|\"tie\", \"why\": \"...\"}\n\nRubric:\n" + rubric}]
            parts += self._run_content(left, "=== A ===")
            parts += self._run_content(right, "=== B ===")
            try:
                ans = parse_score_json(self._chat(parts))
            except (ValueError, httpx.HTTPError):
                continue
            pref = str(ans.get("prefer", "tie")).strip().upper()
            winner = {"A": left, "B": right}.get(pref)
            key = None if winner is None else ("a" if winner == run_a else "b")
            if key:
                tally[key] += 1
            out.append({"left": "a" if left == run_a else "b", "prefer": pref, "why": ans.get("why", "")})
        prefer = "tie" if tally["a"] == tally["b"] else max(tally, key=tally.get)
        return {"prefer": prefer, "votes": out, "tally": tally, "cost_usd": self.cost_usd}

    def calibrate(self, pairs: list[dict], models: list[str]) -> dict:
        res = {}
        for m in models:
            j = OpenRouterJudge(model=m, api_key=self.api_key, transport=self.client._transport, rng=self.rng)
            agree = sum(1 for p in pairs if j.pairwise(Path(p["a"]), Path(p["b"]), votes=3)["prefer"] == p["human"])
            res[m] = agree / len(pairs) if pairs else 0.0
            self.cost_usd += j.cost_usd
        return res
```

CLI additions:
```python
@rig.command()
@click.argument("run_dir", type=click.Path(exists=True))
@click.option("--samples", default=3, show_default=True)
@click.option("--judge-model", default=None)
def judge(run_dir, samples, judge_model):
    """Score a run directory with the vision judge."""
    from obs_lsd.rig.judge import DEFAULT_MODEL, OpenRouterJudge

    j = OpenRouterJudge(model=judge_model or DEFAULT_MODEL)
    s = j.score(Path(run_dir), samples=samples)
    click.echo(f"seamless={s.seamless:.0f} distracting={s.distracting:.0f} ornate={s.ornate:.0f} total={s.total:.0f}/90  (${j.cost_usd:.3f})")
    click.echo(f"notes: {s.notes}\ntop_fix: {s.top_fix}")


@rig.command()
@click.argument("run_a", type=click.Path(exists=True))
@click.argument("run_b", type=click.Path(exists=True))
@click.option("--votes", default=3, show_default=True)
@click.option("--judge-model", default=None)
def pairwise(run_a, run_b, votes, judge_model):
    """A/B two run directories."""
    from obs_lsd.rig.judge import DEFAULT_MODEL, OpenRouterJudge

    j = OpenRouterJudge(model=judge_model or DEFAULT_MODEL)
    res = j.pairwise(Path(run_a), Path(run_b), votes=votes)
    click.echo(json.dumps(res, indent=2))


@rig.command()
@click.argument("pairs_json", type=click.Path(exists=True))
@click.option("--models", default="anthropic/claude-opus-5,google/gemini-3.1-pro-preview,openai/gpt-5.5", show_default=True)
def calibrate(pairs_json, models):
    """Compare judge models against human picks. pairs_json: [{"a":dir,"b":dir,"human":"a"|"b"}]"""
    from obs_lsd.rig.judge import OpenRouterJudge

    pairs = json.loads(Path(pairs_json).read_text())
    j = OpenRouterJudge()
    for m, agree in j.calibrate(pairs, models.split(",")).items():
        click.echo(f"{m}: agreement {agree:.0%}")
    click.echo(f"cost ${j.cost_usd:.2f}")
```
(add `import json` to cli.py.)

- [ ] **Step 4: Tests pass, lint. One real call:** `uv run obs-lsd-rig judge runs/main-<sha> --samples 1` (key via pass; prints a score and cost, no key in output). Record the score in the report. Commit: `git commit -m "Add OpenRouter judge: rubric scores, pairwise A/B, calibration"`

---

### Task 6: Gates and baseline

**Files:**
- Create: `src/obs_lsd/rig/gates.py`, `tests/test_rig_gates.py`
- Modify: `src/obs_lsd/rig/cli.py` (`baseline` command)

**Interfaces:**
- Produces:
  ```python
  MS_FACTOR = 1.15; NO_HEAD_SLACK = 0.01; JUMP_FACTOR = 2.0
  def gate(cand: dict, base: dict, checks_ok: bool = True) -> tuple[bool, list[str]]
  def run_checks(cwd: Path) -> tuple[bool, str]   # uv run pytest -q && ruff check && ruff format --check; returns (ok, tail of output)
  def baseline_path() -> Path   # runs/baseline.json
  def write_baseline(run_dir: Path, stats: dict, score: dict) -> None
  def load_baseline() -> dict   # {"run_dir":..., "stats":..., "score":...}
  ```

- [ ] **Step 1: Failing tests**

```python
from obs_lsd.rig.gates import gate

BASE = {"ms_total_mean": 30.0, "no_head_frac": 0.0, "jump_count": 4}


def test_gate_passes_when_within_limits():
    ok, why = gate({"ms_total_mean": 34.0, "no_head_frac": 0.005, "jump_count": 8}, BASE)
    assert ok and why == []


def test_gate_reports_every_violation():
    ok, why = gate({"ms_total_mean": 40.0, "no_head_frac": 0.05, "jump_count": 9}, BASE)
    assert not ok and len(why) == 3
    assert any("ms/frame" in w for w in why) and any("no-head" in w for w in why) and any("jump" in w for w in why)


def test_gate_fails_on_checks():
    ok, why = gate(BASE, BASE, checks_ok=False)
    assert not ok and why == ["tests/lint failed"]
```

- [ ] **Step 2: Run, expect ImportError**

- [ ] **Step 3: Implement** `src/obs_lsd/rig/gates.py`:

```python
"""Hard gates a candidate must pass before it is worth judging."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from obs_lsd.rig import RUNS_DIR

MS_FACTOR = 1.15
NO_HEAD_SLACK = 0.01
JUMP_FACTOR = 2.0


def gate(cand: dict, base: dict, checks_ok: bool = True) -> tuple[bool, list[str]]:
    why = []
    if not checks_ok:
        why.append("tests/lint failed")
        return False, why
    if cand["ms_total_mean"] > MS_FACTOR * base["ms_total_mean"]:
        why.append(f"ms/frame {cand['ms_total_mean']:.1f} > {MS_FACTOR}x baseline {base['ms_total_mean']:.1f}")
    if cand["no_head_frac"] > base["no_head_frac"] + NO_HEAD_SLACK:
        why.append(f"no-head frac {cand['no_head_frac']:.3f} > baseline {base['no_head_frac']:.3f}+{NO_HEAD_SLACK}")
    if cand["jump_count"] > JUMP_FACTOR * base["jump_count"]:
        why.append(f"jump count {cand['jump_count']} > {JUMP_FACTOR}x baseline {base['jump_count']}")
    return not why, why


def run_checks(cwd: Path) -> tuple[bool, str]:
    cmd = "uv run pytest -q && uv run ruff check . && uv run ruff format --check ."
    r = subprocess.run(["bash", "-lc", cmd], cwd=cwd, capture_output=True, text=True)
    return r.returncode == 0, (r.stdout + r.stderr)[-2000:]


def baseline_path() -> Path:
    return RUNS_DIR / "baseline.json"


def write_baseline(run_dir: Path, stats: dict, score: dict) -> None:
    baseline_path().parent.mkdir(parents=True, exist_ok=True)
    baseline_path().write_text(json.dumps({"run_dir": str(run_dir), "stats": stats, "score": score}, indent=2))


def load_baseline() -> dict:
    return json.loads(baseline_path().read_text())
```

CLI `baseline`:
```python
@rig.command()
@click.argument("clip", type=click.Path(exists=True))
@click.option("--judge-model", default=None)
@click.option("--samples", default=3, show_default=True)
def baseline(clip, judge_model, samples):
    """Render + score the current tree and save runs/baseline.json."""
    import subprocess

    from obs_lsd.pipeline import PipelineConfig
    from obs_lsd.rig.gates import write_baseline
    from obs_lsd.rig.judge import DEFAULT_MODEL, OpenRouterJudge
    from obs_lsd.rig.render import render as _render

    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    run_dir = Path("runs") / f"baseline-{sha}"
    res = _render(clip, run_dir, cfg=PipelineConfig())
    j = OpenRouterJudge(model=judge_model or DEFAULT_MODEL)
    s = j.score(run_dir, samples=samples)
    write_baseline(run_dir, res.stats, s.to_json())
    click.echo(f"baseline {run_dir}: total {s.total:.0f}/90, ms/frame {res.stats['ms_total_mean']:.1f}  (${j.cost_usd:.3f})")
```

- [ ] **Step 4: Tests, lint, then `make baseline CLIP=clips/danny-01.mov` for real. Commit:** `git commit -m "Add rig gates and baseline command"`

---

### Task 7: Evolve (GA driver) — pure parts + dry-run

**Files:**
- Create: `src/obs_lsd/rig/evolve.py`, `tests/test_rig_evolve.py`
- Modify: `src/obs_lsd/rig/cli.py` (`evolve`, `prune`)

**Interfaces:**
- Consumes: render, gates, judge, prompts.
- Produces:
  ```python
  OPERATORS = ["fix_top_complaint", "new_technique", "tune", "simplify", "ornate"]
  def operator_text(name: str) -> str                 # section of operators.md
  def build_brief(branch, generation, operator, parent_score: dict, parent_stats: dict, techniques: str, extra: str = "") -> str
  def select_survivors(records: list[dict], k: int = 3) -> list[dict]   # scored & gated only; by score.total desc, then ms asc
  def child_plan(gen: int, survivors: list[dict]) -> list[dict]         # [{parent, k, operator}] with crossover rule
  class Ledger: (path) .load() .append(rec) .records() .save()
  class Evolution:
      def __init__(self, repo: Path, clip: str, judge=None, dry_run=False, population=3, claude_cmd="claude", timeout_s=1200)
      def run_generation(self, gen: int) -> list[dict]
  ```

- [ ] **Step 1: Failing tests** (pure parts + dry-run on a tmp git repo)

```python
import json
import subprocess
from pathlib import Path

from obs_lsd.rig.evolve import OPERATORS, Evolution, build_brief, child_plan, operator_text, select_survivors


def test_operator_text_sections():
    assert "TOP_FIX" in operator_text("fix_top_complaint")
    assert "diffs" in operator_text("crossover")


def test_build_brief_fills_placeholders():
    b = build_brief("evo/g1-0", 1, "tune", {"seamless": 1, "distracting": 2, "ornate": 3, "total": 6,
                    "notes": "N", "top_fix": "T"}, {"frames": 10}, "- g0: nothing")
    assert "{" not in b.replace("{}", "") or "{operator}" not in b
    assert "evo/g1-0" in b and "TOP_FIX: T" in b and "- g0: nothing" in b and "Do not add techniques" in b


def test_select_survivors_orders_and_filters():
    recs = [
        {"id": "a", "gates": {"ok": True}, "score": {"total": 50}, "metrics_summary": {"ms_total_mean": 30}},
        {"id": "b", "gates": {"ok": True}, "score": {"total": 50}, "metrics_summary": {"ms_total_mean": 20}},
        {"id": "c", "gates": {"ok": False}, "score": None, "metrics_summary": {"ms_total_mean": 5}},
        {"id": "d", "gates": {"ok": True}, "score": {"total": 70}, "metrics_summary": {"ms_total_mean": 40}},
    ]
    assert [r["id"] for r in select_survivors(recs, 3)] == ["d", "b", "a"]


def test_child_plan_rotates_operators_and_crossover():
    surv = [{"id": "p0"}, {"id": "p1"}, {"id": "p2"}]
    plan1 = child_plan(1, surv)
    assert [c["operator"] for c in plan1] == [OPERATORS[1], OPERATORS[2], OPERATORS[3]]
    plan2 = child_plan(2, surv)
    assert plan2[2]["operator"] == "crossover" and plan2[2]["parents"] == ["p0", "p1"]


def _repo(tmp_path):
    r = tmp_path / "repo"
    r.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=r, check=True)
    (r / "techniques.md").write_text("- g0: baseline\n")
    subprocess.run(["git", "add", "-A"], cwd=r, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"], cwd=r, check=True)
    return r


def test_dry_run_generation_creates_branches_and_ledger(tmp_path):
    repo = _repo(tmp_path)
    runs = repo / "runs"
    runs.mkdir()
    base_stats = {"ms_total_mean": 30.0, "no_head_frac": 0.0, "jump_count": 2}
    (runs / "baseline.json").write_text(json.dumps({"run_dir": "runs/b", "stats": base_stats,
        "score": {"seamless": 10, "distracting": 10, "ornate": 10, "total": 30, "notes": "n", "top_fix": "t"}}))
    ev = Evolution(repo=repo, clip="unused.mov", dry_run=True, population=3)
    recs = ev.run_generation(1)
    assert len(recs) == 1  # only one survivor (baseline) -> one child in gen 1
    rec = recs[0]
    assert rec["branch"] == "evo/g1-0" and rec["operator"] in OPERATORS and rec["gates"]["ok"] is None
    assert (repo / ".worktrees" / "evo-g1-0" / "brief.md").exists()
    branches = subprocess.run(["git", "branch", "--list", "evo/*"], cwd=repo, capture_output=True, text=True).stdout
    assert "evo/g1-0" in branches
    ledger = json.loads((runs / "ledger.json").read_text())
    assert ledger[-1]["id"] == rec["id"]
```

- [ ] **Step 2: Run, expect ImportError**

- [ ] **Step 3: Implement** `src/obs_lsd/rig/evolve.py`:

```python
"""Evolutionary loop: branches as genomes, claude -p as the mutation operator."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from obs_lsd.rig import load_prompt
from obs_lsd.rig.metrics import metrics_table

OPERATORS = ["fix_top_complaint", "new_technique", "tune", "simplify", "ornate"]


def operator_text(name: str) -> str:
    text = load_prompt("operators.md")
    m = re.search(rf"^## {re.escape(name)}\n(.*?)(?=^## |\Z)", text, re.S | re.M)
    if not m:
        raise KeyError(name)
    return m.group(1).strip()


def build_brief(branch, generation, operator, parent_score: dict, parent_stats: dict,
                techniques: str, extra: str = "") -> str:
    return load_prompt("mutate.md").format(
        branch=branch, generation=generation, operator=operator,
        operator_text=operator_text(operator) + ("\n\n" + extra if extra else ""),
        seamless=parent_score.get("seamless", "?"), distracting=parent_score.get("distracting", "?"),
        ornate=parent_score.get("ornate", "?"), total=parent_score.get("total", "?"),
        notes=parent_score.get("notes", ""), top_fix=parent_score.get("top_fix", ""),
        metrics_table=metrics_table(parent_stats), techniques=techniques or "(none yet)",
        rubric=load_prompt("rubric.md"),
    )


def select_survivors(records: list[dict], k: int = 3) -> list[dict]:
    ok = [r for r in records if r.get("gates", {}).get("ok") and r.get("score")]
    ok.sort(key=lambda r: (-r["score"]["total"], r["metrics_summary"].get("ms_total_mean", 0.0)))
    return ok[:k]


def child_plan(gen: int, survivors: list[dict]) -> list[dict]:
    plan = []
    for k, parent in enumerate(survivors):
        plan.append({"parent": parent["id"], "k": k, "operator": OPERATORS[(gen + k) % len(OPERATORS)]})
    if gen % 2 == 0 and len(survivors) >= 2 and len(plan) >= 3:
        plan[2] = {"parent": survivors[0]["id"], "k": 2, "operator": "crossover",
                   "parents": [survivors[0]["id"], survivors[1]["id"]]}
    return plan


class Ledger:
    def __init__(self, path: Path):
        self.path = path
        self._recs = json.loads(path.read_text()) if path.exists() else []

    def records(self) -> list[dict]:
        return list(self._recs)

    def append(self, rec: dict) -> None:
        self._recs.append(rec)
        self.path.write_text(json.dumps(self._recs, indent=2))

    def by_id(self, id_: str) -> dict | None:
        return next((r for r in self._recs if r["id"] == id_), None)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()


@dataclass
class Evolution:
    repo: Path
    clip: str
    judge: object = None
    dry_run: bool = False
    population: int = 3
    claude_cmd: str = "claude"
    timeout_s: int = 1200
    log: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.repo = Path(self.repo)
        self.runs = self.repo / "runs"
        self.runs.mkdir(exist_ok=True)
        self.ledger = Ledger(self.runs / "ledger.json")
        self.baseline = json.loads((self.runs / "baseline.json").read_text())
        if not self.ledger.records():
            self.ledger.append({"id": "baseline", "branch": "main", "parent": None, "generation": 0,
                                "operator": "baseline", "sha": self._sha("HEAD"), "gates": {"ok": True, "reasons": []},
                                "score": self.baseline["score"], "metrics_summary": self.baseline["stats"],
                                "run_dir": self.baseline["run_dir"], "cost_usd": 0.0})

    def _sha(self, ref: str) -> str:
        return _git(self.repo, "rev-parse", "--short", ref)

    def _say(self, msg: str) -> None:
        self.log.append(msg)
        print(msg, flush=True)

    def _worktree(self, branch: str, base_sha: str) -> Path:
        wt = self.repo / ".worktrees" / branch.replace("/", "-")
        if wt.exists():
            shutil.rmtree(wt)
            subprocess.run(["git", "worktree", "prune"], cwd=self.repo)
        _git(self.repo, "worktree", "add", "-q", "-B", branch, str(wt), base_sha)
        return wt

    def _techniques(self, wt: Path) -> str:
        p = wt / "techniques.md"
        return p.read_text() if p.exists() else ""

    def _run_agent(self, wt: Path, brief: str) -> tuple[bool, str]:
        (wt / "brief.md").write_text(brief)
        if self.dry_run:
            return True, "dry-run"
        cmd = [self.claude_cmd, "-p", brief, "--output-format", "json", "--dangerously-skip-permissions"]
        try:
            r = subprocess.run(cmd, cwd=wt, capture_output=True, text=True, timeout=self.timeout_s)
        except subprocess.TimeoutExpired:
            return False, "agent timeout"
        (wt / "agent-output.json").write_text(r.stdout)
        return r.returncode == 0, (r.stderr or r.stdout)[-500:]

    def run_generation(self, gen: int) -> list[dict]:
        survivors = select_survivors(self.ledger.records(), self.population)
        results = []
        for child in child_plan(gen, survivors):
            parent = self.ledger.by_id(child["parent"])
            branch = f"evo/g{gen}-{child['k']}"
            wt = self._worktree(branch, parent["sha"])
            extra = ""
            if child["operator"] == "crossover":
                a, b = (self.ledger.by_id(p) for p in child["parents"])
                base = _git(self.repo, "merge-base", a["sha"], b["sha"])
                extra = ("Diff A:\n" + _git(self.repo, "diff", base, a["sha"]) +
                         "\n\nDiff B:\n" + _git(self.repo, "diff", base, b["sha"]))
            brief = build_brief(branch, gen, child["operator"], parent["score"] or {},
                                parent["metrics_summary"], self._techniques(wt), extra)
            self._say(f"[g{gen}] {branch} <- {parent['id']} via {child['operator']}")
            agent_ok, agent_msg = self._run_agent(wt, brief)
            rec = {"id": f"{branch.replace('/', '-')}", "branch": branch, "parent": parent["id"],
                   "generation": gen, "operator": child["operator"], "sha": None,
                   "gates": {"ok": None, "reasons": []}, "score": None, "metrics_summary": {},
                   "run_dir": None, "cost_usd": 0.0, "agent": agent_msg}
            if not self.dry_run:
                sha = self._sha(branch)
                if not agent_ok or sha == parent["sha"]:
                    rec["gates"] = {"ok": False, "reasons": ["agent failed" if not agent_ok else "no change"]}
                else:
                    rec["sha"] = sha
                    rec["id"] = f"{rec['id']}-{sha}"
                    self._evaluate(rec, wt)
            self.ledger.append(rec)
            results.append(rec)
        self._table(gen, results)
        return results

    def _evaluate(self, rec: dict, wt: Path) -> None:
        from obs_lsd.rig.gates import gate, run_checks
        from obs_lsd.rig.render import render

        ok, tail = run_checks(wt)
        run_dir = self.runs / rec["id"]
        # render with the CANDIDATE's code: run obs-lsd-rig render inside the worktree
        r = subprocess.run(["uv", "run", "obs-lsd-rig", "render", str(Path(self.clip).resolve()),
                            "--out", str(run_dir.resolve())], cwd=wt, capture_output=True, text=True)
        if r.returncode != 0:
            rec["gates"] = {"ok": False, "reasons": ["render failed: " + r.stderr[-300:]]}
            return
        stats = json.loads((run_dir / "metrics.json").read_text())
        rec["metrics_summary"] = stats
        rec["run_dir"] = str(run_dir)
        g_ok, why = gate(stats, self.baseline["stats"], checks_ok=ok)
        if not ok:
            why = [f"tests/lint failed: {tail[-200:]}"]
        rec["gates"] = {"ok": g_ok, "reasons": why}
        if g_ok and self.judge is not None:
            s = self.judge.score(run_dir)
            rec["score"] = s.to_json()
            rec["cost_usd"] = s.cost_usd

    def _table(self, gen: int, recs: list[dict]) -> None:
        self._say(f"--- generation {gen} ---")
        for r in recs:
            sc = r["score"]
            s = f"{sc['seamless']:.0f}/{sc['distracting']:.0f}/{sc['ornate']:.0f}={sc['total']:.0f}" if sc else "-"
            ms = r["metrics_summary"].get("ms_total_mean", 0.0)
            gates = "ok" if r["gates"]["ok"] else ("?" if r["gates"]["ok"] is None else "FAIL " + "; ".join(r["gates"]["reasons"]))
            self._say(f"{r['id']:28s} {r['operator']:18s} {gates:40.40s} {s:14s} {ms:5.1f}ms  {r.get('run_dir') or ''}")
        best = select_survivors(self.ledger.records(), 1)
        if best:
            self._say(f"champion: {best[0]['id']} total {best[0]['score']['total']:.0f}  sheet: {best[0].get('run_dir')}/sheet.png")
        self._say(f"openrouter spend so far: ${sum(r.get('cost_usd', 0) for r in self.ledger.records()):.2f}")
```

Note for the implementer: `_evaluate` renders via a subprocess inside the worktree so the *candidate's* code is what runs (the controller process has the main tree's package imported). `uv run` in a worktree uses that worktree's `.venv`; `make install-dev` needs to have been run there once — add to `_worktree()`: if `wt/.venv` is missing, run `uv venv -p 3.12 && uv pip install -e ".[dev]"` in it (skip in dry-run), and symlink `src/obs_lsd/models` from the main repo into the worktree (`os.symlink(self.repo / "src/obs_lsd/models", wt / "src/obs_lsd/models")`) so models aren't re-downloaded. Put this in a helper `_prepare_worktree(wt)`.

CLI `evolve` / `prune`:
```python
@rig.command()
@click.argument("clip", type=click.Path(exists=True))
@click.option("--gens", default=1, show_default=True)
@click.option("--population", default=3, show_default=True)
@click.option("--judge-model", default=None)
@click.option("--dry-run", is_flag=True)
@click.option("--start-gen", default=None, type=int, help="Generation number to start at (default: next)")
def evolve(clip, gens, population, judge_model, dry_run, start_gen):
    """Run generations of the evolutionary loop."""
    from obs_lsd.rig.evolve import Evolution
    from obs_lsd.rig.judge import DEFAULT_MODEL, OpenRouterJudge

    judge_obj = None if dry_run else OpenRouterJudge(model=judge_model or DEFAULT_MODEL)
    ev = Evolution(repo=Path.cwd(), clip=clip, judge=judge_obj, dry_run=dry_run, population=population)
    first = start_gen if start_gen is not None else (max((r["generation"] for r in ev.ledger.records()), default=0) + 1)
    for g in range(first, first + gens):
        ev.run_generation(g)


@rig.command()
def prune():
    """Remove worktrees of non-survivors (branches are kept)."""
    from obs_lsd.rig.evolve import Evolution, select_survivors

    ev = Evolution(repo=Path.cwd(), clip="", dry_run=True)
    keep = {r["branch"].replace("/", "-") for r in select_survivors(ev.ledger.records(), ev.population)}
    for wt in (Path.cwd() / ".worktrees").glob("evo-*"):
        if wt.name not in keep:
            subprocess.run(["git", "worktree", "remove", "--force", str(wt)])
            click.echo(f"removed {wt.name}")
```

- [ ] **Step 4: Tests pass, lint. Dry run for real:** `uv run obs-lsd-rig evolve clips/danny-01.mov --gens 1 --dry-run` → one branch `evo/g1-0`, a brief in `.worktrees/evo-g1-0/brief.md` (read it; it must be a coherent prompt with the rubric and the baseline score). Commit: `git commit -m "Add evolutionary loop driver with dry-run"`

---

### Task 8: First live generation, docs

**Files:**
- Modify: `README.md` (rig section), `CLAUDE.md` (rig modules + make targets), `techniques.md` (create at repo root: `- g0: baseline — polar noise rays, light wrap, slow drift`)

- [ ] **Step 1:** commit `techniques.md` on main (the loop's lineage log starts here).
- [ ] **Step 2:** `make baseline CLIP=clips/danny-01.mov` if not done in Task 6; note total/90.
- [ ] **Step 3:** `uv run obs-lsd-rig evolve clips/danny-01.mov --gens 1` for real (one child, one agent run, one judge score; ~10–20 min). Watch: the agent commits on `evo/g1-0`, render + gates + score land in the ledger, the table prints. If the agent fails, read `.worktrees/evo-g1-0/agent-output.json` and fix the brief/driver, not the agent.
- [ ] **Step 4:** README: a "Evaluation rig" section — recording a clip, `make baseline`, `make eval`, `make evolve GENS=N`, reading `runs/ledger.json`, promoting a champion (`git merge evo/gN-k`), the calibrate command; plain prose. CLAUDE.md: add `rig/` modules and the make targets; note "never run evolve on a dirty main; agents only touch evo/* worktrees".
- [ ] **Step 5:** Commit: `git commit -m "Document the evaluation rig and evolution loop"`.

---

## Self-review notes

- Spec coverage: layout (T1), metrics (T2), images (T3), render (T4), judge incl. pairwise/calibrate/key/cost (T5), gates + baseline (T6), evolve incl. worktrees/briefs/operators/crossover/ledger/selection/table/prune/dry-run (T7), Makefile (T1), docs + first live run (T8). Error handling per spec: judge retry/drop (T5), agent timeout/failure/no-change ledgered (T7), gate reasons human-readable (T6/T7).
- Deliberate deviations: the candidate's code runs in a subprocess inside its worktree (the spec's "rig renders" stays true, but it must use the child's code, not the controller's import); `Pipeline` gains two read-only attributes (`last_box`, `last_mask_out`) so render can compute deltas without private access.
- Type consistency: `Score.to_json()` dict keys = seamless/distracting/ornate/total/notes/top_fix/samples/model/cost_usd, used by ledger records' `score`; `gate(cand, base, checks_ok)`; `select_survivors(records, k)`; `child_plan(gen, survivors)`; `build_brief(branch, generation, operator, parent_score, parent_stats, techniques, extra)`; `render(clip, out_dir, cfg, segmenter, sheet_frames) -> RunResult(run_dir, stats, n_frames)`.
