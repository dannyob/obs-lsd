# Glowing-Head Virtual Camera Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A CLI (`obs-lsd`) that takes a webcam, isolates the head, centers it at constant size on black, wraps it in animated rainbow rays, and publishes it as the OBS Virtual Camera for Zoom.

**Architecture:** Per-frame pipeline of small pure-numpy modules: `capture` -> `segment` (MediaPipe multiclass selfie segmenter + face landmarker chin cut) -> `track` (smoothed affine to center/scale) -> `halo` (tileable noise texture sampled in polar coords via cv2.remap) -> `compose` -> `output` (pyvirtualcam + cv2 preview). `pipeline.py` wires them; `cli.py` is the click entry point.

**Tech Stack:** Python 3.12 (uv, uv_build), mediapipe 0.10.x (NOT 1.x — its FaceLandmarker crashes on macOS), opencv-python, numpy, pyvirtualcam, click, pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-08-18-glowing-head-vcam-design.md`

## Global Constraints

- `requires-python = "==3.12.*"`; dependency pin `mediapipe>=0.10,<1.0`.
- Licence AGPL-3.0-or-later; `LICENSE` file and `license` field in pyproject.
- Output default 1280x720 @ 30 fps, black background, BGR uint8 frames.
- Every module operates on numpy arrays and is testable without a camera.
- Halo interface fixed: `HaloRenderer.render(mask: HxW float32, t: float) -> HxWx3 float32 RGB in [0,1]`.
- Model files live in `src/obs_lsd/models/` (gitignored), fetched by `make models`.
- Verified facts (2026-08-18, this Mac): mediapipe 0.10.35 works; segmenter labels are `['background','hair','body-skin','face-skin','clothes','others']`; `segment_for_video` ~17 ms; `opensimplex` too slow (1.4 s/frame) — do not use it.
- Danny's shell aliases `rm`/`cp`/`mv` to interactive; use `\rm` etc. in Makefiles/scripts.
- All commits: short imperative subject; body says why.

---

### Task 1: Project scaffold

**Files:**
- Create: `pyproject.toml`, `Makefile`, `LICENSE`, `README.md`, `CLAUDE.md`, `.gitignore` (modify), `src/obs_lsd/__init__.py`, `src/obs_lsd/cli.py`, `tests/__init__.py`, `tests/conftest.py`, `tests/test_cli.py`

**Interfaces:**
- Produces: package `obs_lsd`, console script `obs-lsd`, `obs_lsd.MODELS_DIR: pathlib.Path`, Makefile targets `venv install-dev test lint format models smoke run`.

- [ ] **Step 1: Write pyproject.toml**

```toml
[project]
name = "obs-lsd"
version = "0.1.0"
description = "Glowing psychedelic floating-head virtual camera for Zoom/OBS"
requires-python = "==3.12.*"
license = "AGPL-3.0-or-later"
authors = [{ name = "Danny O'Brien", email = "danny@spesh.com" }]
dependencies = [
    "click>=8.0",
    "numpy>=2.0",
    "opencv-python>=4.10",
    "mediapipe>=0.10,<1.0",
    "pyvirtualcam>=0.11",
]

[project.optional-dependencies]
dev = ["pytest>=8", "ruff>=0.5"]

[project.scripts]
obs-lsd = "obs_lsd.cli:cli"

[build-system]
requires = ["uv_build>=0.9.6,<0.14"]
build-backend = "uv_build"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]

[tool.ruff]
line-length = 100
target-version = "py312"
```

- [ ] **Step 2: Write Makefile**

```makefile
.PHONY: help venv install-dev test lint format models smoke run clean
.DEFAULT_GOAL := help
UV := uv
MODELS := src/obs_lsd/models
SEG_URL := https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite
FL_URL := https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
PORTRAIT_URL := https://storage.googleapis.com/mediapipe-assets/portrait.jpg

help: ## Show targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

venv: ## Create .venv with Python 3.12
	$(UV) venv -p 3.12

install-dev: venv ## Install package + dev deps
	$(UV) pip install -e ".[dev]"

models: $(MODELS)/selfie_multiclass_256x256.tflite $(MODELS)/face_landmarker.task ## Download MediaPipe models

$(MODELS)/selfie_multiclass_256x256.tflite:
	mkdir -p $(MODELS) && curl -sSL -o $@ $(SEG_URL)

$(MODELS)/face_landmarker.task:
	mkdir -p $(MODELS) && curl -sSL -o $@ $(FL_URL)

tests/data/portrait.jpg: ## Public MediaPipe sample portrait for tests
	mkdir -p tests/data && curl -sSL -o $@ $(PORTRAIT_URL)

testdata: tests/data/portrait.jpg ## Fetch test assets

test: ## Run tests
	$(UV) run pytest -q

lint: ## ruff check
	$(UV) run ruff check .

format: ## ruff format
	$(UV) run ruff format .

smoke: models testdata ## End-to-end headless run on synthetic clip
	$(UV) run obs-lsd --input tests/data/portrait.jpg --seconds 2 --no-vcam --no-preview --dump-frame /tmp/obs-lsd-smoke.png
	@test -s /tmp/obs-lsd-smoke.png && echo "smoke OK: /tmp/obs-lsd-smoke.png"

run: models ## Run live with preview + virtual camera
	$(UV) run obs-lsd

clean: ## Remove caches
	\rm -rf .pytest_cache .ruff_cache dist build
	find . -name __pycache__ -type d -exec \rm -rf {} +
```

(`uv run` uses `.venv` automatically; the `venv` target pins 3.12.)

- [ ] **Step 3: Write LICENSE, .gitignore, package init, minimal CLI, tests**

`LICENSE`: download the AGPL-3.0 text: `curl -sSL -o LICENSE https://www.gnu.org/licenses/agpl-3.0.txt`.

`.gitignore` (append):
```
src/obs_lsd/models/
tests/data/
*.png
```

`src/obs_lsd/__init__.py`:
```python
"""obs-lsd: glowing floating-head virtual camera."""

from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"
__version__ = "0.1.0"
```

`src/obs_lsd/cli.py` (placeholder, replaced in Task 8):
```python
"""Command-line interface for obs-lsd."""

import click


@click.command()
@click.version_option()
def cli():
    """Glowing floating-head virtual camera."""
    click.echo("obs-lsd: pipeline not wired yet")


if __name__ == "__main__":
    cli()
```

`tests/__init__.py`: empty. `tests/conftest.py`:
```python
"""Shared fixtures."""

import numpy as np
import pytest


@pytest.fixture
def ellipse_mask():
    """A 360x640 float32 mask with an ellipse 'head' centred at (320,180)."""
    import cv2

    m = np.zeros((360, 640), np.float32)
    cv2.ellipse(m, (320, 180), (60, 80), 0, 0, 360, 1.0, -1)
    return m
```

`tests/test_cli.py`:
```python
from click.testing import CliRunner

from obs_lsd.cli import cli


def test_cli_version():
    result = CliRunner().invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output
```

- [ ] **Step 4: Install and run**

Run: `make install-dev && make test && make lint`
Expected: 1 passed, ruff clean.

- [ ] **Step 5: Write CLAUDE.md and README.md**

`CLAUDE.md`: project overview (one paragraph), the module list from the spec, `make` targets, "TDD; run `make smoke` before claiming done; mediapipe must stay <1.0 on macOS". `README.md`: two paragraphs: what it does, and `make install-dev models && make run`, plus the OBS-virtual-camera-extension one-time setup and camera permission note.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "Scaffold obs-lsd package, Makefile, model download"
```

---

### Task 2: Tracker (centering + constant scale)

**Files:**
- Create: `src/obs_lsd/track.py`, `tests/test_track.py`

**Interfaces:**
- Produces:
  ```python
  @dataclass
  class HeadBox:  # what segment.py will emit; defined here to avoid a mediapipe import in track
      cx: float; cy: float; height: float
  @dataclass
  class Transform:
      scale: float; tx: float; ty: float
      def matrix(self) -> np.ndarray  # 2x3 float32 for cv2.warpAffine
      def apply(self, x, y) -> tuple[float, float]
  class Tracker:
      def __init__(self, out_size: tuple[int,int], target_frac=0.55, alpha_pos=0.25, alpha_scale=0.15)
      def update(self, head: HeadBox | None) -> Transform
      frames_missing: int
  ```

- [ ] **Step 1: Failing tests**

`tests/test_track.py`:
```python
import numpy as np
import pytest

from obs_lsd.track import HeadBox, Tracker, Transform


def test_transform_matrix_and_apply():
    t = Transform(scale=2.0, tx=10.0, ty=-5.0)
    m = t.matrix()
    assert m.shape == (2, 3) and m.dtype == np.float32
    assert t.apply(3, 4) == pytest.approx((16.0, 3.0))


def test_first_update_snaps_head_to_center_at_target_height():
    tr = Tracker(out_size=(1280, 720), target_frac=0.5)
    t = tr.update(HeadBox(cx=100, cy=50, height=100))
    assert t.apply(100, 50) == pytest.approx((640, 360))
    assert t.scale == pytest.approx(3.6)  # 0.5*720/100


def test_smoothing_moves_gradually_then_converges():
    tr = Tracker(out_size=(1280, 720), alpha_pos=0.5, alpha_scale=0.5)
    tr.update(HeadBox(cx=0, cy=0, height=100))
    t1 = tr.update(HeadBox(cx=100, cy=0, height=100))
    # after one step at alpha 0.5, the tracked cx is 50, so camera x=50 maps to center
    assert t1.apply(50, 0)[0] == pytest.approx(640)
    for _ in range(40):
        t = tr.update(HeadBox(cx=100, cy=0, height=100))
    assert t.apply(100, 0)[0] == pytest.approx(640, abs=1e-3)


def test_none_holds_last_transform_and_counts_missing():
    tr = Tracker(out_size=(640, 360))
    t0 = tr.update(HeadBox(cx=10, cy=10, height=50))
    assert tr.frames_missing == 0
    t1 = tr.update(None)
    t2 = tr.update(None)
    assert (t1.scale, t1.tx, t1.ty) == (t0.scale, t0.tx, t0.ty) == (t2.scale, t2.tx, t2.ty)
    assert tr.frames_missing == 2
    tr.update(HeadBox(cx=10, cy=10, height=50))
    assert tr.frames_missing == 0


def test_none_before_any_head_gives_identity_centered():
    tr = Tracker(out_size=(640, 360))
    t = tr.update(None)
    assert t.scale == 1.0 and t.tx == 0.0 and t.ty == 0.0
```

- [ ] **Step 2: Run, expect ImportError**

`uv run pytest tests/test_track.py -q`

- [ ] **Step 3: Implement**

`src/obs_lsd/track.py`:
```python
"""Smooth the head position/size and map camera space to output space."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HeadBox:
    """Head centre and height in camera pixels."""

    cx: float
    cy: float
    height: float


@dataclass
class Transform:
    """Uniform scale + translation: out = scale * cam + (tx, ty)."""

    scale: float
    tx: float
    ty: float

    def matrix(self) -> np.ndarray:
        return np.array([[self.scale, 0.0, self.tx], [0.0, self.scale, self.ty]], np.float32)

    def apply(self, x: float, y: float) -> tuple[float, float]:
        return (self.scale * x + self.tx, self.scale * y + self.ty)


class Tracker:
    """Exponentially smooth the head box; emit a Transform centring it at a fixed size."""

    def __init__(
        self,
        out_size: tuple[int, int],
        target_frac: float = 0.55,
        alpha_pos: float = 0.25,
        alpha_scale: float = 0.15,
    ):
        self.out_w, self.out_h = out_size
        self.target_frac = target_frac
        self.alpha_pos = alpha_pos
        self.alpha_scale = alpha_scale
        self._state: HeadBox | None = None
        self.frames_missing = 0

    def update(self, head: HeadBox | None) -> Transform:
        if head is None:
            self.frames_missing += 1
        else:
            self.frames_missing = 0
            if self._state is None:
                self._state = HeadBox(head.cx, head.cy, head.height)
            else:
                s, ap, asc = self._state, self.alpha_pos, self.alpha_scale
                s.cx += ap * (head.cx - s.cx)
                s.cy += ap * (head.cy - s.cy)
                s.height += asc * (head.height - s.height)
        if self._state is None:
            return Transform(1.0, 0.0, 0.0)
        s = self._state
        scale = self.target_frac * self.out_h / max(s.height, 1.0)
        return Transform(scale, self.out_w / 2 - scale * s.cx, self.out_h / 2 - scale * s.cy)
```

- [ ] **Step 4: Run tests, expect 5 passed. Commit**

```bash
git add src/obs_lsd/track.py tests/test_track.py && git commit -m "Add Tracker: smoothed centre/scale transform"
```

---

### Task 3: Halo renderer (rays from tileable noise, polar remap)

**Files:**
- Create: `src/obs_lsd/halo.py`, `tests/test_halo.py`

**Interfaces:**
- Produces:
  ```python
  @dataclass
  class HaloParams:
      reach: float = 260.0      # px beyond contour where rays die
      density: float = 48.0     # angular repeats of the noise texture
      speed: float = 120.0      # px/s outward scroll
      flicker: float = 0.8      # texture-slice cycles per second
      hue_speed: float = 0.08   # hue cycles per second
      hue_twist: float = 0.0015 # hue per px of radius
      sharpness: float = 2.5    # exponent on noise -> ray contrast
      rim_width: float = 6.0    # px, white-ish rim glow
      rim_gain: float = 0.7
      scale: float = 0.5        # internal render resolution factor
  class HaloRenderer:
      def __init__(self, params: HaloParams | None = None, seed: int = 0)
      def render(self, mask: np.ndarray, t: float) -> np.ndarray  # HxWx3 float32 RGB 0..1
  def tileable_noise(h: int, w: int, octaves: int = 4, seed: int = 0) -> np.ndarray  # HxW float32 0..1, wraps on both axes
  ```

- [ ] **Step 1: Failing tests**

`tests/test_halo.py`:
```python
import numpy as np

from obs_lsd.halo import HaloParams, HaloRenderer, tileable_noise


def test_tileable_noise_wraps_and_ranges():
    n = tileable_noise(64, 128, seed=1)
    assert n.shape == (64, 128) and n.dtype == np.float32
    assert 0.0 <= n.min() and n.max() <= 1.0 and n.std() > 0.05
    # continuity across the seam: last col ~ first col, last row ~ first row
    assert np.abs(n[:, -1] - n[:, 0]).mean() < np.abs(n[:, 64] - n[:, 0]).mean()
    assert np.abs(n[-1, :] - n[0, :]).mean() < np.abs(n[32, :] - n[0, :]).mean()


def test_render_shape_dtype_range(ellipse_mask):
    out = HaloRenderer().render(ellipse_mask, 0.0)
    assert out.shape == (360, 640, 3) and out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_render_is_black_inside_head_and_bright_near_contour(ellipse_mask):
    out = HaloRenderer(HaloParams(rim_gain=1.0)).render(ellipse_mask, 0.3)
    inside = out[180, 320]
    assert inside.max() < 0.02
    ring = out[180 - 90 : 180 - 82, 320].max()  # just outside the top of the ellipse
    assert ring > 0.3
    far = out[5, 5].max()  # ~ (315,175) px away, beyond reach=260
    assert far < 0.05


def test_render_deterministic_and_time_varying(ellipse_mask):
    r = HaloRenderer(seed=3)
    a = r.render(ellipse_mask, 1.0)
    b = HaloRenderer(seed=3).render(ellipse_mask, 1.0)
    c = r.render(ellipse_mask, 1.5)
    assert np.array_equal(a, b)
    assert np.abs(a - c).mean() > 0.005


def test_render_empty_mask_returns_black():
    out = HaloRenderer().render(np.zeros((90, 160), np.float32), 0.0)
    assert out.max() == 0.0
```

- [ ] **Step 2: Run, expect ImportError**

- [ ] **Step 3: Implement**

`src/obs_lsd/halo.py`:
```python
"""Psychedelic ray halo around a head mask.

Rays are read from a tileable 2D noise texture indexed by (angle, radius) via
cv2.remap, scrolled outward over time; two texture rows of a small stack are
blended for flicker. Everything is numpy/OpenCV, no per-pixel Python.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


def tileable_noise(h: int, w: int, octaves: int = 4, seed: int = 0) -> np.ndarray:
    """Value noise that wraps on both axes, in [0,1]. Built by upsampling
    random grids with cv2.resize on a 3x3 tiled copy and cropping the centre."""
    rng = np.random.default_rng(seed)
    out = np.zeros((h, w), np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        gh, gw = max(2, 4 * 2**o), max(2, 8 * 2**o)
        g = rng.random((gh, gw), dtype=np.float32)
        big = np.tile(g, (3, 3))
        up = cv2.resize(big, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)
        out += amp * up[h : 2 * h, w : 2 * w]
        total += amp
        amp *= 0.5
    out /= total
    out -= out.min()
    out /= max(out.max(), 1e-6)
    return out.astype(np.float32)


@dataclass
class HaloParams:
    reach: float = 260.0
    density: float = 48.0
    speed: float = 120.0
    flicker: float = 0.8
    hue_speed: float = 0.08
    hue_twist: float = 0.0015
    sharpness: float = 2.5
    rim_width: float = 6.0
    rim_gain: float = 0.7
    scale: float = 0.5


class HaloRenderer:
    TEX_H, TEX_W, TEX_N = 256, 1024, 8  # radius rows, angle cols, flicker slices

    def __init__(self, params: HaloParams | None = None, seed: int = 0):
        self.p = params or HaloParams()
        self.tex = np.stack(
            [tileable_noise(self.TEX_H, self.TEX_W, seed=seed * 100 + i) for i in range(self.TEX_N)]
        )

    def render(self, mask: np.ndarray, t: float) -> np.ndarray:
        H, W = mask.shape
        s = self.p.scale
        h, w = max(1, int(H * s)), max(1, int(W * s))
        m = cv2.resize(mask, (w, h), interpolation=cv2.INTER_AREA) if s != 1.0 else mask
        if m.max() <= 0.0:
            return np.zeros((H, W, 3), np.float32)
        binary = (m > 0.5).astype(np.uint8)
        # distance (in full-res px) from the head contour, outward
        dist = cv2.distanceTransform(1 - binary, cv2.DIST_L2, 3).astype(np.float32) / s
        ys, xs = np.nonzero(binary)
        cx, cy = xs.mean(), ys.mean()
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        theta = np.arctan2(yy - cy, xx - cx)  # -pi..pi
        # texture lookup: angle wraps around TEX_W, radius scrolls with time
        map_x = ((theta / (2 * np.pi) + 0.5) * self.p.density * self.TEX_W / self.p.density) % self.TEX_W
        map_x = ((theta / (2 * np.pi) + 0.5) * self.TEX_W) % self.TEX_W
        map_y = ((dist * (self.TEX_H / self.p.reach)) - t * self.p.speed * (self.TEX_H / self.p.reach)) % self.TEX_H
        f = t * self.p.flicker * self.TEX_N
        i0 = int(f) % self.TEX_N
        i1 = (i0 + 1) % self.TEX_N
        frac = np.float32(f - int(f))
        n0 = cv2.remap(self.tex[i0], map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        n1 = cv2.remap(self.tex[i1], map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        n = n0 * (1 - frac) + n1 * frac
        # angular fine detail: multiply by a second, denser lookup
        n_fine = cv2.remap(
            self.tex[(i0 + 3) % self.TEX_N],
            (map_x * self.p.density / 8.0) % self.TEX_W,
            map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_WRAP,
        )
        rays = np.clip(n * (0.5 + n_fine), 0, 1) ** self.p.sharpness
        falloff = np.clip(1.0 - dist / self.p.reach, 0, 1) ** 1.5
        value = rays * falloff
        rim = self.p.rim_gain * np.exp(-dist / self.p.rim_width)
        hue = (theta / (2 * np.pi) + t * self.p.hue_speed + dist * self.p.hue_twist) % 1.0
        hsv = np.dstack([hue * 360.0, np.clip(1.0 - rim, 0.2, 1.0), np.clip(value + rim, 0, 1)]).astype(np.float32)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        rgb[binary.astype(bool)] = 0.0
        if s != 1.0:
            rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_LINEAR)
            rgb[mask > 0.5] = 0.0
        return np.clip(rgb, 0, 1).astype(np.float32)
```

Note: remove the duplicated first `map_x =` line when implementing (kept here only to show the simplification — the second line is the one to use).

- [ ] **Step 4: Run tests until they pass; if the "bright near contour" or "far is dark" thresholds fail, adjust the *test sample points*, not the algorithm, unless the algorithm is clearly wrong. Then time it:**

`uv run python -c "import numpy as np,cv2,time; from obs_lsd.halo import HaloRenderer; m=np.zeros((720,1280),np.float32); cv2.ellipse(m,(640,360),(120,160),0,0,360,1.0,-1); r=HaloRenderer(); r.render(m,0); t=time.time(); [r.render(m,i/30) for i in range(30)]; print((time.time()-t)/30*1000,'ms')"`
Expected: < 12 ms at scale 0.5. If more, drop `HaloParams.scale` default to 0.4 and note it in the commit.

- [ ] **Step 5: Commit**

```bash
git add src/obs_lsd/halo.py tests/test_halo.py && git commit -m "Add HaloRenderer: polar noise rays via cv2.remap"
```

---

### Task 4: Compositor

**Files:**
- Create: `src/obs_lsd/compose.py`, `tests/test_compose.py`

**Interfaces:**
- Produces: `compose(frame_bgr: HxWx3 uint8, mask: HxW float32, halo_rgb: HxWx3 float32, halo_gain: float = 1.0) -> HxWx3 uint8 BGR`

- [ ] **Step 1: Failing tests**

`tests/test_compose.py`:
```python
import numpy as np

from obs_lsd.compose import compose


def test_compose_head_over_black_with_halo():
    frame = np.full((4, 4, 3), 200, np.uint8)
    mask = np.zeros((4, 4), np.float32)
    mask[1:3, 1:3] = 1.0
    halo = np.zeros((4, 4, 3), np.float32)
    halo[0, 0] = (1.0, 0.0, 0.0)  # red in RGB
    out = compose(frame, mask, halo)
    assert out.dtype == np.uint8 and out.shape == (4, 4, 3)
    assert tuple(out[1, 1]) == (200, 200, 200)  # head pixels kept
    assert tuple(out[0, 0]) == (0, 0, 255)  # RGB red -> BGR
    assert tuple(out[3, 3]) == (0, 0, 0)  # black background


def test_compose_halo_gain_and_soft_mask():
    frame = np.full((2, 2, 3), 100, np.uint8)
    mask = np.array([[0.5, 0.0], [0.0, 0.0]], np.float32)
    halo = np.ones((2, 2, 3), np.float32)
    out = compose(frame, mask, halo, halo_gain=0.5)
    assert tuple(out[1, 1]) == (128, 128, 128)  # 0.5 * 255 rounded
    assert 100 < out[0, 0, 0] < 200  # blend of head 100 and halo 0.5*255*0.5
```

- [ ] **Step 2: Run, expect ImportError**

- [ ] **Step 3: Implement**

`src/obs_lsd/compose.py`:
```python
"""Composite head over black with the halo behind it."""

from __future__ import annotations

import cv2
import numpy as np


def compose(
    frame_bgr: np.ndarray, mask: np.ndarray, halo_rgb: np.ndarray, halo_gain: float = 1.0
) -> np.ndarray:
    m = mask[..., None]
    halo_bgr = cv2.cvtColor(halo_rgb, cv2.COLOR_RGB2BGR) * halo_gain
    out = halo_bgr * (1.0 - m) * 255.0 + frame_bgr.astype(np.float32) * m
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)
```

- [ ] **Step 4: Run tests, expect pass. Commit**

```bash
git add src/obs_lsd/compose.py tests/test_compose.py && git commit -m "Add compose: head over black plus halo"
```

---

### Task 5: Head segmenter (MediaPipe)

**Files:**
- Create: `src/obs_lsd/segment.py`, `tests/test_segment.py`
- Needs: `make models testdata` (models in `src/obs_lsd/models/`, `tests/data/portrait.jpg`)

**Interfaces:**
- Consumes: `obs_lsd.MODELS_DIR`, `obs_lsd.track.HeadBox`
- Produces:
  ```python
  @dataclass
  class HeadMask:
      mask: np.ndarray        # HxW float32 0..1, camera resolution
      box: HeadBox
  class HeadSegmenter:
      def __init__(self, models_dir: Path = MODELS_DIR, hold_frames: int = 10, feather: int = 5, chin_margin: float = 0.06)
      def mask(self, frame_bgr: np.ndarray, timestamp_ms: int) -> HeadMask | None
      def close(self) -> None
  class ModelsMissing(RuntimeError)
  def head_from_masks(hair, face, chin_line, feather) -> np.ndarray   # pure numpy helper, unit-testable
  def chin_halfplane(shape, chin_xy, roll_deg, margin_px) -> np.ndarray  # HxW float32 1 above line, 0 below
  ```

- [ ] **Step 1: Failing tests**

`tests/test_segment.py`:
```python
from pathlib import Path

import cv2
import numpy as np
import pytest

from obs_lsd import MODELS_DIR
from obs_lsd.segment import HeadSegmenter, ModelsMissing, chin_halfplane, head_from_masks

PORTRAIT = Path(__file__).parent / "data" / "portrait.jpg"
have_models = (MODELS_DIR / "face_landmarker.task").exists()


def test_chin_halfplane_level():
    hp = chin_halfplane((100, 100), chin_xy=(50, 60), roll_deg=0.0, margin_px=5)
    assert hp[50, 50] == 1.0 and hp[64, 50] == 1.0 and hp[66, 50] == 0.0
    assert hp[66, 10] == 0.0 and hp[50, 90] == 1.0


def test_chin_halfplane_rolled():
    hp = chin_halfplane((100, 100), chin_xy=(50, 50), roll_deg=45.0, margin_px=0)
    # with the head rolled 45deg, points 'below' along the rolled down-axis are cut
    assert hp[80, 80] == 0.0 or hp[80, 20] == 0.0
    assert hp[20, 50] == 1.0


def test_head_from_masks_unions_and_cuts_below_chin():
    hair = np.zeros((10, 10), np.float32); hair[0:3, :] = 1
    face = np.zeros((10, 10), np.float32); face[3:8, :] = 1
    hp = np.ones((10, 10), np.float32); hp[6:, :] = 0
    m = head_from_masks(hair, face, hp, feather=0)
    assert m[1, 5] == 1.0 and m[4, 5] == 1.0
    assert m[7, 5] == 0.0 and m[9, 5] == 0.0


def test_missing_models_raises(tmp_path):
    with pytest.raises(ModelsMissing):
        HeadSegmenter(models_dir=tmp_path)


@pytest.mark.skipif(not (have_models and PORTRAIT.exists()), reason="run: make models testdata")
def test_portrait_mask_covers_face_not_neck():
    img = cv2.imread(str(PORTRAIT))
    seg = HeadSegmenter()
    try:
        hm = seg.mask(img, 0)
    finally:
        seg.close()
    assert hm is not None
    H, W = img.shape[:2]
    assert hm.mask.shape == (H, W)
    assert 50 < hm.box.height < H
    # centre of the box is inside the mask; well below the box, nothing
    assert hm.mask[int(hm.box.cy), int(hm.box.cx)] > 0.5
    below = int(min(H - 1, hm.box.cy + hm.box.height))
    assert hm.mask[below, int(hm.box.cx)] < 0.1


@pytest.mark.skipif(not have_models, reason="run: make models")
def test_blank_frame_returns_none_after_hold():
    seg = HeadSegmenter(hold_frames=2)
    try:
        blank = np.zeros((240, 320, 3), np.uint8)
        assert seg.mask(blank, 0) is None
    finally:
        seg.close()
```

- [ ] **Step 2: Run, expect ImportError**

- [ ] **Step 3: Implement**

`src/obs_lsd/segment.py`:
```python
"""Head-only mask from a camera frame using MediaPipe segmenter + face landmarks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from obs_lsd import MODELS_DIR
from obs_lsd.track import HeadBox

SEG_MODEL = "selfie_multiclass_256x256.tflite"
FL_MODEL = "face_landmarker.task"
HAIR, FACE = 1, 3  # confidence_masks indices (labels: background,hair,body-skin,face-skin,clothes,others)
CHIN_LM, LEFT_EYE_LM, RIGHT_EYE_LM = 152, 33, 263


class ModelsMissing(RuntimeError):
    pass


@dataclass
class HeadMask:
    mask: np.ndarray
    box: HeadBox


def chin_halfplane(shape, chin_xy, roll_deg: float, margin_px: float) -> np.ndarray:
    """1.0 on the 'above chin' side of a line through chin_xy tilted by roll_deg, else 0."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = math.radians(roll_deg)
    # 'down' unit vector rotated by roll
    dx, dy = -math.sin(r), math.cos(r)
    d = (xx - chin_xy[0]) * dx + (yy - chin_xy[1]) * dy - margin_px
    return (d <= 0).astype(np.float32)


def head_from_masks(hair, face, halfplane, feather: int) -> np.ndarray:
    m = np.maximum(hair, face) * halfplane
    if feather > 0:
        k = feather * 2 + 1
        m = cv2.GaussianBlur(m, (k, k), 0)
    return np.clip(m, 0, 1).astype(np.float32)


class HeadSegmenter:
    def __init__(
        self,
        models_dir: Path = MODELS_DIR,
        hold_frames: int = 10,
        feather: int = 5,
        chin_margin: float = 0.06,
    ):
        seg_path, fl_path = models_dir / SEG_MODEL, models_dir / FL_MODEL
        if not (seg_path.exists() and fl_path.exists()):
            raise ModelsMissing(f"MediaPipe models not found in {models_dir}; run `make models`")
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        self._seg = vision.ImageSegmenter.create_from_options(
            vision.ImageSegmenterOptions(
                base_options=BaseOptions(model_asset_path=str(seg_path)),
                running_mode=vision.RunningMode.VIDEO,
                output_confidence_masks=True,
            )
        )
        self._fl = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(fl_path)),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1,
            )
        )
        self.hold_frames = hold_frames
        self.feather = feather
        self.chin_margin = chin_margin
        self._last: HeadMask | None = None
        self._missing = 0
        self._ts = -1

    def close(self) -> None:
        self._seg.close()
        self._fl.close()

    def mask(self, frame_bgr: np.ndarray, timestamp_ms: int) -> HeadMask | None:
        # MediaPipe VIDEO mode needs strictly increasing timestamps
        self._ts = max(self._ts + 1, int(timestamp_ms))
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        h, w = rgb.shape[:2]
        fl = self._fl.detect_for_video(img, self._ts)
        if not fl.face_landmarks:
            return self._hold()
        lm = fl.face_landmarks[0]
        chin = (lm[CHIN_LM].x * w, lm[CHIN_LM].y * h)
        le, re = lm[LEFT_EYE_LM], lm[RIGHT_EYE_LM]
        roll = math.degrees(math.atan2((re.y - le.y) * h, (re.x - le.x) * w))
        seg = self._seg.segment_for_video(img, self._ts)
        cm = seg.confidence_masks
        hair = cm[HAIR].numpy_view()[..., 0]
        face = cm[FACE].numpy_view()[..., 0]
        face_h = abs(chin[1] - min(le.y, re.y) * h) * 2.0
        hp = chin_halfplane((h, w), chin, roll, self.chin_margin * face_h)
        m = head_from_masks(hair, face, hp, self.feather)
        ys, xs = np.nonzero(m > 0.5)
        if len(ys) < 50:
            return self._hold()
        box = HeadBox(cx=float(xs.mean()), cy=float(ys.mean()), height=float(ys.max() - ys.min() + 1))
        self._last = HeadMask(mask=m, box=box)
        self._missing = 0
        return self._last

    def _hold(self) -> HeadMask | None:
        self._missing += 1
        if self._last is not None and self._missing <= self.hold_frames:
            return self._last
        return None
```

- [ ] **Step 4: `make models testdata`, run tests, expect 6 passed. Look at the mask by eye:**

`uv run python -c "import cv2; from obs_lsd.segment import HeadSegmenter; im=cv2.imread('tests/data/portrait.jpg'); s=HeadSegmenter(); hm=s.mask(im,0); cv2.imwrite('/tmp/mask.png',(hm.mask*255).astype('uint8')); cv2.imwrite('/tmp/cut.png',(im*hm.mask[...,None]).astype('uint8'))"` then `open /tmp/cut.png` — the neck must be gone, hair kept. Tune `chin_margin` if the cut is too high/low.

- [ ] **Step 5: Commit**

```bash
git add src/obs_lsd/segment.py tests/test_segment.py && git commit -m "Add HeadSegmenter: MediaPipe hair+face mask cut at the chin"
```

---

### Task 6: Frame sources and sinks

**Files:**
- Create: `src/obs_lsd/capture.py`, `src/obs_lsd/output.py`, `tests/test_capture.py`

**Interfaces:**
- Produces:
  ```python
  # capture.py
  def list_cameras() -> list[str]                       # names from system_profiler, index order
  def resolve_camera(spec: str | None) -> int | str     # int index for cameras, str path for files/images
  class FrameSource:
      def __init__(self, spec: str | None, width: int, height: int, fps: float)
      def __iter__(self) -> Iterator[tuple[np.ndarray, float]]  # (bgr frame, seconds since start)
      def close(self)
  class CameraError(RuntimeError)
  # output.py
  class VirtualCam:  # context manager; .send(bgr uint8)
      def __init__(self, width, height, fps)
  class VirtualCamMissing(RuntimeError)
  class Preview:  # .show(bgr) -> str | None action in {"quit","size+","size-","hue+","hue-","reach+","reach-","freeze","print"}
      def __init__(self, title="obs-lsd")
      def close(self)
  ```

- [ ] **Step 1: Failing tests (file/image source and camera name parsing only; live devices are not unit-tested)**

`tests/test_capture.py`:
```python
import cv2
import numpy as np

from obs_lsd.capture import FrameSource, parse_system_profiler, resolve_camera

SP_TEXT = """Camera:

    MacBook Pro Camera:

      Model ID: MacBook Pro Camera
      Unique ID: 6C707041

    Yawl Camera:

      Model ID: iPhone18,3
      Unique ID: 908AC892
"""


def test_parse_system_profiler():
    assert parse_system_profiler(SP_TEXT) == ["MacBook Pro Camera", "Yawl Camera"]


def test_resolve_camera_index_name_and_path(tmp_path):
    p = tmp_path / "x.png"
    p.write_bytes(b"x")
    assert resolve_camera("1") == 1
    assert resolve_camera(str(p)) == str(p)
    assert resolve_camera(None) == 0
    assert resolve_camera("yawl", names=["MacBook Pro Camera", "Yawl Camera"]) == 1


def test_image_source_loops_and_times(tmp_path):
    p = tmp_path / "f.png"
    cv2.imwrite(str(p), np.full((20, 30, 3), 7, np.uint8))
    src = FrameSource(str(p), width=30, height=20, fps=100)
    frames = []
    for frame, t in src:
        frames.append((frame.shape, t))
        if len(frames) == 3:
            break
    src.close()
    assert [f[0] for f in frames] == [(20, 30, 3)] * 3
    assert frames[2][1] > frames[0][1] >= 0.0
```

- [ ] **Step 2: Run, expect ImportError**

- [ ] **Step 3: Implement capture.py**

```python
"""Frame sources: webcam by index/name, video file, or a still image (looped)."""

from __future__ import annotations

import re
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np


class CameraError(RuntimeError):
    pass


def parse_system_profiler(text: str) -> list[str]:
    """Camera names from `system_profiler SPCameraDataType` output, in order."""
    return [m.group(1) for m in re.finditer(r"^    (\S.*?):\s*$", text, re.M)]


def list_cameras() -> list[str]:
    try:
        out = subprocess.run(
            ["system_profiler", "SPCameraDataType"], capture_output=True, text=True, timeout=10
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return parse_system_profiler(out)


def resolve_camera(spec: str | None, names: list[str] | None = None) -> int | str:
    if spec is None:
        return 0
    if spec.isdigit():
        return int(spec)
    if Path(spec).exists():
        return spec
    names = list_cameras() if names is None else names
    for i, n in enumerate(names):
        if spec.lower() in n.lower():
            return i
    raise CameraError(f"No camera matching {spec!r}. Available: {names or 'none found'}")


IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp"}


class FrameSource:
    def __init__(self, spec: str | None, width: int, height: int, fps: float):
        self.fps = fps
        self._still: np.ndarray | None = None
        self._cap: cv2.VideoCapture | None = None
        target = resolve_camera(spec)
        if isinstance(target, str) and Path(target).suffix.lower() in IMAGE_EXT:
            self._still = cv2.imread(target)
            if self._still is None:
                raise CameraError(f"Could not read image {target}")
        else:
            self._cap = cv2.VideoCapture(target)
            if isinstance(target, int):
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                self._cap.set(cv2.CAP_PROP_FPS, fps)
            if not self._cap.isOpened():
                raise CameraError(
                    f"Could not open camera {target!r}. Available: {list_cameras()}. "
                    "On macOS check System Settings > Privacy & Security > Camera."
                )
        self._t0 = time.monotonic()

    def __iter__(self) -> Iterator[tuple[np.ndarray, float]]:
        period = 1.0 / self.fps if self.fps > 0 else 0.0
        while True:
            if self._still is not None:
                frame = self._still.copy()
                time.sleep(period)
            else:
                ok, frame = self._cap.read()
                if not ok:
                    return
            yield frame, time.monotonic() - self._t0

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
```

- [ ] **Step 4: Implement output.py**

```python
"""Sinks: OBS virtual camera and a cv2 preview window with hotkeys."""

from __future__ import annotations

import cv2
import numpy as np


class VirtualCamMissing(RuntimeError):
    pass


class VirtualCam:
    def __init__(self, width: int, height: int, fps: float):
        import pyvirtualcam

        try:
            self._cam = pyvirtualcam.Camera(
                width=width, height=height, fps=fps, fmt=pyvirtualcam.PixelFormat.BGR
            )
        except RuntimeError as e:
            raise VirtualCamMissing(
                "Could not open the OBS Virtual Camera. Launch OBS once, click 'Start Virtual "
                "Camera', approve the system extension in System Settings if asked, then retry."
                f"\n({e})"
            ) from e
        self.device = self._cam.device

    def send(self, frame_bgr: np.ndarray) -> None:
        self._cam.send(frame_bgr)
        self._cam.sleep_until_next_frame()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._cam.close()


KEYS = {
    ord("q"): "quit", 27: "quit",
    ord("]"): "size+", ord("["): "size-",
    ord("="): "hue+", ord("-"): "hue-",
    ord("."): "reach+", ord(","): "reach-",
    ord("s"): "freeze", ord("p"): "print",
}


class Preview:
    def __init__(self, title: str = "obs-lsd"):
        self.title = title
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)

    def show(self, frame_bgr: np.ndarray) -> str | None:
        cv2.imshow(self.title, frame_bgr)
        k = cv2.waitKey(1) & 0xFF
        return KEYS.get(k)

    def close(self) -> None:
        cv2.destroyWindow(self.title)
```

- [ ] **Step 5: Run tests, expect pass. Commit**

```bash
git add src/obs_lsd/capture.py src/obs_lsd/output.py tests/test_capture.py && git commit -m "Add frame sources and virtual camera / preview sinks"
```

---

### Task 7: Pipeline

**Files:**
- Create: `src/obs_lsd/pipeline.py`, `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `HeadSegmenter`, `Tracker`, `HaloRenderer`, `HaloParams`, `compose`
- Produces:
  ```python
  @dataclass
  class PipelineConfig:
      width: int = 1280; height: int = 720; head_frac: float = 0.55
      halo: HaloParams = field(default_factory=HaloParams)
      fade_seconds: float = 1.0
  class Pipeline:
      def __init__(self, cfg: PipelineConfig, segmenter=None)  # segmenter injectable for tests
      def step(self, frame_bgr, t: float) -> np.ndarray       # out BGR uint8 (height,width,3)
      frozen: bool                                             # when True, reuse last mask
      timings: dict[str, float]                                # last-step ms per stage
      def close(self)
  ```

- [ ] **Step 1: Failing tests (fake segmenter, no models needed)**

`tests/test_pipeline.py`:
```python
import cv2
import numpy as np

from obs_lsd.pipeline import Pipeline, PipelineConfig
from obs_lsd.segment import HeadMask
from obs_lsd.track import HeadBox


class FakeSeg:
    def __init__(self, present=True):
        self.present = present

    def mask(self, frame, ts):
        if not self.present:
            return None
        h, w = frame.shape[:2]
        m = np.zeros((h, w), np.float32)
        cv2.ellipse(m, (100, 120), (30, 40), 0, 0, 360, 1.0, -1)
        return HeadMask(mask=m, box=HeadBox(cx=100, cy=120, height=80))

    def close(self):
        pass


def test_step_centres_head_and_draws_halo():
    cfg = PipelineConfig(width=320, height=180, head_frac=0.5)
    p = Pipeline(cfg, segmenter=FakeSeg())
    frame = np.full((240, 320, 3), (10, 200, 10), np.uint8)
    out = p.step(frame, 0.0)
    assert out.shape == (180, 320, 3) and out.dtype == np.uint8
    assert tuple(out[90, 160]) == (10, 200, 10)  # head colour at centre
    # head height should be ~0.5*180 = 90 px: green at 90-40, black beyond
    assert out[90 - 40, 160, 1] == 200 and out[90 - 55, 160, 1] < 200
    assert out[2, 2].max() == 0  # far corner black
    assert set(p.timings) >= {"segment", "warp", "halo", "compose"}


def test_lost_head_fades_halo_to_black():
    cfg = PipelineConfig(width=320, height=180, fade_seconds=0.5)
    seg = FakeSeg()
    p = Pipeline(cfg, segmenter=seg)
    p.step(np.zeros((240, 320, 3), np.uint8), 0.0)
    seg.present = False
    bright = p.step(np.zeros((240, 320, 3), np.uint8), 0.05).max()
    gone = p.step(np.zeros((240, 320, 3), np.uint8), 2.0).max()
    assert bright > 0 and gone == 0
```

- [ ] **Step 2: Run, expect ImportError**

- [ ] **Step 3: Implement**

`src/obs_lsd/pipeline.py`:
```python
"""Wire capture-independent stages: segment -> track -> warp -> halo -> compose."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from obs_lsd.compose import compose
from obs_lsd.halo import HaloParams, HaloRenderer
from obs_lsd.track import Tracker


@dataclass
class PipelineConfig:
    width: int = 1280
    height: int = 720
    head_frac: float = 0.55
    halo: HaloParams = field(default_factory=HaloParams)
    fade_seconds: float = 1.0


class Pipeline:
    def __init__(self, cfg: PipelineConfig, segmenter=None):
        self.cfg = cfg
        if segmenter is None:
            from obs_lsd.segment import HeadSegmenter

            segmenter = HeadSegmenter()
        self.seg = segmenter
        self.tracker = Tracker((cfg.width, cfg.height), target_frac=cfg.head_frac)
        self.halo = HaloRenderer(cfg.halo)
        self.frozen = False
        self.timings: dict[str, float] = {}
        self._last_head = None
        self._lost_since: float | None = None

    def step(self, frame_bgr: np.ndarray, t: float) -> np.ndarray:
        tm: dict[str, float] = {}
        t0 = time.perf_counter()
        if self.frozen and self._last_head is not None:
            head = self._last_head
        else:
            head = self.seg.mask(frame_bgr, int(t * 1000))
        tm["segment"] = (time.perf_counter() - t0) * 1000

        if head is not None:
            self._last_head = head
            self._lost_since = None
            gain = 1.0
        else:
            self._lost_since = t if self._lost_since is None else self._lost_since
            gain = max(0.0, 1.0 - (t - self._lost_since) / max(self.cfg.fade_seconds, 1e-6))
        tr = self.tracker.update(head.box if head is not None else None)

        t1 = time.perf_counter()
        size = (self.cfg.width, self.cfg.height)
        if self._last_head is None:
            return np.zeros((self.cfg.height, self.cfg.width, 3), np.uint8)
        src_mask = self._last_head.mask
        m = cv2.warpAffine(src_mask, tr.matrix(), size, flags=cv2.INTER_LINEAR, borderValue=0)
        f = cv2.warpAffine(frame_bgr, tr.matrix(), size, flags=cv2.INTER_LINEAR, borderValue=0)
        tm["warp"] = (time.perf_counter() - t1) * 1000

        t2 = time.perf_counter()
        halo = self.halo.render(m, t) if gain > 0 else np.zeros((*m.shape, 3), np.float32)
        tm["halo"] = (time.perf_counter() - t2) * 1000

        t3 = time.perf_counter()
        out = compose(f, m if gain > 0 else np.zeros_like(m), halo, halo_gain=gain)
        tm["compose"] = (time.perf_counter() - t3) * 1000
        tm["total"] = (time.perf_counter() - t0) * 1000
        self.timings = tm
        return out

    def close(self) -> None:
        self.seg.close()
```

Note on the lost-head fade: the head is also faded out (mask zeroed once gain hits 0) so a stale frozen head does not linger forever; the test `gone == 0` checks the whole frame is black.

- [ ] **Step 4: Run tests, expect pass. Commit**

```bash
git add src/obs_lsd/pipeline.py tests/test_pipeline.py && git commit -m "Add Pipeline: segment, track, warp, halo, compose"
```

---

### Task 8: CLI, smoke test, and end-to-end run

**Files:**
- Modify: `src/obs_lsd/cli.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `obs-lsd` options exactly as the spec lists; exit codes 1 with messages for `CameraError`, `VirtualCamMissing`, `ModelsMissing`.

- [ ] **Step 1: Failing test — headless run on a still image writes a frame**

Append to `tests/test_cli.py`:
```python
import cv2
import numpy as np
from click.testing import CliRunner

from obs_lsd.cli import cli


def test_headless_run_dumps_frame(tmp_path, monkeypatch):
    # fake pipeline so no models are needed
    import obs_lsd.cli as climod

    class FakePipeline:
        timings = {"total": 1.0}
        frozen = False

        def __init__(self, cfg, segmenter=None):
            self.cfg = cfg

        def step(self, frame, t):
            out = np.zeros((self.cfg.height, self.cfg.width, 3), np.uint8)
            out[0, 0] = 255
            return out

        def close(self):
            pass

    monkeypatch.setattr(climod, "Pipeline", FakePipeline)
    src = tmp_path / "in.png"
    cv2.imwrite(str(src), np.zeros((20, 30, 3), np.uint8))
    dump = tmp_path / "out.png"
    r = CliRunner().invoke(
        cli,
        ["--input", str(src), "--seconds", "0.2", "--fps", "50", "--no-vcam", "--no-preview",
         "--dump-frame", str(dump), "--width", "64", "--height", "36"],
    )
    assert r.exit_code == 0, r.output
    img = cv2.imread(str(dump))
    assert img.shape == (36, 64, 3) and img[0, 0].max() == 255
```

- [ ] **Step 2: Run, expect failure (no such options)**

- [ ] **Step 3: Implement cli.py**

```python
"""Command-line interface for obs-lsd."""

from __future__ import annotations

import click

from obs_lsd.capture import CameraError, FrameSource, list_cameras
from obs_lsd.halo import HaloParams
from obs_lsd.pipeline import Pipeline, PipelineConfig


@click.command()
@click.version_option()
@click.option("--camera", default=None, help="Camera index or name substring (default 0)")
@click.option("--input", "input_", default=None, help="Video/image file instead of a camera")
@click.option("--list-cameras", "list_cams", is_flag=True, help="List cameras and exit")
@click.option("--width", default=1280, show_default=True)
@click.option("--height", default=720, show_default=True)
@click.option("--fps", default=30.0, show_default=True)
@click.option("--head-frac", default=0.55, show_default=True, help="Head height as fraction of frame")
@click.option("--no-vcam", is_flag=True, help="Do not open the virtual camera")
@click.option("--no-preview", is_flag=True, help="No preview window")
@click.option("--seconds", default=None, type=float, help="Stop after N seconds")
@click.option("--dump-frame", default=None, type=click.Path(), help="Write the last frame as PNG")
@click.option("--stats", is_flag=True, help="Print per-stage timings once a second")
@click.option("--halo-reach", default=HaloParams.reach, show_default=True)
@click.option("--halo-density", default=HaloParams.density, show_default=True)
@click.option("--halo-speed", default=HaloParams.speed, show_default=True)
@click.option("--halo-flicker", default=HaloParams.flicker, show_default=True)
@click.option("--halo-hue-speed", default=HaloParams.hue_speed, show_default=True)
@click.option("--halo-scale", default=HaloParams.scale, show_default=True)
def cli(camera, input_, list_cams, width, height, fps, head_frac, no_vcam, no_preview, seconds,
        dump_frame, stats, halo_reach, halo_density, halo_speed, halo_flicker, halo_hue_speed,
        halo_scale):
    """Glowing floating-head virtual camera."""
    if list_cams:
        for i, n in enumerate(list_cameras()):
            click.echo(f"{i}: {n}")
        return
    halo = HaloParams(reach=halo_reach, density=halo_density, speed=halo_speed,
                      flicker=halo_flicker, hue_speed=halo_hue_speed, scale=halo_scale)
    cfg = PipelineConfig(width=width, height=height, head_frac=head_frac, halo=halo)
    code = run(cfg, input_ if input_ else camera, fps, no_vcam, no_preview, seconds, dump_frame, stats)
    if code:
        raise SystemExit(code)


def run(cfg: PipelineConfig, source_spec, fps, no_vcam, no_preview, seconds, dump_frame, stats) -> int:
    try:
        src = FrameSource(source_spec, cfg.width, cfg.height, fps)
    except CameraError as e:
        click.echo(f"error: {e}", err=True)
        return 1
    try:
        pipe = Pipeline(cfg)
    except Exception as e:  # ModelsMissing and friends
        click.echo(f"error: {e}", err=True)
        src.close()
        return 1
    vcam = preview = None
    try:
        if not no_vcam:
            from obs_lsd.output import VirtualCam, VirtualCamMissing

            try:
                vcam = VirtualCam(cfg.width, cfg.height, fps)
                click.echo(f"virtual camera: {vcam.device}", err=True)
            except VirtualCamMissing as e:
                click.echo(f"error: {e}", err=True)
                return 1
        if not no_preview:
            from obs_lsd.output import Preview

            preview = Preview()
        last = None
        last_stat = 0.0
        for frame, t in src:
            out = pipe.step(frame, t)
            last = out
            if vcam:
                vcam.send(out)
            if preview:
                action = preview.show(out)
                if action == "quit":
                    break
                _apply_action(action, pipe)
            if stats and t - last_stat >= 1.0:
                last_stat = t
                click.echo(" ".join(f"{k}={v:.1f}ms" for k, v in pipe.timings.items()), err=True)
            if seconds is not None and t >= seconds:
                break
        if dump_frame and last is not None:
            import cv2

            cv2.imwrite(dump_frame, last)
        return 0
    finally:
        if preview:
            preview.close()
        if vcam:
            vcam.__exit__(None, None, None)
        pipe.close()
        src.close()


def _apply_action(action: str | None, pipe: Pipeline) -> None:
    if action is None:
        return
    p = pipe.halo.p
    if action == "size+":
        pipe.tracker.target_frac = min(0.95, pipe.tracker.target_frac + 0.05)
    elif action == "size-":
        pipe.tracker.target_frac = max(0.1, pipe.tracker.target_frac - 0.05)
    elif action == "hue+":
        p.hue_speed *= 1.25
    elif action == "hue-":
        p.hue_speed /= 1.25
    elif action == "reach+":
        p.reach *= 1.15
    elif action == "reach-":
        p.reach /= 1.15
    elif action == "freeze":
        pipe.frozen = not pipe.frozen
    elif action == "print":
        click.echo(f"head_frac={pipe.tracker.target_frac:.2f} {p}", err=True)
```

`HaloParams.reach` etc. work as click defaults because dataclass fields with defaults are class attributes. The test monkeypatches `obs_lsd.cli.Pipeline`, which works because `run` looks the name up in module globals at call time.

- [ ] **Step 4: Run all tests + smoke**

`make test && make lint && make smoke` — smoke must print `smoke OK` and `open /tmp/obs-lsd-smoke.png` must show a centred portrait head with rays on black.

- [ ] **Step 5: Live run with preview only**

`uv run obs-lsd --no-vcam --stats` — head centred, rays animate, `q` quits, `[`/`]` resize. Check `--stats` total < 33 ms. If over, lower `--halo-scale`. Fix anything that's visibly wrong (mask flicker -> raise `hold_frames`/feather; jitter -> lower Tracker alphas) with a test if the fix is in a pure module.

- [ ] **Step 6: Live run with virtual camera + Zoom**

`make run` then Zoom > Settings > Video > Camera: "OBS Virtual Camera". Confirm the glowing head appears. (Requires the OBS camera extension approved and camera permission granted for the terminal app.)

- [ ] **Step 7: Update README with usage/keys, commit**

```bash
git add -A && git commit -m "Add obs-lsd CLI, smoke target, README usage"
```

---

## Self-review notes

- Spec coverage: capture/segment/track/halo/compose/output/pipeline/cli all have tasks; error handling (camera, vcam, models, lost head, drop-frames-by-not-queuing) in Tasks 5-8; tests per spec section; perf budget checked in Task 3 step 4 and Task 8 step 5. `--list-cameras`, `--input`, `--seconds`, `--no-vcam`, `--dump-frame`, `--stats`, hotkeys all present.
- Deviation from spec, deliberate: noise comes from a precomputed tileable texture rather than opensimplex (measured too slow); `hue_twist`/`sharpness`/`rim_width` are in `HaloParams` but only the six most useful get CLI flags (YAGNI; hotkeys cover hue and reach).
- Type consistency: `HeadBox` lives in `track.py` and is used by `segment.py`/`pipeline.py`; `HeadMask.box: HeadBox`; `Tracker.update(HeadBox|None)`; `HaloRenderer.render(mask, t)`; `compose(frame, mask, halo, halo_gain)`.
