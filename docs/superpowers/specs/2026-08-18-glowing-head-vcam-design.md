# obs-lsd: glowing-head virtual camera — design

Date: 2026-08-18
Status: approved in brainstorm, awaiting implementation plan

## Goal

A macOS command-line program that reads a webcam, isolates the user's head
(cut at the neck), centers it at a constant size on a black frame, surrounds
it with an animated psychedelic rainbow halo of wispy rays, and publishes the
result as a virtual camera that Zoom (first) and OBS (later, as a Video
Capture Device source) can pick up.

## Decisions made

| Question | Decision |
|---|---|
| Zoom and OBS simultaneously? | No, one at a time. Zoom first. |
| Background | Solid black. |
| How much of the person | Head only, cut around the neck/chin. Floating head. |
| Framing | Head centered, scaled to a constant fraction of frame height; movement smoothed. |
| Halo style | Wispy noise-driven radial rays that flicker and rotate, rainbow hue cycling. |
| Stack | Python 3.12, MediaPipe, OpenCV, numpy, pyvirtualcam (OBS Virtual Camera extension). |
| Platform | macOS (Apple Silicon) to start. |
| Licence | AGPL-3.0-or-later. |

## Non-goals (for now)

- Transparent output, GPU shader halo, other halo styles, a GUI, Linux/Windows,
  Syphon/NDI output, multiple heads. The halo module boundary is chosen so a
  shader backend can be added later without touching the rest.

## Project layout

uv-managed package based on `~/Public/src/dob/py-template/template`
(click CLI, uv_build, pytest, Makefile).

```
obs-lsd/
  pyproject.toml          name obs-lsd, package obs_lsd, requires-python ==3.12.*
  Makefile                run, test, lint, smoke
  src/obs_lsd/
    cli.py                click entry point `obs-lsd`
    capture.py            camera / video-file frame source
    segment.py            head mask + chin line via MediaPipe
    track.py              smoothing + camera->output transform
    halo.py               ray renderer (numpy backend)
    compose.py            black + halo + head compositing
    output.py             pyvirtualcam sink, cv2 preview, key handling
    pipeline.py           wires the above; one `step(frame, t)` function
    models/               downloaded MediaPipe .task files (gitignored, fetched by Makefile)
  tests/
  docs/superpowers/specs/
```

Dependencies: `mediapipe`, `opencv-python`, `numpy`, `pyvirtualcam`, `click`,
`opensimplex` (or vectorised own noise if it is too slow). Dev: `pytest`,
`ruff`.

Homebrew: OBS (installed) provides the virtual camera system extension. The
user must launch OBS once and Start/Stop Virtual Camera to register it.

## Components

Each module exposes plain functions/classes operating on numpy arrays so it
can be tested on still images without a camera.

### capture.py

`FrameSource(spec, width, height, fps)` — `spec` is a camera index, a
substring of a camera name, or a path to a video file. Iterates BGR uint8
frames. `list_cameras()` returns available devices (via
`system_profiler SPCameraDataType` names mapped to OpenCV indices by probing)
for error messages and `--list-cameras`.

### segment.py

`HeadSegmenter.mask(frame_bgr) -> HeadMask | None`

- MediaPipe Image Segmenter, multiclass selfie model (classes: background,
  hair, body-skin, face-skin, clothes, others). Head classes = hair ∪
  face-skin.
- MediaPipe Face Landmarker gives the chin point and jawline. The head mask
  is intersected with the half-plane above a line through the chin (offset a
  little below the chin, tilted with the face roll) so the neck is dropped.
- Mask is float32 in [0,1], lightly blurred at the edge (feathering).
- `HeadMask` carries: `mask` (HxW float32), `bbox` (x0,y0,x1,y1), `center`
  (cx,cy of the mask), `height` (bbox height).
- Returns None when no face is found. Holds the last mask for up to
  N (default 10) frames of missing face before returning None.

### track.py

`Tracker(target_frac, alpha_pos, alpha_scale)`

- `update(head: HeadMask | None, out_size) -> Transform`
- Exponential smoothing on center and height. `Transform` is a 2x3 affine
  (scale + translate) mapping camera pixels to output pixels such that the
  smoothed head center lands at the output center and the smoothed head
  height equals `target_frac * out_height` (default 0.55).
- On None input, keep returning the last transform; expose `frames_missing`
  so compose can fade the halo.
- Pure math; fully unit-testable.

### halo.py

`HaloRenderer(params).render(mask_out: HxW float32, t: float) -> HxWx3 float32 (RGB, 0..1)`

Algorithm (numpy):
1. Distance transform of `1 - mask` gives `r` = distance from head contour.
2. Angle map `theta` from the mask centroid.
3. Ray field: `n = simplex3(theta*density, r*stretch - t*speed, t*flicker)`;
   intensity `I = clamp(n)^sharpness * falloff(r, reach)`.
4. Hue = `(theta/2pi + t*hue_speed + r*hue_twist) mod 1`, saturation 1,
   value = I; convert HSV->RGB.
5. Small bright rim: `exp(-r/rim_width)` in white/pale rainbow added.

Params (all CLI-adjustable): `reach` px, `density`, `speed`, `flicker`,
`hue_speed`, `hue_twist`, `sharpness`, `rim_width`. Rendered at
`halo_scale` (default 0.5) resolution and upscaled with cv2 to hit 30 fps.
Deterministic for a given (mask, t) — testable.

### compose.py

`compose(frame_out_bgr, mask_out, halo_rgb, halo_gain) -> HxWx3 uint8 BGR`

`black + halo * halo_gain * (1 - mask)` then head pixels blended in by mask.
`halo_gain` is used by the pipeline to fade the halo over ~1 s when the head
is lost.

### output.py

- `VirtualCam(width, height, fps)` — pyvirtualcam sink; raises a friendly
  error if the OBS extension is not present.
- `Preview` — cv2 window; polls keys and returns actions. Keys:
  `q` quit, `[` `]` head size, `-` `=` hue speed, `,` `.` ray reach,
  `s` freeze/unfreeze mask (hold last mask), `p` print current params.

### pipeline.py

`Pipeline(config)` with `step(frame_bgr, t) -> out_bgr`:
segment → track → `cv2.warpAffine` frame and mask into output space → halo →
compose. Also holds the halo fade state. The CLI's main loop is:
capture → step → vcam.send + preview.show. Timing is logged (`--stats`).

### cli.py

```
obs-lsd [--camera SPEC] [--width 1280 --height 720 --fps 30]
        [--head-frac 0.55] [--no-vcam] [--no-preview] [--seconds N]
        [--input FILE.mp4] [--dump-frame out.png] [--stats]
        [--halo-reach ... --halo-speed ... etc]
obs-lsd --list-cameras
```

`--input`, `--seconds`, `--no-vcam`, `--dump-frame` exist so the whole
pipeline can be run end to end headlessly (`make smoke`).

## Error handling

- Camera missing/in use: exit 1 with the list of cameras.
- Virtual camera missing: exit 1 with instructions to launch OBS and
  start/stop Virtual Camera once.
- Segmenter models missing: exit 1 with `make models` hint (Makefile
  downloads them from Google's model URLs into `src/obs_lsd/models/`).
- No head detected: hold last transform, fade halo out, then black frame.
- Slow frames: never queue up; drop and carry on. `--stats` shows per-stage ms.

## Testing

- `tests/test_track.py`: smoothing converges, transform puts a synthetic
  bbox at the center with the requested height, None input holds.
- `tests/test_halo.py`: output shape/dtype/range, deterministic for same
  (mask,t), changes with t, zero inside the mask region.
- `tests/test_compose.py`: golden-image comparison on a synthetic ellipse
  head at fixed t.
- `tests/test_segment.py`: on a bundled test photo (a public-domain face
  image checked into `tests/data/`), mask covers the face bbox and is
  empty well below the chin. Skipped if models absent.
- `make smoke`: `obs-lsd --input tests/data/clip.mp4 --seconds 2 --no-vcam
  --no-preview --dump-frame /tmp/out.png` must exit 0 and produce a non-black
  PNG. This runs the real built package, not just units.
- Manual acceptance: Zoom → Settings → Video → "OBS Virtual Camera" shows the
  glowing head at ~30 fps.

## Performance budget

720p30 → 33 ms/frame. Expected: segmentation ~8-12 ms, landmarks ~5 ms,
warp ~1 ms, halo at half res ~8-12 ms, compose ~2 ms. If over budget, first
lower `halo_scale`, then run segmentation on a downscaled frame, then move
halo to moderngl.
