# obs-lsd

obs-lsd reads your webcam, cuts out your head at the neck, centers it on a
black background at a steady size, surrounds it with an animated rainbow
halo, and publishes the result as a virtual camera. Point Zoom (or OBS, as a
Video Capture Device source) at it instead of your real camera.

Requires Python 3.12 and macOS. Run `make install-dev models && make run` to
set up the environment, fetch the MediaPipe models, and start a live preview
with the virtual camera. OBS's virtual camera system extension provides the
video device this tool publishes to — install OBS and, once, launch it and
click Start Virtual Camera so macOS registers the device. macOS will also
prompt for camera access the first time obs-lsd runs; grant it in System
Settings > Privacy & Security > Camera.

## Usage

`obs-lsd` opens a camera (or a file given with `--input`), runs it through
the segment/track/halo pipeline, and sends the result to the OBS virtual
camera and a preview window. Options:

- `--camera TEXT` -- camera index or a substring of its name (default: 0,
  the first camera). Run with `--list-cameras` to see what's connected and
  exit.
- `--input PATH` -- read from a video or still-image file instead of a
  camera. A still is looped at `--fps`.
- `--width`, `--height`, `--fps` -- output frame size and rate (default
  1280x720 at 30fps).
- `--float` -- drift amplitude as a fraction of frame height (0 disables)
- `--head-frac` -- target head height as a fraction of frame height
  (default 0.55).
- `--no-vcam` -- skip opening the OBS virtual camera; useful when you only
  want the preview or a `--dump-frame`.
- `--no-preview` -- run without the preview window, for headless use.
- `--seconds N` -- stop after N seconds instead of running until quit or the
  input runs out.
- `--dump-frame PATH` -- write the last rendered frame to PATH as a PNG on
  exit.
- `--stats` -- print per-stage timings (segment/warp/halo/compose/total, in
  ms) to stderr once a second.
- `--halo-reach`, `--halo-density`, `--halo-speed`, `--halo-flicker`,
  `--halo-hue-speed`, `--halo-spill`, `--halo-scale` -- tune the halo; see `HaloParams` in
  `src/obs_lsd/halo.py` for what each does and its default.

With the preview window focused, these keys work live:

- `q` or Esc -- quit
- `]` / `[` -- grow / shrink the target head size
- `=` / `-` -- speed up / slow down the hue cycle
- `.` / `,` -- increase / decrease halo reach
- `s` -- freeze/unfreeze the head cutout (halo keeps animating)
- `p` -- print the current head-size fraction and halo params to stderr

`make smoke` runs the pipeline headless against `tests/data/portrait.jpg`
and writes `/tmp/obs-lsd-smoke.png`, so you can check the segmenter and halo
are working without a camera or a preview window.
