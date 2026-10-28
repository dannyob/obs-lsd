# Speed on pequod: design

## Intent

obs-lsd renders a frame in about 161ms on pequod (Apple M2 Max, macOS
26.6), which is 6 fps, against 42ms on the M5 Max it was developed on.
The goal is 60 fps on pequod, 16.7ms a frame, without the halo looking
materially worse.

Measured on 2026-10-28 with the still portrait at 1280x720:

```
pequod:  segment=137.7ms warp=0.9ms halo=14.7ms compose=7.7ms total=161.0ms
M5 Max:  segment=27.0ms  warp=0.3ms halo=8.6ms  compose=4.6ms total=40.5ms
```

The segmenter is 85% of the frame on pequod. The test clip is 1620x1080,
and the whole frame goes to MediaPipe, which returns six confidence masks
at that resolution; that is most of the cost.

## Goals, non-goals, anti-goals

- Goal: ms/frame on pequod as low as it will go, target 16.7ms, on the
  recorded clip and in live use.
- Goal: every kept change has tests, passes lint, and keeps `make smoke`
  working.
- Non-goal: a prettier halo. The aesthetic score only has to hold.
- Non-goal: GPU rendering of the halo or a rewrite in another language.
  Stay in Python, numpy, OpenCV and MediaPipe.
- Anti-goal: speed bought by dropping the head cutout more often, or by a
  halo that the judge scores more than five points below the baseline.

## Constraints

- pequod has uv, Python and the models, but no `claude` and no `pass`.
  Timing runs there; the judge and the editing run on the M5 Max.
- Timing is only meaningful on pequod. The stage proportions differ
  between the two machines, so an M5 number is not a proxy.
- MediaPipe stays below 1.0 on macOS.

## Approaches considered

1. Run the rounds in this session by hand, with a benchmark target that
   syncs the tree to pequod and times a render there. Chosen: it is one
   lineage and one branch, and the rig's population machinery buys
   nothing for hill climbing.
2. A `obs-lsd-rig optimize` driver running `claude -p` rounds unattended
   on a `perf/pequod` branch. Rejected for now as more rig code than the
   job needs; the by-hand rounds can become its brief later.
3. Reuse `evolve` with a speed fitness. Rejected: populations, worktrees
   and crossover are for exploring an aesthetic space, not for
   optimizing one implementation.

## How a round works

1. Work on branch `perf/pequod`, forked from `main`.
2. `make bench` rsyncs `src/obs_lsd` (not the models) and the bench clip
   to pequod, renders the clip there with `obs-lsd-rig render`, and
   prints the per-stage mean ms and the no-head fraction from the
   resulting `metrics.json`.
3. Pick one change, write its test, make it, run tests and lint, bench.
4. Keep the commit if ms/frame fell and the no-head fraction did not rise.
   Otherwise revert it and note what was tried in `docs/perf-log.md`.
5. After each kept change that touches the look (segmenter input size,
   mask reuse, halo or compose maths), render the full clip on the M5 and
   judge it. Score must be at least baseline minus 5. If not, revert.
6. Stop at 16.7ms a frame on the bench clip, or after 60 rounds, or
   when the remaining ideas are exhausted. Report where it landed.

Baseline judge score for `main` is 39.5 in `runs/baseline.json`
(rendered at 451c852, same pipeline code as `main` today).

## Candidate changes, in expected order of payoff

- Downscale the frame before the face landmarker and segmenter, and
  upscale the mask. The models are 256x256 and 192x192 inside.
- Reuse the segmentation mask across N frames (the uncommitted
  `seg_stride` experiment on pequod from 2026-08-27, brought over as a
  patch), with the landmarker still every frame.
- Run the segmenter on its own thread, so the frame loop composes with
  the latest mask rather than waiting.
- Halo: lower `halo_scale`, cache what does not change per frame.
- Compose: avoid float temporaries, work in uint8 where the maths allows.

## Left to the builder

The order above is a guess; the bench decides. Any change that alters
defaults visible to users (a new `--seg-stride` flag, a different halo
scale) is documented in the README options list.

## Outcome (2026-10-28)

Ten rounds, eight kept, on branch `perf/pequod`; the per-round numbers
are in `docs/perf-log.md`. The bench clip went from 163ms a frame (6 fps)
to 12ms (83 fps) on pequod, and the paced live path with the virtual
camera on runs at 12 to 15ms a frame, so 60 fps holds. Round 10, a halo
worker one frame behind, reached 4.9ms but the judge preferred the inline
halo 5-0 (a dark band where the cutout had moved off the stale halo's
black interior), so it is an opt-in flag, off by default.

What mattered, in order:

1. The Metal GPU delegate: the multiclass segmenter is 120ms on the M2
   Max CPU and 6ms on its GPU. The image must be SRGBA; SRGB aborts the
   process. 163 to 49ms.
2. Compose in uint8 with `cv2.blendLinear`: 9.6 to 3.4ms.
3. A worker thread one frame behind the loop for the segmenter. MediaPipe
   and OpenCV release the GIL, so it overlaps with the loop almost
   completely. 29.5 to 14.9ms.
4. Replacing numpy temporaries with in-place cv2 ops in the segmenter's
   post-processing and the halo, each change checked bit-identical
   against a saved render. About 12ms in total.

Rejected: shrinking the frame before inference with `INTER_AREA` (round
2) cost more than it saved; the same change with a linear resize (round
5b) saved 2.4ms. The `seg_stride` experiment from pequod's checkout was
not needed; it is still in pequod's stash.

The judge scored the full clip 37, 36, 38 after rounds 3, 5 and 6
against the baseline's 39.5, all inside the five-point gate, and a
pairwise A/B of round 6 against the baseline split 2 to 1. The A/B was
the instrument that caught round 10; absolute scores did not.
