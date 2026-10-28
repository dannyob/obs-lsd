# obs-lsd

A macOS command-line tool that reads a webcam, isolates the user's head at
the neck, centers and stabilizes it on a black frame, wraps it in an
animated psychedelic halo, and publishes the result as a virtual camera for
Zoom and OBS.

## Modules (src/obs_lsd/)

- `capture.py` — camera / video-file frame source
- `segment.py` — head mask + chin line via MediaPipe
- `track.py` — smoothing + camera-to-output transform
- `halo.py` — ray renderer (numpy backend)
- `compose.py` — black + halo + head compositing
- `output.py` — pyvirtualcam sink, Syphon sink (optional `syphon` extra), cv2 preview, key handling
- `pipeline.py` — wires the above modules together
- `cli.py` — click entry point (`obs-lsd`)

### Evaluation rig (src/obs_lsd/rig/)

- `render.py` — runs the pipeline headless over a clip, writes a run dir
  (`sheet.png`, `crops.png`, `motion.png`, `out.mp4`, `metrics.json`)
- `metrics.py` — per-frame stats (head-loss fraction, jump/jitter, timings)
  that feed the gates and the judge's notes
- `images.py` — builds the sheet/crops/motion composite images from a run
- `judge.py` — OpenRouter LLM judge: rubric scoring, pairwise A/B, cost
  tracking; reads the API key with `pass show openrouter/api-key`
- `gates.py` — pass/fail checks (head still detected, ms/frame against
  parent and baseline, halo-delta jump count, tests/lint clean) with
  human-readable reasons
- `evolve.py` — the evolutionary loop driver: worktrees, agent briefs,
  mutation operators, crossover, `runs/ledger.json`, survivor selection,
  the results table, `prune`
- `cli.py` — click entry point (`obs-lsd-rig`: render, judge, pairwise,
  calibrate, baseline, evolve, prune)

## Make targets

`venv install-dev test lint format models smoke run baseline eval evolve
judge-smoke bench clean`

`make bench` rsyncs `src/` and a 10s clip to another Mac (`BENCH_HOST`,
default pequod) and times a render there; the per-round history is in
`docs/perf-log.md`. Speed only means anything measured on the slower
machine: the stage proportions differ from the M5 Max.

## Working here

TDD: write a failing test, then the code, then run it. Run `make smoke`
before claiming a change works end to end — passing unit tests alone is not
enough. mediapipe must stay `<1.0` on macOS; 1.x crashes.

Speed notes. MediaPipe runs on the Metal GPU delegate (`segment.py`); the
frame must be handed over as SRGBA, since SRGB makes the delegate abort the
process with a CHECK failure, not an exception. The segmenter and the halo
each run on a worker thread one frame behind the loop (`seg_lag`,
`halo_lag` in `PipelineConfig`; `--no-seg-thread`, `--no-halo-thread` on
the CLI). Each step uses the previous frame's mask and halo, so a replay is
still deterministic. The pure-numpy stages were rewritten as in-place cv2
ops; keep it that way, a full-frame float temporary costs about 1ms on an
M2 Max.

Children fork from the sha recorded in `runs/baseline.json`, not from
`HEAD`, so re-run `make baseline CLIP=…` after merging anything into
`main`; `evolve` refuses a stale baseline unless you pass
`--allow-stale-baseline`. Agents spawned by `evolve` are meant to stay in
their own `evo/*` worktree, and a child whose commit touches anything
outside `ALLOWED_PATHS` (`evolve.py`) is discarded — but that is a
convention plus a diff check, not a sandbox: the worktree shares the
repo's `.git` and the agent runs with `--dangerously-skip-permissions`.
What holds is that the driver never checks out `main`, and promotion is a
`git merge` a human types. Judge notes, `techniques.md` and sibling diffs
are model-written text pasted into the next agent's brief, so treat the
loop as a prompt-injection path and don't aim it at anything untrusted.
`runs/` and `clips/` are local and gitignored, not shipped with the repo.

OBS cannot capture its own virtual camera, so `--syphon` (`SyphonSink` in
`output.py`) is how obs-lsd gets into an OBS scene. The sink pumps the
Cocoa run loop every frame; without that the server is never announced
and no client lists it.
