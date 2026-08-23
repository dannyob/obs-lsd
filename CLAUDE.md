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
- `output.py` — pyvirtualcam sink, cv2 preview, key handling
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
- `gates.py` — pass/fail checks (head detected, no scale blowups, timing
  budget, tests/lint clean) with human-readable reasons
- `evolve.py` — the evolutionary loop driver: worktrees, agent briefs,
  mutation operators, crossover, `runs/ledger.json`, survivor selection,
  the results table, `prune`
- `cli.py` — click entry point (`obs-lsd-rig`: render, judge, pairwise,
  calibrate, baseline, evolve, prune)

## Make targets

`venv install-dev test lint format models smoke run baseline eval evolve
judge-smoke clean`

## Working here

TDD: write a failing test, then the code, then run it. Run `make smoke`
before claiming a change works end to end — passing unit tests alone is not
enough. mediapipe must stay `<1.0` on macOS; 1.x crashes.

Never run `evolve` on a dirty `main` — the driver takes `HEAD` as the
parent sha for the first generation, so uncommitted changes there leak
into every child. Agents spawned by `evolve` only ever touch their own
`evo/*` worktree under `.worktrees/`; they never edit the controller or
`main` directly — promotion is a normal `git merge` done by a human.
`runs/` and `clips/` are local and gitignored, not shipped with the repo.
