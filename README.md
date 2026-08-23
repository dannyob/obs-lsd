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
- `--output PATH` -- write every composed frame to an mp4. With `--input
  clip.mov --no-vcam --no-preview` this replays a recorded clip flat out on
  the file's own clock, so the same clip gives the same video every time.
- `--stats` -- print per-stage timings (segment/warp/halo/compose/total, in
  ms) to stderr once a second.
- `--halo-reach`, `--halo-density`, `--halo-speed`, `--halo-flicker`,
  `--halo-hue-speed`, `--halo-spill`, `--halo-scale` -- tune the halo; see `HaloParams` in
  `src/obs_lsd/halo.py` for what each does and its default.

If Zoom lists "OBS Virtual Camera" but shows only black, check that OBS
itself is not running: only one program can feed the virtual camera at a
time, and an open OBS (with its Virtual Camera started) wins, sending its
own — usually empty — scene. Quit OBS, restart `obs-lsd`, and re-select the
camera in Zoom.

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

## Evaluation rig

`obs-lsd-rig` scores how the halo actually looks and can mutate the code
that draws it. It needs a recorded clip of a face to work from: open
QuickTime Player, File > New Movie Recording, pick your camera, record 20-30
seconds moving your head around under normal lighting, and save it under
`clips/` (`clips/danny-01.mov`, say). The directory is gitignored; it's
local test data.

`make baseline CLIP=clips/danny-01.mov` renders that clip through `main`,
scores the result with an LLM judge, and writes `runs/baseline.json`. Every
later generation is compared against this. `make eval CLIP=clips/x.mov`
does the same render-then-judge pass against whatever is currently checked
out, without touching the baseline. It's useful for scoring a branch by
hand before committing to a full evolution run.

`make evolve CLIP=clips/danny-01.mov GENS=N` runs the actual loop: each
generation spins up one or more `evo/gN-k` branches in their own worktrees
under `.worktrees/`, child k seeded from survivor k of the last generation.
Inside each worktree a `claude -p --dangerously-skip-permissions` agent
reads a brief (parent score, judge notes, the running `techniques.md` log)
and edits the halo/compose code to try one new idea, commits it, then the
rig diffs that commit against its parent and throws the child away if it
touched anything outside `ALLOWED_PATHS` in `evolve.py` (the halo/compose/
segment/track/pipeline/cli modules, their tests, and `techniques.md`).
Children that stayed in scope get their tests and lint run, the clip
rendered with their own code, and the render checked against the gates:
the head still found on as many frames as the baseline, ms/frame within
15% of the parent and 1.5x of the baseline, and no more halo-delta jumps
than twice the baseline's count (floor of 5). Whatever gates clean is
judged. `--agent-budget` (default $5) is passed to each agent as
`--max-budget-usd`.

Children fork from the sha in `runs/baseline.json`, not from `HEAD`, so
re-run `make baseline CLIP=…` after merging anything into `main` —
otherwise every child forks from the old commit. `evolve` refuses to start
when that sha is not `HEAD` unless you pass `--allow-stale-baseline`.

Confinement here is by convention plus that scope check, not by a sandbox.
Each worktree lives inside the repo and shares its `.git`, the agent runs
with `--dangerously-skip-permissions`, and it inherits your environment
minus `OPENROUTER_API_KEY` — so it can in principle reach anything you
can. What the code actually guarantees is narrower: the driver never
checks out `main`, and a champion reaches `main` only through a `git
merge` you type yourself. Three streams of model-written text also land
verbatim in the next agent's brief — the judge's notes, `techniques.md`,
and the sibling diffs pasted in for crossover — so the loop is a
prompt-injection path by construction. That's acceptable for a local
single-user tool; don't point it at a clip or a branch you don't trust.

Every attempt, whether it gates clean or not, gets a record appended to
`runs/ledger.json`: branch, parent, generation, operator, gate result and
reasons, judge score (seamless/distracting/ornate/total plus notes and a
top-fix suggestion), the render's metrics summary, and both costs (judge
and agent). The render itself lands in `runs/<branch>-<sha>/`, with
`sheet.png` (a timestamped grid of frames), `crops.png` (centre crops at
25/50/75% of the clip), `motion.png`, the composited `out.mp4`,
`metrics.json`, and `score.json`. The brief the agent was given and its
`total_cost_usd` JSON land in `runs/agents/<branch>/`, outside the
worktree so they can't be committed by accident.

To promote a generation's winner into `main`, just merge its branch:
`git merge evo/g3-1`. Nothing about the rig is special about that branch
once you do — it's ordinary code, and the merge commit is the record of
why it won.

`obs-lsd-rig calibrate PAIRS_JSON [--models a,b,c]` checks whether the
judge's pairwise picks agree with yours. Make `PAIRS_JSON` by hand: render
two variants you want to compare (`make eval` on each, or two `render`
calls), look at their `sheet.png`s, and write a JSON list of
`{"a": run_dir, "b": run_dir, "human": "a"|"b"}` objects, one per pair you
ranked yourself. `calibrate` then runs a pairwise A/B on each pair for
every model in `--models` and prints, per model, how often its pick
matched your `human` pick. Run it before trusting small score deltas
between generations. Every rig command that calls the judge also takes
`--judge-model` to swap the model for a single call (it defaults to a
fixed OpenRouter model id in `judge.py`).

The judge reads its OpenRouter key with `pass show openrouter/api-key`;
set that up first (`pass insert openrouter/api-key`). There are two
separate bills here. The judge is billed to the OpenRouter key: about
$0.09 per call, and a score samples 3 calls by default, so about $0.30 to
score one child. The coding agent's turns are `claude -p` and are billed
to whatever authenticates `claude` on your machine (your Claude plan or an
API key), not OpenRouter; that side runs a few dollars per child in
practice. The generation table prints both, per child and cumulative, and
`--agent-budget` caps the `claude -p` side.
