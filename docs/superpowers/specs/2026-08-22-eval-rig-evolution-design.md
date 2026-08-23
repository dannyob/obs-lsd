# obs-lsd evaluation rig and evolutionary self-improvement loop — design

Date: 2026-08-22
Status: approved in brainstorm, awaiting implementation plan
Builds on: docs/superpowers/specs/2026-08-18-glowing-head-vcam-design.md

## Goal

Replay a recorded clip through obs-lsd, score the result automatically against
Danny's aesthetic rubric (`loop.md`) plus computed stability metrics, and drive
an evolutionary loop in which coding agents propose changes to the halo/compose
code on branches, the rig gates and scores them, and the best survive.

## Decisions made

| Question | Decision |
|---|---|
| Judge | Vision-language model via OpenRouter; key from `pass show openrouter/api-key`; default `anthropic/claude-opus-5`, `--judge-model` flag; calibration mode to compare judges against Danny's picks. |
| Scoring | Absolute rubric score (3 samples, median per category, /90) for ranking; pairwise A/B only for champion-vs-baseline. Category 2 gets computed motion metrics in the prompt. |
| What the loop changes | Code (halo/compose/segment/track) via `claude -p` on branches; params implicitly. |
| Search shape | Population 3, one child per parent per generation, crossover child every second generation, survivors = top 3 of parents ∪ children. |
| Gates | tests+ruff green; ms/frame ≤ 1.15× baseline; no-head ≤ baseline+1%; jump count ≤ 2× baseline. |
| Isolation | Agents run only in `evo/*` worktrees; main untouched; human merges champions. |

Already shipped (commit 790cde7): `--input clip` replays on the file clock,
`--output out.mp4` writes every frame, headless runs flat out, `clips/` gitignored.

## Layout

```
src/obs_lsd/rig/
  __init__.py
  render.py     clip -> runs/<id>/{out.mp4, metrics.json, sheet.png, crops.png, motion.png}
  metrics.py    per-frame numbers from the pipeline run (pure functions, testable)
  images.py     contact sheet / crops / motion strip builders (pure, testable)
  judge.py      OpenRouter client: score(run_dir) -> score.json; pairwise(a,b); calibrate(pairs)
  gates.py      gate(candidate_metrics, baseline_metrics) -> (ok, reasons)
  evolve.py     GA driver: worktrees, claude -p, render+gate+judge, ledger, selection
  prompts/
    rubric.md   copy of loop.md (the scoring rubric, verbatim)
    judge.md    judge instructions + JSON schema
    mutate.md   coding-agent brief template with {operator}, {parent_notes}, {techniques}
    operators.md the five operators + crossover text
  cli.py        click group `obs-lsd-rig`: render, judge, pairwise, calibrate, evolve, baseline
runs/            gitignored; runs/<id>/..., runs/ledger.json, runs/baseline.json
clips/           gitignored
```

`<id>` = `<branch-or-label>-<short sha>` (e.g. `main-23d5b99`, `evo-g2-1-a1b2c3d`).

## Components

### render.py — `render(clip, out_dir, cfg_args) -> RunResult`
Runs `Pipeline` over the clip (via the same code path as the CLI: FrameSource
realtime=False, file clock), writes `out.mp4`, collects per-frame data:
`HeadBox` (or None), `pipe.timings`, the warped mask and the halo region delta
(mean |frame_t - frame_{t-1}| over pixels outside the head mask, uint8 scale /255).
Calls metrics + images, writes `metrics.json`. Also callable as
`obs-lsd-rig render CLIP --out runs/<id> [--width 1280 --height 720] [pipeline flags...]`.
Default output size 1280x720 (so ms/frame is comparable to the live target
regardless of clip resolution).

### metrics.py — pure
```
frame_stats(head_boxes, timings, deltas) -> dict:
  frames, no_head, no_head_frac,
  delta_median, delta_p95, jump_count (delta > 3*delta_median), jump_frac,
  centre_jump_p95 (px, raw HeadBox cx,cy frame-to-frame), height_std,
  ms_total_mean, ms_segment_mean, ms_halo_mean, ms_compose_mean
```

### images.py — pure
- `contact_sheet(frames: list[bgr], cols=4) -> bgr` (8 frames sampled evenly from output, 480 px wide each, time label)
- `crops(frames, size=560)` -> 3 centre crops at 25/50/75 % of the clip, hstacked
- `motion_strip(frames_at_100ms, n=8)` -> hstack of 8 consecutive output frames 100 ms apart from the middle of the clip, each 320 px wide
All written as PNG into the run dir.

### judge.py
- `OpenRouterJudge(model, api_key)`; `score(run_dir, samples=3) -> Score` where
  `Score = {seamless, distracting, ornate, total, notes, top_fix, samples:[...], model}`;
  medians per category, total = sum of medians. Images sent as base64 data URLs
  (sheet, crops, motion), metrics pasted as a table, rubric verbatim, response
  forced to JSON (instruction + parse with fallback regex; one retry on parse
  failure).
- `pairwise(run_a, run_b, votes=3) -> {"prefer": "a"|"b"|"tie", "votes": [...]}`;
  sides randomised per vote; same images.
- `calibrate(pairs_json, models=[...])`: pairs_json is a list of
  `{"a": run_dir, "b": run_dir, "human": "a"|"b"}`; prints agreement % per model.
- API key: `OPENROUTER_API_KEY` env if set, else `pass show openrouter/api-key`.
  Never printed. Cost: usage tokens from the response are summed and printed.
- HTTP via `httpx` (dependency, rig extra); transport injectable for tests.

### gates.py — pure
`gate(cand, base) -> (ok: bool, reasons: list[str])` with the four rules above;
constants at module top.

### evolve.py
State in `runs/ledger.json`: list of records
`{id, branch, parent, generation, operator, sha, gates: {ok, reasons}, score: Score|null, metrics_summary, cost_usd}`.
`runs/baseline.json` = the main run's metrics + score (made by `obs-lsd-rig baseline CLIP`).

One generation:
1. survivors = top-3 scored records (initially just baseline on `main`).
2. for each survivor k: operator = OPERATORS[(gen + k) % 5]; every second
   generation the third child is `crossover` of the two best instead.
3. child branch `evo/g{gen}-{k}` from survivor's sha; `git worktree add .worktrees/<branch>`.
4. write brief from `mutate.md` (rubric, parent score+notes+top_fix, lineage
   `techniques.md`, operator text, hard rules: tests+ruff green, no new deps
   without saying so, commit with message, append to techniques.md).
5. `claude -p "$(cat brief)" --max-turns 40 --dangerously-skip-permissions` run
   with cwd = worktree; timeout 20 min. Capture transcript to run dir.
6. if no new commit → ledger "no change", discard. Else render → gate → score
   (skip score if gate fails) → ledger.
7. selection; print table (id, operator, gates, seamless/distracting/ornate/total,
   ms/frame, sheet path); print cumulative API cost.
`obs-lsd-rig evolve CLIP --gens N [--population 3] [--judge-model M] [--dry-run]`
`--dry-run` builds briefs and skips claude/judge (for tests and sanity).

Worktrees live in `.worktrees/` (gitignored). Dead branches are kept (cheap, auditable);
`obs-lsd-rig prune` removes worktrees of non-survivors.

### Makefile
`baseline CLIP=…`, `eval CLIP=… [REF=…]`, `judge RUN=…`, `evolve CLIP=… GENS=…`, `judge-smoke`.

## Error handling
- Missing key → clear message naming the pass path. HTTP/parse failure → one
  retry, then the sample is dropped; a score with <2 samples is invalid and the
  child is ledgered "judge failed".
- claude -p non-zero or timeout → ledgered "agent failed"; loop continues.
- Gate failure reasons are human-readable strings in the ledger and table.
- The loop never touches `main`; Ctrl-C between children leaves the ledger consistent.

## Testing
- metrics: synthetic sequences with known jumps → counts exact.
- images: shapes/labels; deterministic.
- judge: fake transport returning canned JSON; median logic; side randomisation
  with seeded RNG; parse fallback; key resolution order.
- gates: table-driven.
- evolve: `--dry-run` over a tmp git repo with two commits produces briefs,
  branches, worktrees, ledger entries; selection function tested on synthetic ledgers.
- end-to-end: `make baseline CLIP=clips/danny-01.mov` (real render, real judge, one call).

## Budget
Per child ≈ 3 judge samples ≈ $0.15 on opus-5 (+ agent tokens on Danny's Claude
plan). A generation ≈ $0.5; 8 generations ≈ $4 of OpenRouter spend.

## Non-goals (now)
Multiple clips per eval (design allows a list later), a web dashboard, GPU
rendering, automatic merging into main.
