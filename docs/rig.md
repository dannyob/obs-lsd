# The evaluation rig

`obs-lsd-rig` replays a recorded clip through the pipeline, scores the
result with a vision model, and can run an evolutionary loop in which coding
agents mutate the halo code on branches and the best survive. The README has
the short version; this is the rest.

## Commands

```
obs-lsd-rig render CLIP --out DIR      replay a clip and write a run directory
obs-lsd-rig judge DIR                  score a run directory with the vision judge
obs-lsd-rig pairwise DIR_A DIR_B       A/B two run directories
obs-lsd-rig baseline CLIP              render + score the current tree, write runs/baseline.json
obs-lsd-rig evolve CLIP --gens N       run N generations of the loop
obs-lsd-rig calibrate PAIRS_JSON       compare judge models against your own picks
obs-lsd-rig prune                      remove worktrees of non-survivors (branches are kept)
```

The Makefile wraps the common ones: `make baseline CLIP=clips/me-01.mov`,
`make eval CLIP=clips/me-01.mov` (render plus judge on whatever is checked
out, without touching the baseline), `make evolve CLIP=clips/me-01.mov
GENS=N`, and `make judge-smoke` for a single judge call against the latest
run directory.

## The judge

The judge is a vision-language model called over OpenRouter. It reads its
key with `pass show openrouter/api-key`. The default model is set in
`judge.py`; every command that judges takes `--judge-model` to swap it for
one call.

It scores against the rubric in `loop.md`: three criteria, 30 points each,
90 total.

- Seamless: the halo and the head should look like one object, with no
  visible dividing line.
- Not distracting: visible, but slow to change; no abrupt jumps.
- Ornate: detailed, closer to Byzantine iconography or a woodcut than to a
  stock lens flare.

Each score is the average of three judge calls (`--samples`), each about
$0.09, so a scored child costs around $0.30 on the OpenRouter side. The
judge sees three composite images built from the render: `sheet.png` (a
timestamped grid of frames), `crops.png` (centre crops at 25/50/75% of the
clip) and `motion.png` (a strip that shows how the halo changes frame to
frame), plus a summary of the metrics.

## Calibrating the judge

Before trusting small score differences between generations, check that the
judge's pairwise picks agree with yours. Render two variants you can tell
apart (`make eval` on each branch, or two `render` calls), look at their
`sheet.png`s, and write a JSON list of

```json
[{"a": "runs/evo-g3-1-7057f2d", "b": "runs/evo-g3-2-7c61768", "human": "a"}]
```

one object per pair you ranked yourself. Then:

```sh
obs-lsd-rig calibrate pairs.json --models anthropic/claude-opus-5,google/gemini-2.5-pro
```

runs a pairwise A/B on each pair for every model listed and prints, per
model, how often its pick matched yours.

## The evolutionary loop

`evolve` needs a baseline first. `make baseline CLIP=clips/me-01.mov`
renders the clip through the current tree, scores it, and writes
`runs/baseline.json` with the score and the commit sha. Children fork from
that sha, not from `HEAD`, so re-run `make baseline` after merging anything
into `main`; `evolve` refuses a stale baseline unless you pass
`--allow-stale-baseline`.

Each generation makes one or more `evo/gN-k` branches (`--population` sets
how many) in worktrees under `.worktrees/`, child k seeded from survivor k
of the previous generation. Inside each worktree a `claude -p
--dangerously-skip-permissions` agent reads a brief and is asked to try one
idea. The brief contains the rubric, the parent's score and the judge's
notes on it, the running `techniques.md` log of what has been tried, and the
operator: the kind of change the agent is asked to make (fix_top_complaint,
new_technique, tune, simplify, ornate, or crossover, which also pastes in a
sibling's diff). The operators are spelled out in
`src/obs_lsd/rig/prompts/operators.md`. The agent edits, commits, and exits;
`--agent-budget` (default $5) is passed to it as `--max-budget-usd`.

The rig then diffs the child's commit against its parent and discards the
child if it touched anything outside `ALLOWED_PATHS` in `evolve.py` (the
halo, compose, segment, track, pipeline and cli modules, their tests, and
`techniques.md`). Children that stayed in scope get tests and lint run, the
clip rendered with their own code, and the render checked against the gates:

- the head is still found on at least as many frames as in the baseline;
- ms/frame is within 15% of the parent's and 1.5x the baseline's;
- the count of abrupt halo changes is no more than twice the baseline's,
  with a floor of 5 so a very calm baseline does not fail everything.

Children that pass all three gates are judged. The generation's survivors
are the highest-scoring children, and the next generation forks from them.

Every attempt, gated out or not, gets a record appended to
`runs/ledger.json`: branch, parent, generation, operator, gate result and
reasons, judge score (seamless/distracting/ornate/total, notes and a top-fix
suggestion), the render's metrics summary, and both costs. The render lands
in `runs/<branch>-<sha>/` with the three composite images, the composited
`out.mp4`, `metrics.json` and `score.json`. The brief the agent was given
and its `total_cost_usd` JSON land in `runs/agents/<branch>/`, outside the
worktree so they cannot be committed by accident. `evolve` prints a results
table per generation with the judge cost and agent cost per child and
cumulative.

To promote a winner into `main`, merge its branch: `git merge evo/g3-1`.
`obs-lsd-rig prune` removes the worktrees of non-survivors and keeps the
branches.

## Bills

The judge is billed to your OpenRouter key, about $0.30 per scored child.
The agent's turns are `claude -p` and are billed to whatever authenticates
`claude` on your machine, your Claude plan or an API key, and run a few
dollars per child in practice. The generation table prints both.
`--agent-budget` caps the agent side; nothing caps the judge side except
`--gens` and `--population`.

## Confinement

Confinement is a convention plus the diff check, not a sandbox. Each
worktree lives inside the repo and shares its `.git`. The agent runs with
`--dangerously-skip-permissions` and inherits your environment minus the
OpenRouter key and Claude Code's own variables, so it can in principle reach
anything you can. Three streams of model-written text land verbatim in the
next agent's brief: the judge's notes, `techniques.md`, and the sibling
diffs pasted in for crossover. Any of them could carry instructions to the
next agent.

The driver never checks out `main`, and a winner reaches `main` only through
a `git merge` you type. That is acceptable for a local single-user tool.
Don't point it at a clip or a branch you don't trust.

## Design notes

The design spec and implementation plan that produced the rig are in
`docs/superpowers/specs/2026-08-22-eval-rig-evolution-design.md` and
`docs/superpowers/plans/2026-08-22-eval-rig-evolution.md`.
