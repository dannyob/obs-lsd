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
- Your commit may only touch these paths (`ALLOWED_PATHS` in
  `src/obs_lsd/rig/evolve.py`): src/obs_lsd/halo.py, src/obs_lsd/compose.py,
  src/obs_lsd/segment.py, src/obs_lsd/track.py, src/obs_lsd/pipeline.py,
  src/obs_lsd/cli.py, tests/test_halo.py, tests/test_compose.py,
  tests/test_segment.py, tests/test_track.py, tests/test_pipeline.py,
  tests/test_cli.py, techniques.md. The rig checks
  `git diff --name-only <parent>..<yours>` after you commit and fails the whole
  round if anything else changed, so a stray file wastes the entire run.
- Keep `uv run pytest -q` and `uv run ruff check . && uv run ruff format --check .` green.
- Keep per-frame cost roughly flat: `make smoke` must still pass; do not add
  dependencies.
- Make ONE coherent change. Add or adjust tests for it.
- Append a line to techniques.md at the repo root: `- g{generation}: <what you did, one line>`.
- Commit everything with a short imperative subject and a body saying why.
- When done, print one line starting with `RESULT:` summarising the change.
- Do not commit brief.md or agent-output.json (they're gitignored scratch files).
