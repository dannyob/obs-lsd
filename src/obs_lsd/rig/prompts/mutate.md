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
- Keep `uv run pytest -q` and `uv run ruff check . && uv run ruff format --check .` green.
- Keep per-frame cost roughly flat: `make smoke` must still pass; do not add
  dependencies.
- Make ONE coherent change. Add or adjust tests for it.
- Append a line to techniques.md at the repo root: `- g{generation}: <what you did, one line>`.
- Commit everything with a short imperative subject and a body saying why.
- When done, print one line starting with `RESULT:` summarising the change.
