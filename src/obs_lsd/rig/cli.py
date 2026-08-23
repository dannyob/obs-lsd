"""obs-lsd-rig: evaluation and evolution commands."""

from __future__ import annotations

import json
from pathlib import Path

import click


@click.group()
def rig():
    """Replay, score and evolve obs-lsd."""


@rig.command()
@click.argument("clip", type=click.Path(exists=True))
@click.option("--out", "out_dir", required=True, type=click.Path())
@click.option("--width", default=1280, show_default=True)
@click.option("--height", default=720, show_default=True)
def render(clip, out_dir, width, height):
    """Replay a clip and write a run directory."""
    from obs_lsd.pipeline import PipelineConfig
    from obs_lsd.rig.render import render as _render

    res = _render(clip, Path(out_dir), cfg=PipelineConfig(width=width, height=height))
    click.echo(f"{res.run_dir}: {res.n_frames} frames, ms/frame {res.stats['ms_total_mean']:.1f}")


@rig.command()
@click.argument("run_dir", type=click.Path(exists=True))
@click.option("--samples", default=3, show_default=True)
@click.option("--judge-model", default=None)
def judge(run_dir, samples, judge_model):
    """Score a run directory with the vision judge."""
    from obs_lsd.rig.judge import DEFAULT_MODEL, OpenRouterJudge

    j = OpenRouterJudge(model=judge_model or DEFAULT_MODEL)
    s = j.score(Path(run_dir), samples=samples)
    click.echo(
        f"seamless={s.seamless:.0f} distracting={s.distracting:.0f} ornate={s.ornate:.0f} "
        f"total={s.total:.0f}/90  (${j.cost_usd:.3f})"
    )
    click.echo(f"notes: {s.notes}\ntop_fix: {s.top_fix}")


@rig.command()
@click.argument("run_a", type=click.Path(exists=True))
@click.argument("run_b", type=click.Path(exists=True))
@click.option("--votes", default=3, show_default=True)
@click.option("--judge-model", default=None)
def pairwise(run_a, run_b, votes, judge_model):
    """A/B two run directories."""
    from obs_lsd.rig.judge import DEFAULT_MODEL, OpenRouterJudge

    j = OpenRouterJudge(model=judge_model or DEFAULT_MODEL)
    res = j.pairwise(Path(run_a), Path(run_b), votes=votes)
    click.echo(json.dumps(res, indent=2))


@rig.command()
@click.argument("pairs_json", type=click.Path(exists=True))
@click.option(
    "--models",
    default="anthropic/claude-opus-5,google/gemini-3.1-pro-preview,openai/gpt-5.5",
    show_default=True,
)
def calibrate(pairs_json, models):
    """Compare judge models against human picks. pairs_json: [{"a":dir,"b":dir,"human":"a"|"b"}]"""
    from obs_lsd.rig.judge import OpenRouterJudge

    pairs = json.loads(Path(pairs_json).read_text())
    j = OpenRouterJudge()
    for m, agree in j.calibrate(pairs, models.split(",")).items():
        click.echo(f"{m}: agreement {agree:.0%}")
    click.echo(f"cost ${j.cost_usd:.2f}")


@rig.command()
@click.argument("clip", type=click.Path(exists=True))
@click.option("--judge-model", default=None)
@click.option("--samples", default=3, show_default=True)
def baseline(clip, judge_model, samples):
    """Render + score the current tree and save runs/baseline.json."""
    import subprocess

    from obs_lsd.pipeline import PipelineConfig
    from obs_lsd.rig.gates import write_baseline
    from obs_lsd.rig.judge import DEFAULT_MODEL, OpenRouterJudge
    from obs_lsd.rig.render import render as _render

    sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    run_dir = Path("runs") / f"baseline-{sha}"
    res = _render(clip, run_dir, cfg=PipelineConfig())
    j = OpenRouterJudge(model=judge_model or DEFAULT_MODEL)
    s = j.score(run_dir, samples=samples)
    write_baseline(run_dir, res.stats, s.to_json())
    click.echo(
        f"baseline {run_dir}: total {s.total:.0f}/90, ms/frame {res.stats['ms_total_mean']:.1f}  (${j.cost_usd:.3f})"
    )


@rig.command()
@click.argument("clip", type=click.Path(exists=True))
@click.option("--gens", default=1, show_default=True)
@click.option("--population", default=3, show_default=True)
@click.option("--judge-model", default=None)
@click.option("--dry-run", is_flag=True)
@click.option(
    "--start-gen", default=None, type=int, help="Generation number to start at (default: next)"
)
@click.option(
    "--agent-budget",
    default=5.0,
    show_default=True,
    help="Per-child spend cap handed to claude -p as --max-budget-usd",
)
def evolve(clip, gens, population, judge_model, dry_run, start_gen, agent_budget):
    """Run generations of the evolutionary loop."""
    from obs_lsd.rig.evolve import Evolution
    from obs_lsd.rig.judge import DEFAULT_MODEL, OpenRouterJudge

    ev = Evolution(
        repo=Path.cwd(),
        clip=clip,
        judge=None,
        dry_run=dry_run,
        population=population,
        max_budget_usd=agent_budget,
    )
    if not dry_run:
        ev.judge = OpenRouterJudge(model=judge_model or DEFAULT_MODEL)
    first = (
        start_gen
        if start_gen is not None
        else (max((r["generation"] for r in ev.ledger.records()), default=0) + 1)
    )
    for g in range(first, first + gens):
        ev.run_generation(g)


@rig.command()
def prune():
    """Remove worktrees of non-survivors (branches are kept)."""
    import subprocess

    from obs_lsd.rig.evolve import Evolution, select_survivors

    ev = Evolution(repo=Path.cwd(), clip="", dry_run=True)
    keep = {
        r["branch"].replace("/", "-") for r in select_survivors(ev.ledger.records(), ev.population)
    }
    for wt in (Path.cwd() / ".worktrees").glob("evo-*"):
        if wt.name not in keep:
            subprocess.run(["git", "worktree", "remove", "--force", str(wt)], check=False)
            click.echo(f"removed {wt.name}")
