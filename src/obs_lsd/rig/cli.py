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
def evolve():
    """Run generations of the evolutionary loop."""
    raise click.ClickException("not implemented yet")


@rig.command()
def prune():
    """Remove worktrees of non-survivors."""
    raise click.ClickException("not implemented yet")
