"""obs-lsd-rig: evaluation and evolution commands."""

from __future__ import annotations

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
def judge():
    """Score a run directory with the vision judge."""
    raise click.ClickException("not implemented yet")


@rig.command()
def pairwise():
    """A/B two run directories."""
    raise click.ClickException("not implemented yet")


@rig.command()
def calibrate():
    """Compare judge models against human picks."""
    raise click.ClickException("not implemented yet")


@rig.command()
def baseline():
    """Render + score main and save runs/baseline.json."""
    raise click.ClickException("not implemented yet")


@rig.command()
def evolve():
    """Run generations of the evolutionary loop."""
    raise click.ClickException("not implemented yet")


@rig.command()
def prune():
    """Remove worktrees of non-survivors."""
    raise click.ClickException("not implemented yet")
