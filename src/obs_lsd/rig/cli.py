"""obs-lsd-rig: evaluation and evolution commands."""

from __future__ import annotations

import click


@click.group()
def rig():
    """Replay, score and evolve obs-lsd."""


@rig.command()
def render():
    """Replay a clip and write a run directory."""
    raise click.ClickException("not implemented yet")


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
