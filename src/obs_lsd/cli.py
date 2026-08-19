"""Command-line interface for obs-lsd."""

import click


@click.command()
@click.version_option()
def cli():
    """Glowing floating-head virtual camera."""
    click.echo("obs-lsd: pipeline not wired yet")


if __name__ == "__main__":
    cli()
