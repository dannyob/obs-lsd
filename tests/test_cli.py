from click.testing import CliRunner

from obs_lsd.cli import cli


def test_cli_version():
    result = CliRunner().invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output
