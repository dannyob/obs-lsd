from click.testing import CliRunner

from obs_lsd.rig import load_prompt
from obs_lsd.rig.cli import rig


def test_rig_group_lists_commands():
    r = CliRunner().invoke(rig, ["--help"])
    assert r.exit_code == 0
    for cmd in ("render", "judge", "pairwise", "calibrate", "baseline", "evolve", "prune"):
        assert cmd in r.output


def test_prompts_are_packaged():
    rubric = load_prompt("rubric.md")
    assert "Seamless" in rubric and "Ornate" in rubric and "30 points" in rubric
    assert "{operator}" in load_prompt("mutate.md")
