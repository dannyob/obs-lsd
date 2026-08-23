import subprocess

from obs_lsd.rig import gates
from obs_lsd.rig.gates import gate

BASE = {"ms_total_mean": 30.0, "no_head_frac": 0.0, "jump_count": 4}


def test_gate_passes_when_within_limits():
    ok, why = gate({"ms_total_mean": 34.0, "no_head_frac": 0.005, "jump_count": 8}, BASE)
    assert ok and why == []


def test_gate_reports_every_violation():
    ok, why = gate({"ms_total_mean": 40.0, "no_head_frac": 0.05, "jump_count": 9}, BASE)
    assert not ok and len(why) == 3
    assert (
        any("ms/frame" in w for w in why)
        and any("no-head" in w for w in why)
        and any("jump" in w for w in why)
    )


def test_gate_fails_on_checks():
    ok, why = gate(BASE, BASE, checks_ok=False)
    assert not ok and why == ["tests/lint failed"]


def test_run_checks_timeout(monkeypatch, tmp_path):
    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="bash", timeout=1)

    monkeypatch.setattr(gates.subprocess, "run", fake_run)
    ok, tail = gates.run_checks(tmp_path, timeout_s=1)
    assert ok is False and tail == "tests timed out"
