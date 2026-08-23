import subprocess

import pytest

from obs_lsd.rig import gates
from obs_lsd.rig.gates import gate

BASE = {"ms_total_mean": 30.0, "no_head_frac": 0.0, "jump_count": 4, "delta_median": 0.01}


def cand(**over):
    return {"ms_total_mean": 30.0, "no_head_frac": 0.0, "jump_count": 0, **over}


# (name, candidate, parent, expected ok, substring expected in the reasons)
CASES = [
    ("within limits", cand(ms_total_mean=34.0, no_head_frac=0.005, jump_count=8), None, True, ""),
    ("slower than parent", cand(ms_total_mean=40.0), None, False, "ms/frame"),
    ("loses the head", cand(no_head_frac=0.05), None, False, "no-head"),
    ("too many jumps", cand(jump_count=9), None, False, "jump count"),
    # a parent that already spent its budget lets the child spend 15% more...
    ("within the parent's budget", cand(ms_total_mean=45.0), {"ms_total_mean": 40.0}, True, ""),
    (
        "over the parent's budget",
        cand(ms_total_mean=47.0),
        {"ms_total_mean": 40.0},
        False,
        "parent",
    ),
    # ...but never past 1.5x the baseline, however far the lineage has crept
    (
        "over the absolute ceiling",
        cand(ms_total_mean=46.0),
        {"ms_total_mean": 44.0},
        False,
        "baseline",
    ),
]


@pytest.mark.parametrize("name,c,parent,ok,reason", CASES, ids=[c[0] for c in CASES])
def test_gate_table(name, c, parent, ok, reason):
    got, why = gate(c, BASE, parent=parent)
    assert got is ok, why
    assert ok or any(reason in w for w in why)


def test_gate_reports_every_violation():
    ok, why = gate(cand(ms_total_mean=40.0, no_head_frac=0.05, jump_count=9), BASE)
    assert not ok and len(why) == 3


def test_gate_fails_on_checks():
    ok, why = gate(BASE, BASE, checks_ok=False)
    assert not ok and why == ["tests/lint failed"]


def test_jump_count_is_recomputed_against_the_baseline_median():
    # the candidate's own median (0.001) would call all four of these jumps;
    # against the baseline's 0.01 threshold (0.03) only the last one is
    c = cand(jump_count=4, deltas=[0.001, 0.002, 0.001, 0.05])
    assert gates.candidate_jumps(c, BASE) == 1
    assert gate(c, BASE)[0] is True


def test_jump_gate_has_a_floor_so_a_calm_baseline_is_not_fatal():
    calm = {**BASE, "jump_count": 0}
    assert gate(cand(jump_count=5), calm)[0] is True
    ok, why = gate(cand(jump_count=6), calm)
    assert not ok and "jump count 6" in why[0]


def test_run_checks_timeout(monkeypatch, tmp_path):
    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="bash", timeout=1)

    monkeypatch.setattr(gates.subprocess, "run", fake_run)
    ok, tail = gates.run_checks(tmp_path, timeout_s=1)
    assert ok is False and tail == "tests timed out"
