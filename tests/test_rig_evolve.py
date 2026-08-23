import json
import subprocess
from pathlib import Path

from obs_lsd.rig.evolve import (
    OPERATORS,
    Evolution,
    build_brief,
    child_plan,
    operator_text,
    select_survivors,
)


def test_operator_text_sections():
    assert "TOP_FIX" in operator_text("fix_top_complaint")
    assert "diffs" in operator_text("crossover")


def test_build_brief_fills_placeholders():
    b = build_brief(
        "evo/g1-0",
        1,
        "tune",
        {"seamless": 1, "distracting": 2, "ornate": 3, "total": 6, "notes": "N", "top_fix": "T"},
        {"frames": 10},
        "- g0: nothing",
    )
    assert "{" not in b.replace("{}", "") or "{operator}" not in b
    assert (
        "evo/g1-0" in b
        and "TOP_FIX: T" in b
        and "- g0: nothing" in b
        and "Do not add techniques" in b
    )


def test_select_survivors_orders_and_filters():
    recs = [
        {
            "id": "a",
            "gates": {"ok": True},
            "score": {"total": 50},
            "metrics_summary": {"ms_total_mean": 30},
        },
        {
            "id": "b",
            "gates": {"ok": True},
            "score": {"total": 50},
            "metrics_summary": {"ms_total_mean": 20},
        },
        {"id": "c", "gates": {"ok": False}, "score": None, "metrics_summary": {"ms_total_mean": 5}},
        {
            "id": "d",
            "gates": {"ok": True},
            "score": {"total": 70},
            "metrics_summary": {"ms_total_mean": 40},
        },
    ]
    assert [r["id"] for r in select_survivors(recs, 3)] == ["d", "b", "a"]


def test_child_plan_rotates_operators_and_crossover():
    surv = [{"id": "p0"}, {"id": "p1"}, {"id": "p2"}]
    plan1 = child_plan(1, surv)
    assert [c["operator"] for c in plan1] == [OPERATORS[1], OPERATORS[2], OPERATORS[3]]
    plan2 = child_plan(2, surv)
    assert plan2[2]["operator"] == "crossover" and plan2[2]["parents"] == ["p0", "p1"]


def _repo(tmp_path):
    r = tmp_path / "repo"
    r.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=r, check=True)
    (r / "techniques.md").write_text("- g0: baseline\n")
    subprocess.run(["git", "add", "-A"], cwd=r, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"],
        cwd=r,
        check=True,
    )
    return r


def test_dry_run_generation_creates_branches_and_ledger(tmp_path):
    repo = _repo(tmp_path)
    runs = repo / "runs"
    runs.mkdir()
    base_stats = {"ms_total_mean": 30.0, "no_head_frac": 0.0, "jump_count": 2}
    (runs / "baseline.json").write_text(
        json.dumps(
            {
                "run_dir": "runs/b",
                "stats": base_stats,
                "score": {
                    "seamless": 10,
                    "distracting": 10,
                    "ornate": 10,
                    "total": 30,
                    "notes": "n",
                    "top_fix": "t",
                },
            }
        )
    )
    ev = Evolution(repo=repo, clip="unused.mov", dry_run=True, population=3)
    recs = ev.run_generation(1)
    assert len(recs) == 1  # only one survivor (baseline) -> one child in gen 1
    rec = recs[0]
    assert (
        rec["branch"] == "evo/g1-0" and rec["operator"] in OPERATORS and rec["gates"]["ok"] is None
    )
    assert (repo / ".worktrees" / "evo-g1-0" / "brief.md").exists()
    branches = subprocess.run(
        ["git", "branch", "--list", "evo/*"], cwd=repo, capture_output=True, text=True, check=False
    ).stdout
    assert "evo/g1-0" in branches
    ledger = json.loads((runs / "ledger.json").read_text())
    assert ledger[-1]["id"] == rec["id"]


def test_run_generation_survives_judge_failure(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    runs = repo / "runs"
    runs.mkdir()
    base_stats = {"ms_total_mean": 30.0, "no_head_frac": 0.0, "jump_count": 2}
    (runs / "baseline.json").write_text(
        json.dumps(
            {
                "run_dir": "runs/baseline-abc1234",
                "stats": base_stats,
                "score": {
                    "seamless": 10,
                    "distracting": 10,
                    "ornate": 10,
                    "total": 30,
                    "notes": "n",
                    "top_fix": "t",
                },
            }
        )
    )

    class BoomJudge:
        cost_usd = 0.0

        def score(self, run_dir):
            raise RuntimeError("boom")

    ev = Evolution(repo=repo, clip="unused.mov", dry_run=False, population=3, judge=BoomJudge())
    monkeypatch.setattr(ev, "_prepare_worktree", lambda wt: None)
    monkeypatch.setattr(ev, "_run_agent", lambda wt, brief: (True, "ok"))
    monkeypatch.setattr(ev, "_sha", lambda ref: "cafefeed")
    monkeypatch.setattr("obs_lsd.rig.gates.run_checks", lambda cwd, timeout_s=600: (True, ""))

    real_run = subprocess.run

    def fake_run(cmd, *args, **kwargs):
        if isinstance(cmd, list) and cmd[:3] == ["uv", "run", "obs-lsd-rig"]:
            out_dir = Path(cmd[cmd.index("--out") + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "metrics.json").write_text(
                json.dumps({"ms_total_mean": 10.0, "no_head_frac": 0.0, "jump_count": 0})
            )
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", fake_run)

    recs = ev.run_generation(1)
    assert len(recs) == 1
    rec = recs[0]
    assert rec["gates"]["ok"] is False
    assert rec["gates"]["reasons"][0].startswith("judge failed")
    ledger = json.loads((runs / "ledger.json").read_text())
    assert ledger[-1]["id"] == rec["id"]
