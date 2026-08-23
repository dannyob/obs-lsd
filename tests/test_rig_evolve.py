import json
import re
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
    assert re.search(r"\{[a-z_]+\}", b) is None
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


def test_child_plan_crossover_lands_on_last_child_with_two_survivors():
    surv = [{"id": "p0"}, {"id": "p1"}]
    plan = child_plan(2, surv)
    assert plan[1]["operator"] == "crossover" and plan[1]["parents"] == ["p0", "p1"]


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
    head = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    runs = repo / "runs"
    runs.mkdir()
    base_stats = {"ms_total_mean": 30.0, "no_head_frac": 0.0, "jump_count": 2}
    (runs / "baseline.json").write_text(
        json.dumps(
            {
                "run_dir": f"runs/baseline-{head}",
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


def test_baseline_sha_derived_from_run_dir(tmp_path):
    repo = _repo(tmp_path)
    runs = repo / "runs"
    runs.mkdir()
    (runs / "baseline.json").write_text(
        json.dumps(
            {
                "run_dir": "runs/baseline-abc1234",
                "stats": {"ms_total_mean": 1.0, "no_head_frac": 0.0, "jump_count": 0},
                "score": {
                    "seamless": 1,
                    "distracting": 1,
                    "ornate": 1,
                    "total": 3,
                    "notes": "",
                    "top_fix": "",
                },
            }
        )
    )
    ev = Evolution(repo=repo, clip="unused.mov", dry_run=True, population=3)
    assert ev.ledger.by_id("baseline")["sha"] == "abc1234"


def test_baseline_sha_falls_back_to_head_when_run_dir_unrecognised(tmp_path):
    repo = _repo(tmp_path)
    runs = repo / "runs"
    runs.mkdir()
    (runs / "baseline.json").write_text(
        json.dumps(
            {
                "run_dir": "runs/b",
                "stats": {"ms_total_mean": 1.0, "no_head_frac": 0.0, "jump_count": 0},
                "score": {
                    "seamless": 1,
                    "distracting": 1,
                    "ornate": 1,
                    "total": 3,
                    "notes": "",
                    "top_fix": "",
                },
            }
        )
    )
    ev = Evolution(repo=repo, clip="unused.mov", dry_run=True, population=3)
    head = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert ev.ledger.by_id("baseline")["sha"] == head


def test_worktree_rerun_suffixes_branch_and_leaves_original(tmp_path):
    repo = _repo(tmp_path)
    base_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()
    subprocess.run(["git", "branch", "evo/g1-0"], cwd=repo, check=True)
    subprocess.run(["git", "checkout", "-q", "evo/g1-0"], cwd=repo, check=True)
    (repo / "extra.txt").write_text("x\n")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "advance"],
        cwd=repo,
        check=True,
    )
    subprocess.run(["git", "checkout", "-q", "main"], cwd=repo, check=True)
    tip_before = subprocess.run(
        ["git", "rev-parse", "evo/g1-0"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()

    runs = repo / "runs"
    runs.mkdir()
    (runs / "baseline.json").write_text(
        json.dumps(
            {
                "run_dir": "runs/baseline-deadbee",
                "stats": {"ms_total_mean": 1.0, "no_head_frac": 0.0, "jump_count": 0},
                "score": {
                    "seamless": 1,
                    "distracting": 1,
                    "ornate": 1,
                    "total": 3,
                    "notes": "",
                    "top_fix": "",
                },
            }
        )
    )
    ev = Evolution(repo=repo, clip="unused.mov", dry_run=True, population=3)
    wt, branch = ev._worktree("evo/g1-0", base_sha)
    assert branch == "evo/g1-0-r1"
    assert wt == repo / ".worktrees" / "evo-g1-0-r1"
    assert wt.exists()
    tip_after = subprocess.run(
        ["git", "rev-parse", "evo/g1-0"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()
    assert tip_after == tip_before


def test_prepare_worktree_failure_is_ledgered(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    head = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    runs = repo / "runs"
    runs.mkdir()
    base_stats = {"ms_total_mean": 30.0, "no_head_frac": 0.0, "jump_count": 2}
    (runs / "baseline.json").write_text(
        json.dumps(
            {
                "run_dir": f"runs/baseline-{head}",
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
    ev = Evolution(repo=repo, clip="unused.mov", dry_run=False, population=3)

    real_run = subprocess.run

    def fake_run(cmd, *args, **kwargs):
        if isinstance(cmd, list) and cmd[:2] == ["uv", "venv"]:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="boom: no python")
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", fake_run)

    recs = ev.run_generation(1)
    assert len(recs) == 1
    rec = recs[0]
    assert rec["gates"]["ok"] is False
    assert rec["gates"]["reasons"][0].startswith("worktree setup failed")
    assert "boom" in rec["gates"]["reasons"][0]


def test_crossover_falls_back_to_ornate_when_one_survivor_is_an_ancestor(tmp_path):
    repo = _repo(tmp_path)
    sha0 = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    (repo / "a.txt").write_text("a\n")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "c1"],
        cwd=repo,
        check=True,
    )
    sha1 = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    (repo / "b.txt").write_text("b\n")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "c2"],
        cwd=repo,
        check=True,
    )
    sha2 = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    runs = repo / "runs"
    runs.mkdir()
    (runs / "baseline.json").write_text(
        json.dumps(
            {
                "run_dir": f"runs/baseline-{sha0}",
                "stats": {"ms_total_mean": 1.0, "no_head_frac": 0.0, "jump_count": 0},
                "score": {
                    "seamless": 1,
                    "distracting": 1,
                    "ornate": 1,
                    "total": 3,
                    "notes": "",
                    "top_fix": "",
                },
            }
        )
    )
    common = {"gates": {"ok": True, "reasons": []}, "metrics_summary": {"ms_total_mean": 1.0}}
    ledger_seed = [
        {
            "id": "hi",
            "branch": "evo/hi",
            "parent": "baseline",
            "generation": 1,
            "operator": "tune",
            "sha": sha1,  # ancestor of sha2
            "score": {"total": 50},
            "run_dir": "runs/hi",
            "cost_usd": 0.0,
            **common,
        },
        {
            "id": "lo",
            "branch": "evo/lo",
            "parent": "baseline",
            "generation": 1,
            "operator": "tune",
            "sha": sha2,
            "score": {"total": 40},
            "run_dir": "runs/lo",
            "cost_usd": 0.0,
            **common,
        },
    ]
    (runs / "ledger.json").write_text(json.dumps(ledger_seed))

    ev = Evolution(repo=repo, clip="unused.mov", dry_run=True, population=3)
    recs = ev.run_generation(2)
    assert len(recs) == 2
    fallback_rec = recs[1]
    assert fallback_rec["operator"] == "ornate"
    assert fallback_rec["note"].startswith("crossover skipped")
    assert fallback_rec["parent"] == "hi"
