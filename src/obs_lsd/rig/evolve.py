"""Evolutionary loop: branches as genomes, claude -p as the mutation operator."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from obs_lsd.rig import load_prompt
from obs_lsd.rig.metrics import metrics_table

OPERATORS = ["fix_top_complaint", "new_technique", "tune", "simplify", "ornate"]


def operator_text(name: str) -> str:
    text = load_prompt("operators.md")
    m = re.search(rf"^## {re.escape(name)}\n(.*?)(?=^## |\Z)", text, re.DOTALL | re.MULTILINE)
    if not m:
        raise KeyError(name)
    return m.group(1).strip()


def build_brief(
    branch,
    generation,
    operator,
    parent_score: dict,
    parent_stats: dict,
    techniques: str,
    extra: str = "",
) -> str:
    return load_prompt("mutate.md").format(
        branch=branch,
        generation=generation,
        operator=operator,
        operator_text=operator_text(operator) + ("\n\n" + extra if extra else ""),
        seamless=parent_score.get("seamless", "?"),
        distracting=parent_score.get("distracting", "?"),
        ornate=parent_score.get("ornate", "?"),
        total=parent_score.get("total", "?"),
        notes=parent_score.get("notes", ""),
        top_fix=parent_score.get("top_fix", ""),
        metrics_table=metrics_table(parent_stats),
        techniques=techniques or "(none yet)",
        rubric=load_prompt("rubric.md"),
    )


def select_survivors(records: list[dict], k: int = 3) -> list[dict]:
    ok = [r for r in records if r.get("gates", {}).get("ok") and r.get("score")]
    ok.sort(key=lambda r: (-r["score"]["total"], r["metrics_summary"].get("ms_total_mean", 0.0)))
    return ok[:k]


def child_plan(gen: int, survivors: list[dict]) -> list[dict]:
    plan = []
    for k, parent in enumerate(survivors):
        plan.append(
            {"parent": parent["id"], "k": k, "operator": OPERATORS[(gen + k) % len(OPERATORS)]}
        )
    if gen % 2 == 0 and len(survivors) >= 2 and len(plan) >= 3:
        plan[2] = {
            "parent": survivors[0]["id"],
            "k": 2,
            "operator": "crossover",
            "parents": [survivors[0]["id"], survivors[1]["id"]],
        }
    return plan


class Ledger:
    def __init__(self, path: Path):
        self.path = path
        self._recs = json.loads(path.read_text()) if path.exists() else []

    def records(self) -> list[dict]:
        return list(self._recs)

    def append(self, rec: dict) -> None:
        self._recs.append(rec)
        self.path.write_text(json.dumps(self._recs, indent=2))

    def by_id(self, id_: str) -> dict | None:
        return next((r for r in self._recs if r["id"] == id_), None)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()


@dataclass
class Evolution:
    repo: Path
    clip: str
    judge: object = None
    dry_run: bool = False
    population: int = 3
    claude_cmd: str = "claude"
    timeout_s: int = 1200
    log: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.repo = Path(self.repo)
        self.runs = self.repo / "runs"
        self.runs.mkdir(exist_ok=True)
        self.ledger = Ledger(self.runs / "ledger.json")
        self.baseline = json.loads((self.runs / "baseline.json").read_text())
        if not self.ledger.records():
            self.ledger.append(
                {
                    "id": "baseline",
                    "branch": "main",
                    "parent": None,
                    "generation": 0,
                    "operator": "baseline",
                    "sha": self._sha("HEAD"),
                    "gates": {"ok": True, "reasons": []},
                    "score": self.baseline["score"],
                    "metrics_summary": self.baseline["stats"],
                    "run_dir": self.baseline["run_dir"],
                    "cost_usd": 0.0,
                }
            )

    def _sha(self, ref: str) -> str:
        return _git(self.repo, "rev-parse", "--short", ref)

    def _say(self, msg: str) -> None:
        self.log.append(msg)
        print(msg, flush=True)

    def _worktree(self, branch: str, base_sha: str) -> Path:
        wt = self.repo / ".worktrees" / branch.replace("/", "-")
        if wt.exists():
            shutil.rmtree(wt)
            subprocess.run(["git", "worktree", "prune"], cwd=self.repo, check=False)
        _git(self.repo, "worktree", "add", "-q", "-B", branch, str(wt), base_sha)
        self._prepare_worktree(wt)
        return wt

    def _prepare_worktree(self, wt: Path) -> None:
        """Give the worktree its own venv and a symlink to the shared models,
        so `uv run` inside it doesn't re-download mediapipe's model files."""
        if self.dry_run:
            return
        models_src = self.repo / "src" / "obs_lsd" / "models"
        models_dst = wt / "src" / "obs_lsd" / "models"
        if models_src.exists() and not models_dst.exists() and not models_dst.is_symlink():
            os.symlink(models_src, models_dst)
        if not (wt / ".venv").exists():
            subprocess.run(["uv", "venv", "-p", "3.12"], cwd=wt, check=False)
            subprocess.run(["uv", "pip", "install", "-e", ".[dev]"], cwd=wt, check=False)

    def _techniques(self, wt: Path) -> str:
        p = wt / "techniques.md"
        return p.read_text() if p.exists() else ""

    def _run_agent(self, wt: Path, brief: str) -> tuple[bool, str]:
        (wt / "brief.md").write_text(brief)
        if self.dry_run:
            return True, "dry-run"
        cmd = [
            self.claude_cmd,
            "-p",
            brief,
            "--output-format",
            "json",
            "--dangerously-skip-permissions",
        ]
        try:
            r = subprocess.run(
                cmd, cwd=wt, capture_output=True, text=True, timeout=self.timeout_s, check=False
            )
        except subprocess.TimeoutExpired:
            return False, "agent timeout"
        (wt / "agent-output.json").write_text(r.stdout)
        return r.returncode == 0, (r.stderr or r.stdout)[-500:]

    def run_generation(self, gen: int) -> list[dict]:
        survivors = select_survivors(self.ledger.records(), self.population)
        results = []
        for child in child_plan(gen, survivors):
            parent = self.ledger.by_id(child["parent"])
            branch = f"evo/g{gen}-{child['k']}"
            rec = {
                "id": branch.replace("/", "-"),
                "branch": branch,
                "parent": parent["id"],
                "generation": gen,
                "operator": child["operator"],
                "sha": None,
                "gates": {"ok": None, "reasons": []},
                "score": None,
                "metrics_summary": {},
                "run_dir": None,
                "cost_usd": 0.0,
                "agent": None,
            }
            stage = "worktree"
            try:
                wt = self._worktree(branch, parent["sha"])

                operator = child["operator"]
                extra = ""
                if operator == "crossover":
                    stage = "crossover"
                    a, b = (self.ledger.by_id(p) for p in child["parents"])
                    base = _git(self.repo, "merge-base", a["sha"], b["sha"])
                    extra = (
                        "Diff A:\n"
                        + _git(self.repo, "diff", base, a["sha"])
                        + "\n\nDiff B:\n"
                        + _git(self.repo, "diff", base, b["sha"])
                    )

                stage = "brief"
                brief = build_brief(
                    branch,
                    gen,
                    operator,
                    parent["score"] or {},
                    parent["metrics_summary"],
                    self._techniques(wt),
                    extra,
                )
                self._say(f"[g{gen}] {branch} <- {parent['id']} via {operator}")

                stage = "agent"
                agent_ok, agent_msg = self._run_agent(wt, brief)
                rec["agent"] = agent_msg
                if not self.dry_run:
                    sha = self._sha(branch)
                    if not agent_ok or sha == parent["sha"]:
                        rec["gates"] = {
                            "ok": False,
                            "reasons": ["agent failed" if not agent_ok else "no change"],
                        }
                    else:
                        rec["sha"] = sha
                        rec["id"] = f"{rec['id']}-{sha}"
                        stage = "evaluate"
                        self._evaluate(rec, wt)
            except Exception as e:  # noqa: BLE001 - any per-child failure must not abort the run
                rec["gates"] = {"ok": False, "reasons": [f"{stage} failed: {e!s:.300}"]}
                self._say(f"[g{gen}] {rec['branch']} FAILED at {stage}: {e!s:.200}")
            self.ledger.append(rec)
            results.append(rec)
        self._table(gen, results)
        return results

    def _evaluate(self, rec: dict, wt: Path) -> None:
        from obs_lsd.rig.gates import gate, run_checks

        ok, tail = run_checks(wt, timeout_s=600)
        if not ok:
            reason = tail if tail == "tests timed out" else f"tests/lint failed: {tail[-200:]}"
            rec["gates"] = {"ok": False, "reasons": [reason]}
            return

        run_dir = self.runs / rec["id"]
        # render with the CANDIDATE's code: run obs-lsd-rig render inside the worktree
        try:
            r = subprocess.run(
                [
                    "uv",
                    "run",
                    "obs-lsd-rig",
                    "render",
                    str(Path(self.clip).resolve()),
                    "--out",
                    str(run_dir.resolve()),
                ],
                cwd=wt,
                capture_output=True,
                text=True,
                check=False,
                timeout=600,
            )
        except subprocess.TimeoutExpired:
            rec["gates"] = {"ok": False, "reasons": ["render timed out"]}
            return
        if r.returncode != 0:
            rec["gates"] = {"ok": False, "reasons": ["render failed: " + r.stderr[-300:]]}
            return
        stats = json.loads((run_dir / "metrics.json").read_text())
        rec["metrics_summary"] = stats
        rec["run_dir"] = str(run_dir)
        g_ok, why = gate(stats, self.baseline["stats"], checks_ok=True)
        rec["gates"] = {"ok": g_ok, "reasons": why}
        if g_ok and self.judge is not None:
            try:
                before = self.judge.cost_usd
                s = self.judge.score(run_dir)
                rec["score"] = s.to_json()
                rec["cost_usd"] = self.judge.cost_usd - before
            except Exception as e:  # noqa: BLE001 - judge.score() can raise anything (httpx, JSON, etc.)
                rec["gates"] = {"ok": False, "reasons": [f"judge failed: {e!s:.300}"]}

    def _table(self, gen: int, recs: list[dict]) -> None:
        self._say(f"--- generation {gen} ---")
        for r in recs:
            sc = r["score"]
            s = (
                f"{sc['seamless']:.0f}/{sc['distracting']:.0f}/{sc['ornate']:.0f}={sc['total']:.0f}"
                if sc
                else "-"
            )
            ms = r["metrics_summary"].get("ms_total_mean", 0.0)
            gates = (
                "ok"
                if r["gates"]["ok"]
                else (
                    "?" if r["gates"]["ok"] is None else "FAIL " + "; ".join(r["gates"]["reasons"])
                )
            )
            self._say(
                f"{r['id']:28s} {r['operator']:18s} {gates:40.40s} {s:14s} {ms:5.1f}ms  {r.get('run_dir') or ''}"
            )
        best = select_survivors(self.ledger.records(), 1)
        if best:
            self._say(
                f"champion: {best[0]['id']} total {best[0]['score']['total']:.0f}  sheet: {best[0].get('run_dir')}/sheet.png"
            )
        self._say(
            f"openrouter spend so far: ${sum(r.get('cost_usd', 0) for r in self.ledger.records()):.2f}"
        )
