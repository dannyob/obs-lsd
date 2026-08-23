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

#: The only paths a child commit may touch. Everything that decides whether a
#: child passes - the rig, the lint config, the packaging - is off limits, so a
#: candidate cannot rewrite its own exam.
ALLOWED_PATHS = {
    "src/obs_lsd/halo.py",
    "src/obs_lsd/compose.py",
    "src/obs_lsd/segment.py",
    "src/obs_lsd/track.py",
    "src/obs_lsd/pipeline.py",
    "src/obs_lsd/cli.py",
    "tests/test_halo.py",
    "tests/test_compose.py",
    "tests/test_segment.py",
    "tests/test_track.py",
    "tests/test_pipeline.py",
    "tests/test_cli.py",
    "techniques.md",
}


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
    if gen % 2 == 0 and len(survivors) >= 2:
        last = len(plan) - 1
        plan[last] = {
            "parent": survivors[0]["id"],
            "k": last,
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
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(self._recs, indent=2))
        os.replace(tmp, self.path)

    def by_id(self, id_: str) -> dict | None:
        return next((r for r in self._recs if r["id"] == id_), None)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()


def agent_cost(path: Path) -> float:
    """`total_cost_usd` out of a claude -p --output-format json dump. This is
    the dominant cost of a run and OpenRouter knows nothing about it."""
    try:
        return float(json.loads(path.read_text()).get("total_cost_usd") or 0.0)
    except (OSError, AttributeError, TypeError, ValueError):
        return 0.0


def baseline_is_stale(repo: Path, baseline_sha: str) -> bool:
    """True when HEAD has moved on since the baseline was captured. Children
    fork from the baseline sha, so a stale baseline silently evolves old code."""
    head = _git(Path(repo), "rev-parse", "--short", "HEAD")
    return not (head.startswith(baseline_sha) or baseline_sha.startswith(head))


def scope_violations(repo: Path, parent_sha: str, child_sha: str) -> list[str]:
    """Paths the child touched that are not in ALLOWED_PATHS. Run against the
    main repo, not the child's worktree - the child controls its worktree."""
    out = _git(repo, "diff", "--name-only", f"{parent_sha}..{child_sha}")
    return sorted(p for p in out.splitlines() if p.strip() and p not in ALLOWED_PATHS)


@dataclass
class Evolution:
    repo: Path
    clip: str
    judge: object = None
    dry_run: bool = False
    population: int = 3
    claude_cmd: str = "claude"
    timeout_s: int = 1200
    max_budget_usd: float = 5.0
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
                    "sha": self.baseline_sha(),
                    "gates": {"ok": True, "reasons": []},
                    "score": self.baseline["score"],
                    "metrics_summary": self.baseline["stats"],
                    "run_dir": self.baseline["run_dir"],
                    "cost_usd": 0.0,
                }
            )

    def _sha(self, ref: str) -> str:
        return _git(self.repo, "rev-parse", "--short", ref)

    def baseline_sha(self) -> str:
        """The baseline record's sha comes from runs/baseline.json's run_dir
        name (written by `obs-lsd-rig baseline` as "baseline-<sha>"), not
        HEAD - HEAD may have moved on since the baseline was captured."""
        name = Path(self.baseline.get("run_dir", "")).name
        m = re.match(r"^baseline-([0-9a-fA-F]+)$", name)
        if m:
            return m.group(1)
        self._say(f"warning: could not derive baseline sha from run_dir {name!r}; using HEAD")
        return self._sha("HEAD")

    def _say(self, msg: str) -> None:
        self.log.append(msg)
        print(msg, flush=True)

    def _branch_exists(self, branch: str) -> bool:
        r = subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", branch],
            cwd=self.repo,
            capture_output=True,
            text=True,
            check=False,
        )
        return r.returncode == 0

    def _worktree(self, branch: str, base_sha: str) -> tuple[Path, str]:
        """Create (or re-create) a worktree for `branch` off `base_sha`. If
        `branch` already exists and points somewhere other than `base_sha`
        (a prior run's work), don't clobber it - suffix with -rN and use
        that branch instead, so re-running a generation never loses work."""
        final_branch = branch
        if self._branch_exists(branch) and self._sha(branch) != base_sha:
            n = 1
            while self._branch_exists(f"{branch}-r{n}"):
                n += 1
            final_branch = f"{branch}-r{n}"
        wt = self.repo / ".worktrees" / final_branch.replace("/", "-")
        if wt.exists():
            shutil.rmtree(wt)
            subprocess.run(["git", "worktree", "prune"], cwd=self.repo, check=False)
        _git(self.repo, "worktree", "add", "-q", "-B", final_branch, str(wt), base_sha)
        return wt, final_branch

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
            try:
                r1 = subprocess.run(
                    ["uv", "venv", "-p", "3.12"],
                    cwd=wt,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=600,
                )
                if r1.returncode != 0:
                    raise RuntimeError(f"uv venv failed: {(r1.stderr or r1.stdout)[-300:]}")
                r2 = subprocess.run(
                    ["uv", "pip", "install", "-e", ".[dev]"],
                    cwd=wt,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=600,
                )
                if r2.returncode != 0:
                    raise RuntimeError(f"uv pip install failed: {(r2.stderr or r2.stdout)[-300:]}")
            except subprocess.TimeoutExpired:
                # a stalled index fetch at 2am must not hang the whole run
                raise RuntimeError("worktree setup timed out") from None

    def _techniques(self, wt: Path) -> str:
        p = wt / "techniques.md"
        return p.read_text() if p.exists() else ""

    def _run_agent(self, wt: Path, brief: str, agent_dir: Path) -> tuple[bool, str, float]:
        """Run the mutation agent in `wt`. The brief and the agent's JSON go to
        `agent_dir` (under runs/), never into the worktree the agent commits."""
        agent_dir.mkdir(parents=True, exist_ok=True)
        (agent_dir / "brief.md").write_text(brief)
        if self.dry_run:
            return True, "dry-run", 0.0
        out_path = agent_dir / "agent-output.json"
        cmd = [
            self.claude_cmd,
            "-p",
            brief,
            "--output-format",
            "json",
            "--max-budget-usd",
            str(self.max_budget_usd),
            "--dangerously-skip-permissions",
        ]
        # don't let this claude's own CLAUDE_CODE_*/CLAUDECODE env, or our
        # OpenRouter key, leak into the child claude -p, and never let it read
        # our stdin. Everything else is inherited: claude needs its own auth.
        env = {
            k: v
            for k, v in os.environ.items()
            if not (k.startswith("CLAUDE_CODE_") or k in ("CLAUDECODE", "OPENROUTER_API_KEY"))
        }
        try:
            r = subprocess.run(
                cmd,
                cwd=wt,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                check=False,
                stdin=subprocess.DEVNULL,
                env=env,
            )
        except subprocess.TimeoutExpired as e:
            if e.stdout:
                out = e.stdout if isinstance(e.stdout, str) else e.stdout.decode("utf-8", "replace")
                out_path.write_text(out)
            return False, "agent timeout", agent_cost(out_path)
        except OSError as e:
            return False, f"agent failed: {e}", 0.0
        out_path.write_text(r.stdout)
        return r.returncode == 0, (r.stderr or r.stdout)[-500:], agent_cost(out_path)

    def run_generation(self, gen: int) -> list[dict]:
        survivors = select_survivors(self.ledger.records(), self.population)
        if not survivors:
            self._say("no survivors to evolve from")
            return []
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
                "agent_cost_usd": 0.0,
                "agent_dir": None,
            }
            stage = "worktree setup"
            try:
                wt, branch = self._worktree(branch, parent["sha"])
                rec["branch"] = branch
                rec["id"] = branch.replace("/", "-")
                self._prepare_worktree(wt)

                operator = child["operator"]
                extra = ""
                if operator == "crossover":
                    stage = "crossover"
                    a, b = (self.ledger.by_id(p) for p in child["parents"])
                    try:
                        is_anc = subprocess.run(
                            ["git", "merge-base", "--is-ancestor", a["sha"], b["sha"]],
                            cwd=self.repo,
                            capture_output=True,
                            text=True,
                            check=False,
                        )
                        if is_anc.returncode == 0:
                            raise RuntimeError("survivors[0] is an ancestor of survivors[1]")
                        base = _git(self.repo, "merge-base", a["sha"], b["sha"])
                        extra = (
                            "Diff A:\n"
                            + _git(self.repo, "diff", base, a["sha"])
                            + "\n\nDiff B:\n"
                            + _git(self.repo, "diff", base, b["sha"])
                        )
                    except (subprocess.CalledProcessError, RuntimeError) as e:
                        operator = "ornate"
                        rec["operator"] = operator
                        rec["note"] = f"crossover skipped: {e!s:.200}"
                        extra = ""

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
                agent_dir = self.runs / "agents" / branch.replace("/", "-")
                rec["agent_dir"] = str(agent_dir)
                agent_ok, agent_msg, spent = self._run_agent(wt, brief, agent_dir)
                rec["agent"] = agent_msg
                rec["agent_cost_usd"] = spent
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
                        self._evaluate(rec, wt, parent)
            except Exception as e:  # noqa: BLE001 - any per-child failure must not abort the run
                rec["gates"] = {"ok": False, "reasons": [f"{stage} failed: {e!s:.300}"]}
                self._say(f"[g{gen}] {rec['branch']} FAILED at {stage}: {e!s:.200}")
            self.ledger.append(rec)
            results.append(rec)
        self._table(gen, results)
        return results

    def _evaluate(self, rec: dict, wt: Path, parent: dict) -> None:
        from obs_lsd.rig.gates import gate, run_checks

        bad = scope_violations(self.repo, parent["sha"], rec["sha"])
        if bad:
            rec["gates"] = {"ok": False, "reasons": ["out-of-scope changes: " + ", ".join(bad)]}
            return

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
        # the per-frame deltas are for the gate only; keep them out of the ledger
        rec["metrics_summary"] = {k: v for k, v in stats.items() if k != "deltas"}
        rec["run_dir"] = str(run_dir)
        g_ok, why = gate(
            stats, self.baseline["stats"], checks_ok=True, parent=parent.get("metrics_summary")
        )
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
            if r["gates"]["ok"]:
                gates = "ok"
            elif r["gates"]["ok"] is None:
                gates = "?"
            else:
                flat = "; ".join(" ".join(reason.split()) for reason in r["gates"]["reasons"])
                gates = f"FAIL {flat}"
            cost = f"judge ${r.get('cost_usd', 0):.2f} / agent ${r.get('agent_cost_usd', 0):.2f}"
            self._say(
                f"{r['id']:28s} {r['operator']:18s} {gates:40.40s} {s:14s} {ms:5.1f}ms  "
                f"{cost:26s} {r.get('run_dir') or ''}"
            )
        best = select_survivors(self.ledger.records(), 1)
        if best and best[0].get("run_dir"):
            self._say(
                f"champion: {best[0]['id']} total {best[0]['score']['total']:.0f}  sheet: {best[0]['run_dir']}/sheet.png"
            )
        recs_all = self.ledger.records()
        judge_total = sum(r.get("cost_usd", 0) or 0 for r in recs_all)
        agent_total = sum(r.get("agent_cost_usd", 0) or 0 for r in recs_all)
        self._say(
            f"spend so far: judge ${judge_total:.2f} (openrouter) / "
            f"agent ${agent_total:.2f} (claude) / total ${judge_total + agent_total:.2f}"
        )
