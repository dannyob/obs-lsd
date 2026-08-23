"""Vision-model judge over OpenRouter: absolute rubric scores and pairwise A/B."""

from __future__ import annotations

import base64
import json
import random
import re
import statistics
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx

from obs_lsd.rig import load_prompt
from obs_lsd.rig.metrics import metrics_table

DEFAULT_MODEL = "anthropic/claude-opus-5"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
IMAGES = ("sheet.png", "crops.png", "motion.png")
# rough $/M tokens for the cost print; exact billing is OpenRouter's business
PRICE = {
    "anthropic/claude-opus-5": (5.0, 25.0),
    "anthropic/claude-sonnet-5": (2.0, 10.0),
    "google/gemini-3.1-pro-preview": (2.0, 12.0),
    "openai/gpt-5.5": (5.0, 30.0),
}


def resolve_api_key(env: dict | None = None, pass_cmd=subprocess.run) -> str:
    import os

    env = os.environ if env is None else env
    if env.get("OPENROUTER_API_KEY"):
        return env["OPENROUTER_API_KEY"]
    r = pass_cmd(["pass", "show", "openrouter/api-key"], capture_output=True, text=True)
    if r.returncode != 0 or not r.stdout.strip():
        raise RuntimeError("No OPENROUTER_API_KEY and `pass show openrouter/api-key` failed")
    return r.stdout.strip().splitlines()[0]


def parse_score_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            raise ValueError(f"no JSON in judge reply: {text[:200]!r}") from None
        return json.loads(m.group(0))


@dataclass
class Score:
    seamless: float
    distracting: float
    ornate: float
    total: float
    notes: str
    top_fix: str
    samples: list[dict] = field(default_factory=list)
    model: str = DEFAULT_MODEL
    cost_usd: float = 0.0

    def to_json(self) -> dict:
        return asdict(self)


def _img_part(path: Path) -> dict:
    b64 = base64.b64encode(path.read_bytes()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}


class OpenRouterJudge:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        transport: httpx.BaseTransport | None = None,
        rng: random.Random | None = None,
    ):
        self.model = model
        self.api_key = api_key or resolve_api_key()
        self._transport = transport
        self.client = httpx.Client(transport=transport, timeout=120.0)
        self.rng = rng or random.Random()
        self.cost_usd = 0.0

    def _chat(self, content: list[dict]) -> str:
        body = {"model": self.model, "messages": [{"role": "user", "content": content}]}
        r = self.client.post(
            API_URL,
            json=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": "https://github.com/dob/obs-lsd",
                "X-Title": "obs-lsd rig",
            },
        )
        r.raise_for_status()
        data = r.json()
        u = data.get("usage") or {}
        pin, pout = PRICE.get(self.model, (5.0, 25.0))
        self.cost_usd += (
            u.get("prompt_tokens", 0) / 1e6 * pin + u.get("completion_tokens", 0) / 1e6 * pout
        )
        return data["choices"][0]["message"]["content"]

    def _run_content(self, run_dir: Path, preamble: str) -> list[dict]:
        parts = [{"type": "text", "text": preamble}]
        for name in IMAGES:
            parts.append({"type": "text", "text": f"Image: {name}"})
            parts.append(_img_part(run_dir / name))
        return parts

    def score(self, run_dir: Path, samples: int = 3) -> Score:
        run_dir = Path(run_dir)
        stats = json.loads((run_dir / "metrics.json").read_text())
        prompt = load_prompt("judge.md").format(
            metrics_table=metrics_table(stats), rubric=load_prompt("rubric.md")
        )
        results = []
        for _ in range(samples):
            for attempt in range(2):
                try:
                    results.append(parse_score_json(self._chat(self._run_content(run_dir, prompt))))
                    break
                except (ValueError, httpx.HTTPError):
                    if attempt == 1:
                        pass
        if len(results) < min(2, samples):
            raise RuntimeError(f"judge failed: only {len(results)} usable samples")
        med = {
            k: float(statistics.median(float(r[k]) for r in results))
            for k in ("seamless", "distracting", "ornate")
        }
        total = sum(med.values())
        best = min(results, key=lambda r: abs(sum(float(r[k]) for k in med) - total))
        s = Score(
            **med,
            total=total,
            notes=str(best.get("notes", "")),
            top_fix=str(best.get("top_fix", "")),
            samples=results,
            model=self.model,
            cost_usd=self.cost_usd,
        )
        (run_dir / "score.json").write_text(json.dumps(s.to_json(), indent=2))
        return s

    def pairwise(self, run_a: Path, run_b: Path, votes: int = 3) -> dict:
        run_a, run_b = Path(run_a), Path(run_b)
        rubric = load_prompt("rubric.md")
        tally = {"a": 0, "b": 0}
        out = []
        for _ in range(votes):
            left, right = (run_a, run_b) if self.rng.random() < 0.5 else (run_b, run_a)
            parts = [
                {
                    "type": "text",
                    "text": (
                        "Compare two outputs of the same filter on the same clip, A then B, "
                        'against the rubric. Reply JSON only: {"prefer": "A"|"B"|"tie", '
                        '"why": "..."}\n\nRubric:\n' + rubric
                    ),
                }
            ]
            parts += self._run_content(left, "=== A ===")
            parts += self._run_content(right, "=== B ===")
            try:
                ans = parse_score_json(self._chat(parts))
            except (ValueError, httpx.HTTPError):
                continue
            pref = str(ans.get("prefer", "tie")).strip().upper()
            winner = {"A": left, "B": right}.get(pref)
            key = None if winner is None else ("a" if winner == run_a else "b")
            if key:
                tally[key] += 1
            out.append(
                {"left": "a" if left == run_a else "b", "prefer": pref, "why": ans.get("why", "")}
            )
        prefer = "tie" if tally["a"] == tally["b"] else max(tally, key=tally.get)
        return {"prefer": prefer, "votes": out, "tally": tally, "cost_usd": self.cost_usd}

    def calibrate(self, pairs: list[dict], models: list[str]) -> dict:
        res = {}
        for m in models:
            j = OpenRouterJudge(
                model=m, api_key=self.api_key, transport=self._transport, rng=self.rng
            )
            agree = sum(
                1
                for p in pairs
                if j.pairwise(Path(p["a"]), Path(p["b"]), votes=3)["prefer"] == p["human"]
            )
            res[m] = agree / len(pairs) if pairs else 0.0
            self.cost_usd += j.cost_usd
        return res
