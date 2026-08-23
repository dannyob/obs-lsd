"""Evaluation rig: replay clips, score with a vision judge, evolve the halo code."""

from __future__ import annotations

from pathlib import Path

PROMPTS_DIR = Path(__file__).parent / "prompts"
RUNS_DIR = Path("runs")


def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text()
