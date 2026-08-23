import json
import random

import cv2
import httpx
import numpy as np
import pytest

from obs_lsd.rig.judge import OpenRouterJudge, parse_score_json, resolve_api_key


def _run_dir(tmp_path, name="r"):
    d = tmp_path / name
    d.mkdir()
    img = np.zeros((20, 20, 3), np.uint8)
    for n in ("sheet.png", "crops.png", "motion.png"):
        cv2.imwrite(str(d / n), img)
    (d / "metrics.json").write_text(json.dumps({"frames": 10, "jump_count": 1}))
    return d


def _transport(replies):
    calls = []

    def handler(request: httpx.Request):
        calls.append(json.loads(request.content))
        body = replies[len(calls) - 1] if len(calls) - 1 < len(replies) else replies[-1]
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": body}}],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 100},
            },
        )

    return httpx.MockTransport(handler), calls


def test_parse_score_json_strict_and_fallback():
    assert (
        parse_score_json(
            '{"seamless": 10, "distracting": 20, "ornate": 5, "notes": "n", "top_fix": "f"}'
        )["ornate"]
        == 5
    )
    assert (
        parse_score_json(
            'Sure! ```json\n{"seamless": 1, "distracting": 2, "ornate": 3, "notes": "", "top_fix": ""}\n```'
        )["seamless"]
        == 1
    )
    with pytest.raises(ValueError):
        parse_score_json("no json here")


def test_score_takes_median_and_writes_file(tmp_path):
    d = _run_dir(tmp_path)
    replies = [
        json.dumps({"seamless": 10, "distracting": 20, "ornate": 12, "notes": "a", "top_fix": "x"}),
        json.dumps({"seamless": 14, "distracting": 18, "ornate": 30, "notes": "b", "top_fix": "y"}),
        json.dumps({"seamless": 12, "distracting": 25, "ornate": 6, "notes": "c", "top_fix": "z"}),
    ]
    tr, calls = _transport(replies)
    j = OpenRouterJudge(model="m", api_key="k", transport=tr)
    s = j.score(d, samples=3)
    assert (s.seamless, s.distracting, s.ornate, s.total) == (12, 20, 12, 44)
    assert s.notes == "c" and s.top_fix == "z"  # from the sample closest to the median total
    assert len(calls) == 3 and calls[0]["model"] == "m"
    # images and metrics made it into the request
    content = calls[0]["messages"][0]["content"]
    assert sum(1 for c in content if c["type"] == "image_url") == 3
    assert any("jump_count" in c.get("text", "") for c in content)
    assert json.loads((d / "score.json").read_text())["total"] == 44
    assert j.cost_usd > 0


def test_score_with_one_sample(tmp_path):
    d = _run_dir(tmp_path)
    reply = json.dumps(
        {"seamless": 10, "distracting": 20, "ornate": 12, "notes": "a", "top_fix": "x"}
    )
    tr, calls = _transport([reply])
    j = OpenRouterJudge(model="m", api_key="k", transport=tr)
    s = j.score(d, samples=1)
    assert (s.seamless, s.distracting, s.ornate, s.total) == (10, 20, 12, 42)
    assert len(calls) == 1
    assert json.loads((d / "score.json").read_text())["total"] == 42


def test_score_drops_a_sample_missing_a_category_and_keeps_the_good_ones(tmp_path):
    d = _run_dir(tmp_path)
    good = json.dumps(
        {"seamless": 12, "distracting": 20, "ornate": 12, "notes": "a", "top_fix": "x"}
    )
    bad = json.dumps({"seamless": 10, "distracting": 20, "notes": "b", "top_fix": "y"})
    # sample 2 comes back without "ornate" on both attempts; 1 and 3 are fine
    tr, calls = _transport([good, bad, bad, good])
    j = OpenRouterJudge(model="m", api_key="k", transport=tr)
    s = j.score(d, samples=3)
    assert len(calls) == 4 and len(s.samples) == 2
    assert (s.seamless, s.distracting, s.ornate, s.total) == (12, 20, 12, 44)


def test_score_cost_is_this_score_not_the_running_total(tmp_path):
    d = _run_dir(tmp_path)
    reply = json.dumps(
        {"seamless": 10, "distracting": 20, "ornate": 12, "notes": "a", "top_fix": "x"}
    )
    tr, _calls = _transport([reply])
    j = OpenRouterJudge(model="m", api_key="k", transport=tr)
    first = j.score(d, samples=1).cost_usd
    second = j.score(d, samples=1).cost_usd
    assert first > 0 and second == pytest.approx(first)
    assert j.cost_usd == pytest.approx(first + second)


def test_score_raises_with_too_few_usable_samples(tmp_path):
    d = _run_dir(tmp_path)
    good = json.dumps(
        {"seamless": 10, "distracting": 20, "ornate": 12, "notes": "a", "top_fix": "x"}
    )
    # two of the three requested samples never produce parseable JSON (both attempts each)
    replies = ["garbage"] * 4 + [good]
    tr, _calls = _transport(replies)
    j = OpenRouterJudge(model="m", api_key="k", transport=tr)
    with pytest.raises(RuntimeError):
        j.score(d, samples=3)


def test_pairwise_randomises_sides_and_counts_votes(tmp_path):
    a, b = _run_dir(tmp_path, "a"), _run_dir(tmp_path, "b")
    # the model always says "A"; with randomised sides the real winner alternates
    tr, _calls = _transport(['{"prefer": "A", "why": "..."}'])
    j = OpenRouterJudge(model="m", api_key="k", transport=tr, rng=random.Random(0))
    res = j.pairwise(a, b, votes=4)
    assert res["prefer"] in ("a", "b", "tie") and len(res["votes"]) == 4
    assert {v["left"] for v in res["votes"]} == {"a", "b"}  # both orders were used


def test_resolve_api_key_env_then_pass():
    assert resolve_api_key({"OPENROUTER_API_KEY": "e"}) == "e"

    class R:
        returncode = 0
        stdout = "frompass\n"

    assert resolve_api_key({}, pass_cmd=lambda *a, **k: R()) == "frompass"
