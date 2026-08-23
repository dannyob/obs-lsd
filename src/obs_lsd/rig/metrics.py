"""Per-frame stability numbers for a replayed run. Pure functions."""

from __future__ import annotations

import numpy as np

from obs_lsd.track import HeadBox


def halo_delta(prev_bgr: np.ndarray, cur_bgr: np.ndarray, head_mask: np.ndarray) -> float:
    outside = head_mask < 0.5
    if not outside.any():
        return 0.0
    d = np.abs(cur_bgr.astype(np.int16) - prev_bgr.astype(np.int16)).mean(axis=2)
    return float(d[outside].mean() / 255.0)


def frame_stats(boxes: list[HeadBox | None], timings: list[dict], deltas: list[float]) -> dict:
    n = len(boxes)
    misses = sum(1 for b in boxes if b is None)
    d = np.asarray(deltas, dtype=float)
    med = float(np.median(d)) if d.size else 0.0
    p95 = float(np.percentile(d, 95)) if d.size else 0.0
    jumps = int((d > 3.0 * med).sum()) if d.size and med > 0 else 0
    pts = [(b.cx, b.cy) for b in boxes if b is not None]
    if len(pts) >= 2:
        p = np.asarray(pts)
        step = np.hypot(*np.diff(p, axis=0).T)
        centre_p95 = float(np.percentile(step, 95))
    else:
        centre_p95 = 0.0
    heights = [b.height for b in boxes if b is not None]
    height_std = float(np.std(heights)) if heights else 0.0

    def mean_ms(key: str) -> float:
        vals = [t[key] for t in timings if key in t]
        return float(np.mean(vals)) if vals else 0.0

    return {
        "frames": n,
        "no_head": misses,
        "no_head_frac": misses / n if n else 0.0,
        "delta_median": med,
        "delta_p95": p95,
        "jump_count": jumps,
        "jump_frac": jumps / n if n else 0.0,
        "centre_jump_p95": centre_p95,
        "height_std": height_std,
        "ms_total_mean": mean_ms("total"),
        "ms_segment_mean": mean_ms("segment"),
        "ms_halo_mean": mean_ms("halo"),
        "ms_compose_mean": mean_ms("compose"),
    }


def metrics_table(stats: dict) -> str:
    return "\n".join(
        f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in stats.items()
    )
