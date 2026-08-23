import numpy as np

from obs_lsd.rig.metrics import frame_stats, halo_delta, metrics_table
from obs_lsd.track import HeadBox


def test_halo_delta_ignores_head_pixels():
    a = np.zeros((4, 4, 3), np.uint8)
    b = a.copy()
    b[0, 0] = 255  # outside head
    b[2, 2] = 255  # inside head
    mask = np.zeros((4, 4), np.float32)
    mask[2, 2] = 1.0
    assert abs(halo_delta(a, b, mask) - (255 / 15) / 255) < 1e-6
    assert halo_delta(a, b, np.ones((4, 4), np.float32)) == 0.0


def test_frame_stats_counts_jumps_and_misses():
    boxes = [HeadBox(100, 100, 50), HeadBox(101, 100, 50), None, HeadBox(110, 100, 52)]
    timings = [{"total": 10.0, "segment": 5.0, "halo": 3.0, "compose": 1.0}] * 4
    deltas = [0.01, 0.01, 0.01, 0.10]  # last one is > 3x median -> one jump
    s = frame_stats(boxes, timings, deltas)
    assert s["frames"] == 4 and s["no_head"] == 1 and abs(s["no_head_frac"] - 0.25) < 1e-9
    assert s["jump_count"] == 1 and abs(s["jump_frac"] - 0.25) < 1e-9
    assert abs(s["delta_median"] - 0.01) < 1e-9
    assert s["centre_jump_p95"] > 1.0  # 101->110 jump dominates (None skipped)
    assert s["ms_total_mean"] == 10.0 and s["ms_segment_mean"] == 5.0


def test_frame_stats_empty_and_table():
    s = frame_stats([], [], [])
    assert s["frames"] == 0 and s["no_head_frac"] == 0.0 and s["jump_count"] == 0
    t = metrics_table(s)
    assert "frames: 0" in t and "ms_total_mean" in t
