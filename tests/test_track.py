import numpy as np
import pytest

from obs_lsd.track import HeadBox, Tracker, Transform


def test_transform_matrix_and_apply():
    t = Transform(scale=2.0, tx=10.0, ty=-5.0)
    m = t.matrix()
    assert m.shape == (2, 3) and m.dtype == np.float32
    assert t.apply(3, 4) == pytest.approx((16.0, 3.0))


def test_first_update_snaps_head_to_center_at_target_height():
    tr = Tracker(out_size=(1280, 720), target_frac=0.5)
    t = tr.update(HeadBox(cx=100, cy=50, height=100))
    assert t.apply(100, 50) == pytest.approx((640, 360))
    assert t.scale == pytest.approx(3.6)  # 0.5*720/100


def test_smoothing_moves_gradually_then_converges():
    tr = Tracker(out_size=(1280, 720), alpha_pos=0.5, alpha_scale=0.5)
    tr.update(HeadBox(cx=0, cy=0, height=100))
    t1 = tr.update(HeadBox(cx=100, cy=0, height=100))
    # after one step at alpha 0.5, the tracked cx is 50, so camera x=50 maps to center
    assert t1.apply(50, 0)[0] == pytest.approx(640)
    for _ in range(40):
        t = tr.update(HeadBox(cx=100, cy=0, height=100))
    assert t.apply(100, 0)[0] == pytest.approx(640, abs=1e-3)


def test_none_holds_last_transform_and_counts_missing():
    tr = Tracker(out_size=(640, 360))
    t0 = tr.update(HeadBox(cx=10, cy=10, height=50))
    assert tr.frames_missing == 0
    t1 = tr.update(None)
    t2 = tr.update(None)
    assert (t1.scale, t1.tx, t1.ty) == (t0.scale, t0.tx, t0.ty) == (t2.scale, t2.tx, t2.ty)
    assert tr.frames_missing == 2
    tr.update(HeadBox(cx=10, cy=10, height=50))
    assert tr.frames_missing == 0


def test_none_before_any_head_gives_identity_centered():
    tr = Tracker(out_size=(640, 360))
    t = tr.update(None)
    assert t.scale == 1.0 and t.tx == 0.0 and t.ty == 0.0
