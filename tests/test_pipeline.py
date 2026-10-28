import cv2
import numpy as np
import pytest

from obs_lsd.halo import HaloParams
from obs_lsd.pipeline import Pipeline, PipelineConfig
from obs_lsd.segment import HeadMask
from obs_lsd.track import HeadBox


class FakeSeg:
    def __init__(self, present=True):
        self.present = present

    def mask(self, frame, ts):
        if not self.present:
            return None
        h, w = frame.shape[:2]
        m = np.zeros((h, w), np.float32)
        cv2.ellipse(m, (100, 120), (30, 40), 0, 0, 360, 1.0, -1)
        return HeadMask(mask=m, box=HeadBox(cx=100, cy=120, height=80))

    def close(self):
        pass


def test_step_centres_head_and_draws_halo():
    # reach=40 keeps the corner black at this size; spill off because the tiny
    # test head is only ~90 px tall, so a 30 px wrap would reach its centre
    cfg = PipelineConfig(
        width=320, height=180, head_frac=0.5, halo=HaloParams(reach=40.0, spill_gain=0.0)
    )
    p = Pipeline(cfg, segmenter=FakeSeg())
    frame = np.full((240, 320, 3), (10, 200, 10), np.uint8)
    out = p.step(frame, 0.0)
    assert out.shape == (180, 320, 3) and out.dtype == np.uint8
    assert tuple(out[90, 160]) == (10, 200, 10)  # head colour at centre
    # head height should be ~0.5*180 = 90 px: green at 90-40, black beyond
    assert out[90 - 40, 160, 1] == 200 and out[90 - 55, 160, 1] < 200
    assert out[2, 2].max() == 0  # far corner black
    assert set(p.timings) >= {"segment", "warp", "halo", "compose"}


def test_lost_head_fades_halo_to_black():
    cfg = PipelineConfig(width=320, height=180, fade_seconds=0.5, seg_lag=0)
    seg = FakeSeg()
    p = Pipeline(cfg, segmenter=seg)
    frame = np.full((240, 320, 3), (10, 200, 10), np.uint8)
    full = p.step(frame, 0.0)
    seg.present = False
    # first missed frame: the fade timer starts *this* step, so gain is still 1.0
    bright = p.step(frame, 0.05)
    # partway through the 0.5s fade window (elapsed since first miss = 0.25s -> gain 0.5)
    faded = p.step(frame, 0.3)
    gone = p.step(frame, 2.0).max()
    assert bright.max() > 0 and gone == 0
    # head cutout should dim gradually with the halo, not snap to black
    assert 0 < faded[90, 160, 1] < full[90, 160, 1]


def test_no_head_yet_returns_black_and_sets_timings():
    cfg = PipelineConfig(width=320, height=180)
    p = Pipeline(cfg, segmenter=FakeSeg(present=False))
    out = p.step(np.zeros((240, 320, 3), np.uint8), 0.0)
    assert out.shape == (180, 320, 3) and out.dtype == np.uint8
    assert out.max() == 0
    assert set(p.timings) >= {"segment", "warp", "halo", "compose", "total"}


def test_frozen_reuses_last_mask():
    cfg = PipelineConfig(
        width=320, height=180, head_frac=0.5, halo=HaloParams(spill_gain=0.0), seg_lag=0
    )
    seg = FakeSeg()
    p = Pipeline(cfg, segmenter=seg)
    frame = np.full((240, 320, 3), (10, 200, 10), np.uint8)
    p.step(frame, 0.0)
    p.frozen = True
    seg.present = False
    out = p.step(frame, 0.5)
    assert tuple(out[90, 160]) == (10, 200, 10)  # still shows head colour, not faded
    assert "segment" in p.timings


def test_float_drifts_head_slowly_around_centre():

    base = {
        "width": 320,
        "height": 180,
        "head_frac": 0.5,
        "halo": HaloParams(reach=40.0, spill_gain=0.0),
    }
    still = Pipeline(PipelineConfig(float_amp=0.0, **base), segmenter=FakeSeg())
    drift = Pipeline(PipelineConfig(float_amp=0.1, float_period=8.0, **base), segmenter=FakeSeg())
    frame = np.full((240, 320, 3), (10, 200, 10), np.uint8)
    # at t=0 the x term is zero, so the drifting head is still horizontally centred
    assert drift.float_offset(0.0)[0] == 0.0
    # a quarter of the x period later it has moved right by about amp*height
    t = 8.0 / 4
    a, b = still.step(frame, t), drift.step(frame, t)
    expected_dx = 0.1 * 180

    # find the head's horizontal centre of mass on the centre row (green channel == 200)
    def cx(img):
        cols = np.nonzero(img[:, :, 1] == 200)[1]
        return cols.mean()

    assert abs((cx(b) - cx(a)) - expected_dx) < 3.0
    assert drift.float_offset(t)[0] == pytest.approx(expected_dx, abs=1e-6)
    assert still.float_offset(t) == (0.0, 0.0)


def test_step_exposes_last_box_and_mask():
    cfg = PipelineConfig(width=320, height=180, halo=HaloParams(spill_gain=0.0))
    p = Pipeline(cfg, segmenter=FakeSeg())
    assert p.last_box is None and p.last_mask_out is None
    p.step(np.zeros((240, 320, 3), np.uint8), 0.0)
    assert p.last_box.height == 80 and p.last_mask_out.shape == (180, 320)


class FrameTagSeg(FakeSeg):
    """Mask centred on x = frame[0, 0, 0], so a test can tell which frame a mask came from."""

    def mask(self, frame, ts):
        h, w = frame.shape[:2]
        x = int(frame[0, 0, 0])
        m = np.zeros((h, w), np.float32)
        cv2.ellipse(m, (x, 120), (30, 40), 0, 0, 360, 1.0, -1)
        return HeadMask(mask=m, box=HeadBox(cx=x, cy=120, height=80))


def _tagged(x):
    f = np.full((240, 320, 3), 50, np.uint8)
    f[0, 0, 0] = x
    return f


def test_segment_thread_composes_with_the_previous_frames_mask():
    p = Pipeline(PipelineConfig(width=320, height=180, seg_lag=1), segmenter=FrameTagSeg())
    try:
        p.step(_tagged(100), 0.0)
        assert p.last_box.cx == 100  # first frame waits for its own mask
        p.step(_tagged(140), 1 / 30)
        assert p.last_box.cx == 100  # one frame behind
        p.step(_tagged(180), 2 / 30)
        assert p.last_box.cx == 140
    finally:
        p.close()


def test_segment_inline_has_no_lag():
    p = Pipeline(PipelineConfig(width=320, height=180, seg_lag=0), segmenter=FrameTagSeg())
    try:
        p.step(_tagged(100), 0.0)
        p.step(_tagged(140), 1 / 30)
        assert p.last_box.cx == 140
    finally:
        p.close()


def test_segment_thread_failure_surfaces_in_step():
    class Boom(FakeSeg):
        def mask(self, frame, ts):
            raise RuntimeError("segmenter died")

    p = Pipeline(PipelineConfig(width=320, height=180, seg_lag=1), segmenter=Boom())
    try:
        with pytest.raises(RuntimeError, match="segmenter died"):
            p.step(_tagged(100), 0.0)
    finally:
        p.close()
