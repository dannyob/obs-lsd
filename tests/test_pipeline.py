import cv2
import numpy as np

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
    cfg = PipelineConfig(width=320, height=180, head_frac=0.5, halo=HaloParams(reach=40.0))
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
    cfg = PipelineConfig(width=320, height=180, fade_seconds=0.5)
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
    cfg = PipelineConfig(width=320, height=180, head_frac=0.5)
    seg = FakeSeg()
    p = Pipeline(cfg, segmenter=seg)
    frame = np.full((240, 320, 3), (10, 200, 10), np.uint8)
    p.step(frame, 0.0)
    p.frozen = True
    seg.present = False
    out = p.step(frame, 0.5)
    assert tuple(out[90, 160]) == (10, 200, 10)  # still shows head colour, not faded
    assert "segment" in p.timings
