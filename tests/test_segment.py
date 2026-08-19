from pathlib import Path

import cv2
import numpy as np
import pytest

from obs_lsd import MODELS_DIR
from obs_lsd.segment import HeadSegmenter, ModelsMissing, chin_halfplane, head_from_masks

PORTRAIT = Path(__file__).parent / "data" / "portrait.jpg"
have_models = (MODELS_DIR / "face_landmarker.task").exists()


def test_chin_halfplane_level():
    hp = chin_halfplane((100, 100), chin_xy=(50, 60), roll_deg=0.0, margin_px=5)
    assert hp[50, 50] == 1.0 and hp[64, 50] == 1.0 and hp[66, 50] == 0.0
    assert hp[66, 10] == 0.0 and hp[50, 90] == 1.0


def test_chin_halfplane_rolled():
    hp = chin_halfplane((100, 100), chin_xy=(50, 50), roll_deg=45.0, margin_px=0)
    # 'down' at 45deg roll is (-sin45, cos45) -> points down-LEFT of the chin.
    # d = (x-50)*(-sin45) + (y-50)*cos45; a down-left point has d > 0 (cut),
    # a down-right point has d < 0 (kept). These pick a side, so they'd fail
    # if the rotation sign in chin_halfplane were flipped.
    assert hp[70, 30] == 0.0  # (x=30,y=70): d = 28.3 > 0 -> cut
    assert hp[60, 80] == 1.0  # (x=80,y=60): d = -14.1 <= 0 -> kept
    assert hp[20, 50] == 1.0  # straight up from chin -> kept


def test_head_from_masks_unions_and_cuts_below_chin():
    hair = np.zeros((10, 10), np.float32)
    hair[0:3, :] = 1
    face = np.zeros((10, 10), np.float32)
    face[3:8, :] = 1
    hp = np.ones((10, 10), np.float32)
    hp[6:, :] = 0
    m = head_from_masks(hair, face, hp, feather=0)
    assert m[1, 5] == 1.0 and m[4, 5] == 1.0
    assert m[7, 5] == 0.0 and m[9, 5] == 0.0


def test_missing_models_raises(tmp_path):
    with pytest.raises(ModelsMissing):
        HeadSegmenter(models_dir=tmp_path)


@pytest.mark.skipif(not (have_models and PORTRAIT.exists()), reason="run: make models testdata")
def test_portrait_mask_covers_face_not_neck():
    img = cv2.imread(str(PORTRAIT))
    seg = HeadSegmenter()
    try:
        hm = seg.mask(img, 0)
    finally:
        seg.close()
    assert hm is not None
    H, W = img.shape[:2]
    assert hm.mask.shape == (H, W)
    assert 50 < hm.box.height < H
    # centre of the box is inside the mask; well below the box, nothing
    assert hm.mask[int(hm.box.cy), int(hm.box.cx)] > 0.5
    below = int(min(H - 1, hm.box.cy + hm.box.height))
    assert hm.mask[below, int(hm.box.cx)] < 0.1


@pytest.mark.skipif(not have_models, reason="run: make models")
def test_blank_frame_returns_none_with_no_prior_frame():
    # cheap no-portrait variant: with no held mask yet, a miss is an
    # immediate None (the hold branch has nothing to hold).
    seg = HeadSegmenter(hold_frames=2)
    try:
        blank = np.zeros((240, 320, 3), np.uint8)
        assert seg.mask(blank, 0) is None
    finally:
        seg.close()


@pytest.mark.skipif(not (have_models and PORTRAIT.exists()), reason="run: make models testdata")
def test_hold_frames_then_none_then_recovers():
    img = cv2.imread(str(PORTRAIT))
    blank = np.zeros((240, 320, 3), np.uint8)
    seg = HeadSegmenter(hold_frames=2)
    try:
        hm0 = seg.mask(img, 0)
        assert hm0 is not None
        # two misses within hold_frames=2: both return the SAME held object
        hm1 = seg.mask(blank, 33)
        hm2 = seg.mask(blank, 66)
        assert hm1 is hm0
        assert hm2 is hm0
        # third miss exceeds the hold -> None
        hm3 = seg.mask(blank, 99)
        assert hm3 is None
        # a good frame after that resets the missing counter
        hm4 = seg.mask(img, 132)
        assert hm4 is not None
    finally:
        seg.close()
