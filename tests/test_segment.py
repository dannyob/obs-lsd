from pathlib import Path

import cv2
import numpy as np
import pytest

from obs_lsd import MODELS_DIR
from obs_lsd.segment import (
    HeadSegmenter,
    ModelsMissing,
    chin_halfplane,
    head_from_masks,
    mask_box,
)

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


def test_head_from_masks_normalises_confidence():
    # the segmenter's confidence masks are soft: the head should reach full
    # opacity and a low-confidence wash over the background should be cut
    hair = np.full((6, 6), 0.10, np.float32)
    face = np.zeros((6, 6), np.float32)
    face[2:4, :] = 0.90
    m = head_from_masks(hair, face, np.ones((6, 6), np.float32), feather=0)
    assert m[0, 0] == 0.0
    assert m[2, 3] == 1.0


@pytest.mark.skipif(not have_models, reason="run: make models")
def test_gpu_and_cpu_delegates_agree_on_the_portrait():
    img = cv2.imread(str(PORTRAIT))
    gpu, cpu = HeadSegmenter(delegate="gpu"), HeadSegmenter(delegate="cpu")
    try:
        a, b = gpu.mask(img, 0), cpu.mask(img, 0)
    finally:
        gpu.close()
        cpu.close()
    assert a is not None and b is not None
    ia, ib = a.mask > 0.5, b.mask > 0.5
    iou = (ia & ib).sum() / (ia | ib).sum()
    assert iou > 0.9, iou


def test_unknown_delegate_is_rejected():
    with pytest.raises(ValueError):
        HeadSegmenter(delegate="tpu")


def test_mask_box_matches_nonzero_arithmetic():
    rng = np.random.default_rng(1)
    m = np.zeros((120, 160), np.float32)
    m[30:91, 40:101] = rng.uniform(0.6, 1.0, (61, 61))  # solid blob
    m[10:20, 10:20] = 0.3  # faint wash, below the 0.5 line
    ys, xs = np.nonzero(m > 0.5)
    box = mask_box(m)
    assert box is not None
    assert abs(box.cx - xs.mean()) < 1e-6 and abs(box.cy - ys.mean()) < 1e-6
    assert box.height == ys.max() - ys.min() + 1 == 61


def test_mask_box_none_when_too_small():
    m = np.zeros((50, 50), np.float32)
    m[10:13, 10:13] = 1.0  # 9 pixels
    assert mask_box(m) is None
    assert mask_box(m, min_pixels=9) is not None


def test_head_from_masks_leaves_inputs_alone_and_returns_float32():
    hair = np.full((8, 8), 0.9, np.float64)
    face = np.zeros((8, 8), np.float64)
    hp = np.ones((8, 8), np.float32)
    m = head_from_masks(hair, face, hp, feather=0)
    assert m.dtype == np.float32
    assert hair[0, 0] == 0.9  # not mutated in place


@pytest.mark.skipif(not have_models, reason="run: make models")
def test_small_inference_width_gives_full_size_mask_close_to_full_res():
    img = cv2.imread(str(PORTRAIT))
    H, W = img.shape[:2]
    full, small = HeadSegmenter(infer_width=0), HeadSegmenter(infer_width=256)
    try:
        a, b = full.mask(img, 0), small.mask(img, 0)
    finally:
        full.close()
        small.close()
    assert a is not None and b is not None
    assert b.mask.shape == (H, W)
    assert b.mask.dtype == np.float32
    ia, ib = a.mask > 0.5, b.mask > 0.5
    iou = (ia & ib).sum() / (ia | ib).sum()
    assert iou > 0.9, iou
    assert abs(a.box.height - b.box.height) < 0.05 * a.box.height


@pytest.mark.skipif(not have_models, reason="run: make models")
def test_segmenter_reopens_its_tasks_in_the_background():
    # MediaPipe's GPU tasks leak an IOSurface per frame on macOS until the
    # task is closed, and the process aborts after a few thousand frames.
    # The segmenter builds replacement tasks on a worker thread every
    # reopen_every frames and swaps them in once they are warm.
    import time

    img = cv2.imread(str(PORTRAIT))
    seg = HeadSegmenter(reopen_every=4)
    try:
        opens = seg.opens
        t = 0
        for _ in range(4):
            seg.mask(img, t)
            t += 33
        deadline = time.monotonic() + 30
        while seg.reopens == 0 and time.monotonic() < deadline:
            seg.mask(img, t)
            t += 33
            time.sleep(0.01)
        assert seg.reopens == 1
        assert seg.opens == opens + 1
        hm = seg.mask(img, t)
        assert hm is not None and hm.box.height > 50
    finally:
        seg.close()


def test_segmenter_reopen_zero_never_reopens(monkeypatch):
    if not have_models:
        pytest.skip("run: make models")
    img = cv2.imread(str(PORTRAIT))
    seg = HeadSegmenter(reopen_every=0)
    try:
        for i in range(6):
            seg.mask(img, i * 33)
        assert seg.reopens == 0 and seg.opens == 1
    finally:
        seg.close()
