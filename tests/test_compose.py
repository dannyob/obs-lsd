import numpy as np

from obs_lsd.compose import compose


def test_compose_head_over_black_with_halo():
    frame = np.full((4, 4, 3), 200, np.uint8)
    mask = np.zeros((4, 4), np.float32)
    mask[1:3, 1:3] = 1.0
    halo = np.zeros((4, 4, 3), np.float32)
    halo[0, 0] = (1.0, 0.0, 0.0)  # red in RGB
    out = compose(frame, mask, halo)
    assert out.dtype == np.uint8 and out.shape == (4, 4, 3)
    assert tuple(out[1, 1]) == (200, 200, 200)  # head pixels kept
    assert tuple(out[0, 0]) == (0, 0, 255)  # RGB red -> BGR
    assert tuple(out[3, 3]) == (0, 0, 0)  # black background


def test_compose_halo_gain_and_soft_mask():
    frame = np.full((2, 2, 3), 100, np.uint8)
    mask = np.array([[0.5, 0.0], [0.0, 0.0]], np.float32)
    halo = np.ones((2, 2, 3), np.float32)
    out = compose(frame, mask, halo, halo_gain=0.5)
    assert tuple(out[1, 1]) == (128, 128, 128)  # 0.5 * 255 rounded
    assert 100 < out[0, 0, 0] < 200  # blend of head 100 and halo 0.5*255*0.5


def test_spill_wraps_halo_colour_onto_head_edge_only():
    import cv2

    frame = np.full((200, 200, 3), 100, np.uint8)
    mask = np.zeros((200, 200), np.float32)
    cv2.ellipse(mask, (100, 100), (60, 80), 0, 0, 360, 1.0, -1)
    halo = np.zeros((200, 200, 3), np.float32)
    halo[..., 0] = 1.0  # RGB red everywhere (compose ignores it under the head)
    plain = compose(frame, mask, halo)
    spilled = compose(frame, mask, halo, spill_gain=1.0, spill_width=10.0)
    # spill_gain=0 path is byte-identical to the old behaviour
    assert np.array_equal(plain, compose(frame, mask, halo, spill_gain=0.0))
    edge = (100, 100 - 75)  # 5 px inside the top of the ellipse (col, row)
    assert spilled[edge[1], edge[0], 2] > plain[edge[1], edge[0], 2] + 40  # red channel (BGR idx 2)
    assert spilled[edge[1], edge[0], 0] == plain[edge[1], edge[0], 0]  # blue untouched
    assert abs(int(spilled[100, 100, 2]) - int(plain[100, 100, 2])) <= 1  # centre unchanged


def test_compose_matches_float_reference_within_one_level():
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, (32, 48, 3), np.uint8)
    mask = rng.random((32, 48), dtype=np.float32)
    halo = rng.random((32, 48, 3), dtype=np.float32)
    for gain in (1.0, 0.4):
        ref = frame * mask[..., None] + halo[..., ::-1] * (255.0 * gain) * (1 - mask)[..., None]
        ref = np.clip(ref + 0.5, 0, 255).astype(np.uint8)
        out = compose(frame, mask, halo, halo_gain=gain)
        assert np.abs(out.astype(int) - ref.astype(int)).max() <= 1


def test_compose_returns_a_fresh_frame_each_call():
    frame = np.full((4, 4, 3), 10, np.uint8)
    mask = np.ones((4, 4), np.float32)
    halo = np.zeros((4, 4, 3), np.float32)
    a = compose(frame, mask, halo)
    b = compose(frame * 2, mask, halo)
    assert tuple(a[0, 0]) == (10, 10, 10) and tuple(b[0, 0]) == (20, 20, 20)
