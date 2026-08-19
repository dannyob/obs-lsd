import numpy as np

from obs_lsd.halo import HaloParams, HaloRenderer, tileable_noise


def test_tileable_noise_wraps_and_ranges():
    n = tileable_noise(64, 128, seed=1)
    assert n.shape == (64, 128) and n.dtype == np.float32
    assert 0.0 <= n.min() and n.max() <= 1.0 and n.std() > 0.05
    # continuity across the seam: last col ~ first col, last row ~ first row
    assert np.abs(n[:, -1] - n[:, 0]).mean() < np.abs(n[:, 64] - n[:, 0]).mean()
    assert np.abs(n[-1, :] - n[0, :]).mean() < np.abs(n[32, :] - n[0, :]).mean()


def test_render_shape_dtype_range(ellipse_mask):
    out = HaloRenderer().render(ellipse_mask, 0.0)
    assert out.shape == (360, 640, 3) and out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_render_is_black_inside_head_and_bright_near_contour(ellipse_mask):
    out = HaloRenderer(HaloParams(rim_gain=1.0)).render(ellipse_mask, 0.3)
    inside = out[180, 320]
    assert inside.max() < 0.02
    ring = out[180 - 90 : 180 - 82, 320].max()  # just outside the top of the ellipse
    assert ring > 0.3
    far = out[5, 5].max()  # ~ (315,175) px away, beyond reach=260
    assert far < 0.05


def test_render_deterministic_and_time_varying(ellipse_mask):
    r = HaloRenderer(seed=3)
    a = r.render(ellipse_mask, 1.0)
    b = HaloRenderer(seed=3).render(ellipse_mask, 1.0)
    c = r.render(ellipse_mask, 1.5)
    assert np.array_equal(a, b)
    assert np.abs(a - c).mean() > 0.005


def test_render_empty_mask_returns_black():
    out = HaloRenderer().render(np.zeros((90, 160), np.float32), 0.0)
    assert out.max() == 0.0
