import numpy as np
import pytest

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


def test_density_changes_the_ray_field(ellipse_mask):
    a = HaloRenderer(HaloParams(density=48), seed=5).render(ellipse_mask, 0.4)
    b = HaloRenderer(HaloParams(density=200), seed=5).render(ellipse_mask, 0.4)
    assert np.abs(a - b).mean() > 0.02


def test_rotation_animates_when_everything_else_is_static(ellipse_mask):
    static = {"speed": 0.0, "flicker": 0.0, "hue_speed": 0.0}
    spun = HaloRenderer(HaloParams(rotate=0.25, **static), seed=7)
    assert np.abs(spun.render(ellipse_mask, 0.0) - spun.render(ellipse_mask, 1.0)).mean() > 0.02
    still = HaloRenderer(HaloParams(rotate=0.0, **static), seed=7)
    assert np.array_equal(still.render(ellipse_mask, 0.0), still.render(ellipse_mask, 1.0))


def test_render_empty_mask_returns_black():
    out = HaloRenderer().render(np.zeros((90, 160), np.float32), 0.0)
    assert out.max() == 0.0


def test_render_tiny_mask_returns_black_not_nan():
    # a single full-res pixel can vanish under downscale + 0.5 threshold,
    # leaving an empty binary mask -- must not NaN out via xs.mean() on
    # an empty array.
    mask = np.zeros((10, 10), np.float32)
    mask[5, 5] = 1
    out = HaloRenderer().render(mask, 0.5)
    assert out.max() == 0.0
    assert not np.isnan(out).any()


@pytest.mark.parametrize("reach", [0.0, -5.0])
def test_non_positive_reach_is_rejected(reach):
    # reach divides the radius map and the falloff, so 0 renders NaN
    with pytest.raises(ValueError, match="reach"):
        HaloRenderer(HaloParams(reach=reach))


def test_no_jump_at_flicker_slice_boundary(ellipse_mask):
    # flicker=0.5 with TEX_N=8 -> a slice boundary every 0.25 s; the change
    # across the boundary must be no bigger than the change well inside a slice
    r = HaloRenderer(HaloParams(flicker=0.5, speed=0.0, rotate=0.0, hue_speed=0.0))
    dt = 1 / 60
    inside = np.abs(r.render(ellipse_mask, 0.10) - r.render(ellipse_mask, 0.10 + dt)).mean()
    across = np.abs(
        r.render(ellipse_mask, 0.25 - dt / 2) - r.render(ellipse_mask, 0.25 + dt / 2)
    ).mean()
    assert across < 2.0 * inside + 1e-4
