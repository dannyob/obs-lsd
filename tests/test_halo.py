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
    ref = ellipse_mask.shape[0]  # halo sizes in this mask's own pixels
    a = HaloRenderer(HaloParams(density=48, ref_height=ref), seed=5).render(ellipse_mask, 0.4)
    b = HaloRenderer(HaloParams(density=200, ref_height=ref), seed=5).render(ellipse_mask, 0.4)
    assert np.abs(a - b).mean() > 0.02


def test_rotation_animates_when_everything_else_is_static(ellipse_mask):
    # halo sizes in this mask's own pixels
    static = {"speed": 0.0, "flicker": 0.0, "hue_speed": 0.0, "ref_height": ellipse_mask.shape[0]}
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


def _head(h, w):
    import cv2

    m = np.zeros((h, w), np.float32)
    cv2.ellipse(m, (w // 2, h // 2), (int(h * 0.2), int(h * 0.27)), 0, 0, 360, 1.0, -1)
    return m


def test_halo_looks_the_same_at_1080p_as_at_720p():
    # reach, rim, speed and twist are in 720p pixels, so a bigger output gets
    # the same halo, not a relatively smaller one
    import cv2

    r = HaloRenderer(HaloParams())
    a = r.render(_head(720, 1280), 1.0)
    b = r.render(_head(1080, 1920), 1.0)
    b_small = cv2.resize(b, (1280, 720), interpolation=cv2.INTER_AREA)
    assert np.abs(a - b_small).mean() < 0.02
    # halo still reaches well out at 1080p: compare brightness at the same
    # relative spot outside the head
    assert b[540, 1920 // 2 + int(1080 * 0.2) + 150].max() > 0.05


def test_720p_render_is_unchanged_by_reference_height():
    r1 = HaloRenderer(HaloParams())
    r2 = HaloRenderer(HaloParams(ref_height=720.0))
    m = _head(720, 1280)
    assert np.array_equal(r1.render(m, 0.5), r2.render(m, 0.5))
