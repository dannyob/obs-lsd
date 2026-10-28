"""Psychedelic ray halo around a head mask.

Rays are read from a tileable 2D noise texture indexed by (angle, radius) via
cv2.remap, scrolled outward over time; two texture rows of a small stack are
blended for flicker. Everything is numpy/OpenCV, no per-pixel Python.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


def tileable_noise(h: int, w: int, octaves: int = 4, seed: int = 0) -> np.ndarray:
    """Value noise that wraps on both axes, in [0,1]. Built by upsampling
    random grids with cv2.resize on a 3x3 tiled copy and cropping the centre."""
    rng = np.random.default_rng(seed)
    out = np.zeros((h, w), np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        gh, gw = max(2, 4 * 2**o), max(2, 8 * 2**o)
        g = rng.random((gh, gw), dtype=np.float32)
        big = np.tile(g, (3, 3))
        up = cv2.resize(big, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)
        out += amp * up[h : 2 * h, w : 2 * w]
        total += amp
        amp *= 0.5
    out /= total
    out -= out.min()
    out /= max(out.max(), 1e-6)
    return out.astype(np.float32)


@dataclass
class HaloParams:
    reach: float = 260.0
    density: float = 48.0
    speed: float = 35.0
    rotate: float = 0.015  # revolutions per second the ray field turns
    flicker: float = 0.05  # noise-slice cycles per second (x TEX_N slices); keep small
    hue_speed: float = 0.03
    hue_twist: float = 0.0015
    sharpness: float = 2.5
    rim_width: float = 6.0
    rim_gain: float = 0.3
    spill_gain: float = 1.0  # halo colour wrapped onto the head edge
    spill_width: float = 30.0  # px inward that the wrap reaches
    scale: float = 0.5


class HaloRenderer:
    TEX_H, TEX_W, TEX_N = 256, 1024, 8  # radius rows, angle cols, flicker slices

    def __init__(self, params: HaloParams | None = None, seed: int = 0):
        self.p = params or HaloParams()
        if self.p.reach <= 0:
            raise ValueError(f"halo reach must be > 0 pixels, got {self.p.reach}")
        self.tex = np.stack(
            [tileable_noise(self.TEX_H, self.TEX_W, seed=seed * 100 + i) for i in range(self.TEX_N)]
        )
        self._grids: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

    def _grid(self, h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
        """Pixel x and y coordinates as float32, cached per render size."""
        g = self._grids.get((h, w))
        if g is None:
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            g = self._grids[(h, w)] = (xx, yy)
        return g

    def render(self, mask: np.ndarray, t: float) -> np.ndarray:
        H, W = mask.shape
        s = self.p.scale
        h, w = max(1, int(H * s)), max(1, int(W * s))
        m = cv2.resize(mask, (w, h), interpolation=cv2.INTER_AREA) if s != 1.0 else mask
        if m.max() <= 0.0:
            return np.zeros((H, W, 3), np.float32)
        binary = (m > 0.5).astype(np.uint8)
        mom = cv2.moments(binary, binaryImage=True)
        if mom["m00"] == 0:
            # mask survived the m.max() check but vanished under downscale +
            # 0.5 threshold (e.g. a single full-res pixel) -- nothing to draw
            return np.zeros((H, W, 3), np.float32)
        cx, cy = mom["m10"] / mom["m00"], mom["m01"] / mom["m00"]
        outside = 1 - binary
        # distance (in full-res px) from the head contour, outward
        dist = cv2.distanceTransform(outside, cv2.DIST_L2, 3)
        dist *= np.float32(1.0 / s)
        xx, yy = self._grid(h, w)
        theta = np.arctan2(yy - cy, xx - cx)  # -pi..pi
        # Everything below is a full half-res float32 pass; each is written in
        # place or through cv2, which is 3-5x faster than numpy's temporaries.
        two_pi = 2.0 * np.pi
        theta_n = theta * np.float32(1.0 / two_pi)  # one temporary, reused below
        theta_n += np.float32(t * self.p.rotate + 0.5)
        # Texture lookup: angle wraps around TEX_W, radius scrolls with time.
        # The coarsest noise octave has 8 cells across TEX_W, so scaling the
        # angle by density/8 makes `density` the number of rays around the head.
        map_x = theta_n * np.float32((self.p.density / 8.0) * self.TEX_W)
        np.mod(map_x, self.TEX_W, out=map_x)
        ky = self.TEX_H / self.p.reach
        map_y = dist * np.float32(ky)
        map_y -= np.float32(t * self.p.speed * ky)
        np.mod(map_y, self.TEX_H, out=map_y)
        f = t * self.p.flicker * self.TEX_N
        i0 = int(f) % self.TEX_N
        i1 = (i0 + 1) % self.TEX_N
        frac = float(f - int(f))
        wrap = cv2.BORDER_WRAP
        n0 = cv2.remap(self.tex[i0], map_x, map_y, cv2.INTER_LINEAR, borderMode=wrap)
        n1 = cv2.remap(self.tex[i1], map_x, map_y, cv2.INTER_LINEAR, borderMode=wrap)
        n = cv2.addWeighted(n0, 1.0 - frac, n1, frac, 0.0)
        # angular fine detail: a second, slow lookup that modulates ray brightness
        # around the circle; fixed rate so it does not fight `density`. Cross-faded
        # between slices like `n`, otherwise it snaps to a new texture every tick.
        map_x_fine = theta_n * np.float32(3.0 * self.TEX_W)
        np.mod(map_x_fine, self.TEX_W, out=map_x_fine)
        f0 = cv2.remap(
            self.tex[(i0 + 3) % self.TEX_N], map_x_fine, map_y, cv2.INTER_LINEAR, borderMode=wrap
        )
        f1 = cv2.remap(
            self.tex[(i1 + 3) % self.TEX_N], map_x_fine, map_y, cv2.INTER_LINEAR, borderMode=wrap
        )
        n_fine = cv2.addWeighted(f0, 1.0 - frac, f1, frac, 0.5)  # + 0.5 folded in
        rays = cv2.multiply(n, n_fine)
        np.clip(rays, 0, 1, out=rays)
        cv2.pow(rays, self.p.sharpness, dst=rays)
        falloff = dist * np.float32(-1.0 / self.p.reach)
        falloff += np.float32(1.0)
        np.clip(falloff, 0, 1, out=falloff)
        cv2.pow(falloff, 1.5, dst=falloff)
        value = cv2.multiply(rays, falloff)
        rim = dist * np.float32(-1.0 / self.p.rim_width)
        cv2.exp(rim, dst=rim)
        rim *= np.float32(self.p.rim_gain)
        hue = theta * np.float32(1.0 / two_pi)
        hue += np.float32(t * self.p.hue_speed)
        if self.p.hue_twist:
            cv2.scaleAdd(dist, self.p.hue_twist, hue, dst=hue)
        np.mod(hue, 1.0, out=hue)
        hsv = np.empty((h, w, 3), np.float32)
        np.multiply(hue, np.float32(360.0), out=hsv[..., 0])
        np.subtract(np.float32(1.0), rim, out=hsv[..., 1])
        np.clip(hsv[..., 1], 0.2, 1.0, out=hsv[..., 1])
        np.add(value, rim, out=hsv[..., 2])
        np.clip(hsv[..., 2], 0.0, 1.0, out=hsv[..., 2])
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        rgb *= outside[..., None]  # black inside the head
        if s != 1.0:
            # No second full-res zeroing pass here: it costs ~2.3 ms at 720p and
            # compose already multiplies the halo by (1 - mask).
            rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_LINEAR)
        # No final clip: HSV2RGB of V in [0,1] and a linear resize both stay in [0,1].
        return rgb
