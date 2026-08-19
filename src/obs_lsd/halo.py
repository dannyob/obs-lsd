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
    rim_gain: float = 0.7
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

    def render(self, mask: np.ndarray, t: float) -> np.ndarray:
        H, W = mask.shape
        s = self.p.scale
        h, w = max(1, int(H * s)), max(1, int(W * s))
        m = cv2.resize(mask, (w, h), interpolation=cv2.INTER_AREA) if s != 1.0 else mask
        if m.max() <= 0.0:
            return np.zeros((H, W, 3), np.float32)
        binary = (m > 0.5).astype(np.uint8)
        ys, xs = np.nonzero(binary)
        if xs.size == 0:
            # mask survived the m.max() check but vanished under downscale +
            # 0.5 threshold (e.g. a single full-res pixel) -- nothing to draw
            return np.zeros((H, W, 3), np.float32)
        # distance (in full-res px) from the head contour, outward
        dist = cv2.distanceTransform(1 - binary, cv2.DIST_L2, 3).astype(np.float32) / s
        cx, cy = np.float32(xs.mean()), np.float32(ys.mean())
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        theta = np.arctan2(yy - cy, xx - cx)  # -pi..pi
        # Texture lookup: angle wraps around TEX_W, radius scrolls with time.
        # The coarsest noise octave has 8 cells across TEX_W, so scaling the
        # angle by density/8 makes `density` the number of rays around the head.
        theta_n = (theta + 2 * np.pi * t * self.p.rotate) / (2 * np.pi) + 0.5
        map_x = (theta_n * (self.p.density / 8.0) * self.TEX_W) % self.TEX_W
        map_y = (
            (dist * (self.TEX_H / self.p.reach)) - t * self.p.speed * (self.TEX_H / self.p.reach)
        ) % self.TEX_H
        f = t * self.p.flicker * self.TEX_N
        i0 = int(f) % self.TEX_N
        i1 = (i0 + 1) % self.TEX_N
        frac = np.float32(f - int(f))
        n0 = cv2.remap(self.tex[i0], map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        n1 = cv2.remap(self.tex[i1], map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        n = n0 * (1 - frac) + n1 * frac
        # angular fine detail: a second, slow lookup that modulates ray brightness
        # around the circle; fixed rate so it does not fight `density`
        n_fine = cv2.remap(
            self.tex[(i0 + 3) % self.TEX_N],
            (theta_n * 3.0 * self.TEX_W) % self.TEX_W,
            map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_WRAP,
        )
        rays = np.clip(n * (0.5 + n_fine), 0, 1) ** self.p.sharpness
        falloff = np.clip(1.0 - dist / self.p.reach, 0, 1) ** 1.5
        value = rays * falloff
        rim = self.p.rim_gain * np.exp(-dist / self.p.rim_width)
        hue = (theta / (2 * np.pi) + t * self.p.hue_speed + dist * self.p.hue_twist) % 1.0
        hsv = np.dstack(
            [hue * 360.0, np.clip(1.0 - rim, 0.2, 1.0), np.clip(value + rim, 0, 1)]
        ).astype(np.float32)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        rgb[binary.astype(bool)] = 0.0
        if s != 1.0:
            # No second full-res zeroing pass here: it costs ~2.3 ms at 720p and
            # compose already multiplies the halo by (1 - mask).
            rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_LINEAR)
        np.clip(rgb, 0, 1, out=rgb)
        return rgb
