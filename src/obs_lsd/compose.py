"""Composite head over black with the halo behind it."""

from __future__ import annotations

import cv2
import numpy as np

# compose runs once per frame at 720p. It works in uint8: the frame already
# is, the halo is converted once, and cv2.blendLinear does frame*m + halo*(1-m)
# in a single pass with the mask as the per-pixel weight. The float version
# of this took ten full-frame passes and 9.6 ms on an M2 Max; this takes 3.
_SCRATCH: dict[str, np.ndarray] = {}


def _scratch(h: int, w: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    halo = _SCRATCH.get("halo")
    if halo is None or halo.shape[:2] != (h, w):
        _SCRATCH["halo"] = np.empty((h, w, 3), np.uint8)
        _SCRATCH["halo_bgr"] = np.empty((h, w, 3), np.uint8)
        _SCRATCH["inv"] = np.empty((h, w), np.float32)
    return _SCRATCH["halo"], _SCRATCH["halo_bgr"], _SCRATCH["inv"]


def _spill(mask: np.ndarray, halo_bgr: np.ndarray, gain: float, width: float) -> np.ndarray:
    """Light wrap: the halo's colour, heavily blurred, bleeding a little way in
    from the head's contour (weight exp(-d_in/width)) so the cut is not hard.
    Everything is done at quarter resolution; the result is blurry by design.
    Returns uint8 BGR at full size, to be added onto the composite."""
    h, w = mask.shape
    sh, sw = max(1, h // 4), max(1, w // 4)
    m_small = cv2.resize(mask, (sw, sh), interpolation=cv2.INTER_AREA)
    inside = (m_small > 0.5).astype(np.uint8)
    d_in = cv2.distanceTransform(inside, cv2.DIST_L2, 3) * 4.0  # back to full-res px
    weight = np.exp(-d_in / np.float32(width)) * m_small * np.float32(gain)
    halo_small = cv2.resize(halo_bgr, (sw, sh), interpolation=cv2.INTER_AREA)
    halo_small = cv2.GaussianBlur(halo_small, (0, 0), max(1.0, width / 4))
    spill_small = cv2.convertScaleAbs(halo_small * weight[..., None])
    return cv2.resize(spill_small, (w, h), interpolation=cv2.INTER_LINEAR)


def compose(
    frame_bgr: np.ndarray,
    mask: np.ndarray,
    halo_rgb: np.ndarray,
    halo_gain: float = 1.0,
    spill_gain: float = 0.0,
    spill_width: float = 30.0,
) -> np.ndarray:
    mask = np.ascontiguousarray(mask, dtype=np.float32)
    halo, halo_bgr, inv = _scratch(*mask.shape)
    # float RGB in [0,1] -> uint8 BGR, with the fade (halo_gain) folded in
    cv2.convertScaleAbs(halo_rgb, dst=halo, alpha=255.0 * halo_gain)
    cv2.cvtColor(halo, cv2.COLOR_RGB2BGR, dst=halo_bgr)
    np.subtract(np.float32(1.0), mask, out=inv)
    # (frame*mask + halo*(1-mask)) / (mask + (1-mask)), rounded to nearest.
    # A fresh array each call: callers keep frames (the rig's contact sheet).
    out = cv2.blendLinear(frame_bgr, halo_bgr, mask, inv)
    if spill_gain > 0:
        spill = _spill(mask, halo_bgr, spill_gain * halo_gain, spill_width)
        cv2.add(out, spill, dst=out)  # saturating
    return out
