"""Composite head over black with the halo behind it."""

from __future__ import annotations

import cv2
import numpy as np

# compose runs once per frame at 720p. numpy broadcasting of a HxW mask over a
# HxWx3 frame costs ~1.5 ms per pass, so the mask is expanded to three channels
# once and every stage is a matched-shape cv2 op writing into a reused buffer.
_SCRATCH: dict[str, np.ndarray] = {}


def _scratch(h: int, w: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    out = _SCRATCH.get("out")
    if out is None or out.shape[:2] != (h, w):
        for k in ("out", "head", "m3", "inv3"):
            _SCRATCH[k] = np.empty((h, w, 3), np.float32)
    return _SCRATCH["out"], _SCRATCH["head"], _SCRATCH["m3"], _SCRATCH["inv3"]


def _spill(mask: np.ndarray, halo_bgr: np.ndarray, gain: float, width: float) -> np.ndarray:
    """Light wrap: the halo's colour, heavily blurred, bleeding a little way in
    from the head's contour (weight exp(-d_in/width)) so the cut is not hard.
    Everything is done at quarter resolution; the result is blurry by design."""
    h, w = mask.shape
    sh, sw = max(1, h // 4), max(1, w // 4)
    m_small = cv2.resize(mask, (sw, sh), interpolation=cv2.INTER_AREA)
    inside = (m_small > 0.5).astype(np.uint8)
    d_in = cv2.distanceTransform(inside, cv2.DIST_L2, 3) * 4.0  # back to full-res px
    weight = np.exp(-d_in / np.float32(width)) * m_small * np.float32(gain)
    halo_small = cv2.resize(halo_bgr, (sw, sh), interpolation=cv2.INTER_AREA)
    halo_small = cv2.GaussianBlur(halo_small, (0, 0), max(1.0, width / 4))
    spill_small = halo_small * weight[..., None]
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
    out, head, m3, inv3 = _scratch(*mask.shape)
    cv2.merge((mask, mask, mask), dst=m3)
    cv2.cvtColor(halo_rgb, cv2.COLOR_RGB2BGR, dst=out)
    spill = _spill(mask, out, spill_gain * halo_gain, spill_width) if spill_gain > 0 else None
    # inv3 = 255 * halo_gain * (1 - mask)
    k = 255.0 * halo_gain
    cv2.addWeighted(m3, -k, m3, 0.0, k, dst=inv3)
    cv2.multiply(out, inv3, dst=out)
    cv2.multiply(frame_bgr, m3, dtype=cv2.CV_32F, dst=head)
    if spill is not None:
        cv2.scaleAdd(spill, 255.0, head, dst=head)
    # out = halo + head + 0.5, so the uint8 cast below truncates to round-half-up
    cv2.addWeighted(out, 1.0, head, 1.0, 0.5, dst=out)
    cv2.threshold(out, 255.0, 0.0, cv2.THRESH_TRUNC, dst=out)
    cv2.threshold(out, 0.0, 0.0, cv2.THRESH_TOZERO, dst=out)
    return out.astype(np.uint8)
