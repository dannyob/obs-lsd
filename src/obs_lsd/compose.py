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


def compose(
    frame_bgr: np.ndarray, mask: np.ndarray, halo_rgb: np.ndarray, halo_gain: float = 1.0
) -> np.ndarray:
    mask = np.ascontiguousarray(mask, dtype=np.float32)
    out, head, m3, inv3 = _scratch(*mask.shape)
    cv2.merge((mask, mask, mask), dst=m3)
    cv2.cvtColor(halo_rgb, cv2.COLOR_RGB2BGR, dst=out)
    # inv3 = 255 * halo_gain * (1 - mask)
    k = 255.0 * halo_gain
    cv2.addWeighted(m3, -k, m3, 0.0, k, dst=inv3)
    cv2.multiply(out, inv3, dst=out)
    cv2.multiply(frame_bgr, m3, dtype=cv2.CV_32F, dst=head)
    # out = halo + head + 0.5, so the uint8 cast below truncates to round-half-up
    cv2.addWeighted(out, 1.0, head, 1.0, 0.5, dst=out)
    cv2.threshold(out, 255.0, 0.0, cv2.THRESH_TRUNC, dst=out)
    cv2.threshold(out, 0.0, 0.0, cv2.THRESH_TOZERO, dst=out)
    return out.astype(np.uint8)
