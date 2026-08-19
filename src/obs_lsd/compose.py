"""Composite head over black with the halo behind it."""

from __future__ import annotations

import cv2
import numpy as np


def compose(
    frame_bgr: np.ndarray, mask: np.ndarray, halo_rgb: np.ndarray, halo_gain: float = 1.0
) -> np.ndarray:
    m = mask[..., None]
    halo_bgr = cv2.cvtColor(halo_rgb, cv2.COLOR_RGB2BGR) * halo_gain
    out = halo_bgr * (1.0 - m) * 255.0 + frame_bgr.astype(np.float32) * m
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)
