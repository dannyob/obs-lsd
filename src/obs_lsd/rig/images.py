"""Images the judge looks at: contact sheet, full-res crops, motion strip. Pure."""

from __future__ import annotations

import cv2
import numpy as np


def _label(img: np.ndarray, text: str) -> np.ndarray:
    cv2.putText(img, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img


def _fit_width(img: np.ndarray, width: int) -> np.ndarray:
    h, w = img.shape[:2]
    return cv2.resize(img, (width, max(1, round(h * width / w))), interpolation=cv2.INTER_AREA)


def sample_indices(n_frames: int, k: int) -> list[int]:
    if n_frames <= 0:
        return [0] * k
    lo, hi = int(n_frames * 0.02), max(int(n_frames * 0.98) - 1, 0)
    return [round(lo + (hi - lo) * i / max(k - 1, 1)) for i in range(k)]


def contact_sheet(
    frames: list[np.ndarray], times: list[float], cols: int = 4, width: int = 480
) -> np.ndarray:
    tiles = [
        _label(_fit_width(f.copy(), width), f"t={t:.1f}s")
        for f, t in zip(frames, times, strict=True)
    ]
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i : i + cols]) for i in range(0, len(tiles), cols)]
    return np.vstack(rows)


def crops(frames: list[np.ndarray], size: int = 560) -> np.ndarray:
    out = []
    for i, f in enumerate(frames):
        h, w = f.shape[:2]
        cy, cx = h // 2, w // 2
        half = min(size // 2, h // 2, w // 2)
        c = f[cy - half : cy + half, cx - half : cx + half]
        if c.shape[0] != size:
            c = cv2.resize(c, (size, size), interpolation=cv2.INTER_AREA)
        out.append(_label(c.copy(), f"crop {i + 1}"))
    return np.hstack(out)


def motion_strip(frames: list[np.ndarray], width: int = 320) -> np.ndarray:
    return np.hstack(
        [_label(_fit_width(f.copy(), width), f"+{i * 100}ms") for i, f in enumerate(frames)]
    )
