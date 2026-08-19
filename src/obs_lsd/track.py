"""Smooth the head position/size and map camera space to output space."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HeadBox:
    """Head centre and height in camera pixels."""

    cx: float
    cy: float
    height: float


@dataclass
class Transform:
    """Uniform scale + translation: out = scale * cam + (tx, ty)."""

    scale: float
    tx: float
    ty: float

    def matrix(self) -> np.ndarray:
        return np.array([[self.scale, 0.0, self.tx], [0.0, self.scale, self.ty]], np.float32)

    def apply(self, x: float, y: float) -> tuple[float, float]:
        return (self.scale * x + self.tx, self.scale * y + self.ty)


class Tracker:
    """Exponentially smooth the head box; emit a Transform centring it at a fixed size."""

    def __init__(
        self,
        out_size: tuple[int, int],
        target_frac: float = 0.55,
        alpha_pos: float = 0.25,
        alpha_scale: float = 0.15,
    ):
        self.out_w, self.out_h = out_size
        self.target_frac = target_frac
        self.alpha_pos = alpha_pos
        self.alpha_scale = alpha_scale
        self._state: HeadBox | None = None
        self.frames_missing = 0

    def update(self, head: HeadBox | None) -> Transform:
        if head is None:
            self.frames_missing += 1
        else:
            self.frames_missing = 0
            if self._state is None:
                self._state = HeadBox(head.cx, head.cy, head.height)
            else:
                s, ap, asc = self._state, self.alpha_pos, self.alpha_scale
                s.cx += ap * (head.cx - s.cx)
                s.cy += ap * (head.cy - s.cy)
                s.height += asc * (head.height - s.height)
        if self._state is None:
            return Transform(1.0, 0.0, 0.0)
        s = self._state
        scale = self.target_frac * self.out_h / max(s.height, 1.0)
        return Transform(scale, self.out_w / 2 - scale * s.cx, self.out_h / 2 - scale * s.cy)
