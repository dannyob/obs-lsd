"""Wire capture-independent stages: segment -> track -> warp -> halo -> compose."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from obs_lsd.compose import compose
from obs_lsd.halo import HaloParams, HaloRenderer
from obs_lsd.track import Tracker


@dataclass
class PipelineConfig:
    width: int = 1280
    height: int = 720
    head_frac: float = 0.55
    halo: HaloParams = field(default_factory=HaloParams)
    fade_seconds: float = 1.0


class Pipeline:
    def __init__(self, cfg: PipelineConfig, segmenter=None):
        self.cfg = cfg
        if segmenter is None:
            from obs_lsd.segment import HeadSegmenter

            segmenter = HeadSegmenter()
        self.seg = segmenter
        self.tracker = Tracker((cfg.width, cfg.height), target_frac=cfg.head_frac)
        self.halo = HaloRenderer(cfg.halo)
        self.frozen = False
        self.timings: dict[str, float] = {}
        self._last_head = None
        self._lost_since: float | None = None

    def step(self, frame_bgr: np.ndarray, t: float) -> np.ndarray:
        tm: dict[str, float] = {}
        t0 = time.perf_counter()
        if self.frozen and self._last_head is not None:
            head = self._last_head
        else:
            head = self.seg.mask(frame_bgr, int(t * 1000))
        tm["segment"] = (time.perf_counter() - t0) * 1000

        if head is not None:
            self._last_head = head
            self._lost_since = None
            gain = 1.0
        else:
            self._lost_since = t if self._lost_since is None else self._lost_since
            gain = max(0.0, 1.0 - (t - self._lost_since) / max(self.cfg.fade_seconds, 1e-6))
        tr = self.tracker.update(head.box if head is not None else None)

        t1 = time.perf_counter()
        size = (self.cfg.width, self.cfg.height)
        if self._last_head is None:
            tm["warp"] = tm["halo"] = tm["compose"] = 0.0
            tm["total"] = (time.perf_counter() - t0) * 1000
            self.timings = tm
            return np.zeros((self.cfg.height, self.cfg.width, 3), np.uint8)
        src_mask = self._last_head.mask
        m = cv2.warpAffine(src_mask, tr.matrix(), size, flags=cv2.INTER_LINEAR, borderValue=0)
        f = cv2.warpAffine(frame_bgr, tr.matrix(), size, flags=cv2.INTER_LINEAR, borderValue=0)
        tm["warp"] = (time.perf_counter() - t1) * 1000

        t2 = time.perf_counter()
        halo = self.halo.render(m, t) if gain > 0 else np.zeros((*m.shape, 3), np.float32)
        tm["halo"] = (time.perf_counter() - t2) * 1000

        t3 = time.perf_counter()
        # Fade the head cutout together with the halo (mask * gain) rather than
        # snapping it from fully visible to black once gain hits 0.
        out = compose(
            f,
            m * gain,
            halo,
            halo_gain=gain,
            spill_gain=self.halo.p.spill_gain,
            spill_width=self.halo.p.spill_width,
        )
        tm["compose"] = (time.perf_counter() - t3) * 1000
        tm["total"] = (time.perf_counter() - t0) * 1000
        self.timings = tm
        return out

    def close(self) -> None:
        self.seg.close()
