"""Frame sources: webcam by index/name, video file, or a still image (looped)."""

from __future__ import annotations

import os
import re
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np


class CameraError(RuntimeError):
    pass


def parse_system_profiler(text: str) -> list[str]:
    """Camera names from `system_profiler SPCameraDataType` output, in order."""
    return [m.group(1) for m in re.finditer(r"^    (\S.*?):\s*$", text, re.MULTILINE)]


def list_cameras() -> list[str]:
    try:
        out = subprocess.run(
            ["system_profiler", "SPCameraDataType"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return parse_system_profiler(out)


def resolve_camera(spec: str | None, names: list[str] | None = None) -> int | str:
    if spec is None:
        return 0
    if spec.isdigit():
        return int(spec)
    if Path(spec).exists():
        return spec
    if "/" in spec or os.sep in spec or Path(spec).suffix:
        raise CameraError(f"File not found: {spec}")
    names = list_cameras() if names is None else names
    for i, n in enumerate(names):
        if spec.lower() in n.lower():
            return i
    raise CameraError(f"No camera matching {spec!r}. Available: {names or 'none found'}")


IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp"}


class FrameSource:
    def __init__(
        self, spec: str | None, width: int, height: int, fps: float, realtime: bool = True
    ):
        self.fps = fps
        self.realtime = realtime
        self.is_file = False
        self.file_fps: float | None = None
        self._still: np.ndarray | None = None
        self._cap: cv2.VideoCapture | None = None
        target = resolve_camera(spec)
        if isinstance(target, str) and Path(target).suffix.lower() in IMAGE_EXT:
            self._still = cv2.imread(target)
            if self._still is None:
                raise CameraError(f"Could not read image {target}")
            self.is_file = True
        else:
            self._cap = cv2.VideoCapture(target)
            if isinstance(target, int):
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                self._cap.set(cv2.CAP_PROP_FPS, fps)
            else:
                self.is_file = True
                f = self._cap.get(cv2.CAP_PROP_FPS)
                self.file_fps = f if f and f > 0 else fps
            if not self._cap.isOpened():
                what = "file" if self.is_file else "camera"
                raise CameraError(
                    f"Could not open {what} {target!r}. Available cameras: {list_cameras()}. "
                    "On macOS check System Settings > Privacy & Security > Camera."
                )
        self._t0 = time.monotonic()

    def __iter__(self) -> Iterator[tuple[np.ndarray, float]]:
        """Yield (bgr frame, t). For files t is the file's own clock (frame
        index / fps) so a replay is deterministic; for cameras it is wall
        clock. With realtime=True file playback is paced to its fps."""
        if not self.is_file:
            while True:
                ok, frame = self._cap.read()
                if not ok:
                    return
                yield frame, time.monotonic() - self._t0
        period = 1.0 / (self.file_fps or self.fps)
        idx = 0
        while True:
            if self._still is not None:
                frame = self._still.copy()
            else:
                ok, frame = self._cap.read()
                if not ok:
                    return
            t = idx * period
            if self.realtime:
                time.sleep(max(0.0, self._t0 + t - time.monotonic()))
            yield frame, t
            idx += 1

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
