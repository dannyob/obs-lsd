"""Sinks: OBS virtual camera and a cv2 preview window with hotkeys."""

from __future__ import annotations

import cv2
import numpy as np


class VirtualCamMissing(RuntimeError):
    pass


class VirtualCam:
    def __init__(self, width: int, height: int, fps: float):
        import pyvirtualcam

        try:
            self._cam = pyvirtualcam.Camera(
                width=width, height=height, fps=fps, fmt=pyvirtualcam.PixelFormat.BGR
            )
        except RuntimeError as e:
            raise VirtualCamMissing(
                "Could not open the OBS Virtual Camera. Launch OBS once, click 'Start Virtual "
                "Camera', approve the system extension in System Settings if asked, then retry."
                f"\n({e})"
            ) from e
        self.device = self._cam.device

    def send(self, frame_bgr: np.ndarray) -> None:
        self._cam.send(frame_bgr)
        self._cam.sleep_until_next_frame()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._cam.close()


KEYS = {
    ord("q"): "quit",
    27: "quit",
    ord("]"): "size+",
    ord("["): "size-",
    ord("="): "hue+",
    ord("-"): "hue-",
    ord("."): "reach+",
    ord(","): "reach-",
    ord("s"): "freeze",
    ord("p"): "print",
}


class Preview:
    def __init__(self, title: str = "obs-lsd"):
        self.title = title
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)

    def show(self, frame_bgr: np.ndarray) -> str | None:
        cv2.imshow(self.title, frame_bgr)
        k = cv2.waitKey(1) & 0xFF
        return KEYS.get(k)

    def close(self) -> None:
        cv2.destroyWindow(self.title)
