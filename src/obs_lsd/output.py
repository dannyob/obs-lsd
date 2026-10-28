"""Sinks: OBS virtual camera, Syphon server, cv2 preview window with hotkeys."""

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


class SyphonMissing(RuntimeError):
    pass


def bgr_to_syphon_rgba(frame_bgr: np.ndarray, out: np.ndarray) -> np.ndarray:
    """BGR frame to the RGBA layout a Syphon texture wants, written into `out`.

    Syphon textures have their origin at the bottom-left, so the rows go in
    bottom to top or OBS shows the frame upside down.
    """
    cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGBA, dst=out)
    cv2.flip(out, 0, dst=out)
    return out


class SyphonSink:
    """Publish frames as a Syphon server so OBS can take them as a Syphon Client source.

    Syphon shares a GPU texture between apps on macOS, so unlike the virtual
    camera it can feed OBS itself. The server announces itself through the
    Cocoa run loop, which has to be pumped or no client ever sees it.
    """

    def __init__(self, width: int, height: int, name: str = "obs-lsd"):
        try:
            import Metal
            import syphon
            from Foundation import NSDate, NSRunLoop
        except ImportError as e:
            raise SyphonMissing(
                "Syphon output needs the syphon-python package: uv pip install -e '.[syphon]'"
            ) from e
        from syphon.utils.raw import create_mtl_texture

        self.name = name
        self.frames = 0
        self._server = syphon.SyphonMetalServer(name)
        self._texture = create_mtl_texture(self._server.device, width, height)
        self._region = Metal.MTLRegion((0, 0, 0), (width, height, 1))
        self._row_bytes = width * 4
        self._rgba = np.empty((height, width, 4), np.uint8)
        self._runloop = NSRunLoop.currentRunLoop()
        self._tick = NSDate.dateWithTimeIntervalSinceNow_
        self._pump()

    def _pump(self) -> None:
        self._runloop.runUntilDate_(self._tick(0.0005))

    def send(self, frame_bgr: np.ndarray) -> None:
        bgr_to_syphon_rgba(frame_bgr, self._rgba)
        self._texture.replaceRegion_mipmapLevel_withBytes_bytesPerRow_(
            self._region, 0, self._rgba.data, self._row_bytes
        )
        self._server.publish_frame_texture(self._texture)
        self._pump()
        self.frames += 1

    def close(self) -> None:
        self._server.stop()


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


class VideoFileSink:
    """Write composed frames to an mp4 (for headless replay/evaluation runs)."""

    def __init__(self, path: str, width: int, height: int, fps: float):
        self.path = path
        self._w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not self._w.isOpened():
            raise RuntimeError(f"Could not open {path} for writing")
        self.frames = 0

    def write(self, frame_bgr: np.ndarray) -> None:
        self._w.write(frame_bgr)
        self.frames += 1

    def close(self) -> None:
        self._w.release()
