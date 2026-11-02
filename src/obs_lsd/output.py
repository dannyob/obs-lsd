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


# (keys, action, help text). Freeze is a capital S: a stray lowercase s used
# to freeze the mask with nothing on screen to say so.
BINDINGS = [
    (("q", 27), "quit", "q/Esc  quit"),
    (("]",), "size+", "] [    head bigger / smaller"),
    (("[",), "size-", None),
    (("=",), "hue+", "= -    colours faster / slower"),
    (("-",), "hue-", None),
    ((".",), "reach+", ". ,    halo longer / shorter"),
    ((",",), "reach-", None),
    (("w",), "up", "w a s d  move head"),
    (("a",), "left", None),
    (("s",), "down", None),
    (("d",), "right", None),
    (("}",), "glide+", "} {    glide faster / slower"),
    (("{",), "glide-", None),
    (("c",), "centre", "c      recentre head, full size"),
    (("S",), "freeze", "S      freeze / unfreeze mask"),
    (("p",), "print", "p      print settings"),
]
HELP_KEY = ord("h")
KEYS = {(k if isinstance(k, int) else ord(k)): action for keys, action, _ in BINDINGS for k in keys}
KEY_HELP = [h for _, _, h in BINDINGS if h] + [
    "click  glide head there (or drag)",
    "right-click  glide there, sized to fit",
    "h      hide / show this",
]


class MouseState:
    """Remembers where a button was last pressed or dragged in the preview.

    Left button: ("place", x, y), move the head there. Right button:
    ("fit", x, y), move it there and resize it to fit.
    """

    def __init__(self):
        self._pending: tuple[str, int, int] | None = None

    def on_mouse(self, event, x, y, flags, _param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN or (
            event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON
        ):
            self._pending = ("place", x, y)
        elif event == cv2.EVENT_RBUTTONDOWN or (
            event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_RBUTTON
        ):
            self._pending = ("fit", x, y)

    def take(self) -> tuple[str, int, int] | None:
        p, self._pending = self._pending, None
        return p


def _text(img, s, org, scale):
    # a dark outline under the white text keeps it legible over the halo
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, s, org, font, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, s, org, font, scale, (255, 255, 255), 1, cv2.LINE_AA)


def draw_overlay(frame_bgr: np.ndarray, frozen: bool, show_keys: bool) -> np.ndarray:
    """The preview's copy of a frame with the key list and a FROZEN flag drawn on.

    Never draws on frame_bgr itself: that frame also goes to the virtual
    camera and Syphon, which must not see the overlay.
    """
    img = frame_bgr.copy()
    scale = img.shape[0] / 720 * 0.55
    line = max(1, round(28 * img.shape[0] / 720))
    if show_keys:
        for i, s in enumerate(KEY_HELP):
            _text(img, s, (line // 2, line * (i + 1)), scale)
    if frozen:
        s = "FROZEN (S)"
        (tw, _), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale * 1.6, 2)
        org = (img.shape[1] - tw - line // 2, line + line // 2)
        cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale * 1.6, (0, 0, 0), 6, cv2.LINE_AA)
        cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale * 1.6, (0, 0, 255), 2, cv2.LINE_AA)
    return img


class Preview:
    def __init__(self, title: str = "obs-lsd"):
        self.title = title
        self.show_keys = True
        self.mouse = MouseState()
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(title, self.mouse.on_mouse)

    def show(self, frame_bgr: np.ndarray, frozen: bool = False):
        """Show a frame; return a key's action, ("place"|"fit", x, y) for a click, or None."""
        cv2.imshow(self.title, draw_overlay(frame_bgr, frozen, self.show_keys))
        k = cv2.waitKey(1) & 0xFF
        if k == HELP_KEY:
            self.show_keys = not self.show_keys
            return None
        action = KEYS.get(k)
        if action is None:
            # the preview shows the output frame 1:1, so window clicks are
            # output pixels (OpenCV maps a resized window back to image pixels)
            click = self.mouse.take()
            if click is not None:
                return click
        return action

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
