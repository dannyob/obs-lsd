"""Head-only mask from a camera frame using MediaPipe segmenter + face landmarks."""

from __future__ import annotations

import math
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from obs_lsd import MODELS_DIR
from obs_lsd.track import HeadBox

SEG_MODEL = "selfie_multiclass_256x256.tflite"
FL_MODEL = "face_landmarker.task"
HAIR, FACE = (
    1,
    3,
)  # confidence_masks indices (labels: background,hair,body-skin,face-skin,clothes,others)
CHIN_LM, LEFT_EYE_LM, RIGHT_EYE_LM = 152, 33, 263


class ModelsMissing(RuntimeError):
    pass


@dataclass
class HeadMask:
    mask: np.ndarray
    box: HeadBox


def chin_halfplane(shape, chin_xy, roll_deg: float, margin_px: float) -> np.ndarray:
    """1.0 on the 'above chin' side of a line through chin_xy tilted by roll_deg, else 0."""
    h, w = shape
    r = math.radians(roll_deg)
    # 'down' unit vector rotated by roll, and the line direction across it
    dx, dy = -math.sin(r), math.cos(r)
    ux, uy = math.cos(r), math.sin(r)
    cx, cy = chin_xy[0] + margin_px * dx, chin_xy[1] + margin_px * dy
    # Fill the kept side as one big quadrilateral: a polygon fill is a single
    # C++ pass, where building the half-plane from broadcast ramps is three
    # full-frame numpy passes (add, compare, cast).
    reach = 4.0 * (w + h)
    pts = np.array(
        [
            [cx + reach * ux, cy + reach * uy],
            [cx - reach * ux, cy - reach * uy],
            [cx - reach * ux - reach * dx, cy - reach * uy - reach * dy],
            [cx + reach * ux - reach * dx, cy + reach * uy - reach * dy],
        ]
    )
    hp = np.zeros((h, w), np.float32)
    cv2.fillConvexPoly(hp, np.rint(pts).astype(np.int32), 1.0)
    return hp


def head_from_masks(hair, face, halfplane, feather: int) -> np.ndarray:
    m = np.maximum(hair, face).astype(np.float32, copy=False)
    m *= halfplane
    # MediaPipe's confidence masks rarely reach 1.0 even on the middle of a
    # face, and leave a low wash over the background; stretch [0.15,0.85] to
    # [0,1] so the head is solid and the wash is gone before feathering.
    # In place: each temporary here is a full camera frame of float32.
    m -= 0.15
    m *= 1.0 / 0.7
    np.clip(m, 0, 1, out=m)
    if feather > 0:
        k = feather * 2 + 1
        m = cv2.GaussianBlur(m, (k, k), 0)  # a blur of [0,1] stays in [0,1]
    return m


def mask_box(m: np.ndarray, min_pixels: int = 50) -> HeadBox | None:
    """Centroid and height of the mask's >0.5 region, or None if it is too small."""
    b = (m > 0.5).view(np.uint8)
    mom = cv2.moments(b, binaryImage=True)
    n = mom["m00"]
    if n < min_pixels:
        return None
    _, _, _, bh = cv2.boundingRect(b)
    return HeadBox(cx=mom["m10"] / n, cy=mom["m01"] / n, height=float(bh))


class HeadSegmenter:
    def __init__(
        self,
        models_dir: Path = MODELS_DIR,
        hold_frames: int = 10,
        feather: int = 9,
        chin_margin: float = 0.06,
        delegate: str = "gpu",
        infer_width: int = 320,
        reopen_every: int = 1000,
    ):
        if delegate not in ("gpu", "cpu"):
            raise ValueError(f"delegate must be 'gpu' or 'cpu', not {delegate!r}")
        seg_path, fl_path = models_dir / SEG_MODEL, models_dir / FL_MODEL
        if not (seg_path.exists() and fl_path.exists()):
            raise ModelsMissing(f"MediaPipe models not found in {models_dir}; run `make models`")
        import mediapipe as mp

        self._mp = mp
        # The multiclass segmenter is 120ms a frame on an M2 Max CPU and 6ms on
        # its GPU via Metal. If the GPU delegate cannot be created (Linux
        # without a GPU, say) fall back to the CPU rather than fail.
        try:
            self._seg, self._fl = self._open_models(seg_path, fl_path, delegate)
            self.delegate = delegate
        except Exception as e:  # mediapipe raises plain RuntimeError
            if delegate == "cpu":
                raise
            print(f"obs-lsd: GPU delegate unavailable ({e}); using CPU", file=sys.stderr)
            self._seg, self._fl = self._open_models(seg_path, fl_path, "cpu")
            self.delegate = "cpu"
        self.hold_frames = hold_frames
        self.feather = feather
        self.chin_margin = chin_margin
        # Frames wider than this are shrunk before inference: the models work
        # at 256x256 and 192x192 inside, and MediaPipe hands its confidence
        # masks back at the input size. 0 disables.
        self.infer_width = infer_width
        self._last: HeadMask | None = None
        self._missing = 0
        self._ts = -1
        # MediaPipe's GPU tasks leak one IOSurface per frame on macOS (about
        # 0.5MB at infer_width 320) until the task is closed; the process
        # aborts with "Error creating pixel buffer" after a few thousand
        # frames. Closing the tasks frees it all, so every reopen_every
        # frames a replacement pair is built on a worker thread and swapped
        # in once ready, which costs the frame loop nothing. 0 disables.
        self._paths = (seg_path, fl_path)
        self.reopen_every = reopen_every if self.delegate == "gpu" else 0
        self.opens = 1
        self.reopens = 0
        self._frames = 0
        self._pool: ThreadPoolExecutor | None = None
        self._replacement: Future | None = None

    def _open_models(self, seg_path: Path, fl_path: Path, delegate: str):
        from mediapipe.tasks.python import BaseOptions, vision

        def base(path):
            if delegate == "gpu":
                return BaseOptions(model_asset_path=str(path), delegate=BaseOptions.Delegate.GPU)
            return BaseOptions(model_asset_path=str(path))

        seg = vision.ImageSegmenter.create_from_options(
            vision.ImageSegmenterOptions(
                base_options=base(seg_path),
                running_mode=vision.RunningMode.VIDEO,
                output_confidence_masks=True,
            )
        )
        fl = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=base(fl_path),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1,
            )
        )
        return seg, fl

    def _build_replacement(self):
        seg, fl = self._open_models(*self._paths, self.delegate)
        self.opens += 1
        return seg, fl

    def _maybe_reopen(self) -> None:
        if self._replacement is not None and self._replacement.done():
            old_seg, old_fl = self._seg, self._fl
            self._seg, self._fl = self._replacement.result()
            self._replacement = None
            old_seg.close()
            old_fl.close()
            self.reopens += 1
        self._frames += 1
        if (
            self.reopen_every
            and self._frames % self.reopen_every == 0
            and self._replacement is None
        ):
            if self._pool is None:
                self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="seg-reopen")
            self._replacement = self._pool.submit(self._build_replacement)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            if self._replacement is not None:
                for task in self._replacement.result():
                    task.close()
                self._replacement = None
        self._seg.close()
        self._fl.close()

    def mask(self, frame_bgr: np.ndarray, timestamp_ms: int) -> HeadMask | None:
        self._maybe_reopen()
        # MediaPipe VIDEO mode needs strictly increasing timestamps
        self._ts = max(self._ts + 1, int(timestamp_ms))
        # The Metal delegate aborts the process (a CHECK failure, not an
        # exception) when handed an SRGB image; SRGBA works on both delegates.
        h, w = frame_bgr.shape[:2]
        sw, sh = w, h
        if 0 < self.infer_width < w:
            sw = self.infer_width
            sh = max(1, round(h * sw / w))
            frame_bgr = cv2.resize(frame_bgr, (sw, sh), interpolation=cv2.INTER_LINEAR)
        rgba = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGBA)
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGBA, data=rgba)
        fl = self._fl.detect_for_video(img, self._ts)
        if not fl.face_landmarks:
            return self._hold()
        lm = fl.face_landmarks[0]
        le, re = lm[LEFT_EYE_LM], lm[RIGHT_EYE_LM]
        roll = math.degrees(math.atan2((re.y - le.y) * h, (re.x - le.x) * w))
        seg = self._seg.segment_for_video(img, self._ts)
        cm = seg.confidence_masks
        hair = cm[HAIR].numpy_view()[..., 0]
        face = cm[FACE].numpy_view()[..., 0]
        # Union, chin cut and box at the inference size, where they are nearly
        # free; one resize up, then the feather at camera size so its width
        # in pixels does not depend on infer_width.
        chin = (lm[CHIN_LM].x * sw, lm[CHIN_LM].y * sh)
        face_h = abs(chin[1] - min(le.y, re.y) * sh) * 2.0
        hp = chin_halfplane((sh, sw), chin, roll, self.chin_margin * face_h)
        m = head_from_masks(hair, face, hp, feather=0)
        scale = w / sw
        box = mask_box(m, min_pixels=max(1, round(50 / (scale * scale))))
        if box is None:
            return self._hold()
        if (sh, sw) != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
            box = HeadBox(cx=box.cx * scale, cy=box.cy * scale, height=box.height * scale)
        if self.feather > 0:
            k = self.feather * 2 + 1
            m = cv2.GaussianBlur(m, (k, k), 0)
        self._last = HeadMask(mask=m, box=box)
        self._missing = 0
        return self._last

    def _hold(self) -> HeadMask | None:
        self._missing += 1
        if self._last is not None and self._missing <= self.hold_frames:
            return self._last
        return None
