"""Head-only mask from a camera frame using MediaPipe segmenter + face landmarks."""

from __future__ import annotations

import math
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
    # 'down' unit vector rotated by roll
    dx, dy = -math.sin(r), math.cos(r)
    # separable: the distance to the line is a sum of a per-column and a per-row
    # term, so build two 1-D ramps instead of a full HxW mgrid every frame
    xs = (np.arange(w, dtype=np.float32) - chin_xy[0]) * dx
    ys = (np.arange(h, dtype=np.float32) - chin_xy[1]) * dy - margin_px
    return (xs[None, :] + ys[:, None] <= 0).astype(np.float32)


def head_from_masks(hair, face, halfplane, feather: int) -> np.ndarray:
    m = np.maximum(hair, face) * halfplane
    if feather > 0:
        k = feather * 2 + 1
        m = cv2.GaussianBlur(m, (k, k), 0)
    return np.clip(m, 0, 1).astype(np.float32)


class HeadSegmenter:
    def __init__(
        self,
        models_dir: Path = MODELS_DIR,
        hold_frames: int = 10,
        feather: int = 5,
        chin_margin: float = 0.06,
    ):
        seg_path, fl_path = models_dir / SEG_MODEL, models_dir / FL_MODEL
        if not (seg_path.exists() and fl_path.exists()):
            raise ModelsMissing(f"MediaPipe models not found in {models_dir}; run `make models`")
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        self._seg = vision.ImageSegmenter.create_from_options(
            vision.ImageSegmenterOptions(
                base_options=BaseOptions(model_asset_path=str(seg_path)),
                running_mode=vision.RunningMode.VIDEO,
                output_confidence_masks=True,
            )
        )
        self._fl = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(fl_path)),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1,
            )
        )
        self.hold_frames = hold_frames
        self.feather = feather
        self.chin_margin = chin_margin
        self._last: HeadMask | None = None
        self._missing = 0
        self._ts = -1

    def close(self) -> None:
        self._seg.close()
        self._fl.close()

    def mask(self, frame_bgr: np.ndarray, timestamp_ms: int) -> HeadMask | None:
        # MediaPipe VIDEO mode needs strictly increasing timestamps
        self._ts = max(self._ts + 1, int(timestamp_ms))
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        h, w = rgb.shape[:2]
        fl = self._fl.detect_for_video(img, self._ts)
        if not fl.face_landmarks:
            return self._hold()
        lm = fl.face_landmarks[0]
        chin = (lm[CHIN_LM].x * w, lm[CHIN_LM].y * h)
        le, re = lm[LEFT_EYE_LM], lm[RIGHT_EYE_LM]
        roll = math.degrees(math.atan2((re.y - le.y) * h, (re.x - le.x) * w))
        seg = self._seg.segment_for_video(img, self._ts)
        cm = seg.confidence_masks
        hair = cm[HAIR].numpy_view()[..., 0]
        face = cm[FACE].numpy_view()[..., 0]
        face_h = abs(chin[1] - min(le.y, re.y) * h) * 2.0
        hp = chin_halfplane((h, w), chin, roll, self.chin_margin * face_h)
        m = head_from_masks(hair, face, hp, self.feather)
        ys, xs = np.nonzero(m > 0.5)
        if len(ys) < 50:
            return self._hold()
        box = HeadBox(
            cx=float(xs.mean()), cy=float(ys.mean()), height=float(ys.max() - ys.min() + 1)
        )
        self._last = HeadMask(mask=m, box=box)
        self._missing = 0
        return self._last

    def _hold(self) -> HeadMask | None:
        self._missing += 1
        if self._last is not None and self._missing <= self.hold_frames:
            return self._last
        return None
