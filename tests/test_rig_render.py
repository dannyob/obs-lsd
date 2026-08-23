import json

import cv2
import numpy as np

from obs_lsd.rig.render import render
from obs_lsd.segment import HeadMask
from obs_lsd.track import HeadBox


class FakeSeg:
    def mask(self, frame, ts):
        h, w = frame.shape[:2]
        m = np.zeros((h, w), np.float32)
        cv2.ellipse(m, (w // 2, h // 2), (w // 8, h // 5), 0, 0, 360, 1.0, -1)
        return HeadMask(mask=m, box=HeadBox(cx=w / 2, cy=h / 2, height=2 * h / 5))

    def close(self):
        pass


def _clip(path, n=30, fps=10.0, size=(160, 90)):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(n):
        f = np.full((size[1], size[0], 3), 40, np.uint8)
        f[:, :, 1] = min(255, 60 + i * 5)
        w.write(f)
    w.release()


def test_render_writes_run_dir(tmp_path):
    from obs_lsd.pipeline import PipelineConfig

    clip = tmp_path / "c.mp4"
    _clip(clip)
    out = tmp_path / "run"
    res = render(str(clip), out, cfg=PipelineConfig(width=320, height=180), segmenter=FakeSeg())
    assert res.n_frames == 30 and res.run_dir == out
    for name in ("out.mp4", "metrics.json", "sheet.png", "crops.png", "motion.png"):
        assert (out / name).exists(), name
    stats = json.loads((out / "metrics.json").read_text())
    assert stats["frames"] == 30 and stats["no_head"] == 0 and stats["ms_total_mean"] > 0
    sheet = cv2.imread(str(out / "sheet.png"))
    assert sheet.shape[1] == 4 * 480
    cap = cv2.VideoCapture(str(out / "out.mp4"))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 30
