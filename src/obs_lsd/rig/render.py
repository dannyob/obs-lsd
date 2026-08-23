"""Replay a clip through the pipeline and write a run directory for judging."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2

from obs_lsd.capture import FrameSource
from obs_lsd.output import VideoFileSink
from obs_lsd.pipeline import Pipeline, PipelineConfig
from obs_lsd.rig.images import contact_sheet, crops, motion_strip, sample_indices
from obs_lsd.rig.metrics import frame_stats, halo_delta


@dataclass
class RunResult:
    run_dir: Path
    stats: dict
    n_frames: int


def render(
    clip: str,
    out_dir: Path,
    cfg: PipelineConfig | None = None,
    segmenter=None,
    sheet_frames: int = 8,
) -> RunResult:
    cfg = cfg or PipelineConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src = FrameSource(clip, cfg.width, cfg.height, 30.0, realtime=False)
    fps = src.file_fps or 30.0
    pipe = Pipeline(cfg, segmenter=segmenter)
    sink = VideoFileSink(str(out_dir / "out.mp4"), cfg.width, cfg.height, fps)
    boxes, timings, deltas, times, kept = [], [], [], [], {}
    prev = None
    # which frame indices to keep in memory for the images
    try:
        n_total = int(cv2.VideoCapture(clip).get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        sheet_idx = set(sample_indices(n_total, sheet_frames))
        crop_idx = {int(n_total * f) for f in (0.25, 0.5, 0.75)}
        step100 = max(1, round(fps / 10))
        mid = n_total // 2
        motion_idx = {mid + i * step100 for i in range(8)}
        want = sheet_idx | crop_idx | motion_idx
        i = 0
        for frame, t in src:
            out = pipe.step(frame, t)
            sink.write(out)
            boxes.append(pipe.last_box)
            timings.append(dict(pipe.timings))
            if prev is not None and pipe.last_mask_out is not None:
                deltas.append(halo_delta(prev, out, pipe.last_mask_out))
            if i in want:
                kept[i] = out.copy()
                times.append(t)
            prev = out
            i += 1
    finally:
        sink.close()
        pipe.close()
        src.close()
    stats = frame_stats(boxes, timings, deltas)
    (out_dir / "metrics.json").write_text(json.dumps(stats, indent=2))
    if kept:
        s_keys = sorted(k for k in sheet_idx if k in kept)[:sheet_frames]
        cv2.imwrite(
            str(out_dir / "sheet.png"),
            contact_sheet([kept[k] for k in s_keys], [k / fps for k in s_keys]),
        )
        c_keys = sorted(k for k in crop_idx if k in kept)
        cv2.imwrite(str(out_dir / "crops.png"), crops([kept[k] for k in c_keys]))
        m_keys = sorted(k for k in motion_idx if k in kept)
        cv2.imwrite(str(out_dir / "motion.png"), motion_strip([kept[k] for k in m_keys]))
    return RunResult(run_dir=out_dir, stats=stats, n_frames=i)
