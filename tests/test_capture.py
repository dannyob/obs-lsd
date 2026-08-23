import itertools

import cv2
import numpy as np
import pytest

from obs_lsd.capture import (
    CameraError,
    FrameSource,
    parse_system_profiler,
    resolve_camera,
)

SP_TEXT = """Camera:

    MacBook Pro Camera:

      Model ID: MacBook Pro Camera
      Unique ID: 6C707041

    Yawl Camera:

      Model ID: iPhone18,3
      Unique ID: 908AC892
"""


def test_parse_system_profiler():
    assert parse_system_profiler(SP_TEXT) == ["MacBook Pro Camera", "Yawl Camera"]


def test_resolve_camera_index_name_and_path(tmp_path):
    p = tmp_path / "x.png"
    p.write_bytes(b"x")
    assert resolve_camera("1") == 1
    assert resolve_camera(str(p)) == str(p)
    assert resolve_camera(None) == 0
    assert resolve_camera("yawl", names=["MacBook Pro Camera", "Yawl Camera"]) == 1


def test_image_source_loops_and_times(tmp_path):
    p = tmp_path / "f.png"
    cv2.imwrite(str(p), np.full((20, 30, 3), 7, np.uint8))
    src = FrameSource(str(p), width=30, height=20, fps=100)
    frames = []
    for frame, t in src:
        frames.append((frame.shape, t))
        if len(frames) == 3:
            break
    src.close()
    assert [f[0] for f in frames] == [(20, 30, 3)] * 3
    assert frames[2][1] > frames[0][1] >= 0.0


def test_resolve_camera_reports_missing_files_as_files(tmp_path):
    with pytest.raises(CameraError, match="File not found"):
        resolve_camera(str(tmp_path / "clip.mp4"))
    with pytest.raises(CameraError, match="File not found"):
        resolve_camera("clips/portrait")
    # a bare word is still treated as a camera name
    with pytest.raises(CameraError, match="No camera matching"):
        resolve_camera("zzz", names=["MacBook Pro Camera"])


def _write_clip(path, n=20, fps=10.0, size=(64, 48)):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(n):
        w.write(np.full((size[1], size[0], 3), i * 10, np.uint8))
    w.release()


def test_video_source_uses_file_time_and_ends(tmp_path):
    import time

    p = tmp_path / "clip.mp4"
    _write_clip(p, n=20, fps=10.0)
    src = FrameSource(str(p), width=64, height=48, fps=30, realtime=False)
    t0 = time.monotonic()
    frames = list(src)
    src.close()
    assert time.monotonic() - t0 < 1.0  # not paced to 2 s of wall clock
    assert len(frames) == 20
    ts = [t for _, t in frames]
    assert ts[0] == 0.0 and ts[10] == 1.0 and all(b > a for a, b in itertools.pairwise(ts))
    assert frames[0][0].shape == (48, 64, 3)


def test_video_source_realtime_paces_to_file_fps(tmp_path):
    import time

    p = tmp_path / "clip.mp4"
    _write_clip(p, n=5, fps=10.0)
    src = FrameSource(str(p), width=64, height=48, fps=30, realtime=True)
    t0 = time.monotonic()
    n = sum(1 for _ in src)
    assert n == 5 and time.monotonic() - t0 >= 0.35  # 5 frames at 10 fps ~ 0.4 s
