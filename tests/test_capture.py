import cv2
import numpy as np

from obs_lsd.capture import FrameSource, parse_system_profiler, resolve_camera

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
