import importlib.util
import sys

import numpy as np
import pytest

from obs_lsd import output


def test_bgr_to_syphon_rgba_swaps_channels_sets_alpha_and_flips():
    frame = np.zeros((4, 6, 3), np.uint8)
    frame[..., 0] = 10  # B
    frame[..., 1] = 20  # G
    frame[..., 2] = 30  # R
    frame[0, 0] = (1, 2, 3)  # top-left pixel
    out = np.empty((4, 6, 4), np.uint8)
    got = output.bgr_to_syphon_rgba(frame, out)
    assert got is out
    assert out.shape == (4, 6, 4)
    assert (
        (out[:-1, :, 0] == 30).all()
        and (out[:-1, :, 1] == 20).all()
        and (out[:-1, :, 2] == 10).all()
    )
    assert (out[..., 3] == 255).all()
    # Syphon textures start at the bottom-left, so the top row lands last
    assert tuple(out[-1, 0, :3]) == (3, 2, 1)
    assert tuple(out[0, 0, :3]) == (30, 20, 10)


def test_syphon_sink_explains_missing_package(monkeypatch):
    monkeypatch.setitem(sys.modules, "syphon", None)
    with pytest.raises(output.SyphonMissing, match="syphon"):
        output.SyphonSink(8, 8)


@pytest.mark.skipif(
    importlib.util.find_spec("syphon") is None, reason="syphon-python not installed"
)
def test_syphon_sink_publishes_frames():
    sink = output.SyphonSink(16, 8, name="obs-lsd-test")
    try:
        frame = np.zeros((8, 16, 3), np.uint8)
        frame[..., 2] = 255
        sink.send(frame)
        sink.send(frame)
        assert sink.frames == 2
        assert sink.name == "obs-lsd-test"
    finally:
        sink.close()
