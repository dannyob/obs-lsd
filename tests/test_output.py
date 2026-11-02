import importlib.util
import sys

import cv2
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


def test_freeze_needs_capital_s():
    # lowercase s froze the mask by accident, with nothing on screen to say so
    assert output.KEYS.get(ord("S")) == "freeze"
    assert output.KEYS.get(ord("s")) != "freeze"


def test_overlay_draws_on_a_copy_and_leaves_the_sent_frame_alone():
    frame = np.zeros((360, 640, 3), np.uint8)
    out = output.draw_overlay(frame, frozen=False, show_keys=True)
    assert out is not frame
    assert not frame.any()  # what went to the vcam/syphon is untouched
    assert out.any()  # the key list is drawn on the preview copy


def test_overlay_flags_freeze_even_with_the_key_list_hidden():
    frame = np.zeros((360, 640, 3), np.uint8)
    assert not output.draw_overlay(frame, frozen=False, show_keys=False).any()
    assert output.draw_overlay(frame, frozen=True, show_keys=False).any()


def test_key_help_names_the_keys():
    text = "\n".join(output.KEY_HELP)
    assert "S" in text and "freeze" in text
    for code in output.KEYS:
        if code != 27:  # Esc is listed as "Esc"
            assert chr(code) in text


def test_wasd_move_and_c_recentres():
    assert [output.KEYS[ord(k)] for k in "wasdc"] == ["up", "left", "down", "right", "centre"]


def test_mouse_press_and_drag_report_where_to_put_the_head():
    m = output.MouseState()
    assert m.take() is None
    m.on_mouse(cv2.EVENT_MOUSEMOVE, 5, 6, 0, None)  # hover alone does nothing
    assert m.take() is None
    m.on_mouse(cv2.EVENT_LBUTTONDOWN, 10, 20, cv2.EVENT_FLAG_LBUTTON, None)
    m.on_mouse(cv2.EVENT_MOUSEMOVE, 30, 40, cv2.EVENT_FLAG_LBUTTON, None)
    assert m.take() == ("place", 30, 40)  # latest drag position wins
    assert m.take() is None  # and is consumed


def test_right_button_asks_for_a_fit():
    m = output.MouseState()
    m.on_mouse(cv2.EVENT_RBUTTONDOWN, 7, 8, cv2.EVENT_FLAG_RBUTTON, None)
    assert m.take() == ("fit", 7, 8)
    m.on_mouse(cv2.EVENT_MOUSEMOVE, 9, 10, cv2.EVENT_FLAG_RBUTTON, None)
    assert m.take() == ("fit", 9, 10)


def test_braces_change_glide_speed():
    assert output.KEYS[ord("{")] == "glide-"
    assert output.KEYS[ord("}")] == "glide+"
