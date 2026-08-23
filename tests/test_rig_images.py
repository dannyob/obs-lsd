from itertools import pairwise

import numpy as np

from obs_lsd.rig.images import contact_sheet, crops, motion_strip, sample_indices


def _frame(v):
    f = np.zeros((90, 160, 3), np.uint8)
    f[:] = v
    return f


def test_sample_indices_spread_and_inside():
    idx = sample_indices(1000, 8)
    assert len(idx) == 8 and idx[0] >= 20 and idx[-1] <= 979
    assert all(b > a for a, b in pairwise(idx))
    assert sample_indices(5, 8) == [0, 1, 2, 3, 4, 4, 4, 4] or len(sample_indices(5, 8)) == 8


def test_contact_sheet_shape():
    sheet = contact_sheet(
        [_frame(i * 30) for i in range(8)], [i * 4.0 for i in range(8)], cols=4, width=160
    )
    assert sheet.shape[0] == 2 * 90 and sheet.shape[1] == 4 * 160 and sheet.dtype == np.uint8


def test_crops_and_motion_strip_shapes():
    big = [np.zeros((720, 1280, 3), np.uint8) for _ in range(3)]
    c = crops(big, size=200)
    assert c.shape == (200, 600, 3)
    s = motion_strip([_frame(i) for i in range(8)], width=80)
    assert s.shape[1] == 8 * 80 and s.shape[2] == 3
