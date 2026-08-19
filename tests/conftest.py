"""Shared fixtures."""

import numpy as np
import pytest


@pytest.fixture
def ellipse_mask():
    """A 360x640 float32 mask with an ellipse 'head' centred at (320,180)."""
    import cv2

    m = np.zeros((360, 640), np.float32)
    cv2.ellipse(m, (320, 180), (60, 80), 0, 0, 360, 1.0, -1)
    return m
