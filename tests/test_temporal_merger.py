# tests/test_temporal_merger.py

import pytest
import numpy as np
from pylipextractor.temporal_merger import TemporalMerger


def test_mean_merge():
    # 300 frames of shape (50, 75, 3)
    frames = np.ones((300, 50, 75, 3), dtype=np.uint8) * 100
    merged = TemporalMerger.merge(frames, method="mean", merge_size=4)

    assert merged.shape == (75, 50, 75, 3)
    assert merged.dtype == np.uint8
    assert np.all(merged == 100)


def test_motion_weighted_merge():
    frames = np.zeros((300, 50, 75, 3), dtype=np.uint8)
    # Add movement every alternate frame
    for i in range(300):
        if i % 2 == 1:
            frames[i] = 200

    merged = TemporalMerger.merge(frames, method="motion_weighted", merge_size=4)

    assert merged.shape == (75, 50, 75, 3)
    assert merged.dtype == np.uint8


def test_short_sequence():
    frames = np.full((3, 50, 75, 3), 128, dtype=np.uint8)
    merged = TemporalMerger.merge(frames, method="mean", merge_size=4)

    assert merged.shape == (3, 50, 75, 3)
    assert merged.dtype == np.uint8
    assert np.array_equal(merged, frames)


def test_remainder_frames():
    # 301 frames -> 75 groups of 4 = 300 usable frames, 1 frame discarded
    frames = np.ones((301, 50, 75, 3), dtype=np.uint8) * 50
    merged = TemporalMerger.merge(frames, method="mean", merge_size=4)

    assert merged.shape == (75, 50, 75, 3)
    assert merged.dtype == np.uint8


def test_invalid_merge_size():
    frames = np.ones((10, 50, 75, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        TemporalMerger.merge(frames, method="mean", merge_size=0)

    with pytest.raises(ValueError):
        TemporalMerger.merge(frames, method="invalid_method", merge_size=4)
