# tests/test_lip_extractor.py

import pytest
import numpy as np
import cv2
from pathlib import Path
from pylipextractor.lip_extractor import LipExtractor


@pytest.fixture
def synthetic_video_path(tmp_path):
    """
    Creates a synthetic video file with face-like structures for testing LipExtractor.
    """
    video_file = tmp_path / "test_video.mp4"
    width, height = 320, 240
    fps = 30
    duration_frames = 30

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_file), fourcc, fps, (width, height))

    for idx in range(duration_frames):
        frame = np.full((height, width, 3), 200, dtype=np.uint8)

        # Draw simple face and lips features
        cv2.ellipse(frame, (160, 120), (50, 70), 0, 0, 360, (150, 150, 150), -1)
        # Mouth / Lips region
        cv2.ellipse(frame, (160, 150), (20, 10), 0, 0, 360, (50, 50, 200), -1)

        out.write(frame)

    out.release()
    return video_file


def test_extraction_without_merge(synthetic_video_path, tmp_path):
    LipExtractor.config.TEMPORAL_MERGE_ENABLED = False
    LipExtractor.config.CALCULATE_RTF = False
    LipExtractor.config.MAX_PROBLEMATIC_FRAMES_PERCENTAGE = 100.0

    extractor = LipExtractor()
    npy_out = tmp_path / "output.npy"

    frames, rtf = extractor.extract_lip_frames(synthetic_video_path, output_npy_path=npy_out)
    extractor.close()

    assert frames is not None
    assert frames.ndim == 4
    assert frames.shape[1] == LipExtractor.config.IMG_H
    assert frames.shape[2] == LipExtractor.config.IMG_W
    assert frames.shape[3] == 3
    assert frames.dtype == np.uint8
    assert npy_out.exists()


def test_extraction_with_merge(synthetic_video_path, tmp_path):
    LipExtractor.config.TEMPORAL_MERGE_ENABLED = True
    LipExtractor.config.TEMPORAL_MERGE_METHOD = "mean"
    LipExtractor.config.TEMPORAL_MERGE_SIZE = 4
    LipExtractor.config.MAX_PROBLEMATIC_FRAMES_PERCENTAGE = 100.0

    extractor = LipExtractor()
    npy_out = tmp_path / "output_merged.npy"

    frames, rtf = extractor.extract_lip_frames(synthetic_video_path, output_npy_path=npy_out)
    extractor.close()

    assert frames is not None
    # 30 frames // 4 = 7 merged frames
    assert frames.shape[0] == 7
    assert frames.shape[1] == LipExtractor.config.IMG_H
    assert frames.shape[2] == LipExtractor.config.IMG_W
    assert frames.shape[3] == 3
    assert frames.dtype == np.uint8
    assert npy_out.exists()


def test_output_shape(synthetic_video_path):
    LipExtractor.config.TEMPORAL_MERGE_ENABLED = True
    LipExtractor.config.TEMPORAL_MERGE_METHOD = "motion_weighted"
    LipExtractor.config.TEMPORAL_MERGE_SIZE = 4
    LipExtractor.config.MAX_PROBLEMATIC_FRAMES_PERCENTAGE = 100.0

    extractor = LipExtractor()
    frames, _ = extractor.extract_lip_frames(synthetic_video_path)
    extractor.close()

    assert frames is not None
    assert frames.shape == (7, 50, 75, 3)
