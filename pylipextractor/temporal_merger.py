# pylipextractor/pylipextractor/temporal_merger.py

import logging
import numpy as np
from typing import Union

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class TemporalMerger:
    """
    Independent utility for compressing video frame sequences temporally.
    Operates on NumPy arrays of shape (N, H, W, C) or (N, H, W).
    """

    @staticmethod
    def merge(
        frames: np.ndarray,
        method: str = "mean",
        merge_size: int = 4,
        epsilon: float = 1e-6,
    ) -> np.ndarray:
        """
        Compresses frame sequence length from N to M (N // merge_size).

        Args:
            frames (np.ndarray): Input video frames with shape (N, H, W, C) or (N, H, W).
            method (str): Compression method ("mean" or "motion_weighted").
            merge_size (int): Number of consecutive frames to merge into one.
            epsilon (float): Small constant to avoid division by zero in motion weighting.

        Returns:
            np.ndarray: Compressed frames array with dtype uint8.
        """
        if not isinstance(frames, np.ndarray) or frames.ndim not in (3, 4):
            raise ValueError("Input frames must be a 3D or 4D NumPy array.")

        if merge_size <= 0:
            raise ValueError(f"merge_size must be a positive integer, got {merge_size}.")

        num_frames = frames.shape[0]

        if num_frames == 0:
            return frames.copy()

        if num_frames < merge_size:
            logger.warning(
                f"Total frames ({num_frames}) is less than merge_size ({merge_size}). "
                "Skipping temporal merge and returning original frames."
            )
            return frames.astype(np.uint8)

        method = method.lower()
        if method not in ("mean", "motion_weighted"):
            raise ValueError(f"Unsupported merge method '{method}'. Supported methods: 'mean', 'motion_weighted'.")

        # Truncate remainder frames to ensure deterministic chunk size
        usable_frames = (num_frames // merge_size) * merge_size
        if usable_frames < num_frames:
            logger.debug(
                f"Truncating {num_frames - usable_frames} remainder frame(s) "
                f"from total {num_frames} frames to fit merge_size {merge_size}."
            )
        truncated_frames = frames[:usable_frames]

        float_frames = truncated_frames.astype(np.float32)
        num_groups = usable_frames // merge_size

        if method == "mean":
            # Reshape into groups: (num_groups, merge_size, H, W, C) or (num_groups, merge_size, H, W)
            shape = (num_groups, merge_size) + float_frames.shape[1:]
            grouped = float_frames.reshape(shape)
            merged_float = np.mean(grouped, axis=1)

        elif method == "motion_weighted":
            # Calculate frame difference d_t relative to previous actual frame in sequence
            # d_0 = 0
            motion = np.zeros(usable_frames, dtype=np.float32)
            if usable_frames > 1:
                frame_diffs = np.abs(float_frames[1:] - float_frames[:-1])
                # Compute mean absolute pixel difference per frame
                diff_axes = tuple(range(1, float_frames.ndim))
                motion[1:] = np.mean(frame_diffs, axis=diff_axes)

            shape = (num_groups, merge_size) + float_frames.shape[1:]
            grouped_frames = float_frames.reshape(shape)
            grouped_motion = motion.reshape(num_groups, merge_size)

            # Compute weights for each group: alpha_i = (d_i + epsilon) / sum(d_j + epsilon)
            weights = grouped_motion + epsilon
            sum_weights = np.sum(weights, axis=1, keepdims=True)
            normalized_weights = weights / sum_weights

            # Expand weight dimensions to match frame shape: (num_groups, merge_size, 1, 1, 1) or (num_groups, merge_size, 1, 1)
            expand_dims = (slice(None), slice(None)) + (np.newaxis,) * (float_frames.ndim - 1)
            expanded_weights = normalized_weights[expand_dims]

            merged_float = np.sum(grouped_frames * expanded_weights, axis=1)

        merged_uint8 = np.clip(merged_float, 0, 255).astype(np.uint8)
        return merged_uint8
