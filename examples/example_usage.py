# pylipextractor/examples/example_usage.py

import sys
from pathlib import Path
import numpy as np
import shutil
import cv2
import logging
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['GLOG_minloglevel'] = '2'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from pylipextractor.lip_extractor import LipExtractor


def print_section_header(title):
    print(f"\n{'='*50}\n{title}\n{'='*50}", flush=True)


def main():
    logger.info("Welcome to the pylipextractor example!")

    # --- Section 1: Configuration Overview and Customization ---
    print_section_header("1. Configuration Overview")

    logger.info("Default configurations are loaded from pylipextractor/config.py.")
    logger.info("You can inspect and override them directly via LipExtractor.config.")

    logger.info(f"Current default IMG_H: {LipExtractor.config.IMG_H}")
    logger.info(f"Current default IMG_W: {LipExtractor.config.IMG_W}")
    logger.info(f"Current default SAVE_DEBUG_FRAMES: {LipExtractor.config.SAVE_DEBUG_FRAMES}")
    logger.info(f"Current default APPLY_HISTOGRAM_MATCHING: {LipExtractor.config.APPLY_HISTOGRAM_MATCHING}")
    logger.info(f"Current default APPLY_EMA_SMOOTHING: {LipExtractor.config.APPLY_EMA_SMOOTHING}")
    logger.info(f"Current default EMA_ALPHA: {LipExtractor.config.EMA_ALPHA}")
    logger.info(f"Current default CONVERT_TO_MP4_IF_NEEDED: {LipExtractor.config.CONVERT_TO_MP4_IF_NEEDED}")
    logger.info(f"Current default MAX_PROBLEMATIC_FRAMES_PERCENTAGE: {LipExtractor.config.MAX_PROBLEMATIC_FRAMES_PERCENTAGE}")
    logger.info(f"Current default TEMPORAL_MERGE_ENABLED: {LipExtractor.config.TEMPORAL_MERGE_ENABLED}")
    logger.info(f"Current default TEMPORAL_MERGE_METHOD: {LipExtractor.config.TEMPORAL_MERGE_METHOD}")
    logger.info(f"Current default TEMPORAL_MERGE_SIZE: {LipExtractor.config.TEMPORAL_MERGE_SIZE}")

    # Example: Standard extraction run (compression disabled)
    logger.info("\n--- Standard Extraction Example ---")
    LipExtractor.config.TEMPORAL_MERGE_ENABLED = False

    # Example: Research extraction run with temporal compression
    logger.info("\n--- Optional Temporal Compression Configuration Example ---")
    # To enable research-focused temporal compression:
    # LipExtractor.config.TEMPORAL_MERGE_ENABLED = True
    # LipExtractor.config.TEMPORAL_MERGE_METHOD = "motion_weighted" # Or "mean"
    # LipExtractor.config.TEMPORAL_MERGE_SIZE = 4

    input_video_path = Path("bbbmza.mpg")
    if not input_video_path.exists():
        logger.info(f"Input video file '{input_video_path}' not found. Please provide a valid video file.")
        return

    extractor = LipExtractor()
    output_npy_path = Path("./output_data") / (input_video_path.stem + ".npy")

    extracted_frames, rtf_value = extractor.extract_lip_frames(
        video_path=input_video_path,
        output_npy_path=output_npy_path
    )
    extractor.close()

    if extracted_frames is not None:
        logger.info(f"Extraction successful! Extracted shape: {extracted_frames.shape}")
        if rtf_value is not None:
            logger.info(f"RTF: {rtf_value:.4f}")

    logger.info("\nExample script finished. Thank you for using pylipextractor!")


if __name__ == "__main__":
    main()
