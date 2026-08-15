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
    print(f"\n{'='*60}\n{title}\n{'='*60}", flush=True)


def main():
    logger.info("Welcome to the comprehensive pylipextractor configuration and extraction guide!")

    # --- Section 1: Detailed Configuration Overview ---
    print_section_header("1. Complete Configuration Reference & Override Examples")

    logger.info("Configurations are accessed and overridden directly via LipExtractor.config.")

    # ----------------------------------------------------
    # A. Output Frame Dimensions & Max Limit
    # ----------------------------------------------------
    LipExtractor.config.IMG_H = 50                        # Height of output cropped lip frames (pixels)
    LipExtractor.config.IMG_W = 75                        # Width of output cropped lip frames (pixels)
    LipExtractor.config.MAX_FRAMES = None                 # Limit maximum frames processed from video (None = process all)

    # ----------------------------------------------------
    # B. Lip Cropping Margins & Padding
    # ----------------------------------------------------
    LipExtractor.config.LIP_PROPORTIONAL_MARGIN_X = 0.0   # Horizontal margin as proportion of lip width
    LipExtractor.config.LIP_PROPORTIONAL_MARGIN_Y = 0.0   # Vertical margin as proportion of lip height
    LipExtractor.config.LIP_PADDING_LEFT_PX = 0          # Fixed pixel padding left
    LipExtractor.config.LIP_PADDING_RIGHT_PX = 0         # Fixed pixel padding right
    LipExtractor.config.LIP_PADDING_TOP_PX = 0           # Fixed pixel padding top
    LipExtractor.config.LIP_PADDING_BOTTOM_PX = 0        # Fixed pixel padding bottom

    # ----------------------------------------------------
    # C. MediaPipe FaceLandmarker & Auto-Download Options
    # ----------------------------------------------------
    LipExtractor.config.FACE_LANDMARKER_MODEL_PATH = None  # Explicit local path to face_landmarker.task (if available)
    LipExtractor.config.AUTO_DOWNLOAD_FACE_LANDMARKER = True # Auto-download face_landmarker.task if not in cache
    LipExtractor.config.FACE_LANDMARKER_MODEL_DIR = Path("~/.cache/pylipextractor").expanduser() # Cache directory
    LipExtractor.config.MIN_FACE_DETECTION_CONFIDENCE = 0.5
    LipExtractor.config.MIN_FACE_PRESENCE_CONFIDENCE = 0.5
    LipExtractor.config.MIN_TRACKING_CONFIDENCE = 0.5

    # ----------------------------------------------------
    # D. Bounding Box EMA Temporal Smoothing
    # ----------------------------------------------------
    LipExtractor.config.APPLY_EMA_SMOOTHING = True        # Enable Exponential Moving Average smoothing on crop bbox
    LipExtractor.config.EMA_ALPHA = 0.3                   # Factor between 0.0 and 1.0 (lower = smoother)

    # ----------------------------------------------------
    # E. Temporal Frame Compression (Optional Feature)
    # ----------------------------------------------------
    # Enable temporal frame merging to reduce sequence length (e.g., 300 frames -> 75 frames with size=4)
    # Disabled by default for 1:1 standard extraction.
    LipExtractor.config.TEMPORAL_MERGE_ENABLED = False     # Set to True to enable compression
    LipExtractor.config.TEMPORAL_MERGE_METHOD = "mean"    # Options: "mean", "motion_weighted"
    LipExtractor.config.TEMPORAL_MERGE_SIZE = 4           # Number of consecutive frames merged into 1

    # ----------------------------------------------------
    # F. Visual & Illumination Normalization
    # ----------------------------------------------------
    LipExtractor.config.APPLY_HISTOGRAM_MATCHING = True   # Normalize illumination relative to middle video frame
    LipExtractor.config.BLACK_OUT_NON_LIP_AREAS = False   # Set non-lip pixels outside mask to black

    # ----------------------------------------------------
    # G. FFmpeg Automatic MP4 Conversion & GPU Acceleration
    # ----------------------------------------------------
    LipExtractor.config.CONVERT_TO_MP4_IF_NEEDED = True  # Auto-convert MPG/other formats using FFmpeg
    LipExtractor.config.MP4_TEMP_DIR = Path("./temp_mp4_conversions")
    LipExtractor.config.HW_ACCELERATION_DEVICE = "cuda"   # Hardware device for FFmpeg: "cuda", "auto", or "cpu"

    # ----------------------------------------------------
    # H. Debugging, Logging & Profiling
    # ----------------------------------------------------
    LipExtractor.config.SAVE_DEBUG_FRAMES = False          # Set to True to save intermediate debug images
    LipExtractor.config.MAX_DEBUG_FRAMES = 20
    LipExtractor.config.DEBUG_OUTPUT_DIR = Path("./lip_extraction_debug")
    LipExtractor.config.INCLUDE_LANDMARKS_ON_FINAL_OUTPUT = False
    LipExtractor.config.MAX_PROBLEMATIC_FRAMES_PERCENTAGE = 15.0 # Max allowed % of unprocessable/black frames
    LipExtractor.config.CALCULATE_RTF = False              # Calculate Real-Time Factor (RTF)
    LipExtractor.config.PROFILE_CODE = False               # Generate lip_extraction.prof

    # Print log summary of active settings
    logger.info("Active Configuration Summary:")
    logger.info(f" - Output Dimensions: {LipExtractor.config.IMG_W}x{LipExtractor.config.IMG_H}")
    logger.info(f" - EMA Smoothing: {LipExtractor.config.APPLY_EMA_SMOOTHING} (Alpha: {LipExtractor.config.EMA_ALPHA})")
    logger.info(f" - Histogram Matching: {LipExtractor.config.APPLY_HISTOGRAM_MATCHING}")
    logger.info(f" - Temporal Compression: Enabled={LipExtractor.config.TEMPORAL_MERGE_ENABLED}, Method='{LipExtractor.config.TEMPORAL_MERGE_METHOD}', Merge Size={LipExtractor.config.TEMPORAL_MERGE_SIZE}")
    logger.info(f" - FaceLandmarker Auto-Download: {LipExtractor.config.AUTO_DOWNLOAD_FACE_LANDMARKER}")

    # --- Section 2: Execution Example ---
    print_section_header("2. Performing Lip Frame Extraction")

    input_video_path = Path("bbbmza.mpg") # Replace with your input video file
    output_npy_directory = Path("./output_data")
    output_npy_path = output_npy_directory / (input_video_path.stem + ".npy")

    if not input_video_path.exists():
        logger.warning(f"Video file '{input_video_path}' not found.")
        logger.warning("Please update 'input_video_path' to point to an existing video file to execute extraction.")
        return

    # Initialize LipExtractor (will automatically fetch face_landmarker.task model if needed)
    extractor = LipExtractor()

    try:
        extracted_frames, rtf_value = extractor.extract_lip_frames(
            video_path=input_video_path,
            output_npy_path=output_npy_path
        )

        if extracted_frames is not None:
            logger.info(f"Extraction completed successfully!")
            logger.info(f"Output array shape: {extracted_frames.shape} (Frames x Height x Width x Channels)")
            if rtf_value is not None:
                logger.info(f"Real-Time Factor (RTF): {rtf_value:.4f}")
        else:
            logger.error("Lip frame extraction failed or video exceeded problematic frames threshold.")
    finally:
        extractor.close()

    logger.info("\nExample script finished. Thank you for using pylipextractor!")


if __name__ == "__main__":
    main()
