# pylipextractor/examples/save_npy_frames_to_images.py

import sys
from pathlib import Path
import numpy as np
import cv2
import shutil
import logging # NEW: Import the logging module

# Configure logging at the beginning of the script for example purposes.
# Set level to INFO to see general progress, DEBUG for more detailed messages.
# INFO, WARNING, ERROR, CRITICAL
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Get a logger for this specific module
logger = logging.getLogger(__name__)

# Add the parent directory (project root) to sys.path to allow importing package modules
# This specific line is mainly for local development/testing of the example script itself.
# When pylipextractor is installed via pip, this line is not strictly needed.
project_root = Path(__file__).resolve().parent.parent # Adjust if save_npy_frames_to_images.py is not directly in examples/
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from pylipextractor.lip_extractor import LipExtractor

def main():
    """
    Loads an .npy file containing extracted frames and saves each frame
    as a separate image file (e.g., PNG) in a specified output directory.
    """
    logger.info("--- Starting NPY Frame to Image Converter ---")

    # 1. Define the path to your input .npy file
    # Make sure this path points to the .npy file created by example_usage.py
    # This path now expects the NPY file to be named directly after the video (e.g., 'bbar8a.npy')
    input_npy_path = Path("./output_data/swwz9a.npy") 

    # 2. Define the output directory for saving individual image frames
    output_images_dir = Path("./extracted_lip_images")

    # Check if the input NPY file exists
    if not input_npy_path.exists():
        logger.error(f"Input NPY file not found at '{input_npy_path}'. "
                     "Please ensure 'example_usage.py' has been run successfully first.")
        sys.exit(1)

    # Load the NPY file
    logger.info(f"Attempting to load NPY file from: '{input_npy_path}'...")
    loaded_frames = LipExtractor.extract_npy(input_npy_path) # LipExtractor.extract_npy already uses logging internally

    if loaded_frames is None:
        logger.error(f"Failed to load frames from '{input_npy_path}'. Exiting.")
        sys.exit(1)

    # Ensure the output directory exists and is empty (or create it)
    if output_images_dir.exists():
        logger.info(f"Clearing existing output directory: '{output_images_dir}'...")
        try:
            shutil.rmtree(output_images_dir)
        except OSError as e:
            logger.warning(f"Could not clear output directory '{output_images_dir}': {e}")
    output_images_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created output directory for images: '{output_images_dir}'")

    # 3. Save each frame as an image
    logger.info(f"Saving {loaded_frames.shape[0]} frames to '{output_images_dir}'...")
    for i, frame in enumerate(loaded_frames):
        # Convert RGB frame to BGR for OpenCV (cv2.imwrite expects BGR)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Define filename (e.g., frame_0000.png, frame_0001.png)
        image_filename = f"frame_{i:04d}.png" # :04d ensures zero-padding (e.g., 0001, 0010)
        image_path = output_images_dir / image_filename
        
        cv2.imwrite(str(image_path), frame_bgr)
        
        # Optional: Print progress (use logger.info)
        if (i + 1) % 100 == 0 or (i + 1) == loaded_frames.shape[0]:
            logger.info(f"    Saved {i + 1} / {loaded_frames.shape[0]} frames.")

    logger.info(f"Successfully saved all {loaded_frames.shape[0]} frames as images to '{output_images_dir}'.")
    logger.info("--- NPY Frame to Image Conversion Complete ---")

if __name__ == "__main__":
    main()