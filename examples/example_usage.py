# pylipextractor/examples/example_usage.py

import sys
from pathlib import Path
import numpy as np
import shutil
import logging
import os

# Set environment variables to reduce TensorFlow and MediaPipe logging verbosity
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['GLOG_minloglevel'] = '2'

# Configure basic logging for the script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to sys.path to allow importing 'pylipextractor'
# This is mainly for local development. It's not needed if the package is installed via pip.
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from pylipextractor.lip_extractor import LipExtractor

def print_section_header(title):
    """Prints a formatted header for a section to the console."""
    print(f"\n{'='*50}\n{title}\n{'='*50}", flush=True)

def main():
    """
    A comprehensive example demonstrating various functionalities of PyLipExtractor.
    """
    logger.info("Welcome to the comprehensive PyLipExtractor example!")

    # --- Section 1: Comprehensive Configuration ---
    print_section_header("1. Comprehensive Configuration")
    
    # You can override default settings before creating a LipExtractor instance.
    # Here, we'll configure several options to showcase the package's flexibility.
    
    logger.info("Overriding default settings for a customized extraction...")
    
    # --- General Settings ---
    LipExtractor.config.IMG_H = 60  # Set output frame height
    LipExtractor.config.IMG_W = 90  # Set output frame width
    
    # --- Feature Toggles ---
    LipExtractor.config.APPLY_CLAHE = True  # Enable illumination normalization
    LipExtractor.config.APPLY_EMA_SMOOTHING = True  # Enable temporal smoothing
    LipExtractor.config.CONVERT_TO_MP4_IF_NEEDED = True  # Enable automatic video conversion
    LipExtractor.config.BLACK_OUT_NON_LIP_AREAS = False # Black out areas outside the lip region
    
    # --- Fine-Tuning Parameters ---
    LipExtractor.config.EMA_ALPHA = 0.3  # Set smoothing factor (lower is smoother)
    LipExtractor.config.LIP_PROPORTIONAL_MARGIN_X = 0.1  # Add horizontal margin around lips
    LipExtractor.config.LIP_PROPORTIONAL_MARGIN_Y = 0.2  # Add vertical margin around lips
    
    # --- Debugging and Quality Control ---
    LipExtractor.config.SAVE_DEBUG_FRAMES = True  # Save intermediate frames for inspection
    LipExtractor.config.MAX_DEBUG_FRAMES = 50  # Limit the number of saved debug frames
    LipExtractor.config.MAX_PROBLEMATIC_FRAMES_PERCENTAGE = 20.0  # Reject videos with >20% problematic frames
    
    logger.info(f"Custom configuration applied.")
    logger.info(f"  - Output dimensions: {LipExtractor.config.IMG_H}x{LipExtractor.config.IMG_W}")
    logger.info(f"  - CLAHE: {LipExtractor.config.APPLY_CLAHE}, Smoothing: {LipExtractor.config.APPLY_EMA_SMOOTHING} (alpha={LipExtractor.config.EMA_ALPHA})")
    logger.info(f"  - Debug frames: {LipExtractor.config.SAVE_DEBUG_FRAMES}, Blackout: {LipExtractor.config.BLACK_OUT_NON_LIP_AREAS}")

    # --- Section 2: Initialization ---
    print_section_header("2. Initializing LipExtractor")
    try:
        extractor = LipExtractor()
        logger.info("LipExtractor initialized successfully with custom configuration.")
    except Exception as e:
        logger.error(f"Failed to initialize LipExtractor: {e}")
        return

    # --- Section 3: Input and Output Paths ---
    print_section_header("3. Setting Up Input and Output Paths")
    
    # IMPORTANT: Place your video file in the project's root directory or provide a full path.
    input_video_path =  "bbafzp.mpg" #"your_video.mpg"  # <--- CHANGE THIS TO YOUR VIDEO FILE
    output_npy_directory = "output_data"
    
    # The output .npy file will be named based on the video file stem.
    output_npy_path = os.path.join(output_npy_directory, os.path.splitext(os.path.basename(input_video_path))[0] + ".npy")
    
    logger.info(f"Input video: '{input_video_path}'")
    logger.info(f"Output will be saved to: '{output_npy_path}'")

    # Clean up previous output directories for a fresh run
    debug_dir = LipExtractor.config.DEBUG_OUTPUT_DIR
    if LipExtractor.config.SAVE_DEBUG_FRAMES and os.path.exists(debug_dir):
        shutil.rmtree(debug_dir)
        logger.info(f"Cleared debug directory: '{debug_dir}'")
    
    if not os.path.exists(output_npy_directory):
        os.makedirs(output_npy_directory)

    # --- Section 4: Extraction ---
    print_section_header("4. Starting Lip Frame Extraction")
    
    if not os.path.exists(input_video_path):
        logger.warning(f"Input video not found at '{input_video_path}'.")
        logger.warning("Please place a video file in the root directory and update 'input_video_path'.")
    else:
        try:
            extracted_frames = extractor.extract_lip_frames(
                video_path=input_video_path,
                output_npy_path=output_npy_path
            )

            # --- Section 5: Verification ---
            print_section_header("5. Verifying the Output")
            if extracted_frames is not None:
                logger.info("Extraction successful!")
                logger.info(f"  - Extracted {extracted_frames.shape[0]} frames.")
                logger.info(f"  - Frame dimensions: {extracted_frames.shape[1]}x{extracted_frames.shape[2]}")
                
                # Verify the saved file
                if os.path.exists(output_npy_path):
                    loaded_frames = np.load(output_npy_path)
                    logger.info(f"  - Successfully loaded '{output_npy_path}' with {loaded_frames.shape[0]} frames.")
                else:
                    logger.error("  - Output file was not saved.")
            else:
                logger.error("Lip frame extraction failed. Check logs for details.")
        except Exception as e:
            logger.error(f"An error occurred during extraction: {e}")

    logger.info("\nExample finished.")

if __name__ == "__main__":
    main()