# pylipextractor/examples/example_usage.py

import sys
from pathlib import Path
import numpy as np
import shutil
import cv2 # Import OpenCV for displaying an image or saving them easily
import logging # Import the logging module
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['GLOG_minloglevel'] = '2'
# Configure logging at the beginning of the script for example purposes.
# In a larger application, this might be done in a separate utility or entry point.
# Set level to INFO to see general progress, or DEBUG to see more detailed internal messages.
# INFO, WARNING, ERROR, CRITICAL
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Get a logger for this specific module
logger = logging.getLogger(__name__)

# Ensure the project root is in sys.path when running this script directly
# This allows importing 'pylipextractor' as a package.
# This specific line is mainly for local development/testing of the example script itself.
# When pylipexractor is installed via pip, this line is not strictly needed.
project_root = Path(__file__).resolve().parent.parent # Adjust if example_usage.py is not directly in examples/
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from pylipextractor.lip_extractor import LipExtractor

def print_section_header(title):
    """Prints a formatted header for a section to the console (always visible)."""
    # Keeping this as print for clear visual separation in the console output,
    # as these are high-level structural markers for the example.
    print(f"\n{'='*50}\n{title}\n{'='*50}", flush=True)

def main():
    """
    Comprehensive example demonstrating various functionalities of pylipexractor.
    This script covers configuration, extraction, and post-processing steps.
    """
    logger.info("Welcome to the pylipextractor example!")

    # --- Section 1: Configuration Overview and Customization ---
    print_section_header("1. Configuration Overview")

    logger.info("Default configurations are loaded from pylipextractor/config.py.")
    logger.info("You can inspect and override them directly via LipExtractor.config.")

    logger.info(f"Current default IMG_H: {LipExtractor.config.IMG_H}")
    logger.info(f"Current default IMG_W: {LipExtractor.config.IMG_W}")
    logger.info(f"Current default SAVE_DEBUG_FRAMES: {LipExtractor.config.SAVE_DEBUG_FRAMES}")
    logger.info(f"Current default APPLY_HISTOGRAM_MATCHING: {LipExtractor.config.APPLY_HISTOGRAM_MATCHING}")
    
    # Show new config options for EMA
    logger.info(f"Current default APPLY_EMA_SMOOTHING: {LipExtractor.config.APPLY_EMA_SMOOTHING}")
    logger.info(f"Current default EMA_ALPHA: {LipExtractor.config.EMA_ALPHA}")

    logger.info(f"Current default CONVERT_TO_MP4_IF_NEEDED: {LipExtractor.config.CONVERT_TO_MP4_IF_NEEDED}")
    logger.info(f"Current default MAX_PROBLEMATIC_FRAMES_PERCENTAGE: {LipExtractor.config.MAX_PROBLEMATIC_FRAMES_PERCENTAGE}")
    
    # Show new config option for blacking out non-lip areas
    logger.info(f"Current default BLACK_OUT_NON_LIP_AREAS: {LipExtractor.config.BLACK_OUT_NON_LIP_AREAS}")


    # Example: Override some default settings for this specific run
    logger.info("\n--- Overriding Default Settings for this run ---")
    LipExtractor.config.SAVE_DEBUG_FRAMES = True # Set to True to save debug images
    LipExtractor.config.MAX_DEBUG_FRAMES = 20    # Limit debug frames saved
    LipExtractor.config.INCLUDE_LANDMARKS_ON_FINAL_OUTPUT = False # Don't draw landmarks on final output
    LipExtractor.config.APPLY_HISTOGRAM_MATCHING = False      # Apply illumination normalization
    
    # Configure EMA Smoothing
    LipExtractor.config.APPLY_EMA_SMOOTHING = True # Enable EMA smoothing
    LipExtractor.config.EMA_ALPHA = 0.2            # Set EMA smoothing factor (e.g., 0.2 for more smoothing)

    # Enable optional MP4 conversion for input videos that are not already MP4
    # This is highly recommended for MPG files or other problematic formats.
    LipExtractor.config.CONVERT_TO_MP4_IF_NEEDED = True
    LipExtractor.config.MP4_TEMP_DIR = Path("./temp_converted_mp4s") # Directory for temporary converted files

    # Adjust the threshold for rejecting a video based on problematic frames
    # If more than this percentage of frames are black/undecipherable, the entire video will be rejected.
    LipExtractor.config.MAX_PROBLEMATIC_FRAMES_PERCENTAGE = 10.0 # Example: Allow up to 10% problematic frames
    
    # NEW: Enable/Disable blacking out non-lip areas. Set to True to see the effect.
    LipExtractor.config.BLACK_OUT_NON_LIP_AREAS = False
    
    LipExtractor.config.IMG_H = 50              # Uncomment to change output height
    LipExtractor.config.IMG_W = 75             # Uncomment to change output width
    LipExtractor.config.LIP_PROPORTIONAL_MARGIN_X = 0.0 # Adjust horizontal margin
    LipExtractor.config.LIP_PROPORTIONAL_MARGIN_Y = 0.0 # Adjust vertical margin
    LipExtractor.config.CALCULATE_RTF = False # Disable RTF calculation for this run
    # LipExtractor.config.MAX_FRAMES = 100        # Uncomment to limit the total number of frames processed

    logger.info(f"New SAVE_DEBUG_FRAMES setting: {LipExtractor.config.SAVE_DEBUG_FRAMES}")
    logger.info(f"New MAX_DEBUG_FRAMES setting: {LipExtractor.config.MAX_DEBUG_FRAMES}")
    logger.info(f"New INCLUDE_LANDMARKS_ON_FINAL_OUTPUT setting: {LipExtractor.config.INCLUDE_LANDMARKS_ON_FINAL_OUTPUT}")
    logger.info(f"New APPLY_HISTOGRAM_MATCHING setting: {LipExtractor.config.APPLY_HISTOGRAM_MATCHING}")
    
    # Log new EMA settings
    logger.info(f"New APPLY_EMA_SMOOTHING setting: {LipExtractor.config.APPLY_EMA_SMOOTHING}")
    logger.info(f"New EMA_ALPHA setting: {LipExtractor.config.EMA_ALPHA}")

    logger.info(f"New CONVERT_TO_MP4_IF_NEEDED setting: {LipExtractor.config.CONVERT_TO_MP4_IF_NEEDED}")
    logger.info(f"New MP4_TEMP_DIR setting: {LipExtractor.config.MP4_TEMP_DIR}")
    logger.info(f"New MAX_PROBLEMATIC_FRAMES_PERCENTAGE setting: {LipExtractor.config.MAX_PROBLEMATIC_FRAMES_PERCENTAGE}")
    
    # Log new BLACK_OUT_NON_LIP_AREAS setting
    logger.info(f"New BLACK_OUT_NON_LIP_AREAS setting: {LipExtractor.config.BLACK_OUT_NON_LIP_AREAS}")


    # Clear previous debug directory if saving debug frames is enabled
    if LipExtractor.config.SAVE_DEBUG_FRAMES and LipExtractor.config.DEBUG_OUTPUT_DIR.exists():
        try:
            shutil.rmtree(LipExtractor.config.DEBUG_OUTPUT_DIR)
            logger.info(f"\nDebug directory '{LipExtractor.config.DEBUG_OUTPUT_DIR}' cleared for a fresh run.")
        except OSError as e:
            logger.warning(f"Could not clear debug directory '{LipExtractor.config.DEBUG_OUTPUT_DIR}': {e}")

    # Clear temporary MP4 directory if conversion is enabled
    if LipExtractor.config.CONVERT_TO_MP4_IF_NEEDED and LipExtractor.config.MP4_TEMP_DIR.exists():
        try:
            shutil.rmtree(LipExtractor.config.MP4_TEMP_DIR)
            logger.info(f"Temporary MP4 conversion directory '{LipExtractor.config.MP4_TEMP_DIR}' cleared for a fresh run.")
        except OSError as e:
            logger.warning(f"Could not clear temporary MP4 directory '{LipExtractor.config.MP4_TEMP_DIR}': {e}")


    # --- Section 2: LipExtractor Initialization ---
    print_section_header("2. Initializing LipExtractor")
    logger.info("Creating an instance of LipExtractor. It will automatically use the current configuration.")
    extractor = LipExtractor()
    logger.info("LipExtractor instance created successfully.")

    # --- Section 3: Input Video and Output Path Setup ---
    print_section_header("3. Setting Up Input and Output Paths")

    # Define the path to the input video.
    # !!! IMPORTANT: REPLACE "path/to/your/video.mp4" WITH THE ACTUAL PATH TO YOUR VIDEO FILE. !!!
    # For this example, you can place a short video file in the 'examples' directory.
    input_video_path = Path("bbbmza.mpg")
    
    if not input_video_path.exists():
        logger.error(f"Error: Video file not found at '{input_video_path}'.")
        logger.error("Please update the 'input_video_path' variable in this script to point to your video file.")
        sys.exit(1)
    
    # Define the path for the output .npy file.
    # The output directory 'output_data' will be created if it doesn't exist.
    output_npy_directory = Path("./output_data")
    output_npy_filename = input_video_path.stem + ".npy" # Naming from video's stem (e.g., 'bbar8a.npy')
    output_npy_path = output_npy_directory / output_npy_filename
    
    logger.info(f"Input video: '{input_video_path.name}'")
    logger.info(f"Output .npy file will be saved to: '{output_npy_path}' if extraction is successful.")

    # --- Section 4: Performing Lip Frame Extraction ---
    print_section_header("4. Starting Lip Frame Extraction")
    logger.info(f"Extracting lip frames from '{input_video_path.name}'. This may take a moment...")
    
    # Changed: Unpack both extracted_frames and rtf_value
    extracted_frames, rtf_value = extractor.extract_lip_frames(
        video_path=input_video_path,
        output_npy_path=output_npy_path
    )

    # --- Section 5: Post-Extraction Actions and RTF Display ---
    print_section_header("5. Post-Extraction Actions and RTF Display")
    if extracted_frames is not None:
        logger.info(f"Extraction successful! Total extracted frames: {extracted_frames.shape[0]}")
        logger.info(f"Dimensions of each extracted frame: {extracted_frames.shape[1]}x{extracted_frames.shape[2]}x{extracted_frames.shape[3]} (HWC, RGB)")

        # Display the RTF value
        if rtf_value is not None:
            logger.info(f"Real-Time Factor (RTF) for '{input_video_path.name}': {rtf_value:.4f}")
        else:
            logger.info(f"RTF for '{input_video_path.name}' not calculated (either disabled or video duration was zero).")

        # Optional: Demonstrate collecting RTF for multiple videos and calculating average
        logger.info("\n--- Example: Collecting RTF for multiple videos ---")
        # In a real scenario, you would loop through a list of video paths
        # For demonstration, we'll simulate adding this video's RTF to a list
        
        all_rtf_values = []
        if rtf_value is not None:
            all_rtf_values.append(rtf_value)
        
        # Simulate processing another video (you would replace this with actual video paths)
        # For example:
        # video_path_2 = Path("another_video.mp4")
        # if video_path_2.exists():
        #     logger.info(f"Processing another video: '{video_path_2.name}'...")
        #     _, rtf_value_2 = extractor.extract_lip_frames(video_path=video_path_2, output_npy_path=None)
        #     if rtf_value_2 is not None:
        #         all_rtf_values.append(rtf_value_2)

        if all_rtf_values:
            average_rtf = np.mean(all_rtf_values)
            logger.info(f"Collected RTF values: {all_rtf_values}")
            logger.info(f"Average RTF across processed videos: {average_rtf:.4f}")
        else:
            logger.info("No RTF values collected to calculate an average.")

        # Optional: Load the saved .npy file to verify
        logger.info(f"\nAttempting to load the saved .npy file from '{output_npy_path}'...")
        # The call to extract_npy is already correct as LipExtractor.extract_npy
        loaded_frames = LipExtractor.extract_npy(output_npy_path)

        if loaded_frames is not None:
            logger.info(f"Successfully loaded {loaded_frames.shape[0]} frames from {output_npy_path}.")
            if loaded_frames.shape[0] > 0:
                logger.info(f"First loaded frame shape: {loaded_frames[0].shape}")
                # Optional: Display the first extracted frame using OpenCV
                # You might need to adjust color channels if OpenCV expects BGR (pylipextractor outputs RGB)
                # cv2.imshow("First Extracted Lip Frame (RGB)", cv2.cvtColor(loaded_frames[0], cv2.COLOR_RGB2BGR))
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                # logger.info("First frame displayed. Close the window to continue.")
            else:
                logger.warning("Loaded NPY file is empty.")
        else:
            logger.error(f"Failed to load NPY file from '{output_npy_path}'. There might have been an issue during saving.")

        # --- Section 6: Converting NPY to Image Files ---
        print_section_header("6. Converting .npy to Individual Image Files")
        logger.info("To visually inspect all extracted frames as individual image files (e.g., PNGs),")
        logger.info(f"please run the separate utility script: `python {Path('save_npy_frames_to_images.py').name}`.")
        logger.info("This script will convert the .npy file into a sequence of images in a specified directory.")

    else:
        logger.error("Lip frame extraction failed or the video clip was rejected (e.g., too many invalid frames or no faces detected).")

    logger.info("\nExample script finished. Thank you for using pylipextractor!")

if __name__ == "__main__":
    main()
