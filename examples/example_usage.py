# pylipextractor/examples/example_usage.py

import sys
from pathlib import Path
import numpy as np
import shutil
import cv2 # Import OpenCV for displaying an image or saving them easily

# Ensure the project root is in sys.path when running this script directly
# This allows importing 'pylipextractor' as a package.
# This specific line is mainly for local development/testing of the example script itself.
# When pylipextractor is installed via pip, this line is not strictly needed.
project_root = Path(__file__).resolve().parent.parent # Adjust if example_usage.py is not directly in examples/
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from pylipextractor.lip_extractor import LipExtractor

def print_section_header(title):
    """Prints a formatted header for a section."""
    print(f"\n{'='*50}\n{title}\n{'='*50}", flush=True)

def main():
    """
    Comprehensive example demonstrating various functionalities of pylipextractor.
    This script covers configuration, extraction, and post-processing steps.
    """
    print("Welcome to the pylipextractor example!")

    # --- Section 1: Configuration Overview and Customization ---
    print_section_header("1. Configuration Overview")

    print("Default configurations are loaded from pylipextractor/config.py.")
    print("You can inspect and override them directly via LipExtractor.config.")

    print(f"Current default IMG_H: {LipExtractor.config.IMG_H}")
    print(f"Current default IMG_W: {LipExtractor.config.IMG_W}")
    print(f"Current default SAVE_DEBUG_FRAMES: {LipExtractor.config.SAVE_DEBUG_FRAMES}")
    print(f"Current default APPLY_CLAHE: {LipExtractor.config.APPLY_CLAHE}")

    # Example: Override some default settings for this specific run
    print("\n--- Overriding Default Settings for this run ---")
    LipExtractor.config.SAVE_DEBUG_FRAMES = True # Set to True to save debug images
    LipExtractor.config.MAX_DEBUG_FRAMES = 75   # Limit debug frames saved
    LipExtractor.config.INCLUDE_LANDMARKS_ON_FINAL_OUTPUT = False # Don't draw landmarks on final output
    LipExtractor.config.APPLY_CLAHE = True      # Apply illumination normalization
    # LipExtractor.config.IMG_H = 64             # Uncomment to change output height
    # LipExtractor.config.IMG_W = 128            # Uncomment to change output width
    # LipExtractor.config.LIP_PROPORTIONAL_MARGIN_X = 0.20 # Adjust horizontal margin
    # LipExtractor.config.LIP_PROPORTIONAL_MARGIN_Y = 0.30 # Adjust vertical margin
    # LipExtractor.config.MAX_BLACK_FRAMES_PERCENTAGE = 15.0
    # LipExtractor.config.NUM_CPU_CORES = 4s

    print(f"New SAVE_DEBUG_FRAMES setting: {LipExtractor.config.SAVE_DEBUG_FRAMES}")
    print(f"New MAX_DEBUG_FRAMES setting: {LipExtractor.config.MAX_DEBUG_FRAMES}")
    print(f"New INCLUDE_LANDMARKS_ON_FINAL_OUTPUT setting: {LipExtractor.config.INCLUDE_LANDMARKS_ON_FINAL_OUTPUT}")
    print(f"New APPLY_CLAHE setting: {LipExtractor.config.APPLY_CLAHE}")

    # Clear previous debug directory if saving debug frames is enabled
    if LipExtractor.config.SAVE_DEBUG_FRAMES and LipExtractor.config.DEBUG_OUTPUT_DIR.exists():
        try:
            shutil.rmtree(LipExtractor.config.DEBUG_OUTPUT_DIR)
            print(f"\nDebug directory '{LipExtractor.config.DEBUG_OUTPUT_DIR}' cleared for a fresh run.", flush=True)
        except OSError as e:
            print(f"Warning: Could not clear debug directory '{LipExtractor.config.DEBUG_OUTPUT_DIR}': {e}", flush=True)

    # --- Section 2: LipExtractor Initialization ---
    print_section_header("2. Initializing LipExtractor")
    print("Creating an instance of LipExtractor. It will automatically use the current configuration.", flush=True)
    extractor = LipExtractor()
    print("LipExtractor instance created successfully.", flush=True)

    # --- Section 3: Input Video and Output Path Setup ---
    print_section_header("3. Setting Up Input and Output Paths")

    # Define the path to the input video.
    # For this example, place a short video file (e.g., 'bbar8a.mpg')
    # in the 'examples' directory, next to this script.
    input_video_path = Path("swwz9a.mp4") # !!! IMPORTANT: CHANGE THIS TO YOUR VIDEO FILE NAME !!!
    
    if not input_video_path.exists():
        print(f"Error: Video file '{input_video_path.name}' not found.", flush=True)
        print(f"Please place a video file (e.g., 'bbar8a.mpg') in the '{Path(__file__).parent}' directory, or update 'input_video_path'.", flush=True)
        sys.exit(1)
    
    # Define the path for the output .npy file.
    # The output directory 'output_data' will be created if it doesn't exist.
    output_npy_directory = Path("./output_data")
    output_npy_filename = input_video_path.stem + ".npy" # Naming from video's stem (e.g., 'bbar8a.npy')
    output_npy_path = output_npy_directory / output_npy_filename
    
    print(f"Input video: '{input_video_path.name}'")
    print(f"Output .npy file will be saved to: '{output_npy_path}' if extraction is successful.", flush=True)

    # --- Section 4: Performing Lip Frame Extraction ---
    print_section_header("4. Starting Lip Frame Extraction")
    print(f"Extracting lip frames from '{input_video_path.name}'. This may take a moment...", flush=True)
    
    extracted_frames = extractor.extract_lip_frames(
        video_path=input_video_path,
        output_npy_path=output_npy_path
    )

    if extracted_frames is not None:
        print(f"\nExtraction successful! Total extracted frames: {extracted_frames.shape[0]}", flush=True)
        print(f"Dimensions of each extracted frame: {extracted_frames.shape[1]}x{extracted_frames.shape[2]}x{extracted_frames.shape[3]} (HWC, RGB)", flush=True)

        # --- Section 5: Post-Extraction Verification ---
        print_section_header("5. Post-Extraction Verification")
        
        # Optionally check for potential issues (e.g., entirely black frames)
        num_black_frames = sum(1 for frame in extracted_frames if np.sum(frame) == 0)
        if num_black_frames > 0:
            print(f"Warning: {num_black_frames} completely black frames found in the output. "
                  "This might indicate issues during extraction or an invalid input segment.", flush=True)

        # Demonstrate loading the .npy file (to confirm it saved correctly)
        print(f"\nAttempting to load the saved .npy file from '{output_npy_path}'...", flush=True)
        loaded_frames = LipExtractor.extract_npy(output_npy_path)

        if loaded_frames is not None:
            print(f"Successfully loaded {loaded_frames.shape[0]} frames from {output_npy_path}.", flush=True)
            if loaded_frames.shape[0] > 0:
                print(f"First loaded frame shape: {loaded_frames[0].shape}", flush=True)
                # Optional: Display the first extracted frame using OpenCV
                # You might need to adjust color channels if OpenCV expects BGR (pylipextractor outputs RGB)
                # cv2.imshow("First Extracted Lip Frame (RGB)", cv2.cvtColor(loaded_frames[0], cv2.COLOR_RGB2BGR))
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                # print("First frame displayed. Close the window to continue.", flush=True)
            else:
                print("Loaded NPY file is empty.", flush=True)
        else:
            print(f"Failed to load NPY file from '{output_npy_path}'. There might have been an issue during saving.", flush=True)

        # --- Section 6: Converting NPY to Image Files ---
        print_section_header("6. Converting .npy to Individual Image Files")
        print("To visually inspect all extracted frames as individual image files (e.g., PNGs),")
        print(f"please run the separate utility script: `python {Path('save_npy_frames_to_images.py').name}`.")
        print("This script will convert the .npy file into a sequence of images in a specified directory.")

    else:
        print("Lip frame extraction failed or the video clip was rejected (e.g., too many invalid frames or no faces detected).", flush=True)
        print("Please check your input video and consider adjusting LipExtractor.config settings.", flush=True)

    print("\nExample script finished. Thank you for using pylipextractor!", flush=True)

if __name__ == "__main__":
    main()