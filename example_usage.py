# pylibextractor_project/example_usage.py

import sys
from pathlib import Path
import numpy as np
import shutil

# Add the parent directory (project root) to sys.path to allow importing package modules
sys.path.append(str(Path(__file__).resolve().parent))

from pylibextractor.lip_extractor import LipExtractor
from pylibextractor.config import MainConfig

def main():
    """
    Main entry point for demonstrating lip frame extraction.
    This script will extract lip frames and save them to a .npy file.
    """
    # 1. Initialize configuration


    config = MainConfig().lip_extraction
    
    #config.IMG_H = 48 # Desired height for the output lip frames
    #config.IMG_W = 96 # Desired width for the output lip frames

    #config.LIP_PROPORTIONAL_MARGIN_X = 0.20 # Horizontal margin as a proportion of lip width
    #config.LIP_PROPORTIONAL_MARGIN_Y = 0.30 # Vertical margin as a proportion of lip height

    #config.LIP_PADDING_LEFT_PX = 0
    #config.LIP_PADDING_RIGHT_PX = 0
    #config.LIP_PADDING_TOP_PX = 0
    #config.LIP_PADDING_BOTTOM_PX = 0

    #NUM_CPU_CORES = 4 # Number of CPU cores for parallel processing (if implemented in batch mode).
    #MAX_BLACK_FRAMES_PERCENTAGE = 15.0 # Max allowed percentage of black frames in the output clip.

    # --- Configure Debugging and Output Landmarks ---
    # Set to True to save intermediate debug frames (original, landmarks, cropped, resized).
    # Debug frames will be saved in the directory specified by config.DEBUG_OUTPUT_DIR.

    config.SAVE_DEBUG_FRAMES = True
    #config.DEBUG_OUTPUT_DIR = Path("./my_custom_debug_output") # Directory to save debug frames
    #config.MAX_DEBUG_FRAMES = 5 # Maximum number of debug frames to save per video.

    # Set to True if you want MediaPipe lip landmarks to be drawn directly ON the FINAL
    # extracted NPY frames. These frames will then be saved with dots on them.
    # This is useful for visual inspection of the output, but typically NOT for training
    # models that expect clean, unannotated lip images.
    config.INCLUDE_LANDMARKS_ON_FINAL_OUTPUT = True
    # If you enable landmarks on final output, it's recommended to also save debug frames


    # Clear previous debug directory if saving debug frames is enabled
    if config.SAVE_DEBUG_FRAMES and config.DEBUG_OUTPUT_DIR.exists():
        shutil.rmtree(config.DEBUG_OUTPUT_DIR)
        print(f"Debug directory '{config.DEBUG_OUTPUT_DIR}' cleared.")

    # 2. Create an instance of LipExtractor
    extractor = LipExtractor(config)

    # 3. Define the path to the input video (ensure this file exists)
    # Place a short video file (e.g., 'bbar8a.mpg') in the same directory as this script.
    input_video_path = Path("bbar8a.mpg") # Use your specific video file name here
    
    if not input_video_path.exists():
        print(f"Error: Video file '{input_video_path.name}' not found. "
              "Please place a video file next to 'example_usage.py'.", flush=True)
        sys.exit(1)

    # 4. Define the path for the output .npy file
    # The output directory 'output_data' will be created if it doesn't exist.
    output_npy_directory = Path("./output_data")
    output_npy_filename = input_video_path.stem + "_lips.npy" # Creates filename like 'bbar8a_lips.npy'
    output_npy_path = output_npy_directory / output_npy_filename
    
    print(f"Output .npy file will be saved to: '{output_npy_path}' if extraction is successful.", flush=True)

    # 5. Start the lip frame extraction process
    print(f"Starting lip frame extraction from '{input_video_path.name}'...", flush=True)
    extracted_frames = extractor.extract_lip_frames(input_video_path, output_npy_path=output_npy_path)

    if extracted_frames is not None:
        print(f"Extraction successful. Number of extracted frames: {extracted_frames.shape[0]}", flush=True)
        print(f"Dimensions of each frame: {extracted_frames.shape[1]}x{extracted_frames.shape[2]}x{extracted_frames.shape[3]} (HWC, RGB)", flush=True)

        # Optionally check for black frames in the output
        num_black_frames = sum(1 for frame in extracted_frames if np.sum(frame) == 0)
        if num_black_frames > 0:
            print(f"Note: {num_black_frames} black frames found in the output.", flush=True)

        # 6. Demonstrate loading the .npy file (to confirm it saved correctly)
        print(f"\nAttempting to load the saved .npy file from '{output_npy_path}'...", flush=True)
        loaded_frames = LipExtractor.extract_npy(output_npy_path)

        if loaded_frames is not None:
            print(f"Successfully loaded {loaded_frames.shape[0]} frames from {output_npy_path}.", flush=True)
            print("\nTo view these extracted frames as individual image files (e.g., PNGs),")
            print(f"please run the script: `python {Path('save_npy_frames_to_images.py').name}`.")
        else:
            print(f"Failed to load NPY file from '{output_npy_path}'.", flush=True)

    else:
        print("Lip frame extraction failed or the clip was rejected (e.g., too many black frames).", flush=True)

if __name__ == "__main__":
    main()