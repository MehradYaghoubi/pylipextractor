# pylibextractor_project/example_usage.py

import sys
from pathlib import Path
import numpy as np
import shutil

# Add the parent directory (project root) to sys.path to allow importing package modules
sys.path.append(str(Path(__file__).resolve().parent))

# Import LipExtractor only; config access is now via LipExtractor.config
from pylipextractor.lip_extractor import LipExtractor

def main():
    """
    Main entry point for demonstrating lip frame extraction.
    This script will extract lip frames and save them to a .npy file.
    """
    # 1. Access and optionally modify configuration directly via LipExtractor.config
    # Default configurations are automatically loaded from pylibextractor/config.py
    # To override settings, simply assign new values to LipExtractor.config attributes:
    
    # Example: Override some default settings for this run
    LipExtractor.config.SAVE_DEBUG_FRAMES = True
    LipExtractor.config.MAX_DEBUG_FRAMES = 5 
    LipExtractor.config.INCLUDE_LANDMARKS_ON_FINAL_OUTPUT = False
    LipExtractor.config.APPLY_CLAHE = True 
    # LipExtractor.config.IMG_H = 48 # Example: Change output height
    # LipExtractor.config.IMG_W = 96 # Example: Change output width
    # LipExtractor.config.LIP_PROPORTIONAL_MARGIN_X = 0.20 # Example: Adjust margin

    # Clear previous debug directory if saving debug frames is enabled
    if LipExtractor.config.SAVE_DEBUG_FRAMES and LipExtractor.config.DEBUG_OUTPUT_DIR.exists():
        shutil.rmtree(LipExtractor.config.DEBUG_OUTPUT_DIR)
        print(f"Debug directory '{LipExtractor.config.DEBUG_OUTPUT_DIR}' cleared.")

    # 2. Create an instance of LipExtractor (no arguments needed for config)
    extractor = LipExtractor()

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
    # Naming the NPY file directly from the video's stem (e.g., 'bbar8a.npy')
    output_npy_filename = input_video_path.stem + ".npy" 
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