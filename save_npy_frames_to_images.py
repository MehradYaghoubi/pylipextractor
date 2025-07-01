# pylibextractor_project/save_npy_frames_to_images.py

import sys
from pathlib import Path
import numpy as np
import cv2
import shutil

# Add the parent directory (project root) to sys.path to allow importing package modules
sys.path.append(str(Path(__file__).resolve().parent))

from pylipextractor.lip_extractor import LipExtractor

def main():
    """
    Loads an .npy file containing extracted frames and saves each frame
    as a separate image file (e.g., PNG) in a specified output directory.
    """
    print("--- Starting NPY Frame to Image Converter ---", flush=True)

    # 1. Define the path to your input .npy file
    # Make sure this path points to the .npy file created by example_usage.py
    # This path now expects the NPY file to be named directly after the video (e.g., 'bbar8a.npy')
    input_npy_path = Path("./output_data/bbar8a.npy") 

    # 2. Define the output directory for saving individual image frames
    output_images_dir = Path("./extracted_lip_images") 

    # Check if the input NPY file exists
    if not input_npy_path.exists():
        print(f"Error: Input NPY file not found at '{input_npy_path}'. "
              "Please ensure 'example_usage.py' has been run successfully first.", flush=True)
        sys.exit(1)

    # Load the NPY file
    print(f"Attempting to load NPY file from: '{input_npy_path}'...", flush=True)
    loaded_frames = LipExtractor.extract_npy(input_npy_path)

    if loaded_frames is None:
        print(f"Failed to load frames from '{input_npy_path}'. Exiting.", flush=True)
        sys.exit(1)

    # Ensure the output directory exists and is empty (or create it)
    if output_images_dir.exists():
        print(f"Clearing existing output directory: '{output_images_dir}'...", flush=True)
        shutil.rmtree(output_images_dir)
    output_images_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created output directory for images: '{output_images_dir}'", flush=True)

    # 3. Save each frame as an image
    print(f"Saving {loaded_frames.shape[0]} frames to '{output_images_dir}'...", flush=True)
    for i, frame in enumerate(loaded_frames):
        # Convert RGB frame to BGR for OpenCV (cv2.imwrite expects BGR)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Define filename (e.g., frame_0000.png, frame_0001.png)
        image_filename = f"frame_{i:04d}.png" # :04d ensures zero-padding (e.g., 0001, 0010)
        image_path = output_images_dir / image_filename
        
        cv2.imwrite(str(image_path), frame_bgr)
        
        # Optional: Print progress
        if (i + 1) % 100 == 0 or (i + 1) == loaded_frames.shape[0]:
            print(f"    Saved {i + 1} / {loaded_frames.shape[0]} frames.", flush=True)

    print(f"Successfully saved all {loaded_frames.shape[0]} frames as images to '{output_images_dir}'.", flush=True)
    print("--- NPY Frame to Image Conversion Complete ---", flush=True)

if __name__ == "__main__":
    main()