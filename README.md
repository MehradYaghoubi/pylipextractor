# PyLipExtractor

A Python package for robust lip frame extraction from videos using MediaPipe, designed specifically for visual speech recognition (VSR) and lip-reading tasks. It provides a streamlined, configurable process to convert raw video into ready-to-use lip sequences, ideal for deep learning model training.

## Features

* **Accurate Lip Landmark Detection:** Leverages MediaPipe Face Mesh for precise identification of 3D lip contours, ensuring high fidelity in extraction.
* **Configurable Lip Region Extraction:** Offers fine-grained control over the bounding box around detected lips, allowing for custom proportional margins and padding to capture the desired context.
* **Temporal Smoothing:** Implements a moving average filter on bounding box coordinates to ensure stable and consistent lip frame extraction across video sequences, minimizing jitter.
* **Illumination Normalization (CLAHE):** Applies Adaptive Histogram Equalization (CLAHE) to enhance contrast and normalize illumination, improving the robustness of extracted frames to varying lighting conditions.
* **Flexible Output:** Extracts processed lip frames as NumPy arrays (.npy format), making them directly compatible with deep learning model training pipelines.
* **Debugging Visualizations:** Provides options to save intermediate frames with landmarks and bounding boxes, aiding in visual inspection and troubleshooting of the extraction process.
* **Efficient Video Handling:** Utilizes PyAV for robust and efficient video decoding.

# Demo
https://github.com/user-attachments/assets/cfc7ce9d-dfc0-4a8c-9fc1-1a37299437bf


## Installation

You can easily install pylipextractor using pip directly from PyPI. Ensure you have a compatible Python version (3.8 or newer) installed.

```bash
pip install pylipextractor
```
For Development (Optional)
If you plan to contribute to the project or need an editable installation, follow these steps:

```bash
# First, clone the repository
git clone https://github.com/MehradYaghoubi/pylipextractor.git
cd pylipextractor

# Install the package in editable mode along with its dependencies
pip install -e .
```

## Usage
See example_usage.py in the project root for a full demonstration on how to use the LipExtractor class to process a video and save the lip frames.

Example:
```bash
from pathlib import Path
from pylipextractor.lip_extractor import LipExtractor

# Set your video path (e.g., ensure 'bbar8a.mpg' is in your current working directory or adjust path)
input_video_path = Path("bbar8a.mpg")
output_npy_directory = Path("./output_data")
output_npy_filename = input_video_path.stem + ".npy"
output_npy_path = output_npy_directory / output_npy_filename

# --- Configure LipExtractor settings (optional, defaults are from config.py) ---
# You can override any default setting like this:
LipExtractor.config.SAVE_DEBUG_FRAMES = True
LipExtractor.config.MAX_DEBUG_FRAMES = 10 # Save up to 10 debug frames
LipExtractor.config.APPLY_CLAHE = True   # Ensure CLAHE is applied for contrast
# LipExtractor.config.IMG_H = 64         # Example: Change output frame height

# Create an instance of the extractor
extractor = LipExtractor()

# Perform extraction
print(f"Starting extraction for {input_video_path.name}...")
extracted_frames = extractor.extract_lip_frames(input_video_path, output_npy_path=output_npy_path)

if extracted_frames is not None:
    print(f"Successfully extracted {extracted_frames.shape[0]} frames.")
    print(f"Frames saved to {output_npy_path}")
else:
    print("Extraction failed or no frames were extracted.")
```

To convert the extracted .npy file into individual image frames (e.g., PNGs), use the provided save_npy_frames_to_images.py utility script:
```bash
python save_npy_frames_to_images.py
```

## Dependencies

This project heavily relies on the following open-source libraries:

* **opencv-python:** Essential for core image and video processing operations, including frame manipulation, resizing, and color space conversions.
* **numpy:** Fundamental for efficient numerical computations and handling multi-dimensional data arrays (like image frames).
* **mediapipe:** Utilized for its highly accurate and performant Face Mesh solution, enabling robust facial landmark detection for precise lip localization.
* **av (PyAV):** Provides efficient and reliable reading and writing of various video file formats.
* **Pillow:** A fork of the Python Imaging Library (PIL), often used implicitly by other libraries for image file handling.

## Acknowledgements
I sincerely thank the developers and the vibrant open source community behind all the libraries mentioned in the "Dependencies" section for their valuable work.

## Contributing
Contributions are highly welcome! If you encounter any bugs, have feature requests, or wish to contribute code, please feel free to:

Open an Issue on our GitHub repository.

Submit a Pull Request with your proposed changes.

## License
This project is licensed under the MIT License. See the [LICENSE](https://github.com/MehradYaghoubi/pylipextractor/blob/main/LICENSE) file for more details.
