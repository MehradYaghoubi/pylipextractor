# PyLipExtractor

An advanced, customizable lip extraction tool using MediaPipe Face Mesh for precise and stable lip region cropping from videos.

## Features

* **Accurate Lip Landmark Detection:** Leverages MediaPipe Face Mesh for precise identification of 3D lip contours, ensuring high fidelity in extraction.
* **Configurable Lip Region Extraction:** Offers fine-grained control over the bounding box around detected lips, allowing for custom proportional margins and padding to capture the desired context.
* **Temporal Smoothing:** Implements a moving average filter on bounding box coordinates to ensure stable and consistent lip frame extraction across video sequences.
* **Illumination Normalization (CLAHE):** Applies Adaptive Histogram Equalization (CLAHE) to enhance contrast and normalize illumination, improving the robustness of extracted frames to varying lighting conditions.
* **Optional Video Conversion (FFmpeg):** Can automatically convert various video formats (e.g., MPG) to MP4 internally using FFmpeg, enhancing compatibility and robustness with MediaPipe and PyAV. This can resolve issues with specific problematic video codecs.
* **Flexible Output & Quality Control:** Extracts processed lip frames as NumPy arrays (.npy format). Includes a configurable threshold (`MAX_PROBLEMATIC_FRAMES_PERCENTAGE`) to automatically reject video clips with too many unprocessable (black) frames, ensuring output data quality.
* **Debugging Visualizations:** Provides options to save intermediate frames with landmarks and bounding boxes, aiding in visual inspection and troubleshooting of the extraction process.
* **Efficient Video Handling:** Utilizes PyAV for robust and efficient video decoding.

# Demo
https://github.com/user-attachments/assets/a6841309-3e4d-4b7e-bd0f-b56a5cab28e4



Original video by Tima Miroshnichenko

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
Content is how the package is used in the (examples) folder

## Dependencies

This project heavily relies on the following open-source libraries:

* **opencv-python:** Essential for core image and video processing operations, including frame manipulation, resizing, and color space conversions.
* **numpy:** Fundamental for efficient numerical computations and handling multi-dimensional data arrays (like image frames).
* **mediapipe:** Utilized for its highly accurate and performant Face Mesh solution, enabling robust facial landmark detection for precise lip localization.
* **av (PyAV):** Provides efficient and reliable reading and writing of various video file formats.
* **Pillow:** A fork of the Python Imaging Library (PIL), often used implicitly by other libraries for image file handling.
* **FFmpeg (External Tool):** Required for the optional automatic video format conversion feature. It must be installed separately on your system and accessible via the system's PATH.

## Acknowledgements
I sincerely thank the developers and the vibrant open source community behind all the libraries mentioned in the "Dependencies" section for their valuable work.

## Contributing
Contributions are highly welcome! If you encounter any bugs, have feature requests, or wish to contribute code, please feel free to:

Open an Issue on our GitHub repository.

Submit a Pull Request with your proposed changes.

## License
This project is licensed under the MIT License. See the [LICENSE](https://github.com/MehradYaghoubi/pylipextractor/blob/main/LICENSE) file for more details.
