# PyLipExtractor

A Python package for robust lip frame extraction from videos using MediaPipe, designed for visual speech recognition and lip-reading tasks.

## Features

* **Accurate Lip Landmark Detection:** Utilizes MediaPipe Face Mesh for precise identification of lip contours.
* **Customizable Lip Region Extraction:** Allows fine-tuning of the bounding box around the detected lips with proportional margins and padding.
* **Temporal Smoothing:** Implements a moving average filter to ensure stable and consistent lip frame extraction across video sequences.
* **Flexible Output:** Extracts processed lip frames as NumPy arrays (.npy format), ready for deep learning model training.
* **Debugging Visualizations:** Provides options to save intermediate frames with landmarks and bounding boxes for visual inspection.

## Installation

Currently, you can install the dependencies and run the package locally:

```bash
# First, clone the repository
git clone [https://github.com/your_username/pylibextractor.git](https://github.com/your_username/pylibextractor.git)
cd pylibextractor

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate # On Windows: .\venv\Scripts\activate

# Install the required dependencies
pip install -e . # This installs your package in editable mode and its dependencies
```

(Note: Once the package is fully built and potentially published to PyPI, the installation command will be simpler, e.g., pip install pylibextractor)3

## Usage
See example_usage.py in the project root for a demonstration on how to use the LipExtractor class to process a video and save the lip frames.

Example:
```bash
# In example_usage.py (after setting up input_video_path and output_npy_path)
from pylibextractor.lip_extractor import LipExtractor
from pylibextractor.config import MainConfig

config = MainConfig().lip_extraction
extractor = LipExtractor(config)
extracted_frames = extractor.extract_lip_frames(input_video_path, output_npy_path=output_npy_path)
```

To convert the extracted .npy file into individual image frames, use save_npy_frames_to_images.py.

## Dependencies

This project heavily relies on the following open-source libraries:

opencv-python: Used for core image and video processing operations.

numpy: Essential for numerical computations and handling data arrays.

mediapipe: Utilized for robust face detection and facial landmark localization.

av (PyAV): Provides efficient reading and writing of video files.

## Acknowledgements
We sincerely appreciate the developers and the open-source community behind all the libraries mentioned in the "Dependencies" section for their invaluable work, which has made this project possible.

## Contributing
Contributions are welcome! Please feel free to open issues or pull requests.

## License
This project is licensed under the MIT License - see the LICENSE file for details.