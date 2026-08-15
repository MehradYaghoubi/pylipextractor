# PyLipExtractor

PyLipExtractor is a powerful and easy-to-use Python package for extracting lip regions from videos. It leverages the high-precision MediaPipe FaceLandmarker API to accurately detect and crop the lip area, providing a stable and reliable tool for researchers and developers working on lip-reading, Visual Speech Recognition (VSR), facial analysis, and other related fields.

For detailed information about the package, its features, and how to use them, please refer to the [**Full Documentation**](DOCUMENTATION.md).

## Key Features

- **Accurate & Stable Lip Detection:** Powered by MediaPipe's modern Tasks API (`FaceLandmarker`).
- **Optional Temporal Compression:** Supports optional temporal merging (`mean` and `motion_weighted`) to compress sequences (e.g., 300 frames → 75 frames). Disabled by default.
- **Temporal Smoothing:** Applies an Exponential Moving Average (EMA) filter for smooth and consistent lip crops.
- **Illumination Normalization:** Includes an optional histogram matching filter to normalize video brightness.
- **Flexible Configuration & Auto-download:** Downloads and caches required task models automatically with configurable options.
- **Customizable Hardware Acceleration:** Choose your preferred processing device (`auto`, `cuda`, or `cpu`) for video conversion.

## Demo

https://github.com/user-attachments/assets/a6841309-3e4d-4b7e-bd0f-b56a5cab28e4

*Original video by Tima Miroshnichenko*

## Installation

```bash
pip install pylipextractor
```

## Quick Start

### 1. Standard Extraction (300 frames → 300 frames)

```python
from pylipextractor.lip_extractor import LipExtractor

# Create a LipExtractor instance
extractor = LipExtractor()

video_path = "path/to/your/video.mp4"
output_path = "output/lip_frames.npy"

# Extract lip frames (Standard 1:1 temporal resolution)
extracted_frames, rtf_value = extractor.extract_lip_frames(video_path, output_path)

if extracted_frames is not None:
    print(f"Successfully extracted {len(extracted_frames)} frames.")
```

### 2. Temporal Compression (300 frames → 75 frames)

*Temporal frame compression is optional and disabled by default.*

```python
from pylipextractor.lip_extractor import LipExtractor

# Configure optional temporal frame compression
LipExtractor.config.TEMPORAL_MERGE_ENABLED = True
LipExtractor.config.TEMPORAL_MERGE_METHOD = "motion_weighted"  # Or "mean"
LipExtractor.config.TEMPORAL_MERGE_SIZE = 4

extractor = LipExtractor()

extracted_frames, _ = extractor.extract_lip_frames("path/to/video.mp4", "output/lip_frames.npy")
# Input: 300 frames -> Output shape: (75, 50, 75, 3)
```

## Documentation

For a deep dive into all the features, configurations, and technical details, please read our [**Comprehensive Documentation**](DOCUMENTATION.md).

## Contributing

Contributions are welcome! Please see our [Contributing Guidelines](https://github.com/MehradYaghoubi/pylipextractor/blob/main/CONTRIBUTING.md) for more details.

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/MehradYaghoubi/pylipextractor/blob/main/LICENSE) file for details.
