# PyLipExtractor

PyLipExtractor is a powerful and easy-to-use Python package for extracting lip regions from videos. It leverages the high-precision MediaPipe Face Mesh to accurately detect and crop the lip area, providing a stable and reliable tool for researchers and developers working on lip-reading, facial analysis, and other related fields.

For detailed information about the package, its features, and how to use them, please refer to the [**Full Documentation**](DOCUMENTATION.md).

## Key Features

- **Accurate & Stable Lip Detection:** Utilizes MediaPipe Face Mesh for precise landmark detection.
- **Temporal Smoothing:** Applies an Exponential Moving Average (EMA) filter for smooth and consistent lip crops.
- **Illumination Normalization:** Includes an optional histogram matching filter to normalize video brightness.
- **Flexible Configuration:** Offers a wide range of customizable settings.
- **Customizable Hardware Acceleration:** Choose your preferred processing device (`auto`, `cuda`, or `cpu`) for video conversion. The package leverages NVIDIA GPU acceleration (if available) for faster processing and uses lossless compression to ensure no quality is lost.

## Demo

https://github.com/user-attachments/assets/a6841309-3e4d-4b7e-bd0f-b56a5cab28e4

*Original video by Tima Miroshnichenko*

## Installation

```bash
pip install pylipextractor
```

## Quick Start

Here is a basic example of how to use PyLipExtractor. For a more detailed and comprehensive example, please see `examples/example_usage.py`.

```python
from pylipextractor.lip_extractor import LipExtractor

# Create a LipExtractor instance
extractor = LipExtractor()

# Specify the path to your video
video_path = "path/to/your/video.mp4"
output_path = "output/lip_frames.npy"

# Extract the lip frames
extracted_frames, rtf_value = extractor.extract_lip_frames(video_path, output_path)

if extracted_frames is not None:
    print(f"Successfully extracted {len(extracted_frames)} frames.")
    if rtf_value is not None:
        print(f"Real-Time Factor (RTF): {rtf_value:.4f}")
```

## Documentation

For a deep dive into all the features, configurations, and technical details, please read our [**Comprehensive Documentation**](DOCUMENTATION.md).

## Contributing

Contributions are welcome! Please see our [Contributing Guidelines](https://github.com/MehradYaghoubi/pylipextractor/blob/main/CONTRIBUTING.md) for more details.

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/MehradYaghoubi/pylipextractor/blob/main/LICENSE) file for details.
