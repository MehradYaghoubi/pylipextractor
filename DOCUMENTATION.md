# PyLipExtractor Documentation

Welcome to the official documentation for PyLipExtractor. This document provides a comprehensive overview of the package, its features, and how to use them effectively.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
  - [General Settings](#general-settings)
  - [Feature Toggles](#feature-toggles)
  - [Fine-Tuning Parameters](#fine-tuning-parameters)
  - [Debugging and Quality Control](#debugging-and-quality-control)
- [Technical Details](#technical-details)
  - [Lip Detection](#lip-detection)
  - [Temporal Smoothing](#temporal-smoothing)
  - [Illumination Normalization](#illumination-normalization)
  - [Video Conversion](#video-conversion)
- [Examples](#examples)
  - [Basic Usage](#basic-usage)
  - [Custom Configuration](#custom-configuration)
- [Contributing](#contributing)
- [License](#license)

## Introduction

PyLipExtractor is a Python package designed for robust and accurate lip frame extraction from videos. It leverages Google's MediaPipe Face Mesh to provide high-precision lip landmark detection, making it an ideal tool for applications like lip-reading, facial analysis, and video synchronization.

## Installation

You can install PyLipExtractor using `pip`:

```bash
pip install pylipextractor
```

For development purposes, you can clone the repository and install it in editable mode:

```bash
git clone https://github.com/MehradYaghoubi/pylipextractor.git
cd pylipextractor
pip install -e .
```

## Quick Start

Here's a quick example of how to use PyLipExtractor to extract lip frames from a video:

```python
from pylipextractor.lip_extractor import LipExtractor

# Create a LipExtractor instance
extractor = LipExtractor()

# Specify the path to your video and the output path for the .npy file
video_path = "path/to/your/video.mp4"
output_path = "output/lip_frames.npy"

# Extract the lip frames
lip_frames = extractor.extract_lip_frames(video_path, output_path)

if lip_frames is not None:
    print(f"Successfully extracted {len(lip_frames)} frames.")
```

## Configuration

PyLipExtractor is highly configurable. You can change the default settings by modifying the `LipExtractor.config` object before creating an instance of the `LipExtractor` class.

### General Settings

- `IMG_H`: The height of the output frames. (Default: `50`)
- `IMG_W`: The width of the output frames. (Default: `70`)

### Feature Toggles

- `APPLY_HISTOGRAM_MATCHING`: Whether to apply histogram matching for illumination normalization. (Default: `True`)
- `APPLY_EMA_SMOOTHING`: Whether to apply Exponential Moving Average (EMA) smoothing to the bounding box coordinates for temporal stability. (Default: `True`)
- `CONVERT_TO_MP4_IF_NEEDED`: Whether to automatically convert non-MP4 videos to a compatible MP4 format using FFmpeg. (Default: `True`)
- `BLACK_OUT_NON_LIP_AREAS`: Whether to black out the areas outside the lip region in the output frames. (Default: `False`)
- `INCLUDE_LANDMARKS_ON_FINAL_OUTPUT`: Whether to draw the detected lip landmarks on the final output frames. This is useful for debugging and visualization. (Default: `False`)

### Fine-Tuning Parameters

- `EMA_ALPHA`: The alpha value for the EMA smoothing. A lower value results in smoother bounding boxes. (Default: `0.2`)
- `LIP_PROPORTIONAL_MARGIN_X`: The proportional horizontal margin to add around the detected lips. (Default: `0.0`)
- `LIP_PROPORTIONAL_MARGIN_Y`: The proportional vertical margin to add around the detected lips. (Default: `0.02`)

### Debugging and Quality Control

- `SAVE_DEBUG_FRAMES`: Whether to save intermediate frames with landmarks and bounding boxes for debugging. (Default: `False`)
- `MAX_DEBUG_FRAMES`: The maximum number of debug frames to save. (Default: `20`)
- `MAX_PROBLEMATIC_FRAMES_PERCENTAGE`: The maximum percentage of problematic (e.g., black or unprocessable) frames allowed in a video before it's rejected. (Default: `10.0`)
- `DEBUG_OUTPUT_DIR`: The directory where the debug frames will be saved. (Default: `debug_output`)

## Technical Details

### Lip Detection

PyLipExtractor uses the MediaPipe Face Mesh to detect 468 3D face landmarks. From these landmarks, it identifies the ones that correspond to the lips and calculates a bounding box around them.

### Temporal Smoothing

To ensure that the lip crops are stable and consistent across frames, PyLipExtractor applies an Exponential Moving Average (EMA) filter to the bounding box coordinates. This helps to reduce jitter and create a smoother video of the lip movements.

### Illumination Normalization

The package uses histogram matching to normalize the illumination of the video frames. This method works by matching the histogram of each frame to the histogram of a reference frame (the middle frame of the video). This ensures that the color and brightness of the frames are consistent throughout the video, which can significantly improve the performance of lip-reading models.

### Video Conversion

PyLipExtractor can automatically convert videos to a compatible MP4 format using FFmpeg. This is particularly useful for handling videos in formats like MPG, which can sometimes cause issues with video processing libraries.

## Examples

### Basic Usage

This example shows the most basic usage of the package, with default settings.

```python
from pylipextractor.lip_extractor import LipExtractor

extractor = LipExtractor()
lip_frames = extractor.extract_lip_frames("video.mp4", "output.npy")

if lip_frames is not None:
    print(f"Extracted {len(lip_frames)} frames.")
```

### Custom Configuration

This example demonstrates how to customize the extraction process by overriding the default settings.

```python
from pylipextractor.lip_extractor import LipExtractor

# Customize the configuration
LipExtractor.config.IMG_H = 80
LipExtractor.config.IMG_W = 120
LipExtractor.config.APPLY_CLAHE = False
LipExtractor.config.LIP_PROPORTIONAL_MARGIN_Y = 0.3

extractor = LipExtractor()
lip_frames = extractor.extract_lip_frames("video.mp4", "output.npy")

if lip_frames is not None:
    print(f"Extracted {len(lip_frames)} frames with custom settings.")
```

## Contributing

Contributions are welcome! If you have a feature request, bug report, or want to contribute to the code, please open an issue or submit a pull request on our [GitHub repository](https://github.com/MehradYaghoubi/pylipextractor).

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/MehradYaghoubi/pylipextractor/blob/main/LICENSE) file for details.
