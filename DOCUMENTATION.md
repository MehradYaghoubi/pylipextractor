# PyLipExtractor Documentation

Welcome to the official documentation for PyLipExtractor. This document provides a comprehensive overview of the package, its features, and how to use them effectively.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
  - [General Settings](#general-settings)
  - [MediaPipe Tasks Settings](#mediapipe-tasks-settings)
  - [Temporal Frame Compression Settings](#temporal-frame-compression-settings)
  - [Feature Toggles](#feature-toggles)
  - [Fine-Tuning Parameters](#fine-tuning-parameters)
  - [Debugging and Quality Control](#debugging-and-quality-control)
- [Technical Details](#technical-details)
  - [Lip Detection](#lip-detection)
  - [Temporal Frame Compression](#temporal-frame-compression)
  - [Temporal Smoothing](#temporal-smoothing)
  - [Illumination Normalization](#illumination-normalization)
  - [Video Conversion](#video-conversion)
- [Examples](#examples)
  - [Basic Usage](#basic-usage)
  - [Temporal Compression Usage](#temporal-compression-usage)
  - [Custom Configuration](#custom-configuration)
- [Contributing](#contributing)
- [License](#license)

## Introduction

PyLipExtractor is a Python package designed for robust and accurate lip frame extraction from videos. It leverages Google's MediaPipe Tasks `FaceLandmarker` API to provide high-precision lip landmark detection, making it an ideal tool for applications like lip-reading, Visual Speech Recognition (VSR), facial analysis, and video synchronization.

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
extracted_frames, rtf_value = extractor.extract_lip_frames(video_path, output_path)

if extracted_frames is not None:
    print(f"Successfully extracted {len(extracted_frames)} frames.")
    if rtf_value is not None:
        print(f"Real-Time Factor (RTF): {rtf_value:.4f}")
```

## Configuration

PyLipExtractor is highly configurable. You can change the default settings by modifying the `LipExtractor.config` object before creating an instance of the `LipExtractor` class.

### General Settings

- `IMG_H`: The height of the output frames. (Default: `50`)
- `IMG_W`: The width of the output frames. (Default: `75`)

### MediaPipe Tasks Settings

- `FACE_LANDMARKER_MODEL_PATH`: Custom path to the `.task` model file. If `None`, the model is automatically loaded/downloaded from `FACE_LANDMARKER_MODEL_DIR`. (Default: `None`)
- `AUTO_DOWNLOAD_FACE_LANDMARKER`: Whether to automatically download the official Google FaceLandmarker model file if not present locally. (Default: `True`)
- `FACE_LANDMARKER_MODEL_URL`: Google Cloud storage URL for downloading `face_landmarker.task`.
- `FACE_LANDMARKER_MODEL_DIR`: Directory where downloaded models are cached. (Default: `~/.cache/pylipextractor`)

### Temporal Frame Compression Settings

- `TEMPORAL_MERGE_ENABLED`: Set to `True` to enable optional temporal merging on extracted lip ROI frames. (Default: `False`)
- `TEMPORAL_MERGE_METHOD`: Compression algorithm to use (`"mean"` or `"motion_weighted"`). (Default: `"mean"`)
- `TEMPORAL_MERGE_SIZE`: Number of consecutive frames merged into one frame (e.g. `4` reduces 300 frames to 75 frames). (Default: `4`)

### Feature Toggles

- `APPLY_HISTOGRAM_MATCHING`: Whether to apply histogram matching for illumination normalization. (Default: `True`)
- `APPLY_EMA_SMOOTHING`: Whether to apply Exponential Moving Average (EMA) smoothing to the bounding box coordinates for temporal stability. (Default: `True`)
- `CONVERT_TO_MP4_IF_NEEDED`: Whether to automatically convert non-MP4 videos to a compatible MP4 format using FFmpeg. (Default: `True`)
- `BLACK_OUT_NON_LIP_AREAS`: Whether to black out the areas outside the lip region in the output frames. (Default: `False`)
- `INCLUDE_LANDMARKS_ON_FINAL_OUTPUT`: Whether to draw the detected lip landmarks on the final output frames. (Default: `False`)
- `CALCULATE_RTF`: Set to `True` to calculate and log the Real-Time Factor. (Default: `False`)

### Fine-Tuning Parameters

- `EMA_ALPHA`: The alpha value for EMA smoothing (0.0 to 1.0). (Default: `0.3`)
- `LIP_PROPORTIONAL_MARGIN_X`: The proportional horizontal margin to add around the detected lips. (Default: `0.0`)
- `LIP_PROPORTIONAL_MARGIN_Y`: The proportional vertical margin to add around the detected lips. (Default: `0.0`)

### Debugging and Quality Control

- `SAVE_DEBUG_FRAMES`: Whether to save intermediate frames with landmarks and bounding boxes for debugging. (Default: `False`)
- `MAX_DEBUG_FRAMES`: The maximum number of debug frames to save. (Default: `20`)
- `MAX_PROBLEMATIC_FRAMES_PERCENTAGE`: The maximum percentage of problematic frames allowed in a video before it's rejected. (Default: `15.0`)
- `DEBUG_OUTPUT_DIR`: The directory where the debug frames will be saved. (Default: `Path("./lip_extraction_debug")`)

## Technical Details

### Lip Detection

PyLipExtractor uses MediaPipe's modern `FaceLandmarker` API with `RunningMode.VIDEO` and monotonic frame timestamps (`timestamp_ms`). It extracts 3D face landmarks and identifies lip landmark coordinates using independent library index definitions.

### Temporal Frame Compression

When `TEMPORAL_MERGE_ENABLED` is `True`, extracted lip crops are compressed along the time axis post-crop and resize:
- **`mean`**: Computes the pixel-wise mean across chunks of `merge_size` frames.
- **`motion_weighted`**: Computes pixel-wise weighted averages where frame weights are proportional to frame motion relative to the previous frame ($d_t = \frac{1}{N}\sum|x_t - x_{t-1}|$).

If total frames $N < \text{merge\_size}$, a warning is logged and raw extracted frames are preserved to avoid losing short samples. Remainder frames $N \pmod{\text{merge\_size}}$ are truncated deterministically.

### Temporal Smoothing

To ensure that the lip crops are stable and consistent across frames, PyLipExtractor applies an Exponential Moving Average (EMA) filter to the bounding box coordinates.

### Illumination Normalization

The package uses histogram matching to normalize illumination across video frames relative to a reference middle frame.

### Video Conversion

PyLipExtractor automatically converts videos to MP4 format using FFmpeg with customizable hardware acceleration (`cuda`, `auto`, `cpu`) and lossless video encoding.

## Examples

### Basic Usage

```python
from pylipextractor.lip_extractor import LipExtractor

extractor = LipExtractor()
extracted_frames, rtf_value = extractor.extract_lip_frames("video.mp4", "output.npy")

if extracted_frames is not None:
    print(f"Extracted {len(extracted_frames)} frames.")
```

### Temporal Compression Usage

```python
from pylipextractor.lip_extractor import LipExtractor

LipExtractor.config.TEMPORAL_MERGE_ENABLED = True
LipExtractor.config.TEMPORAL_MERGE_METHOD = "motion_weighted"
LipExtractor.config.TEMPORAL_MERGE_SIZE = 4

extractor = LipExtractor()
extracted_frames, _ = extractor.extract_lip_frames("video.mp4", "output_compressed.npy")

if extracted_frames is not None:
    # 300 input frames -> (75, 50, 75, 3)
    print(f"Extracted compressed shape: {extracted_frames.shape}")
```

### Loading Saved Frames

```python
from pylipextractor.lip_extractor import LipExtractor

loaded_frames = LipExtractor.extract_npy("output.npy")

if loaded_frames is not None:
    print(f"Loaded {len(loaded_frames)} frames.")
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on our [GitHub repository](https://github.com/MehradYaghoubi/pylipextractor).

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/MehradYaghoubi/pylipextractor/blob/main/LICENSE) file for details.
