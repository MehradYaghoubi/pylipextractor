# PyLipExtractor

**PyLipExtractor** is an open-source Python toolkit for extracting aligned lip-region frames from video for **visual speech recognition (VSR), lip-reading research, and computer-vision experiments**. It uses MediaPipe's FaceLandmarker Tasks API to detect facial landmarks, then crops and stabilizes the mouth region across frames.

If you are searching for a Python lip-reading preprocessing tool, a MediaPipe lip cropper, or a video-to-mouth-frame pipeline for VSR, PyLipExtractor provides a configurable starting point.

[![PyPI version](https://img.shields.io/pypi/v/pylipextractor)](https://pypi.org/project/pylipextractor/)
[![Python versions](https://img.shields.io/pypi/pyversions/pylipextractor)](https://pypi.org/project/pylipextractor/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## What it does

Given a video, PyLipExtractor detects face and mouth landmarks and produces cropped lip frames for downstream analysis or model training. It is a preprocessing utility; it does not transcribe speech or include a trained lip-reading model.

### Features

- **FaceLandmarker Tasks API** for modern MediaPipe face and lip landmark detection.
- **Temporal smoothing** with an Exponential Moving Average (EMA) to reduce crop jitter.
- **Optional temporal frame compression** with mean or motion-weighted merging.
- **Optional illumination normalization** for more consistent frames.
- **Configurable processing** for different video and research workflows.
- **CPU, CUDA, or automatic device selection** for supported video conversion paths.
- **Automatic model download and cache** for the FaceLandmarker task model.

## Install

Python 3.9 or newer is required. MediaPipe stays on the 0.10 release line for this version of PyLipExtractor; MediaPipe 1.x is outside the supported range.

```bash
python -m pip install pylipextractor
```

To install the compatible MediaPipe version explicitly:

```bash
python -m pip install "mediapipe>=0.10.35,<1.0.0"
```

## Quick start

```python
from pylipextractor.lip_extractor import LipExtractor

extractor = LipExtractor()
video_path = "path/to/video.mp4"
output_path = "output/lip_frames.npy"

frames, real_time_factor = extractor.extract_lip_frames(video_path, output_path)

if frames is not None:
    print(f"Extracted {len(frames)} lip frames")
    if real_time_factor is not None:
        print(f"Real-time factor: {real_time_factor:.4f}")
```

Temporal frame compression is optional and disabled by default:

```python
from pylipextractor.lip_extractor import LipExtractor

LipExtractor.config.TEMPORAL_MERGE_ENABLED = True
LipExtractor.config.TEMPORAL_MERGE_METHOD = "motion_weighted"  # or "mean"
LipExtractor.config.TEMPORAL_MERGE_SIZE = 4

extractor = LipExtractor()
frames, _ = extractor.extract_lip_frames("path/to/video.mp4")
```

## Demo

A short demonstration of the lip-region extraction workflow:

https://github.com/user-attachments/assets/a6841309-3e4d-4b7e-bd0f-b56a5cab28e4

*Original video by Tima Miroshnichenko.*

## Documentation

- [Full documentation](DOCUMENTATION.md)
- [Usage example](examples/example_usage.py)
- [Report a bug or request a feature](https://github.com/MehradYaghoubi/pylipextractor/issues)

## Intended use

PyLipExtractor is designed for researchers and developers preparing video inputs for visual speech recognition, lip-reading, and facial-video analysis. Review the source dataset's terms and obtain appropriate consent before processing video of people.

## Citation and attribution

If this package supports your work, please cite or link to the repository:

> Mehrad Yaghoubi, PyLipExtractor, https://github.com/MehradYaghoubi/pylipextractor

## Contributing

Contributions, bug reports, and feature requests are welcome. Please open an issue before proposing substantial changes.

## License

PyLipExtractor is released under the [MIT License](LICENSE).
