# pylipextractor/pylipextractor/lip_extractor.py

import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import av
import urllib.request
from pathlib import Path
import warnings
import math
import subprocess
import time
import cProfile
import pstats
from typing import Tuple, Optional, List, Union
import logging

from pylipextractor.config import MainConfig, LipExtractionConfig
from pylipextractor.temporal_merger import TemporalMerger

BaseOptions = python.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
RunningMode = vision.RunningMode


def _histogram_matching(src, ref):
    """
    Matches the histogram of a source image to a reference image.
    """
    src_ycrcb = cv2.cvtColor(src, cv2.COLOR_RGB2YCrCb)
    ref_ycrcb = cv2.cvtColor(ref, cv2.COLOR_RGB2YCrCb)

    src_y, src_cr, src_cb = cv2.split(src_ycrcb)
    ref_y, _, _ = cv2.split(ref_ycrcb)

    src_hist, _ = np.histogram(src_y.flatten(), 256, [0, 256])
    ref_hist, _ = np.histogram(ref_y.flatten(), 256, [0, 256])

    src_cdf = src_hist.cumsum()
    ref_cdf = ref_hist.cumsum()

    src_cdf_normalized = src_cdf * ref_hist.max() / src_hist.max()

    lookup_table = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        j = 255
        while j >= 0 and src_cdf_normalized[i] < ref_cdf[j]:
            j -= 1
        lookup_table[i] = j

    src_y_matched = cv2.LUT(src_y, lookup_table)
    src_ycrcb_matched = cv2.merge([src_y_matched, src_cr, src_cb])
    src_matched = cv2.cvtColor(src_ycrcb_matched, cv2.COLOR_YCrCb2RGB)

    return src_matched


# --- Setup for logging ---
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# --- Suppress specific MediaPipe warnings and GLOG messages ---
warnings.filterwarnings("ignore", category=UserWarning, module="mediapipe")
os.environ['GLOG_minloglevel'] = '2'


class LipExtractor:
    """
    A class for extracting lip frames from videos using MediaPipe FaceLandmarker.
    This class crops and resizes lip frames, returning them as a NumPy array.
    It also provides utilities for loading previously saved NPY files.
    """
    config: LipExtractionConfig = MainConfig().lip_extraction

    def __init__(self):
        """
        Initializes the LipExtractor.
        Ensures the MediaPipe FaceLandmarker task model is available and initializes FaceLandmarker.
        """
        model_path = self._ensure_model_asset()

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=self.config.MIN_FACE_DETECTION_CONFIDENCE,
            min_face_presence_confidence=self.config.MIN_FACE_PRESENCE_CONFIDENCE,
            min_tracking_confidence=self.config.MIN_TRACKING_CONFIDENCE,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )

        self.face_landmarker = FaceLandmarker.create_from_options(options)
        logger.debug(f"A new MediaPipe FaceLandmarker instance created for process {os.getpid()}.")

        self.ema_smoothed_bbox = None

    def close(self):
        """
        Explicitly closes the underlying MediaPipe FaceLandmarker task runner.
        """
        if hasattr(self, 'face_landmarker') and self.face_landmarker is not None:
            try:
                self.face_landmarker.close()
            except Exception as e:
                logger.debug(f"Error during face_landmarker close: {e}")
            self.face_landmarker = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()

    def _ensure_model_asset(self) -> Path:
        """
        Ensures the required .task model file is available, downloading it if necessary.
        """
        if self.config.FACE_LANDMARKER_MODEL_PATH is not None:
            model_path = Path(self.config.FACE_LANDMARKER_MODEL_PATH)
            if model_path.exists():
                return model_path
            raise FileNotFoundError(f"Configured FACE_LANDMARKER_MODEL_PATH '{model_path}' does not exist.")

        cache_dir = Path(self.config.FACE_LANDMARKER_MODEL_DIR).expanduser()
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached_model_path = cache_dir / "face_landmarker.task"

        if cached_model_path.exists():
            return cached_model_path

        if self.config.AUTO_DOWNLOAD_FACE_LANDMARKER:
            logger.info(f"Downloading FaceLandmarker model from '{self.config.FACE_LANDMARKER_MODEL_URL}'...")
            try:
                urllib.request.urlretrieve(self.config.FACE_LANDMARKER_MODEL_URL, cached_model_path)
                logger.info(f"Model successfully saved to '{cached_model_path}'.")
                return cached_model_path
            except Exception as e:
                if cached_model_path.exists():
                    cached_model_path.unlink()
                raise RuntimeError(f"Failed to download FaceLandmarker model: {e}")
        else:
            raise FileNotFoundError(
                f"FaceLandmarker model file not found at '{cached_model_path}' and AUTO_DOWNLOAD_FACE_LANDMARKER is False."
            )

    @staticmethod
    def _is_black_frame(frame_np: np.ndarray) -> bool:
        """
        Checks if a frame is completely black.
        """
        if frame_np is None or frame_np.size == 0:
            return True
        return np.sum(frame_np) == 0

    def _debug_frame_processing(self, frame, frame_idx, debug_type, current_lip_bbox_val=None, face_landmarks=None):
        """
        Saves debug frames at various stages of processing for visual inspection.
        """
        if not self.config.SAVE_DEBUG_FRAMES or frame_idx >= self.config.MAX_DEBUG_FRAMES:
            return

        debug_dir = self.config.DEBUG_OUTPUT_DIR
        debug_dir.mkdir(parents=True, exist_ok=True)

        display_frame = frame.copy()
        if len(display_frame.shape) == 2:
            display_frame = cv2.cvtColor(display_frame, cv2.COLOR_GRAY2BGR)

        cv2.putText(display_frame, f"{debug_type.capitalize()} Frame {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if debug_type == 'landmarks' and face_landmarks is not None:
            for lm_idx_all, lm in enumerate(face_landmarks):
                x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                color = (0, 255, 0)
                if lm_idx_all in self.config.LIPS_LANDMARK_INDICES or lm_idx_all in [
                    self.config.NOSE_BOTTOM_LANDMARK_INDEX,
                    self.config.CHIN_BOTTOM_LANDMARK_INDEX,
                ]:
                    color = (255, 0, 0)
                cv2.circle(display_frame, (x, y), 1, color, -1)
            if current_lip_bbox_val is not None and len(current_lip_bbox_val) == 4:
                x1, y1, x2, y2 = [int(val) for val in current_lip_bbox_val]
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        if len(display_frame.shape) == 3 and display_frame.shape[2] == 3:
            display_frame_bgr = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
        else:
            display_frame_bgr = display_frame

        cv2.imwrite(str(debug_dir / f"{debug_type}_{frame_idx:04d}.png"), display_frame_bgr)

    @staticmethod
    def _is_hw_encoder_available(encoder: str) -> bool:
        """
        Checks if a specific hardware encoder is available in FFmpeg.
        """
        try:
            result = subprocess.run(['ffmpeg', '-encoders'], capture_output=True, text=True, check=True)
            return encoder in result.stdout
        except (FileNotFoundError, subprocess.CalledProcessError):
            return False

    def _apply_ema_smoothing(self, current_bbox: Optional[np.ndarray]) -> np.ndarray:
        """
        Applies Exponential Moving Average (EMA) to bounding box coordinates.
        """
        if current_bbox is None:
            if self.ema_smoothed_bbox is not None:
                logger.debug(f"EMA: current_bbox is None, using previous smoothed_bbox: {self.ema_smoothed_bbox}")
                return self.ema_smoothed_bbox
            else:
                default_bbox = np.array([0, 0, self.config.IMG_W, self.config.IMG_H], dtype=np.int32)
                logger.debug(f"EMA: current_bbox is None and no previous smoothed_bbox, returning default black frame bbox: {default_bbox}")
                return default_bbox

        current_bbox_np = np.array(current_bbox, dtype=np.float32)
        logger.debug(f"EMA: current_bbox_np: {current_bbox_np}")

        if self.ema_smoothed_bbox is None:
            self.ema_smoothed_bbox = current_bbox_np
            logger.debug(f"EMA: Initializing smoothed_bbox with current_bbox_np: {self.ema_smoothed_bbox}")
        else:
            self.ema_smoothed_bbox = (
                self.config.EMA_ALPHA * current_bbox_np + (1 - self.config.EMA_ALPHA) * self.ema_smoothed_bbox
            )
            logger.debug(f"EMA: Applying smoothing. Old: {self.ema_smoothed_bbox}, New: {self.ema_smoothed_bbox}")

        return self.ema_smoothed_bbox.astype(np.int32)

    def _convert_video_to_mp4(self, input_filepath: Path, output_directory: Path) -> Optional[Path]:
        """
        Converts a video file to MP4 format using FFmpeg.
        """
        output_directory.mkdir(parents=True, exist_ok=True)
        filename_without_ext = input_filepath.stem
        output_filepath = output_directory / f"{filename_without_ext}.mp4"

        device = self.config.HW_ACCELERATION_DEVICE.lower()
        encoder = 'libx264'
        preset = 'veryfast'

        if device == 'cuda' or (device == 'auto' and self._is_hw_encoder_available('h264_nvenc')):
            if self._is_hw_encoder_available('h264_nvenc'):
                encoder = 'h264_nvenc'
                preset = 'p6'
                logger.info("Using NVIDIA GPU (h264_nvenc) for hardware-accelerated video conversion.")
            else:
                logger.warning("NVIDIA GPU (h264_nvenc) not available. Falling back to CPU (libx264).")
                encoder = 'libx264'
                preset = 'veryfast'
        else:
            logger.info("Using CPU (libx264) for video conversion.")

        ffmpeg_command = (
            f"ffmpeg -i \"{input_filepath}\" -c:v {encoder} -preset {preset} "
            f"-crf 0 -an -y -loglevel quiet \"{output_filepath}\""
        )

        logger.info(f"Attempting to convert '{input_filepath.name}' to MP4...")
        try:
            subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True, shell=True)
            logger.info(f"Conversion successful: '{output_filepath.name}'.")
            return output_filepath
        except FileNotFoundError:
            logger.error("FFmpeg not found. Please ensure FFmpeg is installed and added to your system's PATH to use video conversion. Skipping conversion.")
            return None
        except subprocess.CalledProcessError as e:
            logger.error(f"Error converting '{input_filepath.name}' with FFmpeg: {e}")
            logger.error(f"FFmpeg stdout: {e.stdout}")
            logger.error(f"FFmpeg stderr: {e.stderr}")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred during FFmpeg conversion of '{input_filepath.name}': {e}")
            return None

    def extract_lip_frames(
        self, video_path: Union[str, Path], output_npy_path: Optional[Union[str, Path]] = None
    ) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """
        Extracts and processes lip frames from a video.
        """
        original_video_path = Path(video_path)
        current_video_path = original_video_path
        rtf = None

        converted_temp_mp4_path = None
        if self.config.CONVERT_TO_MP4_IF_NEEDED and original_video_path.suffix.lower() not in ['.mp4', '.mov']:
            logger.info(f"'{original_video_path.name}' is not in MP4/MOV format. Attempting conversion...")
            converted_temp_mp4_path = self._convert_video_to_mp4(original_video_path, self.config.MP4_TEMP_DIR)
            if converted_temp_mp4_path:
                current_video_path = converted_temp_mp4_path
            else:
                logger.warning(f"MP4 conversion failed for '{original_video_path.name}'. Attempting to process original file.")
                current_video_path = original_video_path

        if self.config.PROFILE_CODE:
            profiler = cProfile.Profile()
            profiler.enable()

        if not current_video_path.exists():
            logger.error(f"Video file not found at '{current_video_path}'. Processing stopped.")
            return None, None

        if self.config.CALCULATE_RTF:
            start_time = time.time()
            video_duration_seconds = 0.0

        processed_frames_temp_list = []
        self.ema_smoothed_bbox = None

        try:
            container = av.open(str(current_video_path))
            if self.config.CALCULATE_RTF:
                video_duration_seconds = container.duration / 1000000
        except av.AVError as e:
            logger.error(f"Error opening video '{current_video_path.name}' with PyAV: {e}. Processing stopped.")
            return None, None
        except Exception as e:
            logger.error(f"An unexpected error occurred while opening video '{current_video_path.name}': {e}. Processing stopped.")
            return None, None

        if not container.streams.video:
            logger.error(f"No video stream found in '{current_video_path.name}'. Processing stopped.")
            container.close()
            return None, None

        video_stream = container.streams.video[0]
        fps = float(video_stream.average_rate) if video_stream.average_rate and float(video_stream.average_rate) > 0 else 30.0

        total_frames_to_process = self.config.MAX_FRAMES
        if total_frames_to_process is None:
            try:
                frames_from_av = video_stream.frames
                if frames_from_av is not None and frames_from_av > 0:
                    total_frames_to_process = frames_from_av
                else:
                    total_frames_to_process = float('inf')
            except Exception:
                total_frames_to_process = float('inf')

        logger.info(f"Processing video: '{current_video_path.name}' ({total_frames_to_process if total_frames_to_process != float('inf') else 'all available'} frames)...")

        ref_frame = None
        if self.config.APPLY_HISTOGRAM_MATCHING:
            try:
                container.seek(int(container.duration / 2))
                ref_frame_av = next(container.decode(video=0))
                ref_frame = ref_frame_av.to_rgb().to_ndarray()
                container.seek(0)
            except (StopIteration, av.AVError):
                logger.warning("Could not get reference frame for histogram matching. Histogram matching will be disabled.")
                self.config.APPLY_HISTOGRAM_MATCHING = False

        num_problematic_frames = 0
        black_frame = np.zeros((self.config.IMG_H, self.config.IMG_W, 3), dtype=np.uint8)

        try:
            for frame_idx, frame_av in enumerate(container.decode(video=0)):
                if total_frames_to_process != float('inf') and frame_idx >= total_frames_to_process:
                    logger.debug(f"Max frames limit ({total_frames_to_process}) reached. Stopping video processing.")
                    break

                current_frame_is_problematic = False
                processed_lip_frame = black_frame.copy()

                try:
                    image_rgb = frame_av.to_rgb().to_ndarray()
                    original_frame_height, original_frame_width, _ = image_rgb.shape

                    if self.config.SAVE_DEBUG_FRAMES:
                        self._debug_frame_processing(image_rgb, frame_idx, 'original')

                    timestamp_ms = int(frame_idx * 1000.0 / fps)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
                    result = self.face_landmarker.detect_for_video(mp_image, timestamp_ms)

                    raw_lip_bbox: Optional[np.ndarray] = None
                    face_landmarks = None

                    if result.face_landmarks and len(result.face_landmarks) > 0:
                        face_landmarks = result.face_landmarks[0]

                        lip_x_coords = []
                        lip_y_coords = []
                        for idx in self.config.LIPS_LANDMARK_INDICES:
                            if idx < len(face_landmarks):
                                lip_x_coords.append(face_landmarks[idx].x * original_frame_width)
                                lip_y_coords.append(face_landmarks[idx].y * original_frame_height)

                        if lip_x_coords and lip_y_coords:
                            nose_idx = self.config.NOSE_BOTTOM_LANDMARK_INDEX
                            chin_idx = self.config.CHIN_BOTTOM_LANDMARK_INDEX

                            min_y_tight = (
                                face_landmarks[nose_idx].y * original_frame_height
                                if nose_idx < len(face_landmarks)
                                else np.min(lip_y_coords)
                            )
                            max_y_tight = (
                                face_landmarks[chin_idx].y * original_frame_height
                                if chin_idx < len(face_landmarks)
                                else np.max(lip_y_coords)
                            )
                            min_x_tight = np.min(lip_x_coords)
                            max_x_tight = np.max(lip_x_coords)

                            lip_centroid_x = (min_x_tight + max_x_tight) / 2
                            lip_centroid_y = (min_y_tight + max_y_tight) / 2

                            initial_tight_width = max_x_tight - min_x_tight
                            initial_tight_height = max_y_tight - min_y_tight

                            target_bbox_width = (
                                initial_tight_width * (1 + 2 * self.config.LIP_PROPORTIONAL_MARGIN_X)
                                + self.config.LIP_PADDING_LEFT_PX
                                + self.config.LIP_PADDING_RIGHT_PX
                            )
                            target_bbox_height = (
                                initial_tight_height * (1 + 2 * self.config.LIP_PROPORTIONAL_MARGIN_Y)
                                + self.config.LIP_PADDING_TOP_PX
                                + self.config.LIP_PADDING_BOTTOM_PX
                            )

                            target_aspect_ratio = self.config.IMG_W / self.config.IMG_H
                            current_bbox_aspect_ratio = target_bbox_width / target_bbox_height

                            if current_bbox_aspect_ratio > target_aspect_ratio:
                                target_bbox_height = target_bbox_width / target_aspect_ratio
                            else:
                                target_bbox_width = target_bbox_height * target_aspect_ratio

                            x1_proposed = lip_centroid_x - target_bbox_width / 2
                            y1_proposed = lip_centroid_y - target_bbox_height / 2
                            x2_proposed = lip_centroid_x + target_bbox_width / 2
                            y2_proposed = lip_centroid_y + target_bbox_height / 2

                            x1_final = int(max(0, x1_proposed))
                            y1_final = int(max(0, y1_proposed))
                            x2_final = int(min(original_frame_width, x2_proposed))
                            y2_final = int(min(original_frame_height, y2_proposed))

                            if (x2_final - x1_final) > 0 and (y2_final - y1_final) > 0:
                                raw_lip_bbox = np.array([x1_final, y1_final, x2_final, y2_final], dtype=np.int32)
                            else:
                                logger.warning(
                                    f"Frame {frame_idx}: Calculated final bounding box has zero or negative dimensions after centering/adjustment. Marking as problematic."
                                )
                                current_frame_is_problematic = True
                        else:
                            logger.warning(f"Frame {frame_idx}: No lip coordinates found. Marking as problematic.")
                            current_frame_is_problematic = True
                    else:
                        logger.warning(f"Frame {frame_idx}: No face detected. Marking as problematic.")
                        current_frame_is_problematic = True

                    smoothed_lip_bbox_np = None
                    if self.config.APPLY_EMA_SMOOTHING:
                        smoothed_lip_bbox_np = self._apply_ema_smoothing(raw_lip_bbox)
                    else:
                        smoothed_lip_bbox_np = raw_lip_bbox

                    if self.config.SAVE_DEBUG_FRAMES:
                        self._debug_frame_processing(
                            image_rgb,
                            frame_idx,
                            'landmarks',
                            raw_lip_bbox.tolist() if raw_lip_bbox is not None else None,
                            face_landmarks,
                        )
                        if self.config.APPLY_EMA_SMOOTHING:
                            self._debug_frame_processing(
                                image_rgb,
                                frame_idx,
                                'smoothed_bbox',
                                smoothed_lip_bbox_np.tolist() if smoothed_lip_bbox_np is not None else None,
                                face_landmarks,
                            )

                    if (
                        not current_frame_is_problematic
                        and smoothed_lip_bbox_np is not None
                        and (smoothed_lip_bbox_np[2] > smoothed_lip_bbox_np[0])
                        and (smoothed_lip_bbox_np[3] > smoothed_lip_bbox_np[1])
                    ):
                        x1_smoothed, y1_smoothed, x2_smoothed, y2_smoothed = smoothed_lip_bbox_np.tolist()

                        x1_smoothed = max(0, min(original_frame_width, x1_smoothed))
                        y1_smoothed = max(0, min(original_frame_height, y1_smoothed))
                        x2_smoothed = max(0, min(original_frame_width, x2_smoothed))
                        y2_smoothed = max(0, min(original_frame_height, y2_smoothed))

                        if (x2_smoothed - x1_smoothed) <= 0 or (y2_smoothed - y1_smoothed) <= 0:
                            logger.warning(
                                f"Frame {frame_idx}: Smoothed bounding box became invalid after clamping. Marking as problematic."
                            )
                            current_frame_is_problematic = True
                        else:
                            lip_cropped_frame = image_rgb[y1_smoothed:y2_smoothed, x1_smoothed:x2_smoothed]

                            current_crop_width = lip_cropped_frame.shape[1]
                            current_crop_height = lip_cropped_frame.shape[0]

                            if current_crop_width > self.config.IMG_W or current_crop_height > self.config.IMG_H:
                                interpolation_method = cv2.INTER_AREA
                            else:
                                interpolation_method = cv2.INTER_LANCZOS4

                            final_resized_lip = cv2.resize(
                                lip_cropped_frame, (self.config.IMG_W, self.config.IMG_H), interpolation=interpolation_method
                            )

                            processed_lip_frame = final_resized_lip.copy()

                            if self.config.APPLY_HISTOGRAM_MATCHING and ref_frame is not None:
                                processed_lip_frame = _histogram_matching(processed_lip_frame, ref_frame)

                            if self.config.INCLUDE_LANDMARKS_ON_FINAL_OUTPUT and face_landmarks and smoothed_lip_bbox_np is not None:
                                x_offset_for_mapping = smoothed_lip_bbox_np[0]
                                y_offset_for_mapping = smoothed_lip_bbox_np[1]

                                width_cropped_for_mapping = smoothed_lip_bbox_np[2] - smoothed_lip_bbox_np[0]
                                height_cropped_for_mapping = smoothed_lip_bbox_np[3] - smoothed_lip_bbox_np[1]

                                if width_cropped_for_mapping > 0 and height_cropped_for_mapping > 0:
                                    scale_x_to_output = self.config.IMG_W / width_cropped_for_mapping
                                    scale_y_to_output = self.config.IMG_H / height_cropped_for_mapping

                                    for lm_idx in self.config.LIPS_LANDMARK_INDICES:
                                        if lm_idx < len(face_landmarks):
                                            orig_x_px = face_landmarks[lm_idx].x * original_frame_width
                                            orig_y_px = face_landmarks[lm_idx].y * original_frame_height

                                            relative_x_px = orig_x_px - x_offset_for_mapping
                                            relative_y_px = orig_y_px - y_offset_for_mapping

                                            final_x_lm = int(relative_x_px * scale_x_to_output)
                                            final_y_lm = int(relative_y_px * scale_y_to_output)

                                            final_x_lm = max(0, min(self.config.IMG_W - 1, final_x_lm))
                                            final_y_lm = max(0, min(self.config.IMG_H - 1, final_y_lm))

                                            cv2.circle(processed_lip_frame, (final_x_lm, final_y_lm), 1, (0, 255, 0), -1)
                                else:
                                    logger.warning(
                                        f"Frame {frame_idx}: Cropped dimensions for landmark mapping are invalid. Skipping landmark drawing."
                                    )
                    else:
                        logger.warning(
                            f"Frame {frame_idx}: Problematic frame (face/lip detection failed or invalid bounding box). Appending black frame."
                        )
                        current_frame_is_problematic = True

                except Exception as e:
                    logger.warning(f"Unexpected error processing frame {frame_idx} from '{current_video_path.name}': {e}. Appending black frame.")
                    current_frame_is_problematic = True
                    if self.config.APPLY_EMA_SMOOTHING:
                        self._apply_ema_smoothing(None)

                if current_frame_is_problematic:
                    processed_frames_temp_list.append(black_frame.copy())
                    num_problematic_frames += 1
                    if self.config.SAVE_DEBUG_FRAMES:
                        self._debug_frame_processing(black_frame, frame_idx, 'black_generated')
                else:
                    processed_frames_temp_list.append(processed_lip_frame)

        finally:
            container.close()
            if converted_temp_mp4_path and converted_temp_mp4_path.exists():
                try:
                    os.remove(str(converted_temp_mp4_path))
                    logger.info(f"Cleaned up temporary MP4 file: '{converted_temp_mp4_path.name}'.")
                except Exception as e:
                    logger.warning(f"Could not remove temporary MP4 file '{converted_temp_mp4_path.name}': {e}")

        if not processed_frames_temp_list:
            logger.warning(f"No frames could be processed from video '{current_video_path.name}'. Returning `None, None`.")
            return None, None

        final_processed_np_frames = np.array(processed_frames_temp_list, dtype=np.uint8)

        if self.config.MAX_FRAMES is not None:
            if final_processed_np_frames.shape[0] > self.config.MAX_FRAMES:
                final_processed_np_frames = final_processed_np_frames[: self.config.MAX_FRAMES]
                logger.info(f"Video truncated to {self.config.MAX_FRAMES} frames as per configuration.")
            elif final_processed_np_frames.shape[0] < self.config.MAX_FRAMES:
                padding_needed = self.config.MAX_FRAMES - final_processed_np_frames.shape[0]
                black_padding = np.zeros((padding_needed, self.config.IMG_H, self.config.IMG_W, 3), dtype=np.uint8)
                final_processed_np_frames = np.concatenate((final_processed_np_frames, black_padding), axis=0)
                num_problematic_frames += padding_needed
                logger.info(f"Video padded with {padding_needed} black frames to reach {self.config.MAX_FRAMES} frames as per configuration.")

        total_output_frames = final_processed_np_frames.shape[0]

        if total_output_frames == 0 or (
            total_output_frames > 0 and (num_problematic_frames / total_output_frames) * 100 > self.config.MAX_PROBLEMATIC_FRAMES_PERCENTAGE
        ):
            percentage_problematic_frames = (num_problematic_frames / total_output_frames) * 100 if total_output_frames > 0 else 100
            logger.error(
                f"Clip '{original_video_path.name}' rejected: {percentage_problematic_frames:.2f}% problematic frames (exceeds {self.config.MAX_PROBLEMATIC_FRAMES_PERCENTAGE}% allowed). Returning None, None."
            )
            return None, None
        elif num_problematic_frames > 0:
            percentage_problematic_frames = (num_problematic_frames / total_output_frames) * 100
            logger.info(f"Clip '{original_video_path.name}': {percentage_problematic_frames:.2f}% problematic frames found. Clip retained.")

        # --- Optional Temporal Frame Compression ---
        original_num_frames = final_processed_np_frames.shape[0]
        if self.config.TEMPORAL_MERGE_ENABLED:
            logger.info(
                f"Temporal merge enabled. Processing {original_num_frames} frames "
                f"using method='{self.config.TEMPORAL_MERGE_METHOD}', size={self.config.TEMPORAL_MERGE_SIZE}."
            )
            final_processed_np_frames = TemporalMerger.merge(
                frames=final_processed_np_frames,
                method=self.config.TEMPORAL_MERGE_METHOD,
                merge_size=self.config.TEMPORAL_MERGE_SIZE,
            )
            compressed_num_frames = final_processed_np_frames.shape[0]
            compression_ratio = original_num_frames / compressed_num_frames if compressed_num_frames > 0 else 1.0
            logger.info(
                f"Temporal merge completed. Original frames: {original_num_frames}, "
                f"Output frames: {compressed_num_frames}, Temporal merge: {self.config.TEMPORAL_MERGE_METHOD}, "
                f"Merge size: {self.config.TEMPORAL_MERGE_SIZE}, Compression ratio: {compression_ratio:.2f}x"
            )

        if output_npy_path:
            output_npy_path = Path(output_npy_path)
            output_npy_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(output_npy_path, final_processed_np_frames)
            logger.info(f"Extracted frames saved to '{output_npy_path}'.")

        if self.config.CALCULATE_RTF:
            end_time = time.time()
            processing_time = end_time - start_time
            if video_duration_seconds > 0:
                rtf = processing_time / video_duration_seconds
                logger.info(f"RTF Calculation: Video Duration: {video_duration_seconds:.2f}s, Processing Time: {processing_time:.2f}s, RTF: {rtf:.4f}")
            else:
                logger.warning("RTF Calculation: Could not determine video duration. RTF cannot be calculated.")
                rtf = None
        else:
            rtf = None

        if self.config.PROFILE_CODE:
            profiler.disable()
            stats = pstats.Stats(profiler).sort_stats('cumulative')
            stats.dump_stats('lip_extraction.prof')
            logger.info("Profiling results saved to 'lip_extraction.prof'")

        return final_processed_np_frames, rtf

    @staticmethod
    def extract_npy(npy_path: Union[str, Path]) -> Optional[np.ndarray]:
        """
        Loads a NumPy array from a .npy file.
        """
        npy_path = Path(npy_path)
        if not npy_path.exists():
            logger.error(f"NPY file not found at '{npy_path}'.")
            return None

        try:
            data = np.load(npy_path)
            logger.info(f"Successfully loaded NPY file from '{npy_path}'. Shape: {data.shape}")
            return data
        except Exception as e:
            logger.error(f"Error loading NPY file '{npy_path}': {e}")
            return None
