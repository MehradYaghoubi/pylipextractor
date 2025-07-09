# pylipextractor/pylipextractor/lip_extractor.py

import os
import cv2
import numpy as np
import mediapipe as mp
import av
from pathlib import Path
import warnings
import math
import subprocess
from typing import Tuple, Optional, List, Union
import logging # NEW: Import the logging module

# --- Setup for logging ---
# It's crucial to set up a NullHandler for libraries to prevent "No handlers could be found for logger" messages
# if the main application doesn't configure logging. The user's application will then configure it as desired.
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
# --- End logging setup ---

# --- Suppress specific MediaPipe warnings and GLOG messages ---
warnings.filterwarnings("ignore", category=UserWarning, module="mediapipe")
os.environ['GLOG_minloglevel'] = '2' # Suppress all GLOG messages below WARNING level.
# --- End suppression ---

# Access the pre-defined lip connections from MediaPipe
# FACEMESH_LIPS is a frozenset of tuples, where each tuple represents a connection (start_idx, end_idx)
_LIP_CONNECTIONS = mp.solutions.face_mesh.FACEMESH_LIPS

# Extract all unique landmark indices involved in these connections
LIPS_MESH_LANDMARKS_INDICES = sorted(list(set([
    idx for connection in _LIP_CONNECTIONS for idx in connection
])))

# Import MainConfig here so LipExtractor can manage it as a class-level attribute
from pylipextractor.config import MainConfig, LipExtractionConfig # Import LipExtractionConfig as well


class LipExtractor:
    """
    A class for extracting lip frames from videos using MediaPipe Face Mesh.
    This class crops and resizes lip frames, returning them as a NumPy array.
    It also provides utilities for loading previously saved NPY files.
    """
    # Class-level attribute to hold MediaPipe model instance, initialized once for all objects
    _mp_face_mesh_instance = None 

    # Class-level attribute to hold the configuration.
    # Users can access and modify this directly: LipExtractor.config.IMG_H = ...
    config: LipExtractionConfig = MainConfig().lip_extraction 

    def __init__(self):
        """
        Initializes the LipExtractor.
        Configuration is managed by the class-level attribute `LipExtractor.config`.
        """
        # Ensure MediaPipe model is loaded/initialized for this process
        self._initialize_mediapipe_if_not_set()
        # Assign the class-level MediaPipe instance to the object for convenient access
        self.mp_face_mesh = LipExtractor._mp_face_mesh_instance

        # History for temporal smoothing of bounding boxes
        self.bbox_history = [] 
        # Use config for smoothing window size
        self.SMOOTHING_WINDOW_SIZE = self.config.SMOOTHING_WINDOW_SIZE 

        # Initialize CLAHE object if enabled in config
        self.clahe_obj = None
        if self.config.APPLY_CLAHE:
            self.clahe_obj = cv2.createCLAHE(
                clipLimit=self.config.CLAHE_CLIP_LIMIT,
                tileGridSize=self.config.CLAHE_TILE_GRID_SIZE
            )

    @classmethod
    def _initialize_mediapipe_if_not_set(cls):
        """
        Initializes the MediaPipe Face Mesh model if it hasn't been initialized yet.
        This ensures the model is loaded only once across all instances and processes.
        """
        if cls._mp_face_mesh_instance is None:
            cls._mp_face_mesh_instance = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1, # Assume one dominant face in the video
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                refine_landmarks=True # Use refined landmarks for better accuracy
            )
            logger.debug(f"MediaPipe Face Mesh model loaded for process {os.getpid()}.") # Changed to debug

    @staticmethod
    def _is_black_frame(frame_np: np.ndarray) -> bool:
        """
        Checks if a frame is completely black (all pixel values are zero).
        
        Args:
            frame_np (np.ndarray): NumPy array representing the image frame.
            
        Returns:
            bool: `True` if the frame is black or `None`/empty, otherwise `False`.
        """
        if frame_np is None or frame_np.size == 0:
            return True
        return np.sum(frame_np) == 0

    def _debug_frame_processing(self, frame, frame_idx, debug_type, current_lip_bbox=None, mp_face_landmarks=None):
        """
        Saves debug frames at various stages of processing for visual inspection.
        
        Args:
            frame (np.array): Image frame (assumed RGB format).
            frame_idx (int): Current frame index.
            debug_type (str): Type of debug frame ('original', 'landmarks', 'clahe_applied', 'black_generated').
            current_lip_bbox (tuple, optional): Tuple (x1, y1, x2, y2) of the calculated lip bounding box.
            mp_face_landmarks (mp.solution.face_mesh.NormalizedLandmarkList, optional): Raw MediaPipe landmarks.
        """
        if not self.config.SAVE_DEBUG_FRAMES or frame_idx >= self.config.MAX_DEBUG_FRAMES:
            return

        debug_dir = self.config.DEBUG_OUTPUT_DIR
        debug_dir.mkdir(parents=True, exist_ok=True)

        display_frame = frame.copy()
        # Ensure frame is 3-channel for text overlay if it's grayscale
        if len(display_frame.shape) == 2: # If grayscale, convert to BGR for text overlay and saving
            display_frame = cv2.cvtColor(display_frame, cv2.COLOR_GRAY2BGR)
        
        cv2.putText(display_frame, f"{debug_type.capitalize()} Frame {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if debug_type == 'landmarks' and mp_face_landmarks is not None:
            # Draw all detected face mesh landmarks (for general debug)
            for lm_idx_all, lm in enumerate(mp_face_landmarks.landmark):
                x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                color = (0, 255, 0) # Default green for all landmarks
                if lm_idx_all in LIPS_MESH_LANDMARKS_INDICES:
                    color = (255, 0, 0) # Red for actual lip landmarks to highlight them
                cv2.circle(display_frame, (x, y), 1, color, -1) 

            # Draw the calculated bounding box for the lip
            if current_lip_bbox:
                x1, y1, x2, y2 = current_lip_bbox
                cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2) # Red rectangle for lip bbox
        
        # Convert to BGR for OpenCV saving
        if len(display_frame.shape) == 3 and display_frame.shape[2] == 3: # Only convert if it's already RGB
            display_frame_bgr = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
        else: # Otherwise, it might already be BGR or grayscale, keep as is
            display_frame_bgr = display_frame
            
        cv2.imwrite(str(debug_dir / f"{debug_type}_{frame_idx:04d}.png"), display_frame_bgr)


    def _apply_temporal_smoothing(self, current_bbox: Optional[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
        """
        Applies a moving average to lip bounding boxes to reduce temporal jumps and instability.
        If detection fails, it attempts to retain the last successful bounding box or uses a default.
        
        Args:
            current_bbox (Tuple[int, int, int, int], optional): Bounding box (x1, y1, x2, y2) for the current frame.
                                                            `None` if no face/lip was detected.
        Returns:
            Tuple[int, int, int, int]: The smoothed bounding box.
        """
        # If current detection is None and history is also empty, return a default black frame bbox
        if current_bbox is None and not self.bbox_history:
            # Return a default bounding box that would result in a black image (no specific region)
            # This is a fallback to ensure dimensions are valid, though it signifies a failed detection.
            return (0, 0, self.config.IMG_W, self.config.IMG_H) 

        # Add current bounding box to history.
        # If current_bbox is None, repeat the last successful bbox for smoother transitions,
        # or append a default "empty" bbox if history is empty.
        if current_bbox is None:
            if self.bbox_history:
                self.bbox_history.append(self.bbox_history[-1]) 
            else:
                self.bbox_history.append((0, 0, self.config.IMG_W, self.config.IMG_H)) 
        else:
            self.bbox_history.append(current_bbox)
        
        # Maintain the defined window size for the history
        if len(self.bbox_history) > self.SMOOTHING_WINDOW_SIZE:
            self.bbox_history.pop(0) # Remove the oldest entry

        # Calculate the average for each coordinate across the history window
        x1_smoothed = int(np.mean([bbox[0] for bbox in self.bbox_history]))
        y1_smoothed = int(np.mean([bbox[1] for bbox in self.bbox_history]))
        x2_smoothed = int(np.mean([bbox[2] for bbox in self.bbox_history]))
        y2_smoothed = int(np.mean([bbox[3] for bbox in self.bbox_history]))

        return (x1_smoothed, y1_smoothed, x2_smoothed, y2_smoothed)

    @staticmethod
    def _convert_video_to_mp4(input_filepath: Path, output_directory: Path) -> Optional[Path]:
        """
        Converts a video file to MP4 format using FFmpeg.
        
        Args:
            input_filepath (Path): Path to the input video file.
            output_directory (Path): Directory where the MP4 output file will be saved.
                                     This directory will be created if it does not exist.
                                     
        Returns:
            Optional[Path]: Path to the converted MP4 file, or `None` if conversion fails.
        """
        output_directory.mkdir(parents=True, exist_ok=True)
        filename_without_ext = input_filepath.stem
        output_filepath = output_directory / f"{filename_without_ext}.mp4"

        # FFmpeg command for converting to MP4 with H.264 video and AAC audio
        ffmpeg_command = [
            'ffmpeg',
            '-i', str(input_filepath),
            '-c:v', 'libx264',
            '-preset', 'veryfast',
            '-crf', '23',
            '-c:a', 'aac',
            '-b:a', '128k',
            '-y',
            '-loglevel', 'quiet', # Suppress FFmpeg console output
            str(output_filepath)
        ]

        logger.info(f"Attempting to convert '{input_filepath.name}' to MP4...") # Changed to info
        try:
            subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True)
            logger.info(f"Conversion successful: '{output_filepath.name}'.") # Changed to info
            return output_filepath
        except FileNotFoundError:
            logger.error("FFmpeg not found. Please ensure FFmpeg is installed and added to your system's PATH to use video conversion. Skipping conversion.") # Changed to error
            return None
        except subprocess.CalledProcessError as e:
            logger.error(f"Error converting '{input_filepath.name}' with FFmpeg: {e}") # Changed to error
            logger.error(f"FFmpeg stdout: {e.stdout}") # Changed to error
            logger.error(f"FFmpeg stderr: {e.stderr}") # Changed to error
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred during FFmpeg conversion of '{input_filepath.name}': {e}") # Changed to error
            return None

    def extract_lip_frames(self, video_path: Union[str, Path], output_npy_path: Optional[Union[str, Path]] = None) -> Optional[np.ndarray]:
        """
        Extracts and processes lip frames from a video.
        Uses PyAV for efficient video reading and MediaPipe for accurate facial landmark detection.
        
        Args:
            video_path (Union[str, Path]): Path to the input video file (e.g., MP4, MPG).
            output_npy_path (Union[str, Path], optional): Path to the .npy file where the extracted
                                                         lip frames will be saved. If `None`,
                                                         frames are only returned, not saved.
            
        Returns:
            Optional[np.ndarray]: A NumPy array of processed lip frames in RGB format
                                  (shape: NUM_FRAMES x IMG_H x IMG_W x 3).
                                  Returns `None` if an error occurs during processing or
                                  if the extracted clip is deemed invalid (e.g., too many problematic frames).
        """
        original_video_path = Path(video_path) # Keep original path
        current_video_path = original_video_path # This will be the path used for processing

        # --- NEW: Optional MP4 Conversion ---
        converted_temp_mp4_path = None
        if self.config.CONVERT_TO_MP4_IF_NEEDED and original_video_path.suffix.lower() not in ['.mp4', '.mov']: # Only convert if not already MP4/MOV
            logger.info(f"'{original_video_path.name}' is not in MP4/MOV format. Attempting conversion...") # Changed to info
            converted_temp_mp4_path = self._convert_video_to_mp4(original_video_path, self.config.MP4_TEMP_DIR)
            if converted_temp_mp4_path:
                current_video_path = converted_temp_mp4_path
            else:
                logger.warning(f"MP4 conversion failed for '{original_video_path.name}'. Attempting to process original file.") # Changed to warning
                # Fallback to original path if conversion fails
                current_video_path = original_video_path 
        # --- END NEW: Optional MP4 Conversion ---


        if not current_video_path.exists():
            logger.error(f"Video file not found at '{current_video_path}'. Processing stopped.") # Changed to error
            return None

        processed_frames_temp_list = []
        self.bbox_history = [] # Reset bounding box history for each new video

        try:
            container = av.open(str(current_video_path)) # Use current_video_path here
        except av.AVError as e:
            logger.error(f"Error opening video '{current_video_path.name}' with PyAV: {e}. Processing stopped.") # Changed to error
            return None

        if not container.streams.video:
            logger.error(f"No video stream found in '{current_video_path.name}'. Processing stopped.") # Changed to error
            container.close()
            return None
            
        video_stream = container.streams.video[0]

        # Determine the total number of frames to process
        total_frames_to_process = self.config.MAX_FRAMES
        if total_frames_to_process is None:
            try:
                # Attempt to get exact frame count from PyAV; fallback if unreliable
                frames_from_av = video_stream.frames
                if frames_from_av is not None and frames_from_av > 0:
                    total_frames_to_process = frames_from_av
                else:
                    # If PyAV reports 0 or None frames, process until stream end
                    total_frames_to_process = float('inf') 
            except Exception:
                # Fallback if any error occurs getting frame count
                total_frames_to_process = float('inf') 

        logger.info(f"Processing video: '{current_video_path.name}' ({total_frames_to_process if total_frames_to_process != float('inf') else 'all available'} frames)...") # Changed to info

        num_problematic_frames = 0 # Counter for frames where lip detection/cropping fails

        try:
            for frame_idx, frame_av in enumerate(container.decode(video=0)):
                # Stop if max frames limit is reached
                if total_frames_to_process != float('inf') and frame_idx >= total_frames_to_process:
                    logger.debug(f"Max frames limit ({total_frames_to_process}) reached. Stopping video processing.") # Changed to debug
                    break 

                try:
                    image_rgb = frame_av.to_rgb().to_ndarray()
                    original_frame_height, original_frame_width, _ = image_rgb.shape

                    if self.config.SAVE_DEBUG_FRAMES:
                        self._debug_frame_processing(image_rgb, frame_idx, 'original')
                    
                    results = self.mp_face_mesh.process(image_rgb) 
                    
                    raw_lip_bbox = None
                    mp_face_landmarks = None 
                    
                    # Assume problematic until proven otherwise
                    frame_is_problematic = True

                    if results.multi_face_landmarks:
                        mp_face_landmarks = results.multi_face_landmarks[0]
                        landmarks = mp_face_landmarks.landmark

                        lip_x_coords = []
                        lip_y_coords = []
                        for idx in LIPS_MESH_LANDMARKS_INDICES:
                            if idx < len(landmarks): 
                                lip_x_coords.append(landmarks[idx].x * original_frame_width)
                                lip_y_coords.append(landmarks[idx].y * original_frame_height)

                        if lip_x_coords and lip_y_coords:
                            min_x_tight = min(lip_x_coords)
                            max_x_tight = max(lip_x_coords)
                            min_y_tight = min(lip_y_coords)
                            max_y_tight = max(lip_y_coords)
                            
                            current_lip_width_tight = max_x_tight - min_x_tight
                            current_lip_height_tight = max_y_tight - min_y_tight

                            margin_x = current_lip_width_tight * self.config.LIP_PROPORTIONAL_MARGIN_X
                            margin_y = current_lip_height_tight * self.config.LIP_PROPORTIONAL_MARGIN_Y

                            x1_proposed = min_x_tight - margin_x - self.config.LIP_PADDING_LEFT_PX
                            y1_proposed = min_y_tight - margin_y - self.config.LIP_PADDING_TOP_PX
                            x2_proposed = max_x_tight + margin_x + self.config.LIP_PADDING_RIGHT_PX
                            y2_proposed = max_y_tight + margin_y + self.config.LIP_PADDING_BOTTOM_PX

                            x1_clamped = max(0, int(x1_proposed))
                            y1_clamped = max(0, int(y1_proposed))
                            x2_clamped = min(original_frame_width, int(x2_proposed))
                            y2_clamped = min(original_frame_height, int(y2_proposed))

                            current_width_clamped = x2_clamped - x1_clamped
                            current_height_clamped = y2_clamped - y1_clamped

                            if current_width_clamped > 0 and current_height_clamped > 0:
                                target_aspect_ratio = self.config.IMG_W / self.config.IMG_H 
                                
                                center_x = (x1_clamped + x2_clamped) / 2
                                center_y = (y1_clamped + y2_clamped) / 2

                                if (current_width_clamped / current_height_clamped) > target_aspect_ratio:
                                    needed_height = current_width_clamped / target_aspect_ratio
                                    y1_adjusted = center_y - needed_height / 2
                                    y2_adjusted = center_y + needed_height / 2

                                    y1_final = max(0, int(y1_adjusted))
                                    y2_final = min(original_frame_height, int(y2_adjusted))
                                    x1_final, x2_final = x1_clamped, x2_clamped
                                else:
                                    needed_width = current_height_clamped * target_aspect_ratio
                                    x1_adjusted = center_x - needed_width / 2
                                    x2_adjusted = center_x + needed_width / 2

                                    x1_final = max(0, int(x1_adjusted))
                                    x2_final = min(original_frame_width, int(x2_adjusted))
                                    y1_final, y2_final = y1_clamped, y2_clamped

                                final_bbox_width = x2_final - x1_final
                                final_bbox_height = y2_final - y1_final

                                # No longer checking MIN_VALID_CROP_FACTOR here.
                                # Any positive dimension indicates a valid raw bounding box.
                                if final_bbox_width > 0 and final_bbox_height > 0: 
                                    raw_lip_bbox = (x1_final, y1_final, x2_final, y2_final)
                                    frame_is_problematic = False # Frame is valid!
                                else:
                                    logger.warning(f"Frame {frame_idx}: Calculated final bounding box has zero or negative dimensions after aspect ratio adjustment. Generating black frame.") # Changed to warning
                            else:
                                logger.warning(f"Frame {frame_idx}: Clamped bounding box has zero or negative dimensions. Generating black frame.") # Changed to warning
                        else:
                            logger.warning(f"Frame {frame_idx}: No lip coordinates found. Generating black frame.") # Changed to warning
                    else:
                        logger.warning(f"Frame {frame_idx}: No face detected. Generating black frame.") # Changed to warning

                    # Apply temporal smoothing to the raw bounding box (or its absence)
                    smoothed_lip_bbox = self._apply_temporal_smoothing(raw_lip_bbox)
                    x1_smoothed, y1_smoothed, x2_smoothed, y2_smoothed = smoothed_lip_bbox

                    # Save debug frames if enabled
                    if self.config.SAVE_DEBUG_FRAMES:
                        if mp_face_landmarks: 
                            self._debug_frame_processing(image_rgb, frame_idx, 'landmarks', raw_lip_bbox, mp_face_landmarks)
                        else: 
                            self._debug_frame_processing(image_rgb, frame_idx, 'landmarks_no_detection', None, None) 

                    # Crop and resize the frame using the smoothed bounding box
                    if not frame_is_problematic and x2_smoothed > x1_smoothed and y2_smoothed > y1_smoothed:
                        lip_cropped_frame = image_rgb[y1_smoothed:y2_smoothed, x1_smoothed:x2_smoothed]
                        
                        current_crop_width = lip_cropped_frame.shape[1]
                        current_crop_height = lip_cropped_frame.shape[0]

                        if current_crop_width > self.config.IMG_W or current_crop_height > self.config.IMG_H:
                            interpolation_method = cv2.INTER_AREA
                        else:
                            interpolation_method = cv2.INTER_LANCZOS4 
                        
                        final_resized_lip = cv2.resize(lip_cropped_frame, (self.config.IMG_W, self.config.IMG_H), interpolation=interpolation_method)
                        
                        processed_lip_frame = final_resized_lip.copy() 
                        if self.config.APPLY_CLAHE and self.clahe_obj is not None:
                            ycrcb_image = cv2.cvtColor(processed_lip_frame, cv2.COLOR_RGB2YCrCb)
                            y_channel, cr_channel, cb_channel = cv2.split(ycrcb_image)
                            
                            clahe_y_channel = self.clahe_obj.apply(y_channel)
                            
                            merged_ycrcb = cv2.merge([clahe_y_channel, cr_channel, cb_channel])
                            
                            processed_lip_frame = cv2.cvtColor(merged_ycrcb, cv2.COLOR_YCrCb2RGB)

                            if self.config.SAVE_DEBUG_FRAMES:
                                self._debug_frame_processing(processed_lip_frame, frame_idx, 'clahe_applied')

                        if self.config.INCLUDE_LANDMARKS_ON_FINAL_OUTPUT and mp_face_landmarks:
                            x_offset_for_mapping = smoothed_lip_bbox[0] 
                            y_offset_for_mapping = smoothed_lip_bbox[1]
                            
                            width_cropped_for_mapping = smoothed_lip_bbox[2] - smoothed_lip_bbox[0]
                            height_cropped_for_mapping = smoothed_lip_bbox[3] - smoothed_lip_bbox[1]

                            if width_cropped_for_mapping > 0 and height_cropped_for_mapping > 0:
                                scale_x_to_output = self.config.IMG_W / width_cropped_for_mapping
                                scale_y_to_output = self.config.IMG_H / height_cropped_for_mapping

                                for lm_idx in LIPS_MESH_LANDMARKS_INDICES:
                                    if lm_idx < len(mp_face_landmarks.landmark):
                                        orig_x_px = mp_face_landmarks.landmark[lm_idx].x * original_frame_width
                                        orig_y_px = mp_face_landmarks.landmark[lm_idx].y * original_frame_height

                                        relative_x_px = orig_x_px - x_offset_for_mapping
                                        relative_y_px = orig_y_px - y_offset_for_mapping

                                        final_x_lm = int(relative_x_px * scale_x_to_output)
                                        final_y_lm = int(relative_y_px * scale_y_to_output)
                                        
                                        cv2.circle(processed_lip_frame, (final_x_lm, final_y_lm), 1, (0, 255, 0), -1) 
                                        
                        processed_frames_temp_list.append(processed_lip_frame)
                        if self.config.SAVE_DEBUG_FRAMES and not self.config.APPLY_CLAHE: 
                            self._debug_frame_processing(final_resized_lip, frame_idx, 'resized')
                    else:
                        # If frame is problematic or smoothed bbox is invalid, append a black frame
                        black_frame = np.zeros((self.config.IMG_H, self.config.IMG_W, 3), dtype=np.uint8)
                        processed_frames_temp_list.append(black_frame)
                        num_problematic_frames += 1 # Increment problematic frame counter
                        if self.config.SAVE_DEBUG_FRAMES:
                            self._debug_frame_processing(black_frame, frame_idx, 'black_generated')

                except Exception as e:
                    logger.warning(f"Unexpected error processing frame {frame_idx} from '{current_video_path.name}': {e}. This frame will be treated as problematic.") # Changed to warning
                    black_frame = np.zeros((self.config.IMG_H, self.config.IMG_W, 3), dtype=np.uint8)
                    processed_frames_temp_list.append(black_frame)
                    num_problematic_frames += 1 # Increment problematic frame counter
                    # Append a default black frame bbox to history for smoothing consistency if an error occurs
                    self.bbox_history.append((0, 0, self.config.IMG_W, self.config.IMG_H)) 

        finally:
            container.close()
            # --- NEW: Cleanup temporary MP4 file if it was created ---
            if converted_temp_mp4_path and converted_temp_mp4_path.exists():
                try:
                    os.remove(str(converted_temp_mp4_path))
                    logger.info(f"Cleaned up temporary MP4 file: '{converted_temp_mp4_path.name}'.") # Changed to info
                except Exception as e:
                    logger.warning(f"Could not remove temporary MP4 file '{converted_temp_mp4_path.name}': {e}") # Changed to warning
            # --- END NEW: Cleanup ---

        if not processed_frames_temp_list:
            logger.warning(f"No frames could be processed from video '{current_video_path.name}'. Returning `None`.") # Changed to warning
            return None

        final_processed_np_frames = np.array(processed_frames_temp_list, dtype=np.uint8)

        # Apply MAX_FRAMES limit if specified
        if self.config.MAX_FRAMES is not None and final_processed_np_frames.shape[0] > self.config.MAX_FRAMES:
            final_processed_np_frames = final_processed_np_frames[:self.config.MAX_FRAMES]
            logger.info(f"Video truncated to {self.config.MAX_FRAMES} frames as per configuration.") # Changed to info
        elif self.config.MAX_FRAMES is not None and final_processed_np_frames.shape[0] < self.config.MAX_FRAMES:
            padding_needed = self.config.MAX_FRAMES - final_processed_np_frames.shape[0]
            black_padding = np.zeros((padding_needed, self.config.IMG_H, self.config.IMG_W, 3), dtype=np.uint8)
            final_processed_np_frames = np.concatenate((final_processed_np_frames, black_padding), axis=0)
            # Count these new black frames as problematic if the original video was shorter
            num_problematic_frames += padding_needed
            logger.info(f"Video padded with {padding_needed} black frames to reach {self.config.MAX_FRAMES} frames as per configuration.") # Changed to info


        total_output_frames = final_processed_np_frames.shape[0]
        
        # New: Check percentage of problematic frames (including those generated due to initial issues or padding)
        percentage_problematic_frames = (num_problematic_frames / total_output_frames) * 100

        if total_output_frames == 0 or percentage_problematic_frames > self.config.MAX_PROBLEMATIC_FRAMES_PERCENTAGE: # Renamed config field
            logger.warning(f"Clip '{original_video_path.name}' rejected: {percentage_problematic_frames:.2f}% problematic frames (exceeds {self.config.MAX_PROBLEMATIC_FRAMES_PERCENTAGE}% allowed).") # Changed to warning
            return None
        elif num_problematic_frames > 0:
            logger.info(f"Clip '{original_video_path.name}': {percentage_problematic_frames:.2f}% problematic frames found. Clip retained.") # Changed to info
        
        # Save to .npy file if a path is provided
        if output_npy_path:
            output_npy_path = Path(output_npy_path)
            output_npy_path.parent.mkdir(parents=True, exist_ok=True) 
            np.save(output_npy_path, final_processed_np_frames)
            logger.info(f"Extracted frames saved to '{output_npy_path}'.") # Changed to info

        return final_processed_np_frames

    @staticmethod
    def extract_npy(npy_path: Union[str, Path]) -> Optional[np.ndarray]:
        """
        Loads a NumPy array from a .npy file.

        Args:
            npy_path (Union[str, Path]): Path to the .npy file.

        Returns:
            Optional[np.ndarray]: The loaded NumPy array, or `None` if the file is not found or an error occurs.
        """
        npy_path = Path(npy_path)
        if not npy_path.exists():
            logger.error(f"NPY file not found at '{npy_path}'.") # Changed to error
            return None
        
        try:
            data = np.load(npy_path)
            logger.info(f"Successfully loaded NPY file from '{npy_path}'. Shape: {data.shape}") # Changed to info
            return data
        except Exception as e:
            logger.error(f"Error loading NPY file '{npy_path}': {e}") # Changed to error
            return None