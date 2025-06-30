# pylibextractor_project/pylibextractor/lip_extractor.py

import os
import cv2
import numpy as np
import mediapipe as mp
import av
from pathlib import Path
import warnings
import math
from typing import Tuple, Optional, List, Union

# --- Suppress specific MediaPipe warnings and GLOG messages ---
warnings.filterwarnings("ignore", category=UserWarning, module="mediapipe")
os.environ['GLOG_minloglevel'] = '2' # Suppress all GLOG messages below WARNING level.
# --- End suppression ---

# IMPORTANT: UPDATED AND CORRECTED LIPS_MESH_LANDMARKS_INDICES
# This list is derived from direct visual inspection of the MediaPipe Face Mesh,
# focusing EXCLUSIVELY on the outer (vermilion) and inner lip contours.
LIPS_MESH_LANDMARKS_INDICES = sorted(list(set([
    # Outer Lip (Vermilion Border)
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, # Upper
    291, 375, 321, 314, 405, 304, 303, 302, 292, 306, # Lower
    # Inner Lip (Mouth Opening)
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415, # Upper
    87, 178, 88, 95, 181, 85, 182, 16, 91, 14, 317, 402, 320, 318, 324, 308, # Lower & Sides
    # Mouth Corners (Crucial for definition)
    17, 267
])))


class LipExtractor:
    """
    A class for extracting lip frames from videos using MediaPipe Face Mesh.
    This class crops and resizes lip frames, returning them as a NumPy array.
    It also provides utilities for loading previously saved NPY files.
    """
    def __init__(self, config):
        """
        Initializes the LipExtractor with provided configuration.

        Args:
            config (LipExtractionConfig): Configuration object containing image dimensions,
                                          padding settings, and other parameters.
        """
        self.config = config
        self.mp_face_mesh = None # MediaPipe Face Mesh is initialized lazily when first needed
        self._initialize_mediapipe_if_not_set()
        
        # History for temporal smoothing of bounding boxes
        self.bbox_history = [] 
        self.SMOOTHING_WINDOW_SIZE = 5 # Size of the moving average window for bounding box smoothing

        # Initialize CLAHE object if enabled in config
        self.clahe_obj = None
        if self.config.APPLY_CLAHE:
            # CLAHE (Contrast Limited Adaptive Histogram Equalization) is used for
            # improving the contrast of images, particularly in cases of varying illumination.
            # It operates on small regions of the image (tiles) rather than the entire image.
            self.clahe_obj = cv2.createCLAHE(
                clipLimit=self.config.CLAHE_CLIP_LIMIT,
                tileGridSize=self.config.CLAHE_TILE_GRID_SIZE
            )

    def _initialize_mediapipe_if_not_set(self):
        """
        Initializes the MediaPipe Face Mesh model if it hasn't been initialized yet.
        This ensures the model is loaded only once per process.
        """
        if self.mp_face_mesh is None:
            self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1, # Assume one dominant face in the video
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                refine_landmarks=True # Use refined landmarks for better accuracy
            )
            # print(f"MediaPipe Face Mesh model loaded for process {os.getpid()}.", flush=True) # Uncomment for debug

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
                                  if the extracted clip is deemed invalid (e.g., too many black frames).
        """
        self._initialize_mediapipe_if_not_set() # Ensure MediaPipe is loaded for this process
        
        video_path = Path(video_path) # Convert to Path object for consistent handling

        if not video_path.exists():
            print(f"Error: Video file not found at '{video_path}'. Processing stopped.", flush=True)
            return None

        processed_frames_temp_list = []
        self.bbox_history = [] # Reset bounding box history for each new video

        try:
            container = av.open(str(video_path))
        except av.AVError as e:
            print(f"Error opening video '{video_path.name}' with PyAV: {e}. Processing stopped.", flush=True)
            return None

        if not container.streams.video:
            print(f"Error: No video stream found in '{video_path.name}'. Processing stopped.", flush=True)
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

        print(f"Processing video: '{video_path.name}' ({total_frames_to_process if total_frames_to_process != float('inf') else 'all available'} frames)...", flush=True)

        try:
            for frame_idx, frame_av in enumerate(container.decode(video=0)):
                # Stop if max frames limit is reached
                if total_frames_to_process != float('inf') and frame_idx >= total_frames_to_process:
                    break 

                try:
                    image_rgb = frame_av.to_rgb().to_ndarray()
                    original_frame_height, original_frame_width, _ = image_rgb.shape

                    if self.config.SAVE_DEBUG_FRAMES:
                        self._debug_frame_processing(image_rgb, frame_idx, 'original')
                    
                    results = self.mp_face_mesh.process(image_rgb)
                    
                    raw_lip_bbox = None
                    mp_face_landmarks = None # Store MediaPipe landmarks to potentially draw on output

                    if results.multi_face_landmarks:
                        mp_face_landmarks = results.multi_face_landmarks[0]
                        landmarks = mp_face_landmarks.landmark

                        # Collect only coordinates for the specified lip landmarks
                        lip_x_coords = []
                        lip_y_coords = []
                        for idx in LIPS_MESH_LANDMARKS_INDICES:
                            if idx < len(landmarks): # Ensure landmark index is valid
                                lip_x_coords.append(landmarks[idx].x * original_frame_width)
                                lip_y_coords.append(landmarks[idx].y * original_frame_height)
                            # else: No warning needed for invalid indices, as LIPS_MESH_LANDMARKS_INDICES should be reliable.

                        if lip_x_coords and lip_y_coords:
                            # Calculate the tightest bounding box around the collected lip landmarks
                            min_x_tight = min(lip_x_coords)
                            max_x_tight = max(lip_x_coords)
                            min_y_tight = min(lip_y_coords)
                            max_y_tight = max(lip_y_coords)
                            
                            current_lip_width_tight = max_x_tight - min_x_tight
                            current_lip_height_tight = max_y_tight - min_y_tight

                            # Add proportional margins and fixed paddings to the tight lip box
                            margin_x = current_lip_width_tight * self.config.LIP_PROPORTIONAL_MARGIN_X
                            margin_y = current_lip_height_tight * self.config.LIP_PROPORTIONAL_MARGIN_Y

                            x1_proposed = min_x_tight - margin_x - self.config.LIP_PADDING_LEFT_PX
                            y1_proposed = min_y_tight - margin_y - self.config.LIP_PADDING_TOP_PX
                            x2_proposed = max_x_tight + margin_x + self.config.LIP_PADDING_RIGHT_PX
                            y2_proposed = max_y_tight + margin_y + self.config.LIP_PADDING_BOTTOM_PX

                            # Clamp proposed coordinates to stay within the original frame boundaries
                            x1_clamped = max(0, int(x1_proposed))
                            y1_clamped = max(0, int(y1_proposed))
                            x2_clamped = min(original_frame_width, int(x2_proposed))
                            y2_clamped = min(original_frame_height, int(y2_proposed))

                            # Calculate current dimensions after initial clamping
                            current_width_clamped = x2_clamped - x1_clamped
                            current_height_clamped = y2_clamped - y1_clamped

                            # Check for invalid or too small dimensions immediately after clamping
                            if current_width_clamped <= 0 or current_height_clamped <= 0:
                                raw_lip_bbox = None # Invalid bbox, will result in black frame
                                continue # Skip to processing the next frame
                            
                            # Adjust bounding box to match the target aspect ratio while preserving center
                            target_aspect_ratio = self.config.IMG_W / self.config.IMG_H # e.g., 96/48 = 2.0
                            
                            center_x = (x1_clamped + x2_clamped) / 2
                            center_y = (y1_clamped + y2_clamped) / 2

                            # Determine if we need to adjust width or height to match aspect ratio
                            if (current_width_clamped / current_height_clamped) > target_aspect_ratio:
                                # Current box is wider than desired aspect ratio, so increase height
                                needed_height = current_width_clamped / target_aspect_ratio
                                y1_adjusted = center_y - needed_height / 2
                                y2_adjusted = center_y + needed_height / 2
                                # Clamp adjusted y-coordinates to original frame boundaries
                                y1_final = max(0, int(y1_adjusted))
                                y2_final = min(original_frame_height, int(y2_adjusted))
                                # Keep original clamped x-coordinates
                                x1_final, x2_final = x1_clamped, x2_clamped
                            else:
                                # Current box is taller than desired aspect ratio, so increase width
                                needed_width = current_height_clamped * target_aspect_ratio
                                x1_adjusted = center_x - needed_width / 2
                                x2_adjusted = center_x + needed_width / 2
                                # Clamp adjusted x-coordinates to original frame boundaries
                                x1_final = max(0, int(x1_adjusted))
                                x2_final = min(original_frame_width, int(x2_adjusted))
                                # Keep original clamped y-coordinates
                                y1_final, y2_final = y1_clamped, y2_clamped

                            # Final check on calculated dimensions before creating bbox
                            final_bbox_width = x2_final - x1_final
                            final_bbox_height = y2_final - y1_final

                            # Ensure the final bounding box is large enough to contain the output resolution
                            # (or at least a reasonable portion of it) to avoid excessive upscaling artifacts.
                            # Using a threshold (e.g., 75% of target width/height) helps reject poor crops.
                            if final_bbox_width > 0 and final_bbox_height > 0 and \
                               final_bbox_width >= self.config.IMG_W * 0.75 and \
                               final_bbox_height >= self.config.IMG_H * 0.75: 
                                raw_lip_bbox = (x1_final, y1_final, x2_final, y2_final)
                            else:
                                raw_lip_bbox = None # Bounding box is too small or invalid after adjustments
                        
                    # Apply temporal smoothing to the raw bounding box (or its absence)
                    smoothed_lip_bbox = self._apply_temporal_smoothing(raw_lip_bbox)
                    x1_smoothed, y1_smoothed, x2_smoothed, y2_smoothed = smoothed_lip_bbox

                    # Save debug frames if enabled
                    if self.config.SAVE_DEBUG_FRAMES:
                        # Pass raw_lip_bbox to debug for drawing, not smoothed_lip_bbox, to see raw detection
                        if mp_face_landmarks: 
                            self._debug_frame_processing(image_rgb, frame_idx, 'landmarks', raw_lip_bbox, mp_face_landmarks)
                        else: 
                            self._debug_frame_processing(image_rgb, frame_idx, 'landmarks_no_detection', None, None) 

                    # Crop and resize the frame using the smoothed bounding box
                    if x2_smoothed > x1_smoothed and y2_smoothed > y1_smoothed:
                        lip_cropped_frame = image_rgb[y1_smoothed:y2_smoothed, x1_smoothed:x2_smoothed]
                        
                        # Determine interpolation method for resizing: INTER_AREA for downscaling, INTER_LANCZOS4 for upscaling.
                        current_crop_width = lip_cropped_frame.shape[1]
                        current_crop_height = lip_cropped_frame.shape[0]

                        if current_crop_width > self.config.IMG_W or current_crop_height > self.config.IMG_H:
                            # If the cropped region is larger than target, we are downscaling (shrinking)
                            interpolation_method = cv2.INTER_AREA
                        else:
                            # If the cropped region is smaller or equal to target, we are upscaling (enlarging)
                            interpolation_method = cv2.INTER_LANCZOS4 # Changed from cv2.INTER_CUBIC to INTER_LANCZOS4
                        
                        final_resized_lip = cv2.resize(lip_cropped_frame, (self.config.IMG_W, self.config.IMG_H), interpolation=interpolation_method)
                        
                        # --- Apply CLAHE for illumination/contrast normalization ---
                        processed_lip_frame = final_resized_lip.copy() # Start with the resized frame
                        if self.config.APPLY_CLAHE and self.clahe_obj is not None:
                            # CLAHE needs grayscale image. Convert RGB to YCrCb and apply CLAHE to Y (luminance) channel.
                            # This preserves color information while enhancing contrast.
                            ycrcb_image = cv2.cvtColor(processed_lip_frame, cv2.COLOR_RGB2YCrCb)
                            y_channel, cr_channel, cb_channel = cv2.split(ycrcb_image)
                            
                            # Apply CLAHE to the Y channel
                            clahe_y_channel = self.clahe_obj.apply(y_channel)
                            
                            # Merge the enhanced Y channel back with Cr and Cb channels
                            merged_ycrcb = cv2.merge([clahe_y_channel, cr_channel, cb_channel])
                            
                            # Convert back to RGB
                            processed_lip_frame = cv2.cvtColor(merged_ycrcb, cv2.COLOR_YCrCb2RGB)

                            if self.config.SAVE_DEBUG_FRAMES:
                                self._debug_frame_processing(processed_lip_frame, frame_idx, 'clahe_applied')
                        # --- End CLAHE application ---

                        # Draw landmarks on this final processed frame ONLY if INCLUDE_LANDMARKS_ON_FINAL_OUTPUT is True
                        if self.config.INCLUDE_LANDMARKS_ON_FINAL_OUTPUT and mp_face_landmarks:
                            # Remap original MediaPipe landmarks (from the full frame) to the coordinates
                            # of this specific processed_lip_frame.
                            x_offset_for_mapping = smoothed_lip_bbox[0] 
                            y_offset_for_mapping = smoothed_lip_bbox[1]
                            
                            width_cropped_for_mapping = smoothed_lip_bbox[2] - smoothed_lip_bbox[0]
                            height_cropped_for_mapping = smoothed_lip_bbox[3] - smoothed_lip_bbox[1]

                            if width_cropped_for_mapping > 0 and height_cropped_for_mapping > 0:
                                scale_x_to_output = self.config.IMG_W / width_cropped_for_mapping
                                scale_y_to_output = self.config.IMG_H / height_cropped_for_mapping

                                for lm_idx in LIPS_MESH_LANDMARKS_INDICES:
                                    if lm_idx < len(mp_face_landmarks.landmark):
                                        # Original landmark pixel coordinates (relative to original video frame)
                                        orig_x_px = mp_face_landmarks.landmark[lm_idx].x * original_frame_width
                                        orig_y_px = mp_face_landmarks.landmark[lm_idx].y * original_frame_height

                                        # Landmark coordinates relative to the top-left of the cropped region
                                        relative_x_px = orig_x_px - x_offset_for_mapping
                                        relative_y_px = orig_y_px - y_offset_for_mapping

                                        # Landmark coordinates in the final (resized) lip frame
                                        final_x_lm = int(relative_x_px * scale_x_to_output)
                                        final_y_lm = int(relative_y_px * scale_y_to_output)
                                        
                                        # Draw a small circle for each landmark on the `processed_lip_frame`
                                        cv2.circle(processed_lip_frame, (final_x_lm, final_y_lm), 1, (0, 255, 0), -1) # Green dots
                                        
                        processed_frames_temp_list.append(processed_lip_frame) # Append the processed (CLAHE-applied) frame
                        if self.config.SAVE_DEBUG_FRAMES:
                            # No separate 'resized' debug frame if CLAHE is applied, as 'clahe_applied' is the final stage before landmark drawing
                            if not self.config.APPLY_CLAHE: # Only save 'resized' if CLAHE was NOT applied
                                self._debug_frame_processing(final_resized_lip, frame_idx, 'resized')
                    else:
                        # If the smoothed bounding box is invalid, append a black frame
                        black_frame = np.zeros((self.config.IMG_H, self.config.IMG_W, 3), dtype=np.uint8)
                        processed_frames_temp_list.append(black_frame)
                        if self.config.SAVE_DEBUG_FRAMES:
                            self._debug_frame_processing(black_frame, frame_idx, 'black_generated')

                except Exception as e:
                    print(f"Warning: Unexpected error processing frame {frame_idx} from '{video_path.name}': {e}. This frame will be treated as black.", flush=True)
                    black_frame = np.zeros((self.config.IMG_H, self.config.IMG_W, 3), dtype=np.uint8)
                    processed_frames_temp_list.append(black_frame)
                    # Append a default black frame bbox to history for smoothing consistency if an error occurs
                    # This helps prevent the smoothed bbox from diverging entirely if detection consistently fails.
                    self.bbox_history.append((0, 0, self.config.IMG_W, self.config.IMG_H)) 

        finally:
            container.close() # Ensure the video container is always closed

        if not processed_frames_temp_list:
            print(f"Warning: No frames could be processed from video '{video_path.name}'. Returning `None`.", flush=True)
            return None

        final_processed_np_frames = np.array(processed_frames_temp_list, dtype=np.uint8)

        # Calculate percentage of black frames to check clip validity
        total_output_frames = final_processed_np_frames.shape[0]
        num_black_frames = sum(1 for frame in final_processed_np_frames if self._is_black_frame(frame))
        
        # Reject clip if too many black frames are present
        if total_output_frames == 0 or (num_black_frames / total_output_frames) * 100 > self.config.MAX_BLACK_FRAMES_PERCENTAGE:
            print(f"Clip '{video_path.name}' rejected: {num_black_frames / total_output_frames * 100:.2f}% black frames (exceeds {self.config.MAX_BLACK_FRAMES_PERCENTAGE}% allowed).", flush=True)
            return None
        elif num_black_frames > 0:
            print(f"Clip '{video_path.name}': {num_black_frames / total_output_frames * 100:.2f}% black frames found. Clip retained.", flush=True)
        
        # Save to .npy file if a path is provided
        if output_npy_path:
            output_npy_path = Path(output_npy_path)
            output_npy_path.parent.mkdir(parents=True, exist_ok=True) # Create parent directories if they don't exist
            np.save(output_npy_path, final_processed_np_frames)
            print(f"Extracted frames saved to '{output_npy_path}'.", flush=True)

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
            print(f"Error: NPY file not found at '{npy_path}'.", flush=True)
            return None
        
        try:
            data = np.load(npy_path)
            print(f"Successfully loaded NPY file from '{npy_path}'. Shape: {data.shape}", flush=True)
            return data
        except Exception as e:
            print(f"Error loading NPY file '{npy_path}': {e}", flush=True)
            return None