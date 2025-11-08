# pylipextractor/pylipextractor/config.py

from pathlib import Path

class LipExtractionConfig:
    """
    Configuration for lip extraction and video processing parameters.
    """
    # --- Output frame dimensions ---
    IMG_H = 50 # Desired height for the output lip frames
    IMG_W = 75 # Desired width for the output lip frames
    MAX_FRAMES = None # Maximum frames to extract from a video. If None, extracts all frames.

    # --- Lip Cropping Settings ---
    # These margins are PROPORTIONAL to the tightly calculated lip bounding box.
    # Adjust these to expand/shrink the area around the detected lips.
    LIP_PROPORTIONAL_MARGIN_X = 0.0 # Horizontal margin as a proportion of lip width
    LIP_PROPORTIONAL_MARGIN_Y = 0.0 # Vertical margin as a proportion of lip height
    
    # These are fixed pixel paddings (applied AFTER proportional margins).
    # Use these for minor fine-tuning if needed.
    LIP_PADDING_LEFT_PX = 0
    LIP_PADDING_RIGHT_PX = 0
    LIP_PADDING_TOP_PX = 0
    LIP_PADDING_BOTTOM_PX = 0

    # --- General Processing Settings ---
    # Renamed: Now represents max allowed percentage of ANY problematic frame (not just black)
    MAX_PROBLEMATIC_FRAMES_PERCENTAGE = 15.0 

    # Removed: SMOOTHING_WINDOW_SIZE is replaced by EMA_ALPHA for EMA smoothing
    # New: EMA Smoothing for Bounding Boxes
    APPLY_EMA_SMOOTHING: bool = True # Set to True to apply EMA smoothing to bounding box coordinates
    EMA_ALPHA: float = 0.3 # EMA smoothing factor (0.0 to 1.0, higher means less smoothing, 1.0 means no smoothing)

    # --- Debugging & Output Customization Settings ---
    DEBUG_OUTPUT_DIR = Path("./lip_extraction_debug") # Directory to save debug frames
    MAX_DEBUG_FRAMES = 20 # Maximum number of debug frames to save per video.
    SAVE_DEBUG_FRAMES = False # Set to True to save intermediate debug frames.
    
    # If True, MediaPipe lip landmarks will be drawn on the final extracted lip frames
    # (i.e., the frames saved in the .npy file and later converted to images).
    # This is for visualization/debugging of the final output, not typically for model training.
    INCLUDE_LANDMARKS_ON_FINAL_OUTPUT = False

    # --- Illumination and Contrast Normalization Settings ---
    APPLY_HISTOGRAM_MATCHING = True  # Set to True to apply histogram matching

    # Black out non-lip areas within the cropped frame ---
    # If True, pixels outside the detected lip mask (within the bounding box) will be set to black.
    # This can help models focus exclusively on the lip region.
    BLACK_OUT_NON_LIP_AREAS = False

    # --- New FFmpeg Conversion Options ---
    CONVERT_TO_MP4_IF_NEEDED: bool = True  # Set to True to enable automatic conversion
    MP4_TEMP_DIR: Path = Path("temp_mp4_conversions") # Directory for temporary MP4 files
    HW_ACCELERATION_DEVICE: str = "cuda" # Device for hardware acceleration ('auto', 'cuda', 'cpu')
    # --- End New FFmpeg Conversion Options ---

    # --- Real-Time Factor (RTF) Calculation ---
    CALCULATE_RTF: bool = False # Set to True to calculate and log the Real-Time Factor

    # --- MediaPipe Settings ---
    REFINE_LANDMARKS: bool = False # Set to True for more accurate landmark detection, but slower processing

    # --- Profiling Settings ---
    PROFILE_CODE: bool = True # Set to True to profile the code and save the results to a file

    # --- Output Organization Settings (placeholder for future, currently no effect) ---
    DEFAULT_OUTPUT_BASE_DIR = Path("output_data") # Default directory for saving extracted NPYs
    ORGANIZE_OUTPUT_BY_VIDEO_NAME = True # If True, will save NPYs in subfolders based on video name


class MainConfig:
    """
    Main project configuration containing sub-configurations.
    """
    def __init__(self):
        self.lip_extraction = LipExtractionConfig()