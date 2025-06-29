# pylibextractor_project/pylibextractor/config.py
from pathlib import Path

class LipExtractionConfig:
    """
    Configuration for lip extraction and video processing parameters.
    """
    # --- Output frame dimensions ---
    IMG_H = 48 # Desired height for the output lip frames
    IMG_W = 96 # Desired width for the output lip frames
    MAX_FRAMES = None # Maximum frames to extract from a video. If None, extracts all frames.

    # --- Lip Cropping Settings ---
    # These margins are PROPORTIONAL to the tightly calculated lip bounding box.
    # Adjust these to expand/shrink the area around the detected lips.
    LIP_PROPORTIONAL_MARGIN_X = 0.20 # Horizontal margin as a proportion of lip width
    LIP_PROPORTIONAL_MARGIN_Y = 0.30 # Vertical margin as a proportion of lip height
    
    # These are fixed pixel paddings (applied AFTER proportional margins).
    # Use these for minor fine-tuning if needed.
    LIP_PADDING_LEFT_PX = 0
    LIP_PADDING_RIGHT_PX = 0
    LIP_PADDING_TOP_PX = 0
    LIP_PADDING_BOTTOM_PX = 0

    # --- General Processing Settings ---
    NUM_CPU_CORES = 4 # Number of CPU cores for parallel processing (if implemented in batch mode).
    MAX_BLACK_FRAMES_PERCENTAGE = 15.0 # Max allowed percentage of black frames in the output clip.

    # --- Debugging & Output Customization Settings ---
    DEBUG_OUTPUT_DIR = Path("./lip_extraction_debug") # Directory to save debug frames
    MAX_DEBUG_FRAMES = 5 # Maximum number of debug frames to save per video.
    SAVE_DEBUG_FRAMES = False # Set to True to save intermediate debug frames.
    
    # If True, MediaPipe lip landmarks will be drawn on the final extracted lip frames
    # (i.e., the frames saved in the .npy file and later converted to images).
    # This is for visualization/debugging of the final output, not typically for model training.
    INCLUDE_LANDMARKS_ON_FINAL_OUTPUT = False


class MainConfig:
    """
    Main project configuration containing sub-configurations.
    """
    def __init__(self):
        self.lip_extraction = LipExtractionConfig()