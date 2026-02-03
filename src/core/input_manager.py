"""
Module 1: Input Manager
=======================
This module handles all video input operations for the Intelligent Virtual Fence system.

Responsibilities:
    - Open video source (webcam or video file)
    - Read frames with consistent resolution
    - Control frame rate for stable processing
    - Provide clean resource management

Design Notes:
    - We resize all frames to a fixed resolution for consistent performance
    - FPS control helps in testing and reduces unnecessary processing
    - Webcam fallback to video file makes demos easier
"""

import cv2
import time


# -----------------------------------------------------------------------------
# Configuration Constants
# -----------------------------------------------------------------------------
# These values are set here so we can easily tweak them during testing
# 640x360 is a good balance between quality and processing speed

DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 360
DEFAULT_FPS = 30  # target frames per second


class InputManager:
    """
    Handles video capture from webcam or video file.
    
    Why a class?
        - We need to keep track of capture object, timing, and settings
        - Cleaner than passing multiple variables between functions
        - Easy to extend later if needed
    """
    
    def __init__(self, source=0, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT, fps=DEFAULT_FPS):
        """
        Initialize the input manager.
        
        Args:
            source: Can be 0 for webcam, or a string path like "assets/videos/demo.mp4"
            width: Target frame width (default 640)
            height: Target frame height (default 360)
            fps: Target frame rate (default 30)
        """
        self.source = source
        self.width = width
        self.height = height
        self.target_fps = fps
        
        # Calculate delay between frames to maintain target FPS
        # For example: 30 FPS means ~0.033 seconds between frames
        self.frame_delay = 1.0 / fps
        
        # Will be set when we open the source
        self.cap = None
        self.last_frame_time = 0
        
        # Store original video properties (useful for debugging)
        self.original_width = 0
        self.original_height = 0
        self.original_fps = 0
    
    def open(self):
        """
        Open the video source and prepare for reading.
        
        Returns:
            True if opened successfully, False otherwise
        """
        self.cap = cv2.VideoCapture(self.source)
        
        if not self.cap.isOpened():
            print(f"[InputManager] ERROR: Cannot open video source: {self.source}")
            return False
        
        # Store original properties before we start resizing
        self.original_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.original_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.original_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # For webcam, try to set the resolution directly (may or may not work)
        # We still resize later to guarantee consistent output
        if isinstance(self.source, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        print(f"[InputManager] Opened source: {self.source}")
        print(f"[InputManager] Original resolution: {self.original_width}x{self.original_height}")
        print(f"[InputManager] Target resolution: {self.width}x{self.height}")
        print(f"[InputManager] Target FPS: {self.target_fps}")
        
        return True
    
    def read_frame(self):
        """
        Read a single frame from the video source.
        
        This method:
            1. Reads the raw frame
            2. Resizes to target resolution
            3. Handles end-of-video gracefully
        
        Returns:
            (success, frame) tuple
            - success: True if frame was read, False if end of video or error
            - frame: The resized frame (or None if failed)
        """
        if self.cap is None:
            print("[InputManager] ERROR: Source not opened. Call open() first.")
            return False, None
        
        ok, frame = self.cap.read()
        
        if not ok:
            # Could be end of video or a read error
            return False, None
        
        # Resize frame to our target resolution
        # This ensures all downstream modules get consistent input
        frame = cv2.resize(frame, (self.width, self.height))
        
        return True, frame
    
    def read_frame_with_fps_control(self):
        """
        Read frame with FPS control - waits if we're reading too fast.
        
        Why FPS control?
            - Without it, video files play way too fast
            - Helps keep CPU usage reasonable
            - Makes testing more realistic
        
        Returns:
            (success, frame) tuple, same as read_frame()
        """
        # Calculate how long since last frame
        current_time = time.time()
        elapsed = current_time - self.last_frame_time
        
        # If we're going too fast, wait a bit
        if elapsed < self.frame_delay:
            time.sleep(self.frame_delay - elapsed)
        
        # Update timestamp
        self.last_frame_time = time.time()
        
        # Now read the frame
        return self.read_frame()
    
    def release(self):
        """
        Release the video source and clean up resources.
        Always call this when done to free up the camera/file handle.
        """
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            print("[InputManager] Released video source.")
    
    def is_opened(self):
        """Check if video source is currently open."""
        return self.cap is not None and self.cap.isOpened()
    
    def get_source_info(self):
        """
        Get information about the current video source.
        Useful for debugging and display purposes.
        
        Returns:
            Dictionary with source information
        """
        return {
            "source": self.source,
            "original_resolution": (self.original_width, self.original_height),
            "target_resolution": (self.width, self.height),
            "original_fps": self.original_fps,
            "target_fps": self.target_fps,
            "is_webcam": isinstance(self.source, int)
        }


# -----------------------------------------------------------------------------
# Standalone functions (kept for backward compatibility)
# -----------------------------------------------------------------------------
# These simple functions can still be used if you don't want the full class

def open_video_source(source):
    """
    Simple function to open a video source.
    
    Args:
        source: 0 for webcam, or path string for video file
    
    Returns:
        OpenCV VideoCapture object
    
    Raises:
        RuntimeError if source cannot be opened
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")
    return cap


def read_frame(cap):
    """
    Simple function to read one frame.
    
    Returns:
        (ok, frame) tuple
    """
    return cap.read()


def release_source(cap):
    """Release the video capture object."""
    cap.release()
