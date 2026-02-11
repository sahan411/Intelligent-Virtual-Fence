"""
Utility Functions for Intelligent Virtual Fence
================================================
Contains helper functions for:
    - Configuration loading
    - Logging intrusion events
    - Screenshot capture

These utilities are separate from core modules to keep things clean.
"""

import os
import json
import cv2
from datetime import datetime


# -----------------------------------------------------------------------------
# Configuration Loader
# -----------------------------------------------------------------------------

def load_config(config_path):
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to config.json
    
    Returns:
        Dictionary with all configuration values
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"[Config] Loaded configuration from {config_path}")
        return config
    except FileNotFoundError:
        print(f"[Config] Warning: Config file not found at {config_path}")
        print("[Config] Using default values.")
        return None
    except json.JSONDecodeError as e:
        print(f"[Config] Error parsing config: {e}")
        return None


# -----------------------------------------------------------------------------
# Intrusion Logger
# -----------------------------------------------------------------------------

class IntrusionLogger:
    """
    Logs intrusion events to a file with timestamps.
    
    Creates an audit trail that examiners love to see.
    Log format: [TIMESTAMP] EVENT - Details
    """
    
    def __init__(self, log_file, enabled=True):
        """
        Initialize the logger.
        
        Args:
            log_file: Path to log file
            enabled: Whether logging is enabled
        """
        self.enabled = enabled
        self.log_file = log_file
        
        if not self.enabled:
            print("[Logger] Logging disabled.")
            return
        
        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            print(f"[Logger] Created log directory: {log_dir}")
        
        # Write session start
        self._write_log("="*60)
        self._write_log("SESSION STARTED")
        self._write_log("="*60)
        
        print(f"[Logger] Logging to: {log_file}")
    
    def _get_timestamp(self):
        """Get current timestamp in readable format."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    
    def _write_log(self, message):
        """Write a message to the log file."""
        if not self.enabled:
            return
        
        timestamp = self._get_timestamp()
        line = f"[{timestamp}] {message}\n"
        
        try:
            with open(self.log_file, 'a') as f:
                f.write(line)
        except Exception as e:
            print(f"[Logger] Error writing to log: {e}")
    
    def log_intrusion(self, frame_num, num_intrusions, details=None):
        """
        Log an intrusion event.
        
        Args:
            frame_num: Current frame number
            num_intrusions: Number of persons inside ROI
            details: Optional list of detection details
        """
        message = f"INTRUSION - Frame {frame_num}: {num_intrusions} person(s) inside ROI"
        self._write_log(message)
        
        # Log individual detection details if provided
        if details:
            for det in details:
                if det.get('inside_roi', False):
                    conf = det.get('confidence', 0)
                    foot = det.get('foot_point', (0, 0))
                    self._write_log(f"  -> Person at foot-point {foot}, confidence: {conf:.2f}")
    
    def log_event(self, event_type, message):
        """
        Log a general event.
        
        Args:
            event_type: Type of event (e.g., "INFO", "WARNING")
            message: Event description
        """
        self._write_log(f"{event_type} - {message}")
    
    def log_session_end(self, stats):
        """
        Log session end with summary statistics.
        
        Args:
            stats: Dictionary with session statistics
        """
        self._write_log("="*60)
        self._write_log("SESSION ENDED")
        self._write_log(f"Total frames processed: {stats.get('frames', 0)}")
        self._write_log(f"Total motion triggers: {stats.get('motion_triggers', 0)}")
        self._write_log(f"Total YOLO inferences: {stats.get('yolo_inferences', 0)}")
        self._write_log(f"Total intrusion detections: {stats.get('intrusions', 0)}")
        self._write_log("="*60)


# -----------------------------------------------------------------------------
# Screenshot Capture
# -----------------------------------------------------------------------------

class ScreenshotCapture:
    """
    Captures and saves screenshots when intrusion is detected.
    
    Includes cooldown to avoid saving too many similar frames.
    """
    
    def __init__(self, save_folder, enabled=True, cooldown_frames=30):
        """
        Initialize screenshot capture.
        
        Args:
            save_folder: Folder to save screenshots
            enabled: Whether screenshot capture is enabled
            cooldown_frames: Minimum frames between captures
        """
        self.enabled = enabled
        self.save_folder = save_folder
        self.cooldown_frames = cooldown_frames
        
        self.frames_since_last = cooldown_frames  # Ready to capture immediately
        self.total_captures = 0
        
        if not self.enabled:
            print("[Screenshot] Screenshot capture disabled.")
            return
        
        # Create screenshot directory if it doesn't exist
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
            print(f"[Screenshot] Created folder: {save_folder}")
        
        print(f"[Screenshot] Saving to: {save_folder}")
        print(f"[Screenshot] Cooldown: {cooldown_frames} frames")
    
    def capture(self, frame, frame_num):
        """
        Capture a screenshot if cooldown has passed.
        
        Args:
            frame: The frame to save
            frame_num: Current frame number
        
        Returns:
            True if screenshot was saved, False otherwise
        """
        if not self.enabled:
            return False
        
        # Check cooldown
        if self.frames_since_last < self.cooldown_frames:
            self.frames_since_last += 1
            return False
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"intrusion_{timestamp}_frame{frame_num}.jpg"
        filepath = os.path.join(self.save_folder, filename)
        
        # Save the frame
        try:
            cv2.imwrite(filepath, frame)
            self.total_captures += 1
            self.frames_since_last = 0  # Reset cooldown
            print(f"[Screenshot] Saved: {filename}")
            return True
        except Exception as e:
            print(f"[Screenshot] Error saving: {e}")
            return False
    
    def tick(self):
        """Call every frame to update cooldown counter."""
        if self.frames_since_last < self.cooldown_frames:
            self.frames_since_last += 1
    
    def get_stats(self):
        """Get capture statistics."""
        return {
            'total_captures': self.total_captures
        }
