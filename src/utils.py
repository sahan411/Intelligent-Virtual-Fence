"""
Utility Functions for Intelligent Virtual Fence
================================================
Contains helper functions for:
    - Configuration loading
    - Logging intrusion events
    - Screenshot capture
    - FPS calculation
    - Intrusion duration tracking
    - Sound alerts

These utilities are separate from core modules to keep things clean.
"""

import os
import json
import cv2
import time
import threading
from datetime import datetime

# Try to import sound library (Windows only)
try:
    import winsound
    SOUND_AVAILABLE = True
except ImportError:
    SOUND_AVAILABLE = False


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
        self._write_log(f"Max intrusion duration: {stats.get('max_intrusion_duration', 0):.1f}s")
        self._write_log(f"Total time in zone: {stats.get('total_intrusion_time', 0):.1f}s")
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


# -----------------------------------------------------------------------------
# FPS Calculator
# -----------------------------------------------------------------------------

class FPSCalculator:
    """
    Calculates real-time FPS (frames per second).
    
    Uses a rolling average for smoother display.
    """
    
    def __init__(self, avg_count=30):
        """
        Initialize FPS calculator.
        
        Args:
            avg_count: Number of frames to average over
        """
        self.avg_count = avg_count
        self.frame_times = []
        self.last_time = time.time()
        self.current_fps = 0.0
    
    def tick(self):
        """
        Call this every frame to update FPS calculation.
        
        Returns:
            Current FPS (smoothed)
        """
        current_time = time.time()
        elapsed = current_time - self.last_time
        self.last_time = current_time
        
        # Avoid division by zero
        if elapsed > 0:
            self.frame_times.append(elapsed)
        
        # Keep only recent frames for rolling average
        if len(self.frame_times) > self.avg_count:
            self.frame_times.pop(0)
        
        # Calculate average FPS
        if len(self.frame_times) > 0:
            avg_time = sum(self.frame_times) / len(self.frame_times)
            self.current_fps = 1.0 / avg_time if avg_time > 0 else 0.0
        
        return self.current_fps
    
    def get_fps(self):
        """Get current FPS value."""
        return self.current_fps


# -----------------------------------------------------------------------------
# Intrusion Duration Tracker
# -----------------------------------------------------------------------------

class IntrusionDurationTracker:
    """
    Tracks how long a person has been inside the ROI.
    
    Useful for determining severity of intrusion.
    """
    
    def __init__(self, fps=30):
        """
        Initialize duration tracker.
        
        Args:
            fps: Expected frames per second (for time calculation)
        """
        self.fps = fps
        self.intrusion_start_time = None
        self.current_duration = 0.0
        self.is_active = False
        self.max_duration = 0.0  # Longest intrusion in session
        self.total_intrusion_time = 0.0
    
    def update(self, has_intrusion):
        """
        Update intrusion status.
        
        Args:
            has_intrusion: True if someone is inside ROI right now
        
        Returns:
            Current intrusion duration in seconds (0 if no intrusion)
        """
        if has_intrusion:
            if not self.is_active:
                # Intrusion just started
                self.intrusion_start_time = time.time()
                self.is_active = True
            
            # Calculate duration
            self.current_duration = time.time() - self.intrusion_start_time
            
            # Update max duration
            if self.current_duration > self.max_duration:
                self.max_duration = self.current_duration
        else:
            if self.is_active:
                # Intrusion just ended
                self.total_intrusion_time += self.current_duration
            
            self.is_active = False
            self.current_duration = 0.0
            self.intrusion_start_time = None
        
        return self.current_duration
    
    def get_duration_str(self):
        """Get formatted duration string."""
        if self.current_duration > 0:
            return f"{self.current_duration:.1f}s"
        return ""
    
    def get_stats(self):
        """Get duration statistics."""
        return {
            'current_duration': self.current_duration,
            'max_duration': self.max_duration,
            'total_intrusion_time': self.total_intrusion_time
        }


# -----------------------------------------------------------------------------
# Sound Alert
# -----------------------------------------------------------------------------

class SoundAlert:
    """
    Plays sound alert when intrusion is detected.
    
    Uses Windows beep (winsound) - plays in separate thread to avoid blocking.
    On non-Windows systems, prints a message instead.
    """
    
    def __init__(self, enabled=True, frequency=1000, duration_ms=200, cooldown_seconds=2.0):
        """
        Initialize sound alert.
        
        Args:
            enabled: Whether sound alerts are enabled
            frequency: Beep frequency in Hz (default 1000)
            duration_ms: Beep duration in milliseconds
            cooldown_seconds: Minimum time between beeps
        """
        self.enabled = enabled and SOUND_AVAILABLE
        self.frequency = frequency
        self.duration_ms = duration_ms
        self.cooldown_seconds = cooldown_seconds
        
        self.last_beep_time = 0
        self.beep_count = 0
        
        if enabled and not SOUND_AVAILABLE:
            print("[Sound] Warning: winsound not available (Windows only)")
        elif enabled:
            print(f"[Sound] Alert enabled: {frequency}Hz, {duration_ms}ms")
    
    def _beep_thread(self):
        """Play beep in separate thread (non-blocking)."""
        try:
            winsound.Beep(self.frequency, self.duration_ms)
        except Exception:
            pass
    
    def alert(self):
        """
        Play alert sound if cooldown has passed.
        
        Returns:
            True if sound was played, False otherwise
        """
        if not self.enabled:
            return False
        
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_beep_time < self.cooldown_seconds:
            return False
        
        # Play beep in separate thread to avoid blocking video
        thread = threading.Thread(target=self._beep_thread)
        thread.daemon = True
        thread.start()
        
        self.last_beep_time = current_time
        self.beep_count += 1
        
        return True
    
    def get_stats(self):
        """Get alert statistics."""
        return {
            'beep_count': self.beep_count
        }
