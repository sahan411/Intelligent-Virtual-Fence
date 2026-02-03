"""
Module 3: Preprocessing & Enhancement
======================================
This module prepares frames for stable analysis in the Intelligent Virtual Fence system.

Responsibilities:
    - Convert frames to grayscale (for motion detection)
    - Apply Gaussian blur (reduce noise)
    - Check for low-light conditions
    - Apply CLAHE enhancement only when needed (on grayscale only)

Design Notes:
    - CLAHE is applied ONLY on grayscale images
    - We use CONDITIONAL enhancement to avoid over-processing
    - Original color frame is kept for drawing boxes later
"""

import cv2
import numpy as np


# -----------------------------------------------------------------------------
# Constants (keep it simple, tune later if needed)
# -----------------------------------------------------------------------------
LOW_LIGHT_THRESHOLD = 50   # Below this intensity, we enhance
BLUR_KERNEL = 5            # Gaussian blur kernel size


class Preprocessor:
    """
    Prepares frames for motion detection.
    
    Pipeline:
        1. Convert BGR to grayscale
        2. Apply Gaussian blur (reduce noise)
        3. If low-light, apply CLAHE to grayscale
    """
    
    def __init__(self):
        """Initialize the preprocessor with CLAHE object."""
        # Create CLAHE object once (reuse for performance)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # Simple counters
        self.frames_processed = 0
        self.frames_enhanced = 0
    
    def process(self, frame):
        """
        Main preprocessing function.
        
        Args:
            frame: Input BGR color frame
        
        Returns:
            tuple: (gray_frame, is_low_light, avg_intensity)
                - gray_frame: processed grayscale (ready for motion detection)
                - is_low_light: True if CLAHE was applied
                - avg_intensity: average brightness value
        """
        self.frames_processed += 1
        
        # Step 1: Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Step 2: Check lighting (before blur for accurate reading)
        avg_intensity = np.mean(gray)
        is_low_light = avg_intensity < LOW_LIGHT_THRESHOLD
        
        # Step 3: Apply Gaussian blur
        gray = cv2.GaussianBlur(gray, (BLUR_KERNEL, BLUR_KERNEL), 0)
        
        # Step 4: Apply CLAHE if low-light (on grayscale only)
        if is_low_light:
            gray = self.clahe.apply(gray)
            self.frames_enhanced += 1
        
        return gray, is_low_light, avg_intensity
    
    def get_stats(self):
        """Get processing statistics."""
        rate = 0
        if self.frames_processed > 0:
            rate = (self.frames_enhanced / self.frames_processed) * 100
        return {
            'processed': self.frames_processed,
            'enhanced': self.frames_enhanced,
            'rate': f"{rate:.1f}%"
        }
