"""
Module 3: Preprocessing & Enhancement
======================================
This module prepares frames for stable analysis in the Intelligent Virtual Fence system.

Responsibilities:
    - Convert frames to grayscale (for motion detection)
    - Apply optional Gaussian blur (reduce noise)
    - Check for low-light conditions
    - Apply CLAHE enhancement only when needed

Design Notes:
    - We use CONDITIONAL enhancement to avoid over-processing
    - CLAHE (Contrast Limited Adaptive Histogram Equalization) helps in low-light
    - Grayscale is needed for motion detection (MOG2)
    - We keep both color and grayscale versions available

Why this approach?
    - Not every frame needs enhancement
    - Processing only when needed saves CPU
    - CLAHE is still the standard for low-light surveillance
"""

import cv2
import numpy as np


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# Threshold for low-light detection
# If average pixel intensity is below this, we consider it low-light
# Range is 0-255, typical indoor lighting is around 80-120
LOW_LIGHT_THRESHOLD = 50

# Gaussian blur kernel size (must be odd number)
# Larger = more blur, removes more noise but loses detail
BLUR_KERNEL_SIZE = 5

# CLAHE parameters
# clipLimit: contrast limiting threshold (higher = more contrast)
# tileGridSize: size of grid for local histogram equalization
CLAHE_CLIP_LIMIT = 2.0
CLAHE_GRID_SIZE = (8, 8)


class Preprocessor:
    """
    Handles frame preprocessing for the surveillance pipeline.
    
    This class prepares frames before they go to motion detection:
        1. Convert to grayscale
        2. Optional blur to reduce noise
        3. Enhance if lighting is poor
    """
    
    def __init__(self, low_light_threshold=LOW_LIGHT_THRESHOLD, 
                 blur_kernel=BLUR_KERNEL_SIZE,
                 apply_blur=True):
        """
        Initialize the preprocessor.
        
        Args:
            low_light_threshold: Intensity below which we apply enhancement
            blur_kernel: Size of Gaussian blur kernel
            apply_blur: Whether to apply blur (can disable for testing)
        """
        self.low_light_threshold = low_light_threshold
        self.blur_kernel = blur_kernel
        self.apply_blur = apply_blur
        
        # Create CLAHE object once (reuse for performance)
        # CLAHE = Contrast Limited Adaptive Histogram Equalization
        self.clahe = cv2.createCLAHE(
            clipLimit=CLAHE_CLIP_LIMIT,
            tileGridSize=CLAHE_GRID_SIZE
        )
        
        # Stats for debugging/monitoring
        self.frames_processed = 0
        self.frames_enhanced = 0
    
    def process(self, frame, force_enhance=False):
        """
        Main preprocessing function.
        
        Takes a color frame and returns both color and processed grayscale.
        
        Args:
            frame: Input BGR color frame
            force_enhance: If True, always apply CLAHE (useful for testing)
        
        Returns:
            Dictionary with:
                - 'color': Original color frame (unchanged)
                - 'gray': Grayscale version (blurred + enhanced if needed)
                - 'is_low_light': Boolean indicating if enhancement was applied
                - 'avg_intensity': Average brightness of the frame
        """
        self.frames_processed += 1
        
        # Step 1: Convert to grayscale
        # Motion detection works on grayscale images
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Step 2: Check lighting conditions
        # We use mean intensity to decide if frame is too dark
        avg_intensity = np.mean(gray)
        is_low_light = avg_intensity < self.low_light_threshold
        
        # Step 3: Apply Gaussian blur if enabled
        # This reduces noise which helps motion detection
        if self.apply_blur:
            gray = cv2.GaussianBlur(
                gray, 
                (self.blur_kernel, self.blur_kernel), 
                0  # sigmaX, 0 means auto-calculate from kernel size
            )
        
        # Step 4: Apply CLAHE if low-light or forced
        # CLAHE improves contrast in dark images
        if is_low_light or force_enhance:
            gray = self.clahe.apply(gray)
            self.frames_enhanced += 1
        
        return {
            'color': frame,
            'gray': gray,
            'is_low_light': is_low_light,
            'avg_intensity': avg_intensity
        }
    
    def to_grayscale(self, frame):
        """
        Simple grayscale conversion.
        Use this when you just need grayscale without other processing.
        
        Args:
            frame: BGR color frame
        
        Returns:
            Grayscale image
        """
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    def apply_gaussian_blur(self, frame, kernel_size=None):
        """
        Apply Gaussian blur to a frame.
        
        Args:
            frame: Input frame (color or grayscale)
            kernel_size: Override default kernel size
        
        Returns:
            Blurred frame
        """
        k = kernel_size if kernel_size else self.blur_kernel
        return cv2.GaussianBlur(frame, (k, k), 0)
    
    def check_lighting(self, frame):
        """
        Check if frame is low-light.
        
        Args:
            frame: Grayscale or color frame
        
        Returns:
            Tuple of (is_low_light, average_intensity)
        """
        # If color, convert to grayscale first
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        avg = np.mean(gray)
        is_low = avg < self.low_light_threshold
        
        return is_low, avg
    
    def enhance_contrast(self, gray_frame):
        """
        Apply CLAHE enhancement to a grayscale frame.
        
        CLAHE divides the image into small tiles and equalizes
        histogram locally, which works better than global equalization.
        
        Args:
            gray_frame: Grayscale input frame
        
        Returns:
            Enhanced grayscale frame
        """
        return self.clahe.apply(gray_frame)
    
    def get_stats(self):
        """
        Get preprocessing statistics.
        
        Returns:
            Dictionary with processing stats
        """
        enhancement_rate = 0
        if self.frames_processed > 0:
            enhancement_rate = (self.frames_enhanced / self.frames_processed) * 100
        
        return {
            'frames_processed': self.frames_processed,
            'frames_enhanced': self.frames_enhanced,
            'enhancement_rate': f"{enhancement_rate:.1f}%"
        }
    
    def reset_stats(self):
        """Reset the statistics counters."""
        self.frames_processed = 0
        self.frames_enhanced = 0


# -----------------------------------------------------------------------------
# Standalone functions (for simple use cases)
# -----------------------------------------------------------------------------

def convert_to_gray(frame):
    """Convert BGR frame to grayscale."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def apply_blur(frame, kernel_size=5):
    """Apply Gaussian blur to frame."""
    return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)


def is_low_light(frame, threshold=50):
    """
    Check if frame is low-light.
    
    Returns:
        (is_low_light, average_intensity)
    """
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    
    avg = np.mean(gray)
    return avg < threshold, avg


def apply_clahe(gray_frame, clip_limit=2.0, grid_size=(8, 8)):
    """
    Apply CLAHE enhancement.
    
    Args:
        gray_frame: Grayscale input
        clip_limit: Contrast limit
        grid_size: Tile grid size
    
    Returns:
        Enhanced grayscale frame
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(gray_frame)
