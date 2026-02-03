"""
Module 4: Motion Gate (Efficiency Core)
========================================
This module decides whether expensive YOLO detection should run.

Key Principle:
    If no motion inside ROI → skip YOLO → save CPU

Responsibilities:
    - Detect motion using background subtraction (MOG2)
    - Apply ROI mask to focus only on the virtual fence area
    - Clean up noise with morphological operations
    - Compute motion score (pixel count inside ROI)
    - Trigger detection only when motion exceeds threshold

Design Notes:
    - MOG2 learns the background over time
    - We only care about motion INSIDE the ROI
    - Warm-up period prevents false triggers at startup
    - Debounce prevents rapid on/off flickering
"""

import cv2
import numpy as np


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
# MOG2 parameters
MOG2_HISTORY = 500          # Number of frames for background learning
MOG2_VAR_THRESHOLD = 16     # Threshold for foreground detection
MOG2_DETECT_SHADOWS = False # Ignore shadows (simpler)

# Morphology kernel sizes
MORPH_KERNEL_SIZE = 5       # For opening and closing operations

# Motion threshold - minimum pixels to trigger detection
MOTION_THRESHOLD = 500

# Warm-up: MOG2 needs time to learn background
# During warm-up, we never trigger (avoids false positives at start)
WARMUP_FRAMES = 30

# Debounce: once triggered, stay triggered for this many frames
# Prevents flickering when motion is near threshold
DEBOUNCE_FRAMES = 10


class MotionGate:
    """
    Decides whether to trigger YOLO based on motion inside ROI.
    
    Pipeline:
        1. Apply MOG2 to get foreground mask
        2. Threshold to clean binary
        3. Apply ROI mask (ignore motion outside fence)
        4. Morphological cleanup (remove noise)
        5. Count motion pixels
        6. If count > threshold → trigger YOLO
    
    Features:
        - Warm-up period (first 30 frames: no triggers)
        - Debounce (once triggered, stays on for 10 frames)
    """
    
    def __init__(self, roi_mask, motion_threshold=MOTION_THRESHOLD):
        """
        Initialize the motion gate.
        
        Args:
            roi_mask: Binary mask from ROI Manager (255 inside, 0 outside)
            motion_threshold: Minimum pixels to trigger detection
        """
        self.roi_mask = roi_mask
        self.motion_threshold = motion_threshold
        
        # Create MOG2 background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=MOG2_HISTORY,
            varThreshold=MOG2_VAR_THRESHOLD,
            detectShadows=MOG2_DETECT_SHADOWS
        )
        
        # Create morphology kernel (reuse for performance)
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE)
        )
        
        # Frame counter for warm-up
        self.frame_count = 0
        self.is_warmed_up = False
        
        # Debounce counter
        # When > 0, we force trigger to True
        self.debounce_counter = 0
        
        # Stats
        self.frames_triggered = 0
    
    def check(self, gray_frame):
        """
        Check if there's enough motion to trigger detection.
        
        Args:
            gray_frame: Preprocessed grayscale frame
        
        Returns:
            tuple: (trigger, motion_score, fg_mask_clean)
                - trigger: True if motion exceeds threshold
                - motion_score: Number of motion pixels inside ROI
                - fg_mask_clean: Cleaned foreground mask (for debug display)
        """
        self.frame_count += 1
        
        # Step 1: Get foreground mask from MOG2
        fg_mask = self.bg_subtractor.apply(gray_frame)
        
        # Step 2: Threshold to ensure clean binary
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        
        # Step 3: Apply ROI mask
        fg_mask_roi = cv2.bitwise_and(fg_mask, self.roi_mask)
        
        # Step 4: Morphological cleanup
        fg_mask_clean = cv2.morphologyEx(fg_mask_roi, cv2.MORPH_OPEN, self.kernel)
        fg_mask_clean = cv2.morphologyEx(fg_mask_clean, cv2.MORPH_CLOSE, self.kernel)
        
        # Step 5: Count motion pixels
        motion_score = cv2.countNonZero(fg_mask_clean)
        
        # Step 6: Warm-up check
        # MOG2 needs time to learn background - don't trigger during warm-up
        if self.frame_count <= WARMUP_FRAMES:
            # Still warming up - never trigger
            return False, motion_score, fg_mask_clean
        
        if not self.is_warmed_up:
            self.is_warmed_up = True
            print(f"[MotionGate] Warm-up complete after {WARMUP_FRAMES} frames.")
        
        # Step 7: Check threshold
        raw_trigger = motion_score > self.motion_threshold
        
        # Step 8: Debounce logic
        if raw_trigger:
            # Motion detected - reset debounce counter
            self.debounce_counter = DEBOUNCE_FRAMES
        
        # If debounce counter > 0, we're in "triggered" state
        if self.debounce_counter > 0:
            trigger = True
            self.debounce_counter -= 1
        else:
            trigger = False
        
        if trigger:
            self.frames_triggered += 1
        
        return trigger, motion_score, fg_mask_clean
    
    def update_roi_mask(self, new_mask):
        """
        Update the ROI mask (if user redefines the fence).
        
        Args:
            new_mask: New binary ROI mask
        """
        self.roi_mask = new_mask
    
    def set_threshold(self, threshold):
        """
        Update motion threshold.
        
        Args:
            threshold: New threshold value
        """
        self.motion_threshold = threshold
    
    def get_stats(self):
        """Get motion gate statistics."""
        trigger_rate = 0
        if self.frame_count > WARMUP_FRAMES:
            effective_frames = self.frame_count - WARMUP_FRAMES
            trigger_rate = (self.frames_triggered / effective_frames) * 100 if effective_frames > 0 else 0
        
        return {
            'checked': self.frame_count,
            'triggered': self.frames_triggered,
            'trigger_rate': f"{trigger_rate:.1f}%",
            'threshold': self.motion_threshold,
            'warmed_up': self.is_warmed_up
        }
    
    def reset_background(self):
        """
        Reset the background model and warm-up state.
        """
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=MOG2_HISTORY,
            varThreshold=MOG2_VAR_THRESHOLD,
            detectShadows=MOG2_DETECT_SHADOWS
        )
        self.frame_count = 0
        self.is_warmed_up = False
        self.debounce_counter = 0
        print("[MotionGate] Background model reset. Warm-up restarting.")
        
        Args:
            new_mask: New binary ROI mask
        """
        self.roi_mask = new_mask
    
    def set_threshold(self, threshold):
        """
        Update motion threshold.
        
        Args:
            threshold: New threshold value
        """
        self.motion_threshold = threshold
    
    def get_stats(self):
        """Get motion gate statistics."""
        trigger_rate = 0
        if self.frames_checked > 0:
            trigger_rate = (self.frames_triggered / self.frames_checked) * 100
        
        return {
            'checked': self.frames_checked,
            'triggered': self.frames_triggered,
            'trigger_rate': f"{trigger_rate:.1f}%",
            'threshold': self.motion_threshold
        }
    
    def reset_background(self):
        """
        Reset the background model.
        
        Useful if the scene changes significantly (e.g., lights turned on/off).
        """
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=MOG2_HISTORY,
            varThreshold=MOG2_VAR_THRESHOLD,
            detectShadows=MOG2_DETECT_SHADOWS
        )
        print("[MotionGate] Background model reset.")
