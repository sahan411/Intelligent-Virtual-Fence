"""
Module 7: Visualizer
=====================
This module handles all drawing and visual feedback for the system.

Purpose:
    - Draw bounding boxes (green=safe, red=intrusion)
    - Draw foot-points to show decision basis
    - Display alert overlay when intrusion detected
    - Keep visualization logic separate from decision logic

Design Principle:
    Main loop stays clean - just call visualizer.draw()
    All drawing complexity is hidden inside this module.

Visual Elements:
    - Green box: Person detected OUTSIDE ROI (safe)
    - Red box: Person detected INSIDE ROI (intrusion!)
    - Foot-point dot: Shows the exact point used for decision
    - ALERT banner: Flashes when active intrusion detected
"""

import cv2
import numpy as np


# Colors (BGR format)
COLOR_SAFE = (0, 255, 0)       # Green - outside ROI
COLOR_INTRUSION = (0, 0, 255)  # Red - inside ROI
COLOR_MEDIUM_ALERT = (0, 165, 255)  # Orange - animal intrusion
COLOR_ROI = (255, 255, 0)      # Cyan - ROI boundary
COLOR_ALERT_BG = (0, 0, 200)   # Dark red - alert background
COLOR_MEDIUM_ALERT_BG = (0, 110, 180)
COLOR_WHITE = (255, 255, 255)

# Drawing settings
BOX_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
FONT_THICKNESS = 2
FOOT_POINT_RADIUS = 4


class Visualizer:
    """
    Handles all drawing operations for the virtual fence system.
    
    Keeps visualization separate from logic modules.
    """
    
    def __init__(self, roi_points, alert_hold_frames=15, box_thickness=2,
                 font_scale=0.5, alert_levels=None):
        """
        Initialize visualizer with ROI points.
        
        Args:
            roi_points: List of (x, y) tuples defining the ROI polygon
        """
        self.roi_points = roi_points
        self.box_thickness = int(box_thickness)
        self.font_scale = float(font_scale)
        self.alert_levels = alert_levels or {
            "high": {"label": "HIGH PRIORITY", "classes": [0]},
            "medium": {"label": "MEDIUM PRIORITY", "classes": [14, 15, 16, 17, 18, 19, 20, 21, 22, 23]}
        }
        
        # Alert persistence (so alert doesn't flicker)
        self.alert_frames = 0
        self.alert_hold_time = int(alert_hold_frames)  # Keep alert visible for N frames after intrusion
        self.current_alert_level = None
    
    def draw(self, frame, intrusions, motion_triggered=False):
        """
        Draw all visual elements on frame.
        
        Args:
            frame: The frame to draw on (will be modified!)
            intrusions: List of detection dicts with inside_roi flag
            motion_triggered: Whether motion was detected this frame
        
        Returns:
            The frame with all drawings applied
        """
        # Draw ROI polygon
        self._draw_roi(frame)
        
        # Draw each detection
        has_intrusion = False
        for det in intrusions:
            is_inside = det['inside_roi']
            if is_inside:
                has_intrusion = True
            self._draw_detection(frame, det)
        
        # Update and draw alert if needed
        if has_intrusion:
            self.alert_frames = self.alert_hold_time
            self.current_alert_level = self._get_alert_level(intrusions)
        
        if self.alert_frames > 0:
            self._draw_alert(frame, self.current_alert_level)
            self.alert_frames -= 1
        
        return frame
    
    def _draw_roi(self, frame):
        """Draw the ROI polygon boundary."""
        if len(self.roi_points) < 3:
            return
        
        # Draw polygon outline - convert to numpy array for OpenCV
        pts = np.array(self.roi_points, dtype=np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, 
                      color=COLOR_ROI, thickness=2)
    
    def _draw_detection(self, frame, detection):
        """
        Draw a single detection with box, label, and foot-point.
        
        Args:
            frame: Frame to draw on
            detection: Detection dict with bbox, inside_roi, foot_point, etc.
        """
        x1, y1, x2, y2 = detection['bbox']
        foot_x, foot_y = detection['foot_point']
        is_inside = detection['inside_roi']
        confidence = detection['confidence']
        class_name = detection.get('class_name', 'object')
        track_id = detection.get('track_id')
        
        # Choose color based on intrusion status
        if is_inside:
            alert_level = self._class_alert_level(detection.get('class_id'))
            color = COLOR_INTRUSION if alert_level == "high" else COLOR_MEDIUM_ALERT
            label = f"INTRUSION {class_name} {confidence:.0%}"
        else:
            color = COLOR_SAFE
            label = f"{class_name} {confidence:.0%}"

        if track_id is not None:
            label = f"ID {track_id} {label}"
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.box_thickness)
        
        # Draw label background (makes text readable)
        text_size = cv2.getTextSize(label, FONT, self.font_scale, FONT_THICKNESS)[0]
        cv2.rectangle(frame, 
                      (x1, y1 - text_size[1] - 10),
                      (x1 + text_size[0] + 4, y1),
                      color, -1)  # Filled rectangle
        
        # Draw label text
        cv2.putText(frame, label, (x1 + 2, y1 - 5),
                    FONT, self.font_scale, COLOR_WHITE, FONT_THICKNESS)
        
        # Draw foot-point (the decision point)
        cv2.circle(frame, (foot_x, foot_y), FOOT_POINT_RADIUS, color, -1)
        
        # Draw line from bottom of bbox to foot-point (visual clarity)
        cv2.line(frame, ((x1 + x2) // 2, y2), (foot_x, foot_y), color, 1)
    
    def _draw_alert(self, frame, alert_level):
        """Draw alert banner at top of frame."""
        h, w = frame.shape[:2]
        
        # Alert banner dimensions
        banner_height = 40
        
        # Draw semi-transparent red banner
        overlay = frame.copy()
        bg_color = COLOR_ALERT_BG if alert_level == "high" else COLOR_MEDIUM_ALERT_BG
        cv2.rectangle(overlay, (0, 0), (w, banner_height), bg_color, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw alert text
        level_label = self.alert_levels.get(alert_level or "medium", {}).get("label", "ALERT")
        alert_text = f"! {level_label} INTRUSION !"
        text_size = cv2.getTextSize(alert_text, FONT, 0.8, 2)[0]
        text_x = (w - text_size[0]) // 2
        text_y = (banner_height + text_size[1]) // 2
        
        cv2.putText(frame, alert_text, (text_x, text_y),
                    FONT, 0.8, COLOR_WHITE, 2)
    
    def update_roi(self, roi_points):
        """Update ROI points if changed."""
        self.roi_points = roi_points

    def _get_alert_level(self, detections):
        """Return the highest active alert level for current intrusion detections."""
        for detection in detections:
            if detection.get('inside_roi') and self._class_alert_level(detection.get('class_id')) == "high":
                return "high"
        return "medium"

    def _class_alert_level(self, class_id):
        """Map a COCO class ID to configured alert level."""
        for level_name in ("high", "medium"):
            level = self.alert_levels.get(level_name, {})
            if class_id in level.get("classes", []):
                return level_name
        return "medium"
