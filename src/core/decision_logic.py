"""
Module 6: Decision Logic (Spatial Intrusion Reasoning)
========================================================
This module determines if detected persons are inside the virtual fence.

Key Principle:
    Foot-point based intrusion detection - not center, not bbox overlap.

Why Foot-Point?
    - Intrusion happens on the GROUND PLANE
    - Foot-point approximates physical contact with ground
    - Reduces false positives when only upper body enters ROI
    - This is how real surveillance systems work

Input:
    - detections: list from YOLO detector
    - roi_polygon: polygon points from ROI manager

Output:
    - intrusions: list of detections with inside_roi flag added

Design Notes:
    - This module is PURE logic - no drawing, no YOLO, no UI
    - Polygon converted to numpy array ONCE at init (not every frame)
    - Clean separation: detections → decision → intrusions
"""

import cv2
import numpy as np


class DecisionLogic:
    """
    Determines which detections are inside the virtual fence.
    
    Uses foot-point method:
        - Foot-point = bottom-center of bounding box
        - Check if foot-point is inside ROI polygon
    """
    
    def __init__(self, roi_points):
        """
        Initialize decision logic with ROI polygon.
        
        Args:
            roi_points: List of (x, y) tuples defining the ROI polygon
        
        Note:
            Polygon is converted to numpy array ONCE here,
            not on every frame (performance optimization).
        """
        # Convert polygon points to numpy array once
        # This avoids repeated conversion every frame
        self.roi_polygon = np.array(roi_points, dtype=np.int32)
        
        # Stats
        self.frames_processed = 0
        self.total_intrusions = 0
    
    def process(self, detections):
        """
        Process detections and determine which are inside ROI.
        
        Args:
            detections: List of detection dicts from YOLO detector
                        Each has: class_id, class_name, confidence, bbox
        
        Returns:
            List of detection dicts with 'inside_roi' and 'foot_point' added:
            {
                'class_id': 0,
                'class_name': 'person',
                'confidence': 0.85,
                'bbox': (x1, y1, x2, y2),
                'foot_point': (x, y),
                'inside_roi': True/False
            }
        """
        self.frames_processed += 1
        
        results = []
        
        for det in detections:
            # Calculate foot-point (bottom-center of bounding box)
            x1, y1, x2, y2 = det['bbox']
            foot_x = (x1 + x2) // 2  # Center horizontally
            foot_y = y2              # Bottom of bbox (feet)
            foot_point = (foot_x, foot_y)
            
            # Check if foot-point is inside ROI polygon
            inside = self._point_in_polygon(foot_point)
            
            # Add results to detection
            result = det.copy()
            result['foot_point'] = foot_point
            result['inside_roi'] = inside
            
            if inside:
                self.total_intrusions += 1
            
            results.append(result)
        
        return results
    
    def _point_in_polygon(self, point):
        """
        Check if a point is inside the ROI polygon.
        
        Args:
            point: Tuple (x, y)
        
        Returns:
            True if point is inside polygon
        """
        # pointPolygonTest returns:
        #   > 0 if inside
        #   = 0 if on edge
        #   < 0 if outside
        result = cv2.pointPolygonTest(self.roi_polygon, point, False)
        return result >= 0
    
    def update_roi(self, roi_points):
        """
        Update ROI polygon (if user redefines the fence).
        
        Args:
            roi_points: New list of (x, y) tuples
        """
        self.roi_polygon = np.array(roi_points, dtype=np.int32)
    
    def get_stats(self):
        """Get decision logic statistics."""
        return {
            'frames_processed': self.frames_processed,
            'total_intrusions': self.total_intrusions
        }
