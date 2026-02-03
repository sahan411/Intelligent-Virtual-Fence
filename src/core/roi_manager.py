"""
Module 2: ROI Manager (Configuration Module)
=============================================
This module handles the virtual fence definition for the Intelligent Virtual Fence system.

Responsibilities:
    - Let user draw a polygon ROI with mouse clicks
    - Save/load ROI configuration to/from JSON
    - Generate binary mask for ROI area
    - Provide ROI data to other modules

Design Notes:
    - ROI is defined ONCE at startup (or loaded from file)
    - This is NOT a processing stage - it's configuration data
    - Other modules will use roi_mask to filter their operations

Controls:
    Left Click  : Add a point to the polygon
    Right Click : Undo the last point
    ENTER       : Finish drawing and confirm ROI
    R           : Reset (clear all points)
    S           : Save ROI to JSON file
    L           : Load ROI from JSON file
    Q           : Quit without saving
"""

import cv2
import numpy as np
import json
import os


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DEFAULT_ROI_PATH = "configs/roi_config.json"

# Colors for visualization (BGR format)
COLOR_POINT = (0, 255, 0)       # Green for points
COLOR_LINE = (0, 255, 255)      # Yellow for lines
COLOR_POLYGON = (0, 255, 0)     # Green for completed polygon
COLOR_FILL = (0, 100, 0)        # Dark green for fill (with transparency)
COLOR_TEXT = (255, 255, 255)    # White for text

POINT_RADIUS = 5
LINE_THICKNESS = 2


class ROIManager:
    """
    Manages the Region of Interest (virtual fence) polygon.
    
    This class handles:
        - Interactive polygon drawing via mouse
        - Saving/loading ROI to/from JSON
        - Generating binary mask for the ROI
    """
    
    def __init__(self, frame_width, frame_height, config_path=DEFAULT_ROI_PATH):
        """
        Initialize the ROI Manager.
        
        Args:
            frame_width: Width of the video frames
            frame_height: Height of the video frames
            config_path: Path to save/load ROI configuration
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.config_path = config_path
        
        # List of points that make up the polygon
        # Each point is a tuple (x, y)
        self.roi_points = []
        
        # Binary mask - will be created when ROI is finalized
        # White (255) inside ROI, black (0) outside
        self.roi_mask = None
        
        # Flag to track if ROI is complete
        self.is_complete = False
        
        # Temporary frame for drawing (used during interactive mode)
        self._temp_frame = None
        self._window_name = "Draw ROI - Virtual Fence"
    
    # -------------------------------------------------------------------------
    # Mouse Callback
    # -------------------------------------------------------------------------
    def _mouse_callback(self, event, x, y, flags, param):
        """
        Handle mouse events during ROI drawing.
        
        This is called automatically by OpenCV when mouse events occur.
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            # Left click: Add a new point
            self.roi_points.append((x, y))
            print(f"[ROI] Added point {len(self.roi_points)}: ({x}, {y})")
            self._update_display()
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click: Remove the last point (undo)
            if len(self.roi_points) > 0:
                removed = self.roi_points.pop()
                print(f"[ROI] Removed point: {removed}")
                self._update_display()
    
    def _update_display(self):
        """
        Update the display with current polygon state.
        Called after each point is added/removed.
        """
        if self._temp_frame is None:
            return
        
        # Start with a fresh copy of the original frame
        display = self._base_frame.copy()
        
        # Draw the polygon visualization
        display = self.draw_roi_on_frame(display, show_points=True)
        
        # Add instruction text
        self._draw_instructions(display)
        
        cv2.imshow(self._window_name, display)
    
    def _draw_instructions(self, frame):
        """
        Draw instruction text on the frame.
        """
        instructions = [
            "LEFT CLICK: Add point",
            "RIGHT CLICK: Undo",
            "ENTER: Finish",
            "R: Reset",
            "S: Save | L: Load",
            "Q: Quit"
        ]
        
        y_offset = 25
        for i, text in enumerate(instructions):
            cv2.putText(
                frame, text,
                (10, y_offset + i * 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                COLOR_TEXT,
                1
            )
        
        # Show point count
        point_text = f"Points: {len(self.roi_points)}"
        cv2.putText(
            frame, point_text,
            (self.frame_width - 100, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            COLOR_POINT,
            2
        )
    
    # -------------------------------------------------------------------------
    # Interactive ROI Drawing
    # -------------------------------------------------------------------------
    def draw_roi_interactive(self, frame):
        """
        Start interactive ROI drawing mode.
        
        User can click to add points, and use keyboard controls.
        
        Args:
            frame: A sample frame to draw on (used as background)
        
        Returns:
            True if ROI was successfully defined
            False if user quit without defining ROI
        """
        print("\n" + "=" * 50)
        print("ROI Drawing Mode")
        print("=" * 50)
        print("Draw a polygon around the area you want to monitor.")
        print("You need at least 3 points to create a valid polygon.")
        print()
        
        # Store the base frame (we'll copy from this for each update)
        self._base_frame = frame.copy()
        self._temp_frame = frame.copy()
        
        # Create window and set mouse callback
        cv2.namedWindow(self._window_name)
        cv2.setMouseCallback(self._window_name, self._mouse_callback)
        
        # Initial display
        self._update_display()
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == 13:  # ENTER key
                # Finish drawing
                if len(self.roi_points) >= 3:
                    self._finalize_roi()
                    print("[ROI] ROI finalized successfully!")
                    cv2.destroyWindow(self._window_name)
                    return True
                else:
                    print("[ROI] Need at least 3 points! Keep adding points.")
            
            elif key == ord('r') or key == ord('R'):
                # Reset all points
                self.roi_points = []
                self.is_complete = False
                self.roi_mask = None
                print("[ROI] Reset - all points cleared.")
                self._update_display()
            
            elif key == ord('s') or key == ord('S'):
                # Save ROI to file
                if len(self.roi_points) >= 3:
                    self.save_roi()
                else:
                    print("[ROI] Need at least 3 points to save!")
            
            elif key == ord('l') or key == ord('L'):
                # Load ROI from file
                if self.load_roi():
                    self._update_display()
            
            elif key == ord('q') or key == ord('Q'):
                # Quit without saving
                print("[ROI] Quit - ROI not defined.")
                cv2.destroyWindow(self._window_name)
                return False
        
        return False
    
    def _finalize_roi(self):
        """
        Finalize the ROI after user confirms.
        Creates the binary mask from the polygon points.
        """
        if len(self.roi_points) < 3:
            print("[ROI] Cannot finalize - need at least 3 points.")
            return False
        
        # Create the binary mask
        self._create_mask()
        self.is_complete = True
        
        return True
    
    # -------------------------------------------------------------------------
    # Mask Generation
    # -------------------------------------------------------------------------
    def _create_mask(self):
        """
        Create a binary mask from the polygon points.
        
        The mask is:
            - Same size as the frame
            - White (255) inside the polygon
            - Black (0) outside the polygon
        
        This mask will be used by other modules to check if something
        is inside or outside the ROI.
        """
        # Create a black image (all zeros)
        self.roi_mask = np.zeros((self.frame_height, self.frame_width), dtype=np.uint8)
        
        # Convert points to numpy array format that fillPoly expects
        pts = np.array(self.roi_points, dtype=np.int32)
        pts = pts.reshape((-1, 1, 2))
        
        # Fill the polygon with white (255)
        cv2.fillPoly(self.roi_mask, [pts], 255)
        
        print(f"[ROI] Created mask: {self.frame_width}x{self.frame_height}")
    
    def get_mask(self):
        """
        Get the binary ROI mask.
        
        Returns:
            Binary mask (numpy array) or None if ROI not defined
        """
        return self.roi_mask
    
    def get_points(self):
        """
        Get the ROI polygon points.
        
        Returns:
            List of (x, y) tuples
        """
        return self.roi_points.copy()
    
    # -------------------------------------------------------------------------
    # Save / Load
    # -------------------------------------------------------------------------
    def save_roi(self, path=None):
        """
        Save ROI configuration to a JSON file.
        
        Args:
            path: Optional custom path. Uses default if not specified.
        
        Returns:
            True if saved successfully
        """
        if path is None:
            path = self.config_path
        
        if len(self.roi_points) < 3:
            print("[ROI] Cannot save - need at least 3 points.")
            return False
        
        # Prepare data to save
        data = {
            "roi_points": self.roi_points,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "description": "ROI configuration for Intelligent Virtual Fence"
        }
        
        # Make sure the directory exists
        dir_path = os.path.dirname(path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"[ROI] Created directory: {dir_path}")
        
        # Write to file
        try:
            with open(path, 'w') as f:
                json.dump(data, f, indent=4)
            print(f"[ROI] Saved ROI to: {path}")
            return True
        except Exception as e:
            print(f"[ROI] Error saving ROI: {e}")
            return False
    
    def load_roi(self, path=None):
        """
        Load ROI configuration from a JSON file.
        
        Args:
            path: Optional custom path. Uses default if not specified.
        
        Returns:
            True if loaded successfully
        """
        if path is None:
            path = self.config_path
        
        if not os.path.exists(path):
            print(f"[ROI] File not found: {path}")
            return False
        
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            # Load points
            self.roi_points = [tuple(p) for p in data["roi_points"]]
            
            # Check if frame size matches
            if data.get("frame_width") != self.frame_width or \
               data.get("frame_height") != self.frame_height:
                print("[ROI] Warning: Frame size different from saved config!")
                print(f"[ROI] Saved: {data.get('frame_width')}x{data.get('frame_height')}")
                print(f"[ROI] Current: {self.frame_width}x{self.frame_height}")
                # We still load it, but warn the user
            
            # Recreate mask
            self._create_mask()
            self.is_complete = True
            
            print(f"[ROI] Loaded ROI from: {path}")
            print(f"[ROI] Loaded {len(self.roi_points)} points.")
            return True
            
        except Exception as e:
            print(f"[ROI] Error loading ROI: {e}")
            return False
    
    # -------------------------------------------------------------------------
    # Visualization
    # -------------------------------------------------------------------------
    def draw_roi_on_frame(self, frame, show_points=False, alpha=0.3):
        """
        Draw the ROI polygon on a frame.
        
        This is used both during drawing and during normal operation
        to show the virtual fence on the video.
        
        Args:
            frame: The frame to draw on
            show_points: If True, draw circles at each point
            alpha: Transparency for the fill (0.0 to 1.0)
        
        Returns:
            Frame with ROI drawn on it
        """
        if len(self.roi_points) == 0:
            return frame
        
        output = frame.copy()
        pts = np.array(self.roi_points, dtype=np.int32)
        
        # Draw filled polygon with transparency
        if len(self.roi_points) >= 3:
            # Create overlay for transparency effect
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts.reshape((-1, 1, 2))], COLOR_FILL)
            cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
        
        # Draw the polygon outline
        if len(self.roi_points) >= 2:
            # Draw lines between consecutive points
            for i in range(len(self.roi_points)):
                start = self.roi_points[i]
                end = self.roi_points[(i + 1) % len(self.roi_points)]
                
                # If not complete, don't close the polygon (skip last line)
                if not self.is_complete and i == len(self.roi_points) - 1:
                    continue
                
                cv2.line(output, start, end, COLOR_LINE, LINE_THICKNESS)
        
        # Draw points
        if show_points:
            for i, point in enumerate(self.roi_points):
                cv2.circle(output, point, POINT_RADIUS, COLOR_POINT, -1)
                # Number each point
                cv2.putText(
                    output, str(i + 1),
                    (point[0] + 8, point[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    COLOR_TEXT,
                    1
                )
        
        return output
    
    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------
    def is_point_inside(self, x, y):
        """
        Check if a point is inside the ROI.
        
        This is useful for quick checks without using the mask.
        Uses OpenCV's pointPolygonTest function.
        
        Args:
            x: X coordinate
            y: Y coordinate
        
        Returns:
            True if point is inside the ROI
        """
        if len(self.roi_points) < 3:
            return False
        
        pts = np.array(self.roi_points, dtype=np.int32)
        # pointPolygonTest returns:
        #   > 0 if inside
        #   = 0 if on edge
        #   < 0 if outside
        result = cv2.pointPolygonTest(pts, (x, y), False)
        return result >= 0
    
    def is_defined(self):
        """Check if ROI is properly defined."""
        return self.is_complete and len(self.roi_points) >= 3
    
    def get_roi_info(self):
        """
        Get information about the current ROI.
        
        Returns:
            Dictionary with ROI information
        """
        return {
            "is_defined": self.is_defined(),
            "point_count": len(self.roi_points),
            "points": self.roi_points.copy(),
            "frame_size": (self.frame_width, self.frame_height),
            "config_path": self.config_path
        }
