"""
Intelligent Virtual Fence - Main Entry Point
=============================================
This is the main file that ties all modules together.

Current Status:
    - Module 1 (Input Manager): DONE
    - Module 2 (ROI Manager): DONE
    - Module 3 (Preprocessing): DONE
    - Module 4 (Motion Gate): DONE
    - Module 5 (Object Detection): TODO
    - Module 6 (Decision Logic): TODO
    - Module 7 (Visualizer): TODO

Usage:
    python main.py

Controls (during main loop):
    q - Quit the application

Controls (during ROI drawing):
    Left Click  - Add a point
    Right Click - Undo last point
    ENTER       - Finish ROI
    R           - Reset points
    S           - Save ROI
    L           - Load ROI
    Q           - Quit
"""

import cv2
import sys
import os

# Add src directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.input_manager import InputManager
from core.roi_manager import ROIManager
from core.preprocess import Preprocessor
from core.motion_gate import MotionGate


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
VIDEO_PATH = "assets/videos/demo.mp4"
# VIDEO_PATH = 0  # Uncomment for webcam

FRAME_WIDTH = 640
FRAME_HEIGHT = 360
TARGET_FPS = 30

ROI_CONFIG_PATH = "configs/roi_config.json"

# Motion gate settings
MOTION_THRESHOLD = 500  # Minimum pixels to trigger YOLO


def main():
    """
    Main function - entry point of the application.
    
    Flow:
        1. Initialize Input Manager (video source)
        2. Initialize ROI Manager and let user draw the fence
        3. Run main processing loop
    """
    print("=" * 50)
    print("Intelligent Virtual Fence")
    print("=" * 50)
    print()
    
    # ---------------------------------------------------------------------
    # Step 1: Initialize Input Manager
    # ---------------------------------------------------------------------
    print("[Main] Initializing Input Manager...")
    
    input_mgr = InputManager(
        source=VIDEO_PATH,
        width=FRAME_WIDTH,
        height=FRAME_HEIGHT,
        fps=TARGET_FPS
    )
    
    # Try to open the video source
    if not input_mgr.open():
        print("[Main] Failed to open video source. Exiting.")
        return
    
    # Print source info for debugging
    info = input_mgr.get_source_info()
    print(f"[Main] Source type: {'Webcam' if info['is_webcam'] else 'Video File'}")
    print()
    
    # ---------------------------------------------------------------------
    # Step 2: Initialize ROI Manager
    # ---------------------------------------------------------------------
    print("[Main] Initializing ROI Manager...")
    
    roi_mgr = ROIManager(
        frame_width=FRAME_WIDTH,
        frame_height=FRAME_HEIGHT,
        config_path=ROI_CONFIG_PATH
    )
    
    # Try to load existing ROI first
    # If a saved ROI exists, ask user if they want to use it
    if os.path.exists(ROI_CONFIG_PATH):
        print("[Main] Found existing ROI configuration.")
        use_existing = input("[Main] Use existing ROI? (y/n): ").strip().lower()
        
        if use_existing == 'y':
            if roi_mgr.load_roi():
                print("[Main] Loaded existing ROI successfully.")
            else:
                print("[Main] Failed to load ROI. Will draw new one.")
    
    # If ROI is not defined yet, let user draw it
    if not roi_mgr.is_defined():
        # Read one frame to use as background for ROI drawing
        ok, sample_frame = input_mgr.read_frame()
        
        if not ok:
            print("[Main] Cannot read frame for ROI drawing. Exiting.")
            input_mgr.release()
            return
        
        # Let user draw the ROI
        roi_defined = roi_mgr.draw_roi_interactive(sample_frame)
        
        if not roi_defined:
            print("[Main] ROI not defined. Exiting.")
            input_mgr.release()
            return
        
        # Ask if user wants to save the ROI
        save_roi = input("[Main] Save this ROI for future use? (y/n): ").strip().lower()
        if save_roi == 'y':
            roi_mgr.save_roi()
    
    # Print ROI info
    roi_info = roi_mgr.get_roi_info()
    print(f"[Main] ROI defined with {roi_info['point_count']} points.")
    print()
    
    # Reopen video to start from beginning
    # (we read a frame for ROI drawing, so video position moved)
    input_mgr.release()
    input_mgr.open()
    
    # ---------------------------------------------------------------------
    # Step 3: Initialize Preprocessor
    # ---------------------------------------------------------------------
    print("[Main] Initializing Preprocessor...")
    preprocessor = Preprocessor()
    
    # ---------------------------------------------------------------------
    # Step 4: Initialize Motion Gate
    # ---------------------------------------------------------------------
    print("[Main] Initializing Motion Gate...")
    motion_gate = MotionGate(
        roi_mask=roi_mgr.get_mask(),
        motion_threshold=MOTION_THRESHOLD
    )
    print(f"[Main] Motion threshold: {MOTION_THRESHOLD} pixels")
    print()
    
    # ---------------------------------------------------------------------
    # Step 5: Main Processing Loop
    # ---------------------------------------------------------------------
    print("[Main] Starting main loop. Press 'q' to quit.")
    print("-" * 50)
    
    frame_count = 0
    
    while True:
        # Read frame with FPS control
        ok, frame = input_mgr.read_frame_with_fps_control()
        
        if not ok:
            print("[Main] End of video or cannot read frame.")
            break
        
        frame_count += 1
        
        # -----------------------------------------------------------------
        # Step 5a: Preprocessing (Module 3)
        # -----------------------------------------------------------------
        gray_frame, is_low_light, avg_intensity = preprocessor.process(frame)
        
        # -----------------------------------------------------------------
        # Step 5b: Motion Gate (Module 4)
        # -----------------------------------------------------------------
        trigger, motion_score, fg_mask = motion_gate.check(gray_frame)
        
        # -----------------------------------------------------------------
        # TODO: Future modules
        # -----------------------------------------------------------------
        # Step 6: Object Detection if trigger=True (Module 5)
        # Step 7: Decision logic - check if inside ROI (Module 6)
        # Step 8: Visualization (Module 7)
        # -----------------------------------------------------------------
        
        # Draw ROI on frame
        display_frame = roi_mgr.draw_roi_on_frame(frame)
        
        # Show status line
        status = f"Motion: {motion_score}"
        if trigger:
            status += " [TRIGGER]"
            color = (0, 0, 255)  # Red when triggered
        else:
            color = (0, 255, 0)  # Green when idle
        
        cv2.putText(display_frame, status, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Display main frame
        cv2.imshow("Intelligent Virtual Fence", display_frame)
        
        # Show motion mask in separate window (for debugging)
        cv2.imshow("Motion Mask", fg_mask)
        
        # Check for quit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[Main] User pressed 'q'. Exiting.")
            break
    
    # ---------------------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------------------
    print("-" * 50)
    print(f"[Main] Processed {frame_count} frames.")
    
    # Show stats
    prep_stats = preprocessor.get_stats()
    motion_stats = motion_gate.get_stats()
    print(f"[Main] Enhanced frames: {prep_stats['enhanced']} ({prep_stats['rate']})")
    print(f"[Main] Motion triggers: {motion_stats['triggered']} ({motion_stats['trigger_rate']})")
    
    input_mgr.release()
    cv2.destroyAllWindows()
    print("[Main] Done.")


if __name__ == "__main__":
    main()
