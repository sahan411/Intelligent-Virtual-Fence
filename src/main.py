"""
Intelligent Virtual Fence - Main Entry Point
=============================================
This is the main file that ties all modules together.

Current Status:
    - Module 1 (Input Manager): DONE
    - Module 2 (ROI Manager): DONE
    - Module 3 (Preprocessing): DONE
    - Module 4 (Motion Gate): TODO
    - Module 5 (Object Detection): TODO
    - Module 6 (Spatial Logic): TODO
    - Module 7 (Policy & Visualization): TODO

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
# This is needed because we're running from src folder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.input_manager import InputManager
from core.roi_manager import ROIManager
from core.preprocessor import Preprocessor


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# Change this to 0 to use webcam instead of video file
VIDEO_PATH = "assets/videos/demo.mp4"
# VIDEO_PATH = 0  # Uncomment this line to use webcam

# Frame settings - these should match what InputManager expects
FRAME_WIDTH = 640
FRAME_HEIGHT = 360
TARGET_FPS = 30

# ROI configuration
ROI_CONFIG_PATH = "configs/roi_config.json"

# Preprocessing settings
LOW_LIGHT_THRESHOLD = 50  # Below this intensity, we enhance the frame


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
    
    preprocessor = Preprocessor(
        low_light_threshold=LOW_LIGHT_THRESHOLD,
        apply_blur=True
    )
    
    print(f"[Main] Low-light threshold: {LOW_LIGHT_THRESHOLD}")
    print()
    
    # ---------------------------------------------------------------------
    # Step 4: Main Processing Loop
    # ---------------------------------------------------------------------
    print("[Main] Starting main loop. Press 'q' to quit.")
    print("-" * 50)
    
    frame_count = 0
    
    while True:
        # Read frame with FPS control
        # This prevents video from playing too fast
        ok, frame = input_mgr.read_frame_with_fps_control()
        
        if not ok:
            print("[Main] End of video or cannot read frame.")
            break
        
        frame_count += 1
        
        # -----------------------------------------------------------------
        # Step 4a: Preprocessing (Module 3)
        # -----------------------------------------------------------------
        # Process the frame - converts to grayscale, applies blur,
        # and enhances if lighting is poor
        processed = preprocessor.process(frame)
        
        # Get the results
        gray_frame = processed['gray']          # For motion detection later
        is_low_light = processed['is_low_light']
        avg_intensity = processed['avg_intensity']
        
        # -----------------------------------------------------------------
        # TODO: Future modules will be called here
        # -----------------------------------------------------------------
        # Step 5: Motion Detection (Module 4) - will use gray_frame
        # Step 6: Object Detection if motion detected (Module 5)
        # Step 7: Spatial logic - check if inside ROI (Module 6)
        # Step 8: Policy check and visualization (Module 7)
        # -----------------------------------------------------------------
        
        # Draw ROI on frame (so user can see the virtual fence)
        display_frame = roi_mgr.draw_roi_on_frame(frame)
        
        # Show preprocessing info on frame
        # This helps us see if enhancement is being applied
        status_text = f"Intensity: {avg_intensity:.0f}"
        if is_low_light:
            status_text += " [LOW LIGHT - Enhanced]"
        
        cv2.putText(
            display_frame,
            status_text,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),  # Yellow
            1
        )
        
        # Show frame count at bottom
        cv2.putText(
            display_frame, 
            f"Frame: {frame_count}", 
            (10, display_frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (0, 255, 0),
            1
        )
        
        # Display the frame
        cv2.imshow("Intelligent Virtual Fence", display_frame)
        
        # Check for quit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[Main] User pressed 'q'. Exiting.")
            break
    
    # ---------------------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------------------
    print("-" * 50)
    print(f"[Main] Processed {frame_count} frames total.")
    
    # Show preprocessing stats
    prep_stats = preprocessor.get_stats()
    print(f"[Main] Frames enhanced (low-light): {prep_stats['frames_enhanced']} ({prep_stats['enhancement_rate']})")
    
    input_mgr.release()
    cv2.destroyAllWindows()
    print("[Main] Cleanup complete. Goodbye!")


if __name__ == "__main__":
    main()
