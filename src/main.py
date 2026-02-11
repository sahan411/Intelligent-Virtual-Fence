"""
Intelligent Virtual Fence - Main Entry Point
=============================================
This is the main file that ties all modules together.

Current Status:
    - Module 1 (Input Manager): DONE
    - Module 2 (ROI Manager): DONE
    - Module 3 (Preprocessing): DONE
    - Module 4 (Motion Gate): DONE
    - Module 5 (Object Detection): DONE
    - Module 6 (Decision Logic): DONE
    - Module 7 (Visualizer): DONE

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
from core.detector import Detector
from core.decision_logic import DecisionLogic
from core.visualizer import Visualizer
from utils import (load_config, IntrusionLogger, ScreenshotCapture,
                   FPSCalculator, IntrusionDurationTracker, SoundAlert)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# Project root for resolving paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "config.json")

# Load configuration (or use defaults if not found)
config = load_config(CONFIG_PATH)

if config:
    # Video settings
    video_source = config['video']['source']
    if video_source == 0 or video_source == "0":
        VIDEO_PATH = 0
    else:
        VIDEO_PATH = os.path.join(PROJECT_ROOT, video_source)
    FRAME_WIDTH = config['video']['width']
    FRAME_HEIGHT = config['video']['height']
    TARGET_FPS = config['video']['target_fps']
    PLAYBACK_DELAY = config['video']['playback_delay_ms']
    
    # Motion gate settings
    MOTION_THRESHOLD = config['motion_gate']['threshold']
    
    # Logging settings
    LOG_ENABLED = config['logging']['enabled']
    LOG_FILE = os.path.join(PROJECT_ROOT, config['logging']['log_file'])
    SCREENSHOT_ENABLED = config['logging']['screenshot_on_intrusion']
    SCREENSHOT_FOLDER = os.path.join(PROJECT_ROOT, config['logging']['screenshot_folder'])
    SCREENSHOT_COOLDOWN = config['logging']['screenshot_cooldown_frames']
    
    # ROI path
    ROI_CONFIG_PATH = os.path.join(PROJECT_ROOT, config['paths']['roi_config'])
else:
    # Fallback defaults
    VIDEO_PATH = os.path.join(PROJECT_ROOT, "assets", "videos", "demo.mp4")
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 360
    TARGET_FPS = 30
    PLAYBACK_DELAY = 30
    MOTION_THRESHOLD = 500
    LOG_ENABLED = True
    LOG_FILE = os.path.join(PROJECT_ROOT, "logs", "intrusions.log")
    SCREENSHOT_ENABLED = True
    SCREENSHOT_FOLDER = os.path.join(PROJECT_ROOT, "assets", "screenshots")
    SCREENSHOT_COOLDOWN = 30
    ROI_CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "roi_config.json")


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

    # ---------------------------------------------------------------------
    # Step 5: Initialize YOLO Detector
    # ---------------------------------------------------------------------
    print("[Main] Initializing YOLO Detector...")
    detector = Detector()
    
    # ---------------------------------------------------------------------
    # Step 6: Initialize Decision Logic
    # ---------------------------------------------------------------------
    print("[Main] Initializing Decision Logic...")
    decision_logic = DecisionLogic(roi_points=roi_mgr.roi_points)
    print("[Main] Using foot-point based intrusion detection.")
    
    # ---------------------------------------------------------------------
    # Step 7: Initialize Visualizer
    # ---------------------------------------------------------------------
    print("[Main] Initializing Visualizer...")
    visualizer = Visualizer(roi_points=roi_mgr.roi_points)
    
    # ---------------------------------------------------------------------
    # Step 8: Initialize Logger and Screenshot Capture
    # ---------------------------------------------------------------------
    print("[Main] Initializing Logger...")
    logger = IntrusionLogger(log_file=LOG_FILE, enabled=LOG_ENABLED)
    
    print("[Main] Initializing Screenshot Capture...")
    screenshot = ScreenshotCapture(
        save_folder=SCREENSHOT_FOLDER,
        enabled=SCREENSHOT_ENABLED,
        cooldown_frames=SCREENSHOT_COOLDOWN
    )
    
    # ---------------------------------------------------------------------
    # Step 9: Initialize FPS, Duration Tracker, and Sound Alert
    # ---------------------------------------------------------------------
    fps_calc = FPSCalculator(avg_count=30)
    duration_tracker = IntrusionDurationTracker(fps=TARGET_FPS)
    sound_alert = SoundAlert(enabled=True, frequency=1000, duration_ms=150, cooldown_seconds=2.0)
    print()

    # ---------------------------------------------------------------------
    # Main Processing Loop
    # ---------------------------------------------------------------------
    print("[Main] Starting main loop.")
    print("[Main] Controls: q=quit, SPACE=pause, d=debug, +/-=sensitivity, s=screenshot")
    print("-" * 50)
    
    frame_count = 0
    paused = False
    show_debug = True  # Motion mask window
    current_threshold = MOTION_THRESHOLD
    
    while True:
        # Handle pause state
        if paused:
            key = cv2.waitKey(100) & 0xFF
            if key == ord(' '):
                paused = False
                print("[Main] Resumed.")
            elif key == ord('q'):
                print("[Main] User pressed 'q'. Exiting.")
                break
            continue
        
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
        # Step 6c: Object Detection (Module 5) - only if triggered
        # -----------------------------------------------------------------
        detections = []
        intrusions = []
        has_intrusion = False
        inside_count = 0
        
        if trigger:
            detections = detector.detect(frame)
            
            # ---------------------------------------------------------------
            # Step 6d: Decision Logic (Module 6) - check if inside ROI
            # ---------------------------------------------------------------
            # Process detections through decision logic
            # Each detection gets foot_point and inside_roi added
            intrusions = decision_logic.process(detections)
            
            # Count how many are actually inside ROI
            inside_count = sum(1 for d in intrusions if d.get('inside_roi', False))
            has_intrusion = inside_count > 0

        # -----------------------------------------------------------------
        # Step 6e: Visualization (Module 7)
        # -----------------------------------------------------------------
        # Create display frame and draw all visual elements
        display_frame = frame.copy()
        visualizer.draw(display_frame, intrusions, motion_triggered=trigger)
        
        # -----------------------------------------------------------------
        # Log and capture screenshot on intrusion (after visualization)
        # -----------------------------------------------------------------
        if has_intrusion:
            # Log the intrusion event
            logger.log_intrusion(frame_count, inside_count, intrusions)
            
            # Capture screenshot with annotations (respects cooldown)
            screenshot.capture(display_frame, frame_count)
            
            # Play sound alert
            sound_alert.alert()
        
        # Update screenshot cooldown even when no intrusion
        screenshot.tick()
        
        # -----------------------------------------------------------------
        # Update FPS and Duration Tracker
        # -----------------------------------------------------------------
        current_fps = fps_calc.tick()
        intrusion_duration = duration_tracker.update(has_intrusion)
        
        # After warm-up: show motion score and trigger status
        stats = motion_gate.get_stats()
        
        if not stats['warmed_up']:
            status = f"Warming up... ({frame_count}/30)"
            color = (255, 255, 0)  # Cyan during warm-up
        else:
            status = f"Motion: {motion_score}"
            if trigger:
                status += f" [TRIGGER] Detected: {len(detections)}"
                color = (0, 0, 255)  # Red when triggered
            else:
                color = (0, 255, 0)  # Green when idle
        
        # Draw status at bottom (avoids overlap with alert banner)
        h, w = display_frame.shape[:2]
        cv2.putText(display_frame, status, (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw FPS at top-right corner
        fps_text = f"FPS: {current_fps:.1f}"
        cv2.putText(display_frame, fps_text, (w - 100, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw intrusion duration if active
        if intrusion_duration > 0:
            duration_text = f"In zone: {intrusion_duration:.1f}s"
            # Draw with red background for visibility
            text_size = cv2.getTextSize(duration_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(display_frame, (w - text_size[0] - 15, 30), 
                          (w - 5, 55), (0, 0, 180), -1)
            cv2.putText(display_frame, duration_text, (w - text_size[0] - 10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display main frame
        cv2.imshow("Intelligent Virtual Fence", display_frame)
        
        # Show motion mask in separate window (for debugging)
        if show_debug:
            cv2.imshow("Motion Mask", fg_mask)
        
        # -----------------------------------------------------------------
        # Keyboard Controls
        # -----------------------------------------------------------------
        key = cv2.waitKey(PLAYBACK_DELAY) & 0xFF
        
        if key == ord('q'):
            print("[Main] User pressed 'q'. Exiting.")
            break
        elif key == ord(' '):
            paused = True
            print("[Main] Paused. Press SPACE to resume.")
        elif key == ord('d'):
            show_debug = not show_debug
            if not show_debug:
                cv2.destroyWindow("Motion Mask")
            print(f"[Main] Debug window: {'ON' if show_debug else 'OFF'}")
        elif key == ord('+') or key == ord('='):
            current_threshold += 100
            motion_gate.set_threshold(current_threshold)
            print(f"[Main] Motion threshold: {current_threshold}")
        elif key == ord('-'):
            current_threshold = max(100, current_threshold - 100)
            motion_gate.set_threshold(current_threshold)
            print(f"[Main] Motion threshold: {current_threshold}")
        elif key == ord('s'):
            # Manual screenshot
            timestamp = frame_count
            screenshot.capture(display_frame, timestamp)
            print("[Main] Manual screenshot taken.")
    
    # ---------------------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------------------
    print("-" * 50)
    print(f"[Main] Processed {frame_count} frames.")
    
    # Show stats
    prep_stats = preprocessor.get_stats()
    motion_stats = motion_gate.get_stats()
    detector_stats = detector.get_stats()
    decision_stats = decision_logic.get_stats()
    screenshot_stats = screenshot.get_stats()
    duration_stats = duration_tracker.get_stats()
    sound_stats = sound_alert.get_stats()
    
    print(f"[Main] Average FPS: {fps_calc.get_fps():.1f}")
    print(f"[Main] Enhanced frames: {prep_stats['enhanced']} ({prep_stats['rate']})")
    print(f"[Main] Motion triggers: {motion_stats['triggered']} ({motion_stats['trigger_rate']})")
    print(f"[Main] YOLO inferences: {detector_stats['inferences']}, Detections: {detector_stats['total_detections']}")
    print(f"[Main] Intrusion detections: {decision_stats['total_intrusions']} (frames where person inside ROI)")
    print(f"[Main] Max intrusion duration: {duration_stats['max_duration']:.1f}s")
    print(f"[Main] Total time in zone: {duration_stats['total_intrusion_time']:.1f}s")
    print(f"[Main] Screenshots saved: {screenshot_stats['total_captures']}")
    print(f"[Main] Sound alerts: {sound_stats['beep_count']}")
    
    # Log session end with all stats
    logger.log_session_end({
        'frames': frame_count,
        'motion_triggers': motion_stats['triggered'],
        'yolo_inferences': detector_stats['inferences'],
        'intrusions': decision_stats['total_intrusions'],
        'max_intrusion_duration': duration_stats['max_duration'],
        'total_intrusion_time': duration_stats['total_intrusion_time']
    })
    
    # Release resources
    input_mgr.release()
    cv2.destroyAllWindows()
    
    print("[Main] Done.")


if __name__ == "__main__":
    main()
