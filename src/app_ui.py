import streamlit as st
import cv2
import sys
import os
import time
import numpy as np
from PIL import Image

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.input_manager import InputManager
from core.roi_manager import ROIManager
from core.preprocess import Preprocessor
from core.motion_gate import MotionGate
from core.detector import Detector
from core.decision_logic import DecisionLogic

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROI_CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "roi_config.json")
DEFAULT_VIDEO_PATH = os.path.join(PROJECT_ROOT, "assets", "videos", "demo.mp4")

# Page Config
st.set_page_config(
    page_title="Intelligent Virtual Fence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Sleek UI
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    .status-badge {
        padding: 5px 10px;
        border-radius: 5px;
        font-weight: bold;
        text-align: center;
    }
    .status-idle {
        background-color: #262730;
        color: #00ff00;
        border: 1px solid #00ff00;
    }
    .status-analyzing {
        background-color: #4a1c1c;
        color: #ff4b4b;
        border: 1px solid #ff4b4b;
    }
    .metric-card {
        background-color: #262730;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #4a90e2;
    }
</style>
""", unsafe_allow_html=True)

def run_roi_setup_subprocess():
    """Runs a separate process to handle ROI drawing via OpenCV window"""
    import subprocess
    # We'll assume we can just run a small script or main.py with a flag, 
    # but for now let's just warn the user.
    # Actually, let's try to run main.py in a way that it just does ROI and exits.
    # For now, we'll just instruct the user.
    pass

def main():
    # -------------------------------------------------------------------------
    # Sidebar
    # -------------------------------------------------------------------------
    st.sidebar.title("🛡️ Virtual Fence")
    st.sidebar.markdown("---")
    
    st.sidebar.header("Settings")
    
    # input source
    use_webcam = st.sidebar.checkbox("Use Webcam (Live)", value=False)
    source_path = 0 if use_webcam else DEFAULT_VIDEO_PATH
    
    # Thresholds
    st.sidebar.subheader("Sensitivity")
    motion_thresh = st.sidebar.slider("Motion Threshold", 100, 2000, 500, 50, help="Pixels required to trigger YOLO")
    conf_thresh = st.sidebar.slider("AI Confidence", 0.1, 1.0, 0.4, 0.05)
    
    # Controls
    st.sidebar.markdown("---")
    start_btn = st.sidebar.button("Start Monitoring", type="primary")
    stop_btn = st.sidebar.button("Stop")
    
    roi_btn = st.sidebar.button("🛠️ Configure ROI")
    if roi_btn:
        st.sidebar.warning("Please run 'python src/main.py' locally to configure ROI with mouse interactions.")

    # -------------------------------------------------------------------------
    # Main Layout
    # -------------------------------------------------------------------------
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.title("Intelligent Virtual Fence")
        video_placeholder = st.empty()
    
    with col2:
        st.subheader("System Status")
        status_placeholder = st.empty()
        status_placeholder.markdown('<div class="status-badge status-idle">SYSTEM IDLE</div>', unsafe_allow_html=True)
        
        st.markdown("### detection Logs")
        log_placeholder = st.empty()
        
        st.markdown("### Metrics")
        fps_metric = st.empty()
        motion_metric = st.empty()

    # -------------------------------------------------------------------------
    # Logic
    # -------------------------------------------------------------------------
    if "monitoring" not in st.session_state:
        st.session_state.monitoring = False

    if start_btn:
        st.session_state.monitoring = True
        st.rerun()
    
    if stop_btn:
        st.session_state.monitoring = False
        st.rerun()

    if st.session_state.monitoring:
        # initialize components
        input_mgr = InputManager(source=source_path, width=640, height=360, fps=30)
        
        if not input_mgr.open():
            st.error("Could not open video source.")
            st.session_state.monitoring = False
            return

        roi_mgr = ROIManager(frame_width=640, frame_height=360, config_path=ROI_CONFIG_PATH)
        if not roi_mgr.load_roi():
            st.error("ROI not found! Please run the configuration utility first.")
            input_mgr.release()
            st.session_state.monitoring = False
            return
            
        preprocessor = Preprocessor()
        motion_gate = MotionGate(roi_mask=roi_mgr.get_mask(), motion_threshold=motion_thresh)
        # Pass dynamic confidence
        detector = Detector(confidence=conf_thresh)
        
        decision_logic = DecisionLogic(roi_points=roi_mgr.roi_points)
        
        frame_count = 0
        
        logs = []
        
        # Create a placeholder for the stop button INSIDE the running state 
        # so user can stop without sidebar reload, though sidebar stop works too.
        
        while input_mgr.cap.isOpened() and st.session_state.monitoring:
            ok, frame = input_mgr.read_frame_with_fps_control()
            if not ok:
                st.info("End of video stream.")
                st.session_state.monitoring = False
                st.rerun()
                break

                
            frame_count += 1
            
            # 1. Preprocess
            gray_frame, is_low_light, avg_int = preprocessor.process(frame)
            
            # 2. Motion Gate
            # We need to update motion gate threshold dynamically if we want slider to work live...
            # but object is already created. For now, static config at start is fine.
            trigger, motion_score, fg_mask = motion_gate.check(gray_frame)
            
            # 3. Detection
            intrusions = []
            is_analyzing = False
            
            if trigger:
                is_analyzing = True
                detections = detector.detect(frame)
                intrusions = decision_logic.process(detections)
            
            # 4. Visualization
            # Draw ROI
            display_frame = roi_mgr.draw_roi_on_frame(frame)
            
            # Draw Status on Frame
            if is_analyzing:
                cv2.putText(display_frame, "YOLO ACTIVE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(display_frame, "MONITORING (Motion Gate)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Draw Detections
            for det in intrusions:
                x1, y1, x2, y2 = det['bbox']
                color = (0, 0, 255) if det['inside_roi'] else (0, 255, 0)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                
                label = f"{det['class_name']}"
                if det['inside_roi']:
                    label += " [INTRUSION]"
                    logging_msg = f"Frame {frame_count}: Person detected in ROI!"
                    if not logs or logs[-1] != logging_msg:
                        logs.append(logging_msg)
                        if len(logs) > 5: logs.pop(0)

                cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # 5. UI Updates
            # Convert BGR to RGB
            display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(display_frame_rgb, channels="RGB", use_column_width=True)
            
            # Update Status Badge
            if is_analyzing:
                status_html = '<div class="status-badge status-analyzing">⚠️ ANALYSIS ACTIVE</div>'
            else:
                status_html = '<div class="status-badge status-idle">✅ MONITORING</div>'
            status_placeholder.markdown(status_html, unsafe_allow_html=True)
            
            # Update Metrics
            motion_metric.metric("Motion Score", f"{motion_score}", delta_color="off")
            fps_metric.text(f"Frame: {frame_count}")
            
            # Update Logs
            log_text = "\n".join(logs)
            log_placeholder.code(log_text)
            
            # Small sleep to allow UI to breathe? 
            # In Streamlit loop, it yields back to frontend.
            # check if stop button was pressed? 
            # Not easy.
        
        input_mgr.release()
        # No need to success here, as we loop continuously until stop or end

if __name__ == "__main__":
    main()
