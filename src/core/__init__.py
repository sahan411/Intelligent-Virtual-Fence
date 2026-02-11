"""
Core modules for Intelligent Virtual Fence.

This package contains the main processing modules:
    - input_manager: Video source handling (Module 1) - DONE
    - roi_manager: Virtual fence definition (Module 2) - DONE
    - preprocess: Frame enhancement (Module 3) - DONE
    - motion_gate: Motion detection gate (Module 4) - DONE
    - detector: YOLO object detection (Module 5) - DONE
    - decision_logic: Intrusion detection (Module 6) - DONE
    - visualizer: Drawing and alerts (Module 7) - TODO
"""

from .input_manager import InputManager
from .roi_manager import ROIManager
from .preprocess import Preprocessor
from .motion_gate import MotionGate
from .detector import Detector
from .decision_logic import DecisionLogic
