"""
Core modules for Intelligent Virtual Fence.

This package contains the main processing modules:
    - input_manager: Video source handling (Module 1) - DONE
    - roi_manager: Virtual fence definition (Module 2) - DONE
    - preprocessor: Frame enhancement (Module 3) - DONE
    - motion_gate: Motion detection (Module 4) - TODO
    - detector: YOLO object detection (Module 5) - TODO
    - spatial_logic: Intrusion detection (Module 6) - TODO
    - policy: Alert and visualization (Module 7) - TODO
"""

from .input_manager import InputManager
from .roi_manager import ROIManager
from .preprocessor import Preprocessor
