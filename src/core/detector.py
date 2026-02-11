"""
Module 5: Object Detection (Selective AI)
==========================================
This module handles YOLO-based object detection.

Key Principle:
    YOLO runs ONLY when motion gate triggers - saves CPU/GPU

Responsibilities:
    - Load YOLOv8 Nano model once at startup
    - Run detection only when triggered
    - Return clean list of detections
    - Filter to classes of interest (person by default)

Design Notes:
    - YOLOv8n is smallest/fastest model - good for real-time
    - Runs on full frame (not cropped) to preserve spatial context
    - Returns structured data for decision logic module
"""

from ultralytics import YOLO


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
# Model to use - YOLOv8 Nano is smallest and fastest
MODEL_NAME = "yolov8n.pt"

# Confidence threshold - ignore weak detections
# 0.4 is good for UG project (reduces false positives)
CONFIDENCE_THRESHOLD = 0.4

# Classes we care about (COCO dataset class IDs)
# LOCKED: person only for now (class ID 0)
# Later we can add: 16=dog, 17=horse, 18=sheep, 19=cow
CLASSES_OF_INTEREST = [0]  # person only

# Class name mapping (for display)
CLASS_NAMES = {
    0: "person",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    15: "cat"
}


class Detector:
    """
    YOLO-based object detector.
    
    Usage:
        detector = Detector()
        if trigger:
            detections = detector.detect(frame)
    """
    
    def __init__(self, model_name=MODEL_NAME, confidence=CONFIDENCE_THRESHOLD,
                 classes=CLASSES_OF_INTEREST):
        """
        Initialize the detector.
        
        Args:
            model_name: YOLO model file (will download if not present)
            confidence: Minimum confidence threshold
            classes: List of class IDs to detect
        """
        self.confidence = confidence
        self.classes = classes
        
        # Load YOLO model
        # First run will download the model automatically
        print(f"[Detector] Loading {model_name}...")
        self.model = YOLO(model_name)
        print(f"[Detector] Model loaded successfully.")
        
        # Stats
        self.inference_count = 0
        self.total_detections = 0
    
    def detect(self, frame):
        """
        Run object detection on a frame.
        
        Args:
            frame: BGR color frame (NOT grayscale)
        
        Returns:
            List of detections, each detection is a dict:
            {
                'class_id': 0,
                'class_name': 'person',
                'confidence': 0.85,
                'bbox': (x1, y1, x2, y2)  # Top-left and bottom-right corners
            }
        """
        self.inference_count += 1
        
        # Run YOLO inference
        # verbose=False suppresses the output spam
        results = self.model(
            frame,
            conf=self.confidence,
            classes=self.classes,
            verbose=False
        )
        
        # Parse results into clean format
        detections = []
        
        # results[0] contains the detections for first image
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            for i in range(len(boxes)):
                # Get bounding box coordinates
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                
                # Get class and confidence
                class_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                
                # Get class name
                class_name = CLASS_NAMES.get(class_id, f"class_{class_id}")
                
                detection = {
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': conf,
                    'bbox': (int(x1), int(y1), int(x2), int(y2))
                }
                
                detections.append(detection)
        
        self.total_detections += len(detections)
        
        return detections
    
    def get_stats(self):
        """Get detector statistics."""
        avg_per_frame = 0
        if self.inference_count > 0:
            avg_per_frame = self.total_detections / self.inference_count
        
        return {
            'inferences': self.inference_count,
            'total_detections': self.total_detections,
            'avg_per_frame': f"{avg_per_frame:.2f}"
        }
    
    def set_confidence(self, threshold):
        """Update confidence threshold."""
        self.confidence = threshold
    
    def set_classes(self, classes):
        """
        Update classes to detect.
        
        Args:
            classes: List of class IDs (e.g., [0, 16, 19] for person, dog, cow)
        """
        self.classes = classes
