import json
import os
import tempfile
import unittest

import numpy as np

from src.core.preprocess import Preprocessor
from src.core.tracker import ObjectTracker
from src.utils import IntrusionLogger


class ObjectTrackerTests(unittest.TestCase):
    def test_same_object_keeps_track_id(self):
        tracker = ObjectTracker(max_missing_frames=2, iou_threshold=0.1, distance_threshold=50)

        first = tracker.update([{
            'class_id': 0,
            'class_name': 'person',
            'confidence': 0.9,
            'bbox': (10, 10, 60, 100),
            'foot_point': (35, 100),
            'inside_roi': True
        }])
        second = tracker.update([{
            'class_id': 0,
            'class_name': 'person',
            'confidence': 0.88,
            'bbox': (14, 12, 64, 102),
            'foot_point': (39, 102),
            'inside_roi': True
        }])

        self.assertEqual(first[0]['track_id'], second[0]['track_id'])
        self.assertEqual(tracker.get_stats()['total_tracks_created'], 1)

    def test_different_classes_get_different_track_ids(self):
        tracker = ObjectTracker(max_missing_frames=2)

        detections = tracker.update([
            {
                'class_id': 0,
                'class_name': 'person',
                'confidence': 0.9,
                'bbox': (10, 10, 60, 100),
                'foot_point': (35, 100),
                'inside_roi': True
            },
            {
                'class_id': 16,
                'class_name': 'dog',
                'confidence': 0.7,
                'bbox': (10, 10, 60, 100),
                'foot_point': (35, 100),
                'inside_roi': True
            }
        ])

        self.assertNotEqual(detections[0]['track_id'], detections[1]['track_id'])


class ConfigDrivenPreprocessorTests(unittest.TestCase):
    def test_preprocessor_uses_config_values(self):
        frame = np.full((20, 20, 3), 40, dtype=np.uint8)
        preprocessor = Preprocessor(
            blur_kernel=4,
            low_light_threshold=30,
            clahe_clip_limit=1.5,
            clahe_grid_size=4
        )

        _, is_low_light, _ = preprocessor.process(frame)

        self.assertEqual(preprocessor.blur_kernel, 5)
        self.assertFalse(is_low_light)


class LoggerTests(unittest.TestCase):
    def test_logger_records_intrusion_start_and_end(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = os.path.join(temp_dir, "intrusions.log")
            logger = IntrusionLogger(log_path, enabled=True)
            detection = {
                'track_id': 3,
                'class_name': 'dog',
                'confidence': 0.75,
                'foot_point': (20, 50),
                'alert_level': 'medium'
            }

            logger.log_intrusion_start(10, detection)
            logger.log_intrusion_end(25, 3, detection, 1.5)

            with open(log_path, "r") as file:
                content = file.read()

        self.assertIn("INTRUSION START - Frame 10: Track 3 dog", content)
        self.assertIn("INTRUSION END - Frame 25: Track 3 dog", content)


class ConfigFileTests(unittest.TestCase):
    def test_config_has_required_sections(self):
        with open("configs/config.json", "r") as file:
            config = json.load(file)

        self.assertIn("tracking", config)
        self.assertIn("alerts", config)
        self.assertIn("alert_levels", config["visualizer"])


if __name__ == "__main__":
    unittest.main()
