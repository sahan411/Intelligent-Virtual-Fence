"""
Simple object tracking for intrusion events.

The tracker keeps stable IDs for detected target objects so the same person or
animal is counted as one intruder across frames. It uses class-aware greedy
matching with IoU and foot-point distance, which is enough for this single-camera
demo without adding a heavier tracking dependency.
"""

import math


def bbox_iou(a, b):
    """Calculate intersection-over-union for two (x1, y1, x2, y2) boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0

    return inter_area / union


def point_distance(a, b):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


class ObjectTracker:
    """
    Class-aware tracker for YOLO detections.

    Each active track stores the latest detection and a missing-frame counter.
    Detections are matched to existing tracks if they have the same class and
    either enough bbox overlap or a nearby foot-point.
    """

    def __init__(self, max_missing_frames=15, iou_threshold=0.2, distance_threshold=80):
        self.max_missing_frames = int(max_missing_frames)
        self.iou_threshold = float(iou_threshold)
        self.distance_threshold = float(distance_threshold)

        self.next_track_id = 1
        self.tracks = {}
        self.total_tracks_created = 0

    def update(self, detections):
        """
        Update tracks with current detections.

        Args:
            detections: Detection dicts after decision logic. Each detection should
                        include bbox, class_id, class_name, foot_point, inside_roi.

        Returns:
            Detections with track_id added.
        """
        unmatched_track_ids = set(self.tracks.keys())
        updated_detections = []

        for detection in detections:
            track_id = self._find_best_track(detection, unmatched_track_ids)

            if track_id is None:
                track_id = self._create_track(detection)
            else:
                unmatched_track_ids.discard(track_id)

            detection = detection.copy()
            detection['track_id'] = track_id
            self.tracks[track_id]['detection'] = detection
            self.tracks[track_id]['missed'] = 0
            updated_detections.append(detection)

        for track_id in list(unmatched_track_ids):
            self.tracks[track_id]['missed'] += 1
            if self.tracks[track_id]['missed'] > self.max_missing_frames:
                del self.tracks[track_id]

        return updated_detections

    def mark_missed(self):
        """Advance missing-frame counters when YOLO did not run this frame."""
        for track_id in list(self.tracks.keys()):
            self.tracks[track_id]['missed'] += 1
            if self.tracks[track_id]['missed'] > self.max_missing_frames:
                del self.tracks[track_id]

    def get_active_inside_tracks(self):
        """Return last detections for tracks that are still considered inside ROI."""
        inside = {}
        for track_id, track in self.tracks.items():
            detection = track['detection']
            if detection.get('inside_roi', False):
                inside[track_id] = detection
        return inside

    def get_stats(self):
        """Get tracker statistics."""
        return {
            'active_tracks': len(self.tracks),
            'total_tracks_created': self.total_tracks_created
        }

    def _find_best_track(self, detection, candidate_ids):
        best_track_id = None
        best_score = -1.0

        for track_id in candidate_ids:
            track_detection = self.tracks[track_id]['detection']

            if track_detection.get('class_id') != detection.get('class_id'):
                continue

            iou = bbox_iou(track_detection['bbox'], detection['bbox'])
            distance = point_distance(
                track_detection.get('foot_point', (0, 0)),
                detection.get('foot_point', (0, 0))
            )

            if iou < self.iou_threshold and distance > self.distance_threshold:
                continue

            distance_score = max(0.0, 1.0 - (distance / self.distance_threshold))
            score = iou + (0.25 * distance_score)

            if score > best_score:
                best_score = score
                best_track_id = track_id

        return best_track_id

    def _create_track(self, detection):
        track_id = self.next_track_id
        self.next_track_id += 1
        self.total_tracks_created += 1

        self.tracks[track_id] = {
            'detection': detection.copy(),
            'missed': 0
        }

        return track_id
