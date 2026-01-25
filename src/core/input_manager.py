import cv2


def open_video_source(source):
    """
    source can be:
      - 0 (webcam)
      - "assets/videos/demo.mp4" (video file path)
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")
    return cap


def read_frame(cap):
    """
    Reads one frame.
    Returns (ok, frame).
    """
    ok, frame = cap.read()
    return ok, frame


def release_source(cap):
    cap.release()
