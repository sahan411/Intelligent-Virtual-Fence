"""
Intelligent Virtual Fence — Performance Benchmark
===================================================
Runs the full detection pipeline headlessly (no GUI windows) and collects
detailed per-frame timing data for presentation / report purposes.

Produces:
    1. A JSON file with raw per-frame metrics  (benchmark_results.json)
    2. A human-readable summary printed to console and saved as text

Usage:
    cd src
    python benchmark.py                          # uses config defaults
    python benchmark.py --source ../assets/videos/demo.mp4
    python benchmark.py --all-frames             # run YOLO on every frame (no gate)
"""

import argparse
import json
import os
import sys
import time
import statistics

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Path setup (same as main.py)
# ---------------------------------------------------------------------------
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.input_manager import InputManager
from core.roi_manager import ROIManager
from core.preprocess import Preprocessor
from core.motion_gate import MotionGate
from core.detector import Detector
from core.decision_logic import DecisionLogic
from core.tracker import ObjectTracker
from utils import load_config

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "config.json")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "benchmark_results")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ms(seconds: float) -> float:
    """Convert seconds to milliseconds, rounded to 2 decimals."""
    return round(seconds * 1000, 2)


def _safe_median(values):
    return round(statistics.median(values), 2) if values else 0.0


def _safe_mean(values):
    return round(statistics.mean(values), 2) if values else 0.0


def _safe_stdev(values):
    return round(statistics.stdev(values), 2) if len(values) >= 2 else 0.0


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmark(video_source, config, disable_gate: bool = False):
    """
    Run the full pipeline on *video_source* without opening any GUI windows.

    Args:
        video_source: path to a video file or integer camera index
        config:       loaded config dict (or None for defaults)
        disable_gate: if True, run YOLO on every frame (for comparison)

    Returns:
        dict with all collected metrics
    """

    # ---- Config values -------------------------------------------------
    if config:
        frame_w = config["video"]["width"]
        frame_h = config["video"]["height"]
        target_fps = config["video"]["target_fps"]
        prep_cfg = config.get("preprocessing", {})
        motion_cfg = config.get("motion_gate", {})
        motion_threshold = motion_cfg.get("threshold", 500)
        det_cfg = config.get("detector", {})
        model_name = det_cfg.get("model", "yolov8n.pt")
        det_conf = det_cfg.get("confidence_threshold", 0.4)
        det_classes = det_cfg.get("classes_of_interest", [0])
        profiles = det_cfg.get("target_profiles", {})
        active_profile = det_cfg.get("active_profile", "humans_only")
        if profiles and active_profile in profiles:
            det_classes = profiles[active_profile].get("classes", det_classes)
        tracking_cfg = config.get("tracking", {})
        roi_path = os.path.join(PROJECT_ROOT, config["paths"]["roi_config"])
    else:
        frame_w, frame_h, target_fps = 640, 360, 30
        prep_cfg, motion_cfg = {}, {}
        motion_threshold = 500
        model_name, det_conf, det_classes = "yolov8n.pt", 0.4, [0]
        tracking_cfg = {}
        roi_path = os.path.join(PROJECT_ROOT, "configs", "roi_config.json")

    # ---- Initialise modules --------------------------------------------
    input_mgr = InputManager(source=video_source, width=frame_w,
                             height=frame_h, fps=target_fps)
    if not input_mgr.open():
        print("[Benchmark] ERROR: cannot open video source.")
        return None

    roi_mgr = ROIManager(frame_width=frame_w, frame_height=frame_h,
                         config_path=roi_path)
    if not roi_mgr.load_roi():
        print("[Benchmark] ERROR: no saved ROI found. Run main.py first to draw one.")
        input_mgr.release()
        return None

    preprocessor = Preprocessor(
        blur_kernel=prep_cfg.get("blur_kernel", 5),
        low_light_threshold=prep_cfg.get("low_light_threshold", 50),
        clahe_clip_limit=prep_cfg.get("clahe_clip_limit", 2.0),
        clahe_grid_size=prep_cfg.get("clahe_grid_size", 8),
    )
    motion_gate = MotionGate(
        roi_mask=roi_mgr.get_mask(),
        motion_threshold=motion_threshold,
        warmup_frames=motion_cfg.get("warmup_frames", 30),
        debounce_frames=motion_cfg.get("debounce_frames", 10),
    )
    detector = Detector(model_name=model_name, confidence=det_conf,
                        classes=det_classes)
    decision_logic = DecisionLogic(roi_points=roi_mgr.roi_points)
    tracker = ObjectTracker(
        max_missing_frames=tracking_cfg.get("max_missing_frames", 15),
        iou_threshold=tracking_cfg.get("iou_threshold", 0.2),
        distance_threshold=tracking_cfg.get("distance_threshold", 80),
    )

    # ---- Per-frame collection lists ------------------------------------
    frame_times_ms = []           # total time per frame
    preprocess_times_ms = []      # preprocessing time
    motion_gate_times_ms = []     # motion gate time
    yolo_times_ms = []            # YOLO inference time (only triggered frames)
    decision_times_ms = []        # decision logic time
    tracking_times_ms = []        # tracker update time

    frame_count = 0
    motion_triggered_count = 0
    yolo_inference_count = 0
    total_detections = 0
    total_intrusions = 0
    enhanced_frames = 0
    unique_tracks = set()

    mode_label = "ALL FRAMES (no gate)" if disable_gate else "MOTION-GATED"
    print(f"\n{'='*60}")
    print(f"  Benchmark Mode: {mode_label}")
    print(f"  Video: {video_source}")
    print(f"  Resolution: {frame_w}x{frame_h}")
    print(f"{'='*60}\n")

    # ---- Main loop -----------------------------------------------------
    bench_start = time.perf_counter()

    while True:
        frame_start = time.perf_counter()

        ok, frame = input_mgr.read_frame()
        if not ok:
            break
        frame_count += 1

        # Preprocessing
        t0 = time.perf_counter()
        gray_frame, is_low_light, avg_intensity = preprocessor.process(frame)
        preprocess_times_ms.append(_ms(time.perf_counter() - t0))
        if is_low_light:
            enhanced_frames += 1

        # Motion gate
        t0 = time.perf_counter()
        trigger, motion_score, fg_mask = motion_gate.check(gray_frame)
        motion_gate_times_ms.append(_ms(time.perf_counter() - t0))

        # Override gate if --all-frames mode
        if disable_gate and frame_count > 30:  # still respect warmup
            trigger = True

        if trigger:
            motion_triggered_count += 1

            # YOLO detection
            t0 = time.perf_counter()
            detections = detector.detect(frame)
            yolo_time = time.perf_counter() - t0
            yolo_times_ms.append(_ms(yolo_time))
            yolo_inference_count += 1
            total_detections += len(detections)

            # Decision logic
            t0 = time.perf_counter()
            intrusions = decision_logic.process(detections)
            decision_times_ms.append(_ms(time.perf_counter() - t0))

            # Tracking
            t0 = time.perf_counter()
            intrusions = tracker.update(intrusions)
            tracking_times_ms.append(_ms(time.perf_counter() - t0))

            inside = [d for d in intrusions if d.get("inside_roi")]
            total_intrusions += len(inside)
            for d in intrusions:
                tid = d.get("track_id")
                if tid is not None:
                    unique_tracks.add(tid)
        else:
            tracker.mark_missed()

        frame_times_ms.append(_ms(time.perf_counter() - frame_start))

        # Progress indicator every 100 frames
        if frame_count % 100 == 0:
            print(f"  ... processed {frame_count} frames")

    bench_elapsed = time.perf_counter() - bench_start
    input_mgr.release()

    # ---- Aggregate stats -----------------------------------------------
    frames_skipped = frame_count - motion_triggered_count
    skip_rate = (frames_skipped / frame_count * 100) if frame_count else 0
    avg_fps = frame_count / bench_elapsed if bench_elapsed > 0 else 0

    results = {
        "mode": mode_label,
        "video_source": str(video_source),
        "resolution": f"{frame_w}x{frame_h}",
        "total_frames": frame_count,
        "total_time_seconds": round(bench_elapsed, 2),
        "average_fps": round(avg_fps, 1),
        "enhanced_frames": enhanced_frames,
        "enhanced_rate_pct": round(enhanced_frames / frame_count * 100, 1) if frame_count else 0,
        "motion_triggered_frames": motion_triggered_count,
        "frames_skipped": frames_skipped,
        "skip_rate_pct": round(skip_rate, 1),
        "yolo_inferences": yolo_inference_count,
        "total_detections": total_detections,
        "total_intrusions": total_intrusions,
        "unique_tracks": len(unique_tracks),
        "timing": {
            "frame_total": {
                "mean_ms": _safe_mean(frame_times_ms),
                "median_ms": _safe_median(frame_times_ms),
                "stdev_ms": _safe_stdev(frame_times_ms),
                "min_ms": round(min(frame_times_ms), 2) if frame_times_ms else 0,
                "max_ms": round(max(frame_times_ms), 2) if frame_times_ms else 0,
            },
            "preprocessing": {
                "mean_ms": _safe_mean(preprocess_times_ms),
                "median_ms": _safe_median(preprocess_times_ms),
            },
            "motion_gate": {
                "mean_ms": _safe_mean(motion_gate_times_ms),
                "median_ms": _safe_median(motion_gate_times_ms),
            },
            "yolo_inference": {
                "mean_ms": _safe_mean(yolo_times_ms),
                "median_ms": _safe_median(yolo_times_ms),
                "stdev_ms": _safe_stdev(yolo_times_ms),
                "min_ms": round(min(yolo_times_ms), 2) if yolo_times_ms else 0,
                "max_ms": round(max(yolo_times_ms), 2) if yolo_times_ms else 0,
                "count": len(yolo_times_ms),
            },
            "decision_logic": {
                "mean_ms": _safe_mean(decision_times_ms),
                "median_ms": _safe_median(decision_times_ms),
            },
            "tracking": {
                "mean_ms": _safe_mean(tracking_times_ms),
                "median_ms": _safe_median(tracking_times_ms),
            },
        },
        "per_frame_ms": frame_times_ms,   # raw array for charts
    }

    return results


def print_summary(results):
    """Pretty-print a results dict."""
    if results is None:
        return

    print(f"\n{'='*60}")
    print(f"  BENCHMARK RESULTS — {results['mode']}")
    print(f"{'='*60}")
    print(f"  Video           : {results['video_source']}")
    print(f"  Resolution      : {results['resolution']}")
    print(f"  Total frames    : {results['total_frames']}")
    print(f"  Total time      : {results['total_time_seconds']}s")
    print(f"  Average FPS     : {results['average_fps']}")
    print()
    print(f"  --- Pipeline Stats ---")
    print(f"  Enhanced frames : {results['enhanced_frames']} ({results['enhanced_rate_pct']}%)")
    print(f"  Motion triggers : {results['motion_triggered_frames']}")
    print(f"  Frames skipped  : {results['frames_skipped']} ({results['skip_rate_pct']}%)")
    print(f"  YOLO inferences : {results['yolo_inferences']}")
    print(f"  Total detections: {results['total_detections']}")
    print(f"  Total intrusions: {results['total_intrusions']}")
    print(f"  Unique tracks   : {results['unique_tracks']}")
    print()
    print(f"  --- Timing (ms) ---")
    t = results["timing"]
    print(f"  Frame total     : mean={t['frame_total']['mean_ms']}  "
          f"median={t['frame_total']['median_ms']}  "
          f"stdev={t['frame_total']['stdev_ms']}")
    print(f"  Preprocessing   : mean={t['preprocessing']['mean_ms']}  "
          f"median={t['preprocessing']['median_ms']}")
    print(f"  Motion gate     : mean={t['motion_gate']['mean_ms']}  "
          f"median={t['motion_gate']['median_ms']}")
    yi = t["yolo_inference"]
    print(f"  YOLO inference  : mean={yi['mean_ms']}  median={yi['median_ms']}  "
          f"min={yi['min_ms']}  max={yi['max_ms']}  (n={yi['count']})")
    print(f"  Decision logic  : mean={t['decision_logic']['mean_ms']}  "
          f"median={t['decision_logic']['median_ms']}")
    print(f"  Tracking        : mean={t['tracking']['mean_ms']}  "
          f"median={t['tracking']['median_ms']}")
    print(f"{'='*60}\n")


def save_results(results, label):
    """Save results dict as JSON (without per_frame_ms raw array in summary)."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    filename = f"benchmark_{label}.json"
    filepath = os.path.join(RESULTS_DIR, filename)

    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Benchmark] Results saved to: {filepath}")
    return filepath


def save_comparison(gated, ungated):
    """Save a side-by-side comparison summary."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    filepath = os.path.join(RESULTS_DIR, "comparison_summary.txt")

    lines = []
    lines.append("=" * 70)
    lines.append("  PERFORMANCE COMPARISON: Motion-Gated vs All-Frames")
    lines.append("=" * 70)
    lines.append("")

    header = f"{'Metric':<35} {'Motion-Gated':>15} {'All-Frames':>15}"
    lines.append(header)
    lines.append("-" * 70)

    rows = [
        ("Total frames", gated["total_frames"], ungated["total_frames"]),
        ("Average FPS", gated["average_fps"], ungated["average_fps"]),
        ("YOLO inferences", gated["yolo_inferences"], ungated["yolo_inferences"]),
        ("Frames skipped (no YOLO)", gated["frames_skipped"], ungated["frames_skipped"]),
        ("Skip rate %", f"{gated['skip_rate_pct']}%", f"{ungated['skip_rate_pct']}%"),
        ("Total detections", gated["total_detections"], ungated["total_detections"]),
        ("Total intrusions", gated["total_intrusions"], ungated["total_intrusions"]),
        ("Unique tracks", gated["unique_tracks"], ungated["unique_tracks"]),
        ("", "", ""),
        ("Frame time mean (ms)", gated["timing"]["frame_total"]["mean_ms"],
         ungated["timing"]["frame_total"]["mean_ms"]),
        ("YOLO mean (ms)", gated["timing"]["yolo_inference"]["mean_ms"],
         ungated["timing"]["yolo_inference"]["mean_ms"]),
        ("YOLO calls", gated["timing"]["yolo_inference"]["count"],
         ungated["timing"]["yolo_inference"]["count"]),
    ]

    for label, g_val, u_val in rows:
        lines.append(f"{label:<35} {str(g_val):>15} {str(u_val):>15}")

    # Compute savings
    if ungated["yolo_inferences"] > 0:
        savings = (1 - gated["yolo_inferences"] / ungated["yolo_inferences"]) * 100
        lines.append("")
        lines.append(f"  >> YOLO Compute Savings: {savings:.1f}%")

    if ungated["timing"]["frame_total"]["mean_ms"] > 0:
        speedup = ungated["timing"]["frame_total"]["mean_ms"] / gated["timing"]["frame_total"]["mean_ms"]
        lines.append(f"  >> Frame Processing Speedup: {speedup:.2f}x")

    fps_improvement = gated["average_fps"] - ungated["average_fps"]
    lines.append(f"  >> FPS Improvement: +{fps_improvement:.1f} FPS")

    lines.append("")
    lines.append("=" * 70)

    text = "\n".join(lines)
    with open(filepath, "w") as f:
        f.write(text)
    print(f"\n[Benchmark] Comparison saved to: {filepath}")
    print(text)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run headless performance benchmark for Intelligent Virtual Fence."
    )
    parser.add_argument("--source", help="Video file path (default: from config)")
    parser.add_argument("--all-frames", action="store_true",
                        help="Run YOLO on every frame (no motion gate) for comparison")
    parser.add_argument("--compare", action="store_true",
                        help="Run BOTH gated and ungated, then print comparison")
    args = parser.parse_args()

    config = load_config(CONFIG_PATH)

    # Determine video source
    if args.source:
        source = args.source
        if not os.path.isabs(source):
            source = os.path.join(PROJECT_ROOT, source)
    elif config:
        raw = config["video"]["source"]
        source = raw if (isinstance(raw, int) or os.path.isabs(str(raw))) else os.path.join(PROJECT_ROOT, raw)
    else:
        source = os.path.join(PROJECT_ROOT, "assets", "videos", "demo.mp4")

    if args.compare:
        print("\n>>> Running MOTION-GATED benchmark...")
        gated = run_benchmark(source, config, disable_gate=False)
        print_summary(gated)
        save_results(gated, "motion_gated")

        print("\n>>> Running ALL-FRAMES (no gate) benchmark...")
        ungated = run_benchmark(source, config, disable_gate=True)
        print_summary(ungated)
        save_results(ungated, "all_frames")

        if gated and ungated:
            save_comparison(gated, ungated)
    elif args.all_frames:
        results = run_benchmark(source, config, disable_gate=True)
        print_summary(results)
        save_results(results, "all_frames")
    else:
        results = run_benchmark(source, config, disable_gate=False)
        print_summary(results)
        save_results(results, "motion_gated")


if __name__ == "__main__":
    main()
