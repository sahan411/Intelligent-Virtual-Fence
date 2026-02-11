# Intelligent Virtual Fence

A computer vision-based surveillance system that detects intrusions into user-defined restricted zones using YOLOv8 object detection with an efficiency-optimized architecture.

## Project Architecture

```
Cheap → Smart → Decision
```

| Stage | Module | Purpose |
|-------|--------|---------|
| Cheap | Motion Gate | Filter frames with no activity (saves ~80% compute) |
| Smart | YOLOv8 Detector | Run AI detection only when motion detected |
| Decision | Foot-Point Logic | Determine if person is inside restricted zone |

## Features

- **User-defined ROI**: Draw custom polygon zones interactively
- **Foot-point intrusion detection**: Ground-level spatial reasoning (not bbox center)
- **Motion-gated YOLO**: Efficient - only runs detection when needed
- **Low-light enhancement**: Automatic CLAHE when scene is dark
- **Real-time visualization**: Green (safe) / Red (intrusion) color coding
- **Intrusion logging**: Timestamped audit trail
- **Auto-screenshot**: Captures evidence on intrusion
- **Live controls**: Pause, adjust sensitivity, toggle debug view
- **Real-time FPS display**: Monitor system performance
- **Intrusion duration timer**: Shows how long person has been in zone
- **Sound alert**: Beep notification on intrusion (Windows)

## Requirements

- Python 3.10+
- Webcam or video file
- Windows/Linux/Mac

## Installation

1. **Clone or download the project**

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv .venv
   
   # Windows
   .venv\Scripts\activate
   
   # Linux/Mac
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download YOLOv8 model** (auto-downloads on first run)
   - The system uses `yolov8n.pt` (nano model)
   - It will download automatically when you first run the program

## Usage

### Basic Run
```bash
cd src
python main.py
```

### First Run - ROI Setup
1. When prompted, draw your restricted zone by clicking points
2. Press **ENTER** to finish the polygon
3. Type `y` to save the ROI for future runs

### Keyboard Controls (During Playback)

| Key | Action |
|-----|--------|
| `q` | Quit |
| `SPACE` | Pause / Resume |
| `d` | Toggle debug window (motion mask) |
| `+` or `=` | Increase motion sensitivity |
| `-` | Decrease motion sensitivity |
| `s` | Take manual screenshot |

### ROI Drawing Controls

| Key | Action |
|-----|--------|
| Left Click | Add point |
| Right Click | Undo last point |
| `ENTER` | Finish drawing |
| `R` | Reset all points |
| `S` | Save ROI |
| `L` | Load saved ROI |
| `Q` | Quit |

## Configuration

All settings are in `configs/config.json`:

```json
{
    "video": {
        "source": "assets/videos/demo.mp4",  // or 0 for webcam
        "width": 640,
        "height": 360
    },
    "motion_gate": {
        "threshold": 500  // Lower = more sensitive
    },
    "detector": {
        "confidence_threshold": 0.4
    },
    "logging": {
        "screenshot_on_intrusion": true
    }
}
```

## Project Structure

```
Intelligent Virtual Fence/
├── README.md
├── requirements.txt
├── configs/
│   ├── config.json          # Main configuration
│   └── roi_config.json      # Saved ROI polygon
├── logs/
│   └── intrusions.log       # Intrusion event log
├── assets/
│   ├── videos/              # Input videos
│   └── screenshots/         # Intrusion captures
└── src/
    ├── main.py              # Entry point
    ├── utils.py             # Logger, screenshot, config loader
    └── core/
        ├── input_manager.py   # Module 1: Video source handling
        ├── roi_manager.py     # Module 2: ROI polygon drawing
        ├── preprocess.py      # Module 3: Frame enhancement
        ├── motion_gate.py     # Module 4: Motion detection gate
        ├── detector.py        # Module 5: YOLOv8 detection
        ├── decision_logic.py  # Module 6: Foot-point intrusion
        └── visualizer.py      # Module 7: Drawing & alerts
```

## Output

### Console Output
```
[Main] Processed 376 frames.
[Main] Average FPS: 28.5
[Main] Enhanced frames: 0 (0.0%)
[Main] Motion triggers: 78 (20.7%)
[Main] YOLO inferences: 78, Detections: 143
[Main] Intrusion detections: 58 (frames where person inside ROI)
[Main] Max intrusion duration: 4.2s
[Main] Total time in zone: 12.8s
[Main] Screenshots saved: 2
[Main] Sound alerts: 3
```

### Log File Sample (`logs/intrusions.log`)
```
[2026-02-11 10:30:45.123] SESSION STARTED
[2026-02-11 10:30:47.456] INTRUSION - Frame 85: 1 person(s) inside ROI
  -> Person at foot-point (320, 280), confidence: 0.87
[2026-02-11 10:31:02.789] SESSION ENDED
```

## Technical Details

### Why Foot-Point Detection?

Traditional methods use bounding box center or overlap. We use **foot-point** (bottom-center of bbox):

> "Foot-point better approximates the physical contact location of a person with the ground, reducing false intrusion decisions when only the upper body enters the ROI."

This is how real surveillance systems work - intrusion happens on the **ground plane**.

### Efficiency Metrics

The motion gate typically filters 70-85% of frames:
- Without gate: 30 FPS × YOLO = 30 inferences/second
- With gate: ~7 inferences/second (only when motion)
- **Savings: ~77% compute reduction**

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Cannot open video source" | Check file path in config.json |
| No display window | Install `opencv-python` (not headless) |
| Video plays too fast | Increase `playback_delay_ms` in config |
| Too many false triggers | Increase motion `threshold` (try 800-1000) |
| Missing detections | Lower `confidence_threshold` (try 0.3) |

## License

Educational project - free to use and modify.

## Author

Computer Vision Surveillance System - UG Final Year Project
