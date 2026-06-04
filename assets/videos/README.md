# Demo Video Test Set

`demo.mp4` is the current sample clip.

For final evaluation, add real clips with these filenames:

| Filename | Scenario |
|----------|----------|
| `day_demo.mp4` | Daylight intrusion |
| `night_demo.mp4` | Low-light intrusion |
| `indoor_demo.mp4` | Indoor/corridor intrusion |
| `outdoor_demo.mp4` | Outdoor intrusion with background movement |
| `false_alarm_demo.mp4` | Harmless motion outside ROI or non-target motion |

Run any clip with:

```bash
python src/main.py --source assets/videos/day_demo.mp4
```
