# Stage 1 — Video Tracking (single source, no employee names)

`video_tracker.py` is the earliest version of the
pipeline. It runs on a ** video files**, not a live camera — the
`CAMERA_SOURCES` dict supports multiple entries, but only one video was
active (`cam2` was commented out) since this was the initial single-source
prototype. The body re-ID / face embedding / fused-matching scaffolding was
already in place here so it could be extended to real multi-camera setups
later (see stage 2).

## What it does

- Detects and tracks people frame-by-frame in a video file using YOLO
- Extracts a body appearance embedding per track (OSNet)
- Extracts a face embedding per track (InsightFace), used only for
  re-identifying the same person after they leave and re-enter frame — not
  for naming them
- Persists a gallery of known tracks across a run (`gallery_state.pkl`) so
  IDs are stable within a session

## Configuring

Edit the constants near the top of the file:

```python
CAMERA_SOURCES = {
    "cam1": "your_video_file.mp4",
}
MODEL_PATH = "yolov8m.pt"
```

## Running

```bash
python video_tracker.py
```

Output goes to `outputs/`.
