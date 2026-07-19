# Stage 1 — Single-Camera Tracking

`single_camera_tracker.py` (originally `v15.py`) is the earliest version of
the pipeline: detection + tracking on a single video feed, with the body
re-ID / face embedding / fused-matching scaffolding already in place so it
could later be extended to multiple cameras (see stage 2).

## What it does

- Detects and tracks people frame-by-frame using YOLO
- Extracts a body appearance embedding per track (OSNet)
- Extracts a face embedding per track (InsightFace), used for re-identifying
  the same person after they leave and re-enter frame — not for naming them
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
python single_camera_tracker.py
```

Output goes to `outputs/`.
