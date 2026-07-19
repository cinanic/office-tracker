# Stage 2 — Multi-Camera Tracking (2 cameras, no employee names)

Extends stage 1 to two camera feeds and adds cross-camera identity
re-linking: the same person walking from camera 1's view into camera 2's
view keeps the same track ID, based on fused body + face + clothing-color
similarity.

Two detector variants are kept side by side so their tracking quality could
be compared directly:

- `multi_camera_tracker_yolo.py` — YOLO (Ultralytics) detector
- `multi_camera_tracker_rcnn.py` — Faster R-CNN (torchvision) detector

Both share the same tracking / re-ID / fusion logic; only the person
detector differs.

## What it does

- Runs one detector + tracker per camera (threaded)
- Extracts body (OSNet) and face (InsightFace) embeddings per track
- Fuses body + face + clothing-color similarity to decide whether a track in
  one camera is the same person as a track in another camera, with stricter
  thresholds for cross-camera matches than same-camera matches
- Tracks are anonymous (`ID 1`, `ID 2`, ...) — no names yet, see stage 3 for that

## Configuring

Camera sources default to local webcams (`0`, `0`) for quick testing; point
them at your own video files or RTSP streams by editing the `camera_sources`
/ config section near the bottom of each script.

## Running

```bash
python multi_camera_tracker_yolo.py
# or
python multi_camera_tracker_rcnn.py
```

Output goes to `outputs/`.
