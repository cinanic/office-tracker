# Office Person Tracker

A person detection, tracking, and re-identification pipeline for office
environments, built up in three stages: single-camera tracking, multi-camera
cross-view re-identification, and finally named employee identification via
face recognition.

## Project stages

| Folder | Stage | Input | Output |
|---|---|---|---|
| [`01_video_only/`](01_video_only) | Detection & tracking on video files | 1 or more video files | Anonymous track IDs |
| [`02_multi_camera/`](02_multi_camera) | Cross-camera identity re-linking (YOLO vs. Faster R-CNN detector variants) | 2 live/RTSP cameras | Anonymous track IDs (same person, same ID, across both cameras) |
| [`03_employee_identification/`](03_employee_identification) | Cross-camera tracking + face-based employee recognition | 2 live/RTSP cameras + employee reference photos/videos | The employee's actual name (or `"Unknown"`) |

Each stage builds on the previous one. See the README in each folder for
details on that stage's approach and how to run it.

## Core techniques used across stages

- **Detection**: YOLO (Ultralytics) and/or Faster R-CNN (torchvision), depending on stage/variant
- **Tracking**: ByteTrack / BoT-SORT
- **Body re-identification**: OSNet (torchreid)
- **Face recognition**: InsightFace (buffalo_l / ArcFace embeddings)
- **Cross-camera fusion**: weighted combination of body appearance, face embedding, and clothing color histograms

## Setup

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

You will also need:

- YOLO weights (e.g. `yolov8m.pt` / `yolo26n.pt`, auto-downloaded by
  Ultralytics on first run)
- For stages 2 & 3 (RTSP camera setups): copy `config.example.yaml` to
  `config.yaml` and fill in your own camera URLs/credentials —
  `config.yaml` is gitignored so real credentials never get committed
- For stage 3 only: a folder of employee reference photos/videos (not
  included in this repo — see
  [`03_employee_identification/README.md`](03_employee_identification/README.md))

## Repo contents note

Model weights, video footage, employee photos, and generated `.pkl` gallery
files are **not** tracked in this repo (see `.gitignore`) since they're large
and/or contain personal data. Point each script's config constants at your
own local paths.
