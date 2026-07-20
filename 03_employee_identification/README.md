# Stage 3 — Multi-Camera Tracking + Employee Identification

The final stage: same 2-camera cross-view tracking as stage 2, plus matching
each tracked face against a gallery of known employees, so tracks are
labeled with real names instead of anonymous IDs (falling back to
`"Unknown"` for anyone not enrolled).

As with stage 2, two detector variants are kept side by side:

- `office_tracker_yolo.py` — YOLO detector, standard matching thresholds
- `office_tracker_rcnn.py` — Faster R-CNN detector, noticeably stricter
  matching thresholds (e.g. `FACE_MATCH_THRESHOLD` 0.70 vs. 0.65,
  `BODY_MATCH_THRESHOLD` 0.80 vs. 0.70) — trades some recall for fewer
  false identity merges, at the cost of Faster R-CNN being heavier
  per-frame than YOLO

Both share the same employee-matching logic and threshold
(`KNOWN_EMPLOYEE_MATCH_THRESHOLD = 0.50`) for deciding whether a face
belongs to a known employee — the difference is only in how strictly they
track/re-link people across cameras before that point.

## Files

- `enroll_employees.py` — builds `known_employees.pkl` from a folder of
  employee **photos** (one subfolder per person)
- `enroll_employees_v2.py` — same, but also accepts short **video clips**
  per employee (recommended — a 10–20s clip of someone turning their head
  gives much better angle/lighting coverage than a handful of stills, which
  matters a lot for matching against CCTV-quality footage)
- `office_tracker_yolo.py` / `office_tracker_rcnn.py` — the tracker itself
  (pick one, per above)

## Setup

### 1. Collect reference photos/videos

```
employee_photos/
    John_Doe/
        1.jpg
        clip.mp4
    Jane_Smith/
        turn_head.mov
```

### 2. Build the employee gallery

```bash
python enroll_employees_v2.py   # or enroll_employees.py for photos only
```

This produces `known_employees.pkl` (not tracked in git — see `.gitignore`;
it contains face embeddings derived from personal photos).

### 3. Run the tracker

```bash
python office_tracker_yolo.py
# or, for the stricter Faster R-CNN variant:
python office_tracker_rcnn.py
```

Both load `known_employees.pkl` on startup and label any track whose face
embedding is close enough to a known employee
(`KNOWN_EMPLOYEE_MATCH_THRESHOLD`, default `0.50`) with that person's name;
everyone else is labeled `"Unknown"`.

Camera sources are set via `config.yaml` at the repo root (`gating_camera`
for the entrance camera, `perimeter_cameras` for the rest — see
`config.example.yaml` for the template) or default to local webcams if no
config is found.

## Privacy note

This stage processes and stores biometric data (face embeddings) tied to
named individuals. `employee_photos/`, `known_employees.pkl`, and any
recorded footage are excluded from version control by `.gitignore` — treat
them as sensitive data in your own storage/handling as well.