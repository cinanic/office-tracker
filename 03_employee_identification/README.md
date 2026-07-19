# Stage 3 — Multi-Camera Tracking + Employee Identification

The final stage: same 2-camera cross-view tracking as stage 2, plus matching
each tracked face against a gallery of known employees, so tracks are
labeled with real names instead of anonymous IDs (falling back to
`"Unknown"` for anyone not enrolled).

## Files

- `enroll_employees.py` — builds `known_employees.pkl` from a folder of
  employee **photos** (one subfolder per person)
- `enroll_employees_v2.py` — same, but also accepts short **video clips**
  per employee (recommended — a 10–20s clip of someone turning their head
  gives much better angle/lighting coverage than a handful of stills, which
  matters a lot for matching against CCTV-quality footage)
- `office_tracker.py` — the tracker itself

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
python office_tracker.py
```

`office_tracker.py` loads `known_employees.pkl` on startup and labels any
track whose face embedding is close enough to a known employee
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
