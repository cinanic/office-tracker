"""
Enrollment script (v2): builds known_employees.pkl from a folder of employee
photos AND/OR short videos.

Expected folder layout (mix and match freely):
    employee_photos/
        John_Doe/
            1.jpg
            clip.mp4
        Jane_Smith/
            turn_head.mov
            walk_in.mp4

For video, aim for a 10-20 second clip where the person:
  - slowly turns their head left/right (profile -> frontal -> profile)
  - looks slightly up/down
  - optionally walks toward/away from the camera a bit

This matters more than image *count* — the goal is angle/lighting coverage
that resembles what a CCTV camera will actually see, not a stack of sharp
frontal headshots.

Run this once (and re-run whenever you add/remove employees, or want to add
more footage for someone who keeps getting misrecognized).
"""

import os
import pickle
import numpy as np
import cv2
from insightface.app import FaceAnalysis

PHOTOS_DIR = "employee_photos"
OUTPUT_PATH = "known_employees.pkl"

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm")

MIN_DET_SCORE = 0.5
MIN_FACE_SIZE = 40           # pixels, in the still/video frame (not upscaled CCTV crop)

# --- video sampling controls ---
SAMPLE_EVERY_N_FRAMES = 5    # check every Nth frame for a usable face
MAX_EMBEDDINGS_PER_PERSON = 40   # cap so matching stays fast and file size sane
DEDUP_SIMILARITY_THRESHOLD = 0.90  # skip a new embedding if this similar to one we kept
                                   # (keeps diversity: near-duplicate poses get skipped)


def cosine_sim(a, b):
    a, b = np.asarray(a).flatten(), np.asarray(b).flatten()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def is_diverse_enough(new_emb, kept_embs, threshold=DEDUP_SIMILARITY_THRESHOLD):
    """Only keep new_emb if it's not a near-duplicate of something we already have."""
    if not kept_embs:
        return True
    return max(cosine_sim(new_emb, e) for e in kept_embs) < threshold


def extract_face_embedding(face_app, img):
    faces = face_app.get(img)
    if not faces:
        return None, 0.0

    best = max(faces, key=lambda f: f.det_score)
    face_w = best.bbox[2] - best.bbox[0]
    face_h = best.bbox[3] - best.bbox[1]

    if face_w < MIN_FACE_SIZE or face_h < MIN_FACE_SIZE:
        return None, 0.0
    if best.det_score < MIN_DET_SCORE:
        return None, 0.0

    return best.normed_embedding, float(best.det_score)


def process_image(face_app, path, kept_embs):
    img = cv2.imread(path)
    if img is None:
        print(f"⚠️ Could not read {path}")
        return 0

    emb, score = extract_face_embedding(face_app, img)
    if emb is None:
        print(f"⚠️ No usable face in {os.path.basename(path)}")
        return 0

    if not is_diverse_enough(emb, kept_embs):
        return 0  # near-duplicate of something already kept, skip

    kept_embs.append(emb)
    print(f"✅ enrolled {os.path.basename(path)} (det_score={score:.2f})")
    return 1


def process_video(face_app, path, kept_embs, max_total):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"⚠️ Could not open video {path}")
        return 0

    added = 0
    frame_idx = 0
    checked = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % SAMPLE_EVERY_N_FRAMES != 0:
            continue

        if len(kept_embs) >= max_total:
            break

        checked += 1
        emb, score = extract_face_embedding(face_app, frame)
        if emb is None:
            continue

        if is_diverse_enough(emb, kept_embs):
            kept_embs.append(emb)
            added += 1

    cap.release()
    print(f"✅ {os.path.basename(path)}: sampled {checked} frames, "
          f"added {added} diverse embeddings")
    return added


def build_gallery():
    face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    face_app.prepare(ctx_id=0, det_size=(640, 640))

    known = {}  # name -> list of embeddings

    for person_name in sorted(os.listdir(PHOTOS_DIR)):
        person_dir = os.path.join(PHOTOS_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue

        print(f"\n--- Enrolling {person_name} ---")
        kept_embs = []

        for fname in sorted(os.listdir(person_dir)):
            if len(kept_embs) >= MAX_EMBEDDINGS_PER_PERSON:
                print(f"   reached cap of {MAX_EMBEDDINGS_PER_PERSON} embeddings, stopping")
                break

            path = os.path.join(person_dir, fname)
            ext = os.path.splitext(fname)[1].lower()

            if ext in IMAGE_EXTS:
                process_image(face_app, path, kept_embs)
            elif ext in VIDEO_EXTS:
                process_video(face_app, path, kept_embs, MAX_EMBEDDINGS_PER_PERSON)
            else:
                continue  # skip unrelated files

        if kept_embs:
            known[person_name] = kept_embs
            print(f"=> {person_name}: {len(kept_embs)} total reference embeddings")
        else:
            print(f"❌ No usable photos/video for {person_name} — not enrolled")

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(known, f)

    print(f"\nSaved {len(known)} employees to {OUTPUT_PATH}")
    for name, embs in known.items():
        print(f"  - {name}: {len(embs)} reference embeddings")


if __name__ == "__main__":
    build_gallery()
