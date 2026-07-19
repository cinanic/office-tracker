"""
Enrollment script: builds known_employees.pkl from a folder of employee photos.

Expected folder layout:
    employee_photos/
        John_Doe/
            1.jpg
            2.jpg
        Jane_Smith/
            1.jpg
            2.jpg
            3.jpg

Run this once (and re-run whenever you add/remove employees or want to add
more reference photos for someone who keeps getting misrecognized).
"""

import os
import pickle
import numpy as np
import cv2
from insightface.app import FaceAnalysis

PHOTOS_DIR = "employee_photos"
OUTPUT_PATH = "known_employees.pkl"

MIN_DET_SCORE = 0.5


def build_gallery():
    face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    face_app.prepare(ctx_id=0, det_size=(640, 640))  # bigger det_size for still photos

    known = {}  # name -> list of embeddings (kept as a list, not averaged)

    for person_name in sorted(os.listdir(PHOTOS_DIR)):
        person_dir = os.path.join(PHOTOS_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue

        embeddings = []
        for fname in sorted(os.listdir(person_dir)):
            path = os.path.join(person_dir, fname)
            img = cv2.imread(path)
            if img is None:
                print(f"⚠️ Could not read {path}")
                continue

            faces = face_app.get(img)
            if not faces:
                print(f"⚠️ No face found in {path}")
                continue

            best = max(faces, key=lambda f: f.det_score)
            if best.det_score < MIN_DET_SCORE:
                print(f"⚠️ Low quality face in {path} ({best.det_score:.2f}), skipping")
                continue

            embeddings.append(best.normed_embedding)
            print(f"✅ {person_name}: enrolled {fname} (det_score={best.det_score:.2f})")

        if embeddings:
            known[person_name] = embeddings
        else:
            print(f"❌ No usable photos for {person_name} — not enrolled")

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(known, f)

    print(f"\nSaved {len(known)} employees to {OUTPUT_PATH}")
    for name, embs in known.items():
        print(f"  - {name}: {len(embs)} reference embeddings")


if __name__ == "__main__":
    build_gallery()
