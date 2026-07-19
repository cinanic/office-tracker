import cv2
import torch
import numpy as np
import yaml
import os
import csv
import pickle
import threading
from collections import deque
from ultralytics import YOLO
import torchreid
from insightface.app import FaceAnalysis

# ============================================================
# CONFIG
# ============================================================
CAMERA_SOURCES = {
    "cam1": "videoplayback (5).mp4",
    #"cam2": "videoplayback (5).mp4",
    # add more cameras here
}
OUTPUT_DIR = "outputs"
MODEL_PATH = "yolov8m.pt"
CONF_THRESHOLD = 0.4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FACE_MATCH_THRESHOLD = 0.35
MIN_FACE_SIZE = 30
MIN_FACE_DET_SCORE = 0.65

GALLERY_MAX_EMB_PER_ID = 10
GALLERY_STALE_SECONDS = 300

FUSED_MATCH_THRESHOLD = 0.35
FUSED_MATCH_MARGIN = 0.10
FUSED_WEIGHTS = {"body": 0.55, "color": 0.45, "height": 0.0}

# NEW: per-signal veto thresholds. If a candidate clearly fails on EITHER
# signal individually, reject the match even if the combined fused score
# looked acceptable. This stops a strong body match from "covering for"
# a clearly different colored outfit, and vice versa.
COLOR_VETO_DISTANCE = 0.35
BODY_VETO_DISTANCE = 0.55

CAMERA_TIME_OFFSETS = {
    "cam1": 0.0,
    "cam2": 0.0,
}

SESSION_GAP_THRESHOLD = 5.0
DEBUG_MATCHING = True

DECISION_BUFFER_SIZE = 15
MIN_MARGIN = 0.12

GALLERY_STATE_PATH = "gallery_state.pkl"
LOAD_PREVIOUS_GALLERY = True
SAVE_GALLERY_ON_EXIT = True

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# BODY REID MODEL (OSNet)
# ============================================================
print("Loading body ReID model...")
reid_model = torchreid.models.build_model(
    name="osnet_x1_0",
    num_classes=1000,
    pretrained=True
)
reid_model.eval().to(DEVICE)
REID_INPUT_SIZE = (256, 128)

def extract_body_embedding(crop_bgr):
    if crop_bgr.size == 0:
        return None
    img = cv2.resize(crop_bgr, (REID_INPUT_SIZE[1], REID_INPUT_SIZE[0]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
    with torch.no_grad():
        feat = reid_model(tensor)
    feat = feat.cpu().numpy().flatten()
    norm = np.linalg.norm(feat)
    return feat / norm if norm > 0 else feat

# ============================================================
# FACE RECOGNITION MODEL (InsightFace)
# ============================================================
print("Loading face recognition model...")
face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=0, det_size=(320, 320))

def extract_face_embedding(crop_bgr):
    if crop_bgr.size == 0:
        return None
    faces = face_app.get(crop_bgr)
    if not faces:
        return None
    faces.sort(key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]), reverse=True)
    best = faces[0]
    w = best.bbox[2] - best.bbox[0]
    h = best.bbox[3] - best.bbox[1]
    if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
        return None
    if best.det_score < MIN_FACE_DET_SCORE:
        return None
    return best.normed_embedding

def cosine_distance(a, b):
    return 1.0 - float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

# ============================================================
# CLOTHING COLOR HISTOGRAM
# ============================================================
def extract_clothing_color_histogram(crop_bgr):
    if crop_bgr.size == 0:
        return None
    h, w = crop_bgr.shape[:2]
    if h < 20 or w < 10:
        return None

    upper = crop_bgr[int(h*0.12):int(h*0.50), :]
    lower = crop_bgr[int(h*0.55):int(h*0.90), :]

    def region_hist(region):
        if region.size == 0:
            return np.zeros(32)
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256])
        hist = np.concatenate([hist_h, hist_s]).flatten()
        norm = np.linalg.norm(hist)
        return hist / norm if norm > 0 else hist

    upper_hist = region_hist(upper)
    lower_hist = region_hist(lower)
    return np.concatenate([upper_hist, lower_hist])

def color_distance(a, b):
    if a is None or b is None:
        return None
    return 1.0 - float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

def compute_height_ratio(box_height, frame_height):
    if frame_height <= 0:
        return None
    return box_height / frame_height

# ============================================================
# GLOBAL IDENTITY GALLERY
# ============================================================
class GlobalGallery:
    def __init__(self):
        self.lock = threading.Lock()
        self.next_id = 1
        self.face_gallery = {}
        self.body_gallery = {}
        self.color_gallery = {}
        self.height_gallery = {}
        self.last_seen = {}
        self.track_state = {}

    def load_state(self, path):
        if not os.path.exists(path):
            print(f"No previous gallery state found at {path}, starting fresh.")
            return
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.next_id = data["next_id"]
        self.face_gallery = {gid: deque(embs, maxlen=GALLERY_MAX_EMB_PER_ID)
                              for gid, embs in data["face_gallery"].items()}
        self.body_gallery = {gid: deque(embs, maxlen=GALLERY_MAX_EMB_PER_ID)
                              for gid, embs in data["body_gallery"].items()}
        self.color_gallery = {gid: deque(embs, maxlen=GALLERY_MAX_EMB_PER_ID)
                               for gid, embs in data.get("color_gallery", {}).items()}
        self.height_gallery = {gid: deque(embs, maxlen=GALLERY_MAX_EMB_PER_ID)
                                for gid, embs in data.get("height_gallery", {}).items()}
        self.last_seen = data.get("last_seen", {})
        print(f"Loaded previous gallery state from {path}: "
              f"{len(self.face_gallery)} known identities, next_id={self.next_id}")

    def save_state(self, path):
        with self.lock:
            data = {
                "next_id": self.next_id,
                "face_gallery": {gid: list(embs) for gid, embs in self.face_gallery.items()},
                "body_gallery": {gid: list(embs) for gid, embs in self.body_gallery.items()},
                "color_gallery": {gid: list(embs) for gid, embs in self.color_gallery.items()},
                "height_gallery": {gid: list(embs) for gid, embs in self.height_gallery.items()},
                "last_seen": self.last_seen,
            }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"Saved gallery state to {path}: {len(data['face_gallery'])} identities, next_id={data['next_id']}")

    def _best_two(self, embedding, gallery_dict, exclude_ids):
        scored = []
        for gid, emb_list in gallery_dict.items():
            if gid in exclude_ids:
                continue
            for emb in emb_list:
                scored.append((gid, cosine_distance(embedding, emb)))
        if not scored:
            return None, 999, 999
        scored.sort(key=lambda x: x[1])
        best_id, best_dist = scored[0]
        second_dist = next((d for gid, d in scored[1:] if gid != best_id), 999)
        return best_id, best_dist, second_dist

    def _confident_match(self, embedding, gallery_dict, threshold, margin_required, exclude_ids, label):
        best_id, best_dist, second_dist = self._best_two(embedding, gallery_dict, exclude_ids)
        if best_id is None:
            return None
        margin = second_dist - best_dist
        if DEBUG_MATCHING:
            print(f"    [{label}] best=ID{best_id} d={best_dist:.3f} second_d={second_dist:.3f} "
                  f"margin={margin:.3f} (need d<{threshold}, margin>={margin_required})")
        if best_dist < threshold and margin >= margin_required:
            return best_id
        return None

    def _signal_min_dist(self, gid, embedding, gallery_dict, dist_fn):
        """Min distance from embedding to any stored embedding for gid. None if no data."""
        if gid not in gallery_dict or len(gallery_dict[gid]) == 0 or embedding is None:
            return None
        dists = [dist_fn(embedding, e) for e in gallery_dict[gid]]
        dists = [d for d in dists if d is not None]
        return min(dists) if dists else None

    def _fused_score(self, gid, body_emb, color_hist, height_ratio):
        parts, weights = [], []

        body_d = self._signal_min_dist(gid, body_emb, self.body_gallery, cosine_distance)
        if body_d is not None:
            parts.append(body_d); weights.append(FUSED_WEIGHTS["body"])

        color_d = self._signal_min_dist(gid, color_hist, self.color_gallery, color_distance)
        if color_d is not None:
            parts.append(color_d); weights.append(FUSED_WEIGHTS["color"])

        if FUSED_WEIGHTS["height"] > 0 and height_ratio is not None and \
           gid in self.height_gallery and len(self.height_gallery[gid]) > 0:
            avg_h = float(np.mean(self.height_gallery[gid]))
            d = min(abs(height_ratio - avg_h) / max(avg_h, 0.01), 1.0)
            parts.append(d); weights.append(FUSED_WEIGHTS["height"])

        if not parts:
            return None, None, None
        total_weight = sum(weights)
        fused = sum(p * w for p, w in zip(parts, weights)) / total_weight
        return fused, body_d, color_d

    def _match_fused(self, body_emb, color_hist, height_ratio, exclude_ids, threshold, margin_required, label):
        candidate_ids = set(self.body_gallery.keys()) | set(self.color_gallery.keys())
        scored = []  # (gid, fused_score, body_d, color_d)
        for gid in candidate_ids:
            if gid in exclude_ids:
                continue
            fused, body_d, color_d = self._fused_score(gid, body_emb, color_hist, height_ratio)
            if fused is not None:
                scored.append((gid, fused, body_d, color_d))

        if not scored:
            return None
        scored.sort(key=lambda x: x[1])
        best_id, best_score, best_body_d, best_color_d = scored[0]
        second_score = scored[1][1] if len(scored) > 1 else 999
        margin = second_score - best_score

        if DEBUG_MATCHING:
            print(f"    [{label}] best=ID{best_id} score={best_score:.3f} second={second_score:.3f} "
                  f"margin={margin:.3f} body_d={best_body_d} color_d={best_color_d}")

        # VETO: if either individual signal is a clear mismatch, reject
        # regardless of how good the combined fused score looks.
        if best_color_d is not None and best_color_d > COLOR_VETO_DISTANCE:
            if DEBUG_MATCHING:
                print(f"    [{label}] VETOED -- color_d={best_color_d:.3f} exceeds veto threshold {COLOR_VETO_DISTANCE}")
            return None
        if best_body_d is not None and best_body_d > BODY_VETO_DISTANCE:
            if DEBUG_MATCHING:
                print(f"    [{label}] VETOED -- body_d={best_body_d:.3f} exceeds veto threshold {BODY_VETO_DISTANCE}")
            return None

        if best_score < threshold and margin >= margin_required:
            return best_id
        return None

    def _create_new_id(self):
        gid = self.next_id
        self.next_id += 1
        self.face_gallery[gid] = deque(maxlen=GALLERY_MAX_EMB_PER_ID)
        self.body_gallery[gid] = deque(maxlen=GALLERY_MAX_EMB_PER_ID)
        self.color_gallery[gid] = deque(maxlen=GALLERY_MAX_EMB_PER_ID)
        self.height_gallery[gid] = deque(maxlen=GALLERY_MAX_EMB_PER_ID)
        return gid

    def process_detection(self, cam_name, local_id, face_emb, body_emb, color_hist, height_ratio,
                           timestamp, used_ids_this_frame):
        key = (cam_name, local_id)

        with self.lock:
            state = self.track_state.get(key)
            if state is None:
                state = {
                    "global_id": None,
                    "pending_face": [], "pending_body": [],
                    "pending_color": [], "pending_height": [],
                }
                self.track_state[key] = state

            if state["global_id"] is not None:
                gid = state["global_id"]

                if gid in used_ids_this_frame:
                    new_gid = self._create_new_id()
                    if DEBUG_MATCHING:
                        print(f"  [COLLISION] track {key} was ID{gid}, already used this frame "
                              f"-> reassigned to new ID{new_gid}")
                    state["global_id"] = new_gid
                    gid = new_gid

                if face_emb is not None:
                    self.face_gallery[gid].append(face_emb)
                if body_emb is not None:
                    self.body_gallery[gid].append(body_emb)
                if color_hist is not None:
                    self.color_gallery[gid].append(color_hist)
                if height_ratio is not None:
                    self.height_gallery[gid].append(height_ratio)

                self.last_seen[gid] = timestamp
                used_ids_this_frame.add(gid)
                return gid

            if face_emb is not None:
                state["pending_face"].append(face_emb)
            if body_emb is not None:
                state["pending_body"].append(body_emb)
            if color_hist is not None:
                state["pending_color"].append(color_hist)
            if height_ratio is not None:
                state["pending_height"].append(height_ratio)

            total_evidence = len(state["pending_face"]) + len(state["pending_body"])

            if face_emb is not None:
                gid = self._confident_match(face_emb, self.face_gallery, FACE_MATCH_THRESHOLD,
                                             MIN_MARGIN, used_ids_this_frame, "face-early")
                if gid is not None:
                    state["global_id"] = gid
                    self.face_gallery[gid].append(face_emb)
                    if body_emb is not None:
                        self.body_gallery[gid].append(body_emb)
                    if color_hist is not None:
                        self.color_gallery[gid].append(color_hist)
                    if height_ratio is not None:
                        self.height_gallery[gid].append(height_ratio)
                    self.last_seen[gid] = timestamp
                    used_ids_this_frame.add(gid)
                    return gid

            if total_evidence < DECISION_BUFFER_SIZE:
                return -1 * ((hash(key) % 100000) + 1)

            avg_face = None
            if state["pending_face"]:
                avg_face = np.mean(state["pending_face"], axis=0)
                avg_face = avg_face / (np.linalg.norm(avg_face) + 1e-8)

            avg_body = None
            if state["pending_body"]:
                avg_body = np.mean(state["pending_body"], axis=0)
                avg_body = avg_body / (np.linalg.norm(avg_body) + 1e-8)

            avg_color = None
            if state["pending_color"]:
                avg_color = np.mean(state["pending_color"], axis=0)
                norm = np.linalg.norm(avg_color)
                avg_color = avg_color / norm if norm > 0 else avg_color

            avg_height = float(np.mean(state["pending_height"])) if state["pending_height"] else None

            gid = None
            if avg_face is not None:
                gid = self._confident_match(avg_face, self.face_gallery, FACE_MATCH_THRESHOLD,
                                             MIN_MARGIN, used_ids_this_frame, "face-final")

            if gid is None:
                gid = self._match_fused(avg_body, avg_color, avg_height, used_ids_this_frame,
                                         threshold=FUSED_MATCH_THRESHOLD,
                                         margin_required=FUSED_MATCH_MARGIN,
                                         label="fused-final")

            if gid is None:
                gid = self._create_new_id()

            for f in state["pending_face"]:
                self.face_gallery[gid].append(f)
            for b in state["pending_body"]:
                self.body_gallery[gid].append(b)
            for c in state["pending_color"]:
                self.color_gallery[gid].append(c)
            for h in state["pending_height"]:
                self.height_gallery[gid].append(h)

            state["global_id"] = gid
            state["pending_face"] = []
            state["pending_body"] = []
            state["pending_color"] = []
            state["pending_height"] = []
            self.last_seen[gid] = timestamp
            used_ids_this_frame.add(gid)
            return gid

    def prune_stale(self, now):
        if GALLERY_STALE_SECONDS is None:
            return
        with self.lock:
            stale = [gid for gid, t in self.last_seen.items() if now - t > GALLERY_STALE_SECONDS]
            for gid in stale:
                self.face_gallery.pop(gid, None)
                self.body_gallery.pop(gid, None)
                self.color_gallery.pop(gid, None)
                self.height_gallery.pop(gid, None)
                self.last_seen.pop(gid, None)
            self.track_state = {
                k: v for k, v in self.track_state.items()
                if v["global_id"] is None or v["global_id"] not in stale
            }

global_gallery = GlobalGallery()
if LOAD_PREVIOUS_GALLERY:
    global_gallery.load_state(GALLERY_STATE_PATH)

# ============================================================
# PRESENCE LOGGER
# ============================================================
class PresenceLogger:
    def __init__(self):
        self.lock = threading.Lock()
        self.sessions = {}

    def log(self, global_id, cam_name, timestamp):
        with self.lock:
            key = (global_id, cam_name)
            if key not in self.sessions:
                self.sessions[key] = []
            sessions_for_key = self.sessions[key]
            if sessions_for_key and (timestamp - sessions_for_key[-1]["last"]) <= SESSION_GAP_THRESHOLD:
                sessions_for_key[-1]["last"] = timestamp
            else:
                sessions_for_key.append({"first": timestamp, "last": timestamp})

    def write_csv(self, detailed_path, summary_path):
        with self.lock:
            with open(detailed_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["global_id", "camera", "session_start_sec", "session_end_sec", "duration_sec"])
                for (gid, cam), sessions in sorted(self.sessions.items()):
                    for s in sessions:
                        duration = s["last"] - s["first"]
                        writer.writerow([gid, cam, round(s["first"], 2), round(s["last"], 2), round(duration, 2)])

            totals, first_overall, last_overall = {}, {}, {}
            for (gid, cam), sessions in self.sessions.items():
                for s in sessions:
                    duration = s["last"] - s["first"]
                    totals[gid] = totals.get(gid, 0) + duration
                    first_overall[gid] = min(first_overall.get(gid, s["first"]), s["first"])
                    last_overall[gid] = max(last_overall.get(gid, s["last"]), s["last"])

            with open(summary_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["global_id", "first_seen_sec", "last_seen_sec", "total_duration_sec"])
                for gid in sorted(totals.keys()):
                    writer.writerow([
                        gid, round(first_overall[gid], 2), round(last_overall[gid], 2), round(totals[gid], 2)
                    ])

presence_logger = PresenceLogger()

# ============================================================
# TRACKER CONFIG
# ============================================================
def make_tracker_yaml(path):
    config = {
        "tracker_type": "botsort",
        "track_high_thresh": 0.4,
        "track_low_thresh": 0.1,
        "new_track_thresh": 0.6,
        "track_buffer": 200,
        "match_thresh": 0.9,
        "fuse_score": True,
        "gmc_method": "sparseOptFlow",
        "proximity_thresh": 0.5,
        "appearance_thresh": 0.25,
        "with_reid": False,
        "model": "auto",
    }
    with open(path, "w") as f:
        yaml.dump(config, f)

def get_color(idx):
    import random
    random.seed(int(idx) * 999)
    return tuple(random.randint(60, 255) for _ in range(3))

# ============================================================
# PER-CAMERA WORKER
# ============================================================
def process_camera(cam_name, video_path):
    print(f"[{cam_name}] starting...")
    tracker_yaml = f"tracker_{cam_name}.yaml"
    make_tracker_yaml(tracker_yaml)

    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = os.path.join(OUTPUT_DIR, f"{cam_name}_tracked.mp4")
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    time_offset = CAMERA_TIME_OFFSETS.get(cam_name, 0.0)
    frame_idx = 0

    results_generator = model.track(
        source=video_path,
        tracker=tracker_yaml,
        persist=True,
        classes=[0],
        conf=CONF_THRESHOLD,
        device=DEVICE,
        stream=True,
        verbose=False
    )

    for result in results_generator:
        frame = result.orig_img.copy()
        timestamp = (frame_idx / fps) + time_offset
        used_ids_this_frame = set()

        boxes = result.boxes
        if boxes is not None and boxes.id is not None:
            ids = boxes.id.cpu().numpy().astype(int)
            xyxy = boxes.xyxy.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()

            if DEBUG_MATCHING and len(ids) > 1:
                print(f"[{cam_name}] frame {frame_idx}: {len(ids)} detections")

            for (x1, y1, x2, y2), local_id, conf in zip(xyxy, ids, confs):
                x1c, y1c = max(0, x1), max(0, y1)
                x2c, y2c = min(width, x2), min(height, y2)
                crop = frame[y1c:y2c, x1c:x2c]
                if crop.size == 0:
                    continue

                face_embedding = extract_face_embedding(crop)
                body_embedding = extract_body_embedding(crop)
                color_hist = extract_clothing_color_histogram(crop)
                height_ratio = compute_height_ratio(y2 - y1, height)

                global_id = global_gallery.process_detection(
                    cam_name, local_id, face_embedding, body_embedding,
                    color_hist, height_ratio, timestamp, used_ids_this_frame
                )

                if global_id < 0:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 2)
                    cv2.putText(frame, "ID pending...", (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
                    continue

                presence_logger.log(global_id, cam_name, timestamp)

                color = get_color(global_id)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                tag = "F" if face_embedding is not None else "B"
                label = f"ID {global_id} [{tag}]"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
                cv2.putText(frame, label, (x1 + 3, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        cv2.putText(frame, cam_name, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        out.write(frame)
        frame_idx += 1

        if frame_idx % 50 == 0:
            global_gallery.prune_stale(timestamp)
            print(f"[{cam_name}] processed {frame_idx} frames")

    out.release()
    os.remove(tracker_yaml)
    print(f"[{cam_name}] done -> {out_path}")

# ============================================================
# RUN ALL CAMERAS
# ============================================================
if __name__ == "__main__":
    threads = []
    for cam_name, path in CAMERA_SOURCES.items():
        t = threading.Thread(target=process_camera, args=(cam_name, path))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    detailed_csv = os.path.join(OUTPUT_DIR, "presence_detailed.csv")
    summary_csv = os.path.join(OUTPUT_DIR, "presence_summary.csv")
    presence_logger.write_csv(detailed_csv, summary_csv)

    if SAVE_GALLERY_ON_EXIT:
        global_gallery.save_state(GALLERY_STATE_PATH)

    print("\nAll cameras processed. Outputs in:", OUTPUT_DIR)
    print(f"Detailed presence log: {detailed_csv}")
    print(f"Summary log: {summary_csv}")
