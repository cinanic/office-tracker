import cv2
import torch
import torchvision
import numpy as np
import yaml
import os
import sys
import pickle
import threading
import time
from collections import deque, defaultdict
import torchreid
from insightface.app import FaceAnalysis
from scipy.spatial.distance import cosine
import warnings
warnings.filterwarnings('ignore')

import supervision as sv

# ============================================================
# CONFIGURATION – STRICTEST SETTINGS
# ============================================================
OUTPUT_DIR = "outputs"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 360
BOTTOM_BAR_HEIGHT = 30

# ----- Appearance matching (very strict) -----
FACE_MATCH_THRESHOLD = 0.70
FACE_STRONG_MATCH = 0.85
BODY_MATCH_THRESHOLD = 0.80          # raised
BODY_STRONG_MATCH = 0.90
CROSS_CAMERA_THRESHOLD = 0.70
SAME_CAMERA_BOOST = 1.05

# ----- ID retention -----
ID_RETENTION_SECONDS = 86400 * 7     # 7 days
TRACK_LINK_TIMEOUT = 30              # seconds – short to avoid reuse

# ----- Discrimination -----
MIN_SCORE_GAP = 0.20                 # large gap required

# ----- Confirmation -----
MIN_CONSISTENT_MATCHES = 4           # raised
FORCE_CONFIRM_AFTER_FRAMES = 8
PENDING_EXPIRY_SECONDS = 3.0

# ----- Face detection -----
MIN_FACE_SIZE = 35
MIN_FACE_DET_SCORE = 0.50

# ----- Known employee (named) recognition -----
KNOWN_EMPLOYEES_PATH = "known_employees.pkl"   # built by enroll_employees.py / v2
# Cosine-similarity threshold against enrolled reference face embeddings.
# This is a SEPARATE concern from body-based track identity above — it only
# answers "does this face belong to someone we enrolled." Tune against real
# footage: too high -> employees keep showing "Unknown"; too low -> strangers
# get misnamed.
KNOWN_EMPLOYEE_MATCH_THRESHOLD = 0.50
UNKNOWN_LABEL = "Unknown"

# ----- Gallery -----
GALLERY_MAX_EMB_PER_ID = 50
STABLE_ID_EXPIRY = ID_RETENTION_SECONDS

# ----- Misc -----
DEBUG_MATCHING = True
MIN_DETECTION_CONFIDENCE = 0.30

# ----- PERFORMANCE / LAG CONTROL -----
# Faster R-CNN is much heavier per-frame than YOLO, so decoupling capture from
# processing matters even more here. See FrameGrabber below: it's a dedicated
# thread that ONLY reads frames and always holds just the latest one, so the
# detector/ReID pipeline never blocks or delays the live camera stream.
PROCESS_EVERY_N_CAPTURED_FRAMES = 1  # raise to 2/3 if the detector still can't keep up
REID_EVERY_N_FRAMES_CONFIRMED = 5    # skip re-extracting ReID/face embeddings for already-linked tracks most frames

os.makedirs(OUTPUT_DIR, exist_ok=True)
MODEL_LOCK = threading.Lock()
GALLERY_STATE_PATH = "gallery_state.pkl"

# ============================================================
# DETECTOR – Faster R‑CNN (only people, high confidence)
# ============================================================
class PersonDetector:
    def __init__(self, conf_threshold=0.6):
        print("Loading Faster R‑CNN detector...")
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.eval().to(DEVICE)
        self.conf_threshold = conf_threshold
        self.frame_counter = 0

    @torch.no_grad()
    def detect(self, frame):
        self.frame_counter += 1
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().to(DEVICE) / 255.0
        outputs = self.model([img_tensor])
        boxes = outputs[0]['boxes'].cpu().numpy()
        scores = outputs[0]['scores'].cpu().numpy()
        labels = outputs[0]['labels'].cpu().numpy()

        # Only keep persons (class 1) with high confidence
        valid = (scores >= self.conf_threshold) & (labels == 1)
        xyxy = boxes[valid].astype(np.int32)
        confs = scores[valid]
        cls_ids = labels[valid]   # all 1

        # Debug: print first few detections on frame 1
        if self.frame_counter == 1 and len(confs) > 0:
            print("🔍 First detection stats:")
            for i in range(min(5, len(confs))):
                print(f"   Label: {cls_ids[i]}, Score: {confs[i]:.3f}, Box: {xyxy[i]}")

        return xyxy, confs, cls_ids

# ============================================================
# INFINITE MEMORY GALLERY
# ============================================================
class InfiniteMemoryGallery:
    def __init__(self):
        self.lock = threading.RLock()
        self.next_id = 1
        self.person_count = 0
        self.identities = {}
        self.active_ids = set()
        self.last_seen = {}
        self.first_seen = {}
        self.last_camera = {}
        self.camera_tracks = defaultdict(dict)
        self.track_history = defaultdict(list)
        self.recently_created = {}
        self.track_id_map = {}
        self.embedding_cache = {}
        self.id_reacquisition_count = defaultdict(int)

        # --- known employee (named) recognition ---
        self.known_identities = {}   # name -> list of reference face embeddings
        self.name_by_gid = {}        # global_id -> employee name (once resolved)

        self.load_state(GALLERY_STATE_PATH)
        self.load_known_employees(KNOWN_EMPLOYEES_PATH)

    def load_state(self, path):
        if not os.path.exists(path):
            print("No previous state found, starting fresh")
            return
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.next_id = data.get("next_id", 1)
            self.person_count = data.get("person_count", 0)
            self.identities = data.get("identities", {})
            self.last_seen = data.get("last_seen", {})
            self.first_seen = data.get("first_seen", {})
            self.last_camera = data.get("last_camera", {})
            self.name_by_gid = data.get("name_by_gid", {})
            print(f"✅ Loaded gallery: {len(self.identities)} identities")
        except Exception as e:
            print(f"⚠️ Error loading gallery: {e}")

    def save_state(self, path):
        with self.lock:
            data = {
                "next_id": self.next_id,
                "person_count": self.person_count,
                "identities": self.identities,
                "last_seen": self.last_seen,
                "first_seen": self.first_seen,
                "last_camera": self.last_camera,
                "name_by_gid": self.name_by_gid,
            }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load_known_employees(self, path):
        """Load the enrolled employee face gallery built by enroll_employees.py"""
        if not os.path.exists(path):
            print(f"⚠️ No {path} found — running with no employee gallery "
                  f"(everyone will show as '{UNKNOWN_LABEL}'). Run enroll_employees.py first.")
            return
        try:
            with open(path, "rb") as f:
                self.known_identities = pickle.load(f)
            print(f"✅ Loaded {len(self.known_identities)} known employees from {path}")
            for name, embs in self.known_identities.items():
                print(f"   - {name}: {len(embs)} reference photo(s)/frame(s)")
        except Exception as e:
            print(f"⚠️ Error loading known employees: {e}")

    def match_known_employee(self, face_emb, threshold=KNOWN_EMPLOYEE_MATCH_THRESHOLD):
        """
        Match a live face embedding against enrolled employee reference
        embeddings. Returns (name, score) or (None, 0.0). This is completely
        independent of body-based track identity matching below.
        """
        if face_emb is None or not self.known_identities:
            return None, 0.0

        best_name, best_score = None, 0.0
        for name, ref_embs in self.known_identities.items():
            if not ref_embs:
                continue
            sims = [self._cosine_similarity(face_emb, e) for e in ref_embs]
            score = max(sims)
            if score > best_score:
                best_score, best_name = score, name

        if best_score >= threshold:
            return best_name, best_score
        return None, 0.0

    def assign_name_if_possible(self, gid, face_emb):
        """
        Try to resolve/refresh the employee name attached to a global ID.
        Safe to call whenever a decent face crop is available — once a gid
        has a name, it's only overwritten by a clearly stronger match, so a
        single bad frame can't bump someone back to 'Unknown'.
        """
        if face_emb is None:
            return self.name_by_gid.get(gid)

        name, score = self.match_known_employee(face_emb)
        if name is None:
            return self.name_by_gid.get(gid)

        with self.lock:
            current = self.name_by_gid.get(gid)
            if current is None or current == name:
                self.name_by_gid[gid] = name
            elif score >= 0.65:
                if DEBUG_MATCHING:
                    print(f"⚠️ Re-labeling ID {gid}: {current} -> {name} (score {score:.2f})")
                self.name_by_gid[gid] = name

        return self.name_by_gid.get(gid)

    def get_display_name(self, gid):
        return self.name_by_gid.get(gid, UNKNOWN_LABEL)

    def _cosine_similarity(self, emb1, emb2):
        if emb1 is None or emb2 is None:
            return 0.0
        emb1 = np.array(emb1).flatten()
        emb2 = np.array(emb2).flatten()
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(emb1, emb2) / (norm1 * norm2)

    def _calculate_embedding_variance(self, embeddings):
        if len(embeddings) < 2:
            return 0.0
        emb_array = np.array([e for e in embeddings if e is not None])
        if len(emb_array) < 2:
            return 0.0
        distances = []
        for i in range(len(emb_array)):
            for j in range(i+1, len(emb_array)):
                dist = 1 - self._cosine_similarity(emb_array[i], emb_array[j])
                distances.append(dist)
        return np.mean(distances) if distances else 0.0

    def find_by_body_infinite(self, body_emb, camera_name, exclude_ids=None):
        if body_emb is None:
            return None, 0.0
        exclude_ids = exclude_ids or set()
        now = time.time()
        with self.lock:
            candidates = []
            for gid, identity in self.identities.items():
                if gid in exclude_ids:
                    continue
                if now - self.last_seen.get(gid, 0) > ID_RETENTION_SECONDS:
                    continue
                if 'body_embs' not in identity or not identity['body_embs']:
                    continue
                app_score = 0.0
                for stored_emb in identity['body_embs']:
                    if stored_emb is None:
                        continue
                    score = self._cosine_similarity(body_emb, stored_emb)
                    if score > app_score:
                        app_score = score
                if self.last_camera.get(gid) == camera_name:
                    app_score = min(1.0, app_score * SAME_CAMERA_BOOST)
                if len(identity['body_embs']) > 3:
                    var = self._calculate_embedding_variance(identity['body_embs'])
                    if var > 0.35:
                        app_score *= 0.85
                candidates.append((gid, app_score))
            candidates.sort(key=lambda x: x[1], reverse=True)
            if not candidates:
                return None, 0.0
            top_id, top_score = candidates[0]
            if DEBUG_MATCHING and len(candidates) > 1:
                second_score = candidates[1][1]
                gap = top_score - second_score
                print(f"    Top: {top_score:.3f}, 2nd: {second_score:.3f}, gap: {gap:.3f}")
            if len(candidates) > 1:
                second_score = candidates[1][1]
                if top_score - second_score < MIN_SCORE_GAP:
                    if top_score < BODY_STRONG_MATCH:
                        if DEBUG_MATCHING:
                            print(f"    Rejected: gap too small and top not strong")
                        return None, 0.0
            time_since = now - self.last_seen.get(top_id, now)
            last_cam = self.last_camera.get(top_id)
            is_cross = (last_cam is not None and last_cam != camera_name)

            if time_since < 5.0 and not is_cross:
                min_thresh = BODY_MATCH_THRESHOLD
            elif time_since < 5.0 and is_cross:
                min_thresh = CROSS_CAMERA_THRESHOLD
            elif time_since < 60.0:
                min_thresh = 0.70 if is_cross else 0.75
            elif time_since < 300.0:
                min_thresh = 0.65
            elif time_since < 3600.0:
                min_thresh = 0.60
            else:
                min_thresh = 0.55

            if top_score >= min_thresh:
                if DEBUG_MATCHING:
                    print(f"    ACCEPTED: ID {top_id} score {top_score:.3f} >= {min_thresh:.3f}")
                return top_id, top_score
            else:
                if DEBUG_MATCHING:
                    print(f"    REJECTED: score {top_score:.3f} < {min_thresh:.3f}")
                return None, 0.0

    def find_by_face_infinite(self, face_emb, min_confidence=0.60):
        if face_emb is None:
            return None, 0.0
        now = time.time()
        candidates = []
        with self.lock:
            for gid, identity in self.identities.items():
                if now - self.last_seen.get(gid, 0) > ID_RETENTION_SECONDS:
                    continue
                if 'face_embs' not in identity or not identity['face_embs']:
                    continue
                face_score = 0.0
                for stored_emb in identity['face_embs']:
                    if stored_emb is None:
                        continue
                    score = self._cosine_similarity(face_emb, stored_emb)
                    if score > face_score:
                        face_score = score
                if face_score >= min_confidence:
                    candidates.append((gid, face_score))
            candidates.sort(key=lambda x: x[1], reverse=True)
            if not candidates:
                return None, 0.0
            top_id, top_score = candidates[0]
            if len(candidates) > 1:
                second_score = candidates[1][1]
                if top_score - second_score < 0.15 and top_score < FACE_STRONG_MATCH:
                    return None, 0.0
            return top_id, top_score

    def add_identity_infinite(self, body_emb=None, face_emb=None, camera_name=None, face_quality=0):
        if body_emb is not None:
            for gid, emb in list(self.recently_created.items()):
                if emb is None:
                    continue
                score = self._cosine_similarity(body_emb, emb)
                if score > 0.80:
                    if gid in self.identities:
                        self.update_identity_infinite(gid, body_emb=body_emb, face_emb=face_emb, camera_name=camera_name)
                        return gid
        with self.lock:
            gid = self.next_id
            self.next_id += 1
            self.person_count += 1
            identity = {
                "face_embs": deque(maxlen=GALLERY_MAX_EMB_PER_ID),
                "body_embs": deque(maxlen=GALLERY_MAX_EMB_PER_ID),
                "color_hist": None,
                "height_ratios": deque(maxlen=GALLERY_MAX_EMB_PER_ID),
                "created_at": time.time(),
                "camera_views": [camera_name] if camera_name else [],
                "last_updated": time.time(),
                "detection_count": 1,
                "face_quality": face_quality,
                "primary_identifier": "body" if body_emb is not None else "face"
            }
            if body_emb is not None:
                identity["body_embs"].append(body_emb)
                self.recently_created[gid] = body_emb
            if face_emb is not None:
                identity["face_embs"].append(face_emb)
            self.identities[gid] = identity
            self.active_ids.add(gid)
            self.last_seen[gid] = time.time()
            self.first_seen[gid] = time.time()
            self.last_camera[gid] = camera_name
            if len(self.recently_created) > 50:
                oldest = sorted(self.recently_created.keys())[:25]
                for k in oldest:
                    del self.recently_created[k]

        # Try to resolve this new person against the enrolled employee gallery
        name = self.assign_name_if_possible(gid, face_emb)
        if name and name != UNKNOWN_LABEL:
            print(f"👤 NEW PERSON ID {gid} (cam: {camera_name}) -> recognized as '{name}'")
        else:
            print(f"👤 NEW PERSON ID {gid} (cam: {camera_name}) -> {UNKNOWN_LABEL}")
        return gid

    def update_identity_infinite(self, gid, body_emb=None, face_emb=None,
                                  camera_name=None, face_quality=None):
        with self.lock:
            if gid not in self.identities:
                return False
            identity = self.identities[gid]
            if body_emb is not None:
                identity["body_embs"].append(body_emb)
                self.recently_created[gid] = body_emb
            if face_emb is not None:
                if face_quality is None or face_quality >= 0.30:
                    identity["face_embs"].append(face_emb)
                    # Fresh, decent-quality face crop -> good opportunity to
                    # (re)confirm this person's name against the employee gallery
                    self.assign_name_if_possible(gid, face_emb)
            if camera_name and camera_name not in identity["camera_views"]:
                identity["camera_views"].append(camera_name)
            identity["last_updated"] = time.time()
            identity["detection_count"] += 1
            self.last_seen[gid] = time.time()
            self.last_camera[gid] = camera_name
            self.active_ids.add(gid)
            return True

    def get_track_link(self, cam_name, local_id):
        with self.lock:
            if local_id in self.camera_tracks[cam_name]:
                gid = self.camera_tracks[cam_name][local_id]
                if gid in self.identities:
                    return gid
                else:
                    del self.camera_tracks[cam_name][local_id]
            return None

    def set_track_link(self, cam_name, local_id, global_id, confidence=1.0):
        with self.lock:
            self.camera_tracks[cam_name][local_id] = global_id
            self.track_id_map[local_id] = (global_id, confidence, time.time())

    def get_stable_id_for_track(self, cam_name, local_id):
        with self.lock:
            if local_id in self.track_id_map:
                gid, confidence, last_seen = self.track_id_map[local_id]
                if time.time() - last_seen < TRACK_LINK_TIMEOUT:
                    if gid in self.identities:
                        return gid, confidence
            return None, 0.0

# ============================================================
# INFINITE PENDING IDENTITY
# ============================================================
class InfinitePendingIdentity:
    def __init__(self, temp_id, body_emb=None, face_emb=None, camera_name=None, face_quality=0):
        self.temp_id = temp_id
        self.body_embs = deque(maxlen=20)
        self.face_embs = deque(maxlen=15)
        self.camera_name = camera_name
        self.frame_count = 0
        self.best_match_id = None
        self.best_match_score = 0.0
        self.match_scores = []
        self.created_at = time.time()
        self.last_seen = time.time()
        self.best_body_emb = None
        self.confirmed = False
        self.consistent_matches = 0
        if body_emb is not None:
            self.body_embs.append(body_emb)
            self.best_body_emb = body_emb
        if face_emb is not None:
            self.face_embs.append(face_emb)
        self.frame_count = 1

    def add_observation(self, body_emb=None, face_emb=None,
                       match_id=None, match_score=0):
        if body_emb is not None:
            self.body_embs.append(body_emb)
            self.best_body_emb = body_emb
        if face_emb is not None:
            self.face_embs.append(face_emb)
        self.frame_count += 1
        self.last_seen = time.time()
        if match_id is not None:
            self.match_scores.append(match_score)
            if match_score > self.best_match_score:
                self.best_match_score = match_score
                self.best_match_id = match_id
            if match_score > BODY_MATCH_THRESHOLD:
                self.consistent_matches += 1
            else:
                self.consistent_matches = max(0, self.consistent_matches - 1)

    def is_confirmed(self):
        if self.frame_count >= FORCE_CONFIRM_AFTER_FRAMES:
            self.confirmed = True
            return True
        if self.best_match_score >= 0.85 and self.consistent_matches >= MIN_CONSISTENT_MATCHES:
            self.confirmed = True
            return True
        if self.consistent_matches >= MIN_CONSISTENT_MATCHES:
            self.confirmed = True
            return True
        if len(self.body_embs) >= 4:
            emb_array = np.array([e for e in self.body_embs if e is not None])
            if len(emb_array) >= 4:
                similarities = []
                for i in range(len(emb_array)):
                    for j in range(i+1, len(emb_array)):
                        sim = np.dot(emb_array[i], emb_array[j]) / (np.linalg.norm(emb_array[i]) * np.linalg.norm(emb_array[j]))
                        similarities.append(sim)
                avg_sim = np.mean(similarities) if similarities else 0
                if avg_sim > 0.70 and self.frame_count >= 4:
                    self.confirmed = True
                    return True
        return False

    def get_average_body_embedding(self):
        if not self.body_embs:
            return None
        emb_array = np.array([e for e in self.body_embs if e is not None])
        if len(emb_array) == 0:
            return None
        weights = np.linspace(0.5, 1.0, len(emb_array))
        weights = weights / weights.sum()
        avg_emb = np.average(emb_array, axis=0, weights=weights)
        norm = np.linalg.norm(avg_emb)
        return avg_emb / norm if norm > 0 else avg_emb

    def get_average_face_embedding(self):
        if not self.face_embs:
            return None
        emb_array = np.array([e for e in self.face_embs if e is not None])
        if len(emb_array) == 0:
            return None
        avg_emb = np.mean(emb_array, axis=0)
        norm = np.linalg.norm(avg_emb)
        return avg_emb / norm if norm > 0 else avg_emb

    def is_expired(self):
        return time.time() - self.last_seen > PENDING_EXPIRY_SECONDS

# ============================================================
# FACE RECOGNIZER
# ============================================================
class FaceRecognizer:
    def __init__(self):
        print("Loading face recognition model...")
        self.face_app = None
        try:
            self.face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
            self.face_app.prepare(ctx_id=0, det_size=(320, 320))
            print("✅ Face model loaded")
        except Exception as e:
            print(f"⚠️ Face model load failed: {e}")
        self.min_face_size = MIN_FACE_SIZE
        self.min_det_score = MIN_FACE_DET_SCORE

    def extract_with_quality(self, crop_bgr):
        if crop_bgr is None or crop_bgr.size == 0 or self.face_app is None:
            return None, 0
        try:
            with MODEL_LOCK:
                faces = self.face_app.get(crop_bgr)
            if not faces:
                return None, 0
            best = max(faces, key=lambda f: f.det_score)
            face_w = best.bbox[2] - best.bbox[0]
            face_h = best.bbox[3] - best.bbox[1]
            if face_w < self.min_face_size or face_h < self.min_face_size:
                return None, 0
            if best.det_score < self.min_det_score:
                return None, 0
            return best.normed_embedding, best.det_score
        except Exception:
            return None, 0

# ============================================================
# REID ENSEMBLE
# ============================================================
class ReIDEnsemble:
    def __init__(self):
        print("Loading ReID models...")
        self.reid_model = None
        try:
            self.reid_model = torchreid.models.build_model(
                name="osnet_x1_0",
                num_classes=1000,
                pretrained=True
            )
            self.reid_model.eval().to(DEVICE)
            print("✅ OSNet loaded")
        except Exception as e:
            print(f"⚠️ OSNet load failed: {e}")
        self.reid_input_size = (256, 128)
        self.color_extractor = ClothingColorExtractor()

    @torch.no_grad()
    def extract_embeddings(self, crop_bgr):
        if crop_bgr is None or crop_bgr.size == 0:
            return None, None, None
        try:
            img = cv2.resize(crop_bgr, (self.reid_input_size[1], self.reid_input_size[0]))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_norm = (img_rgb - mean) / std
            tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
            if self.reid_model is not None:
                with torch.no_grad():
                    emb = self.reid_model(tensor).cpu().numpy().flatten()
                norm = np.linalg.norm(emb)
                emb = emb / norm if norm > 0 else emb
            else:
                emb = None
            color_hist = self.color_extractor.extract(crop_bgr)
            return emb, emb, color_hist
        except Exception:
            return None, None, None

class ClothingColorExtractor:
    def __init__(self):
        self.hist_size = 32

    def extract(self, crop_bgr):
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        h, w = crop_bgr.shape[:2]
        if h < 20 or w < 10:
            return None
        try:
            upper = crop_bgr[int(h * 0.1):int(h * 0.45), :]
            lower = crop_bgr[int(h * 0.50):int(h * 0.85), :]
            def region_hist(region):
                if region.size == 0:
                    return np.zeros(self.hist_size)
                hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
                hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180])
                hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256])
                hist = np.concatenate([hist_h, hist_s]).flatten()
                norm = np.linalg.norm(hist)
                return hist / norm if norm > 0 else hist
            upper_hist = region_hist(upper)
            lower_hist = region_hist(lower)
            combined = np.concatenate([upper_hist, lower_hist])
            norm = np.linalg.norm(combined)
            return combined / norm if norm > 0 else combined
        except Exception:
            return None

# ============================================================
# INFINITE TRACKER (Faster R‑CNN + Supervision ByteTrack)
# ============================================================
class InfiniteTracker:
    def __init__(self, detector, conf_threshold=0.25):
        self.detector = detector
        self.conf_threshold = conf_threshold
        self.previous_tracks = {}
        self.pending_identities = {}
        self.temp_id_counter = 10000
        self.track_confidence = defaultdict(float)
        self.frame_counter = 0
        # How many frames each track has been seen for, used to throttle
        # ReID/face re-extraction once a track is already confirmed & linked.
        self.track_frames_seen = defaultdict(int)

        self.byte_tracker = sv.ByteTrack(
            track_activation_threshold=0.5,
            lost_track_buffer=90,
            minimum_matching_threshold=0.6,
            frame_rate=25,
        )

    def track_frame(self, frame, camera_name, identity_gallery, face_recognizer, reid_ensemble):
        self.frame_counter += 1
        expired = [tid for tid, pend in self.pending_identities.items() if pend.is_expired()]
        for tid in expired:
            del self.pending_identities[tid]

        # NOTE: detect() is NOT wrapped in MODEL_LOCK. If multiple cameras
        # share the same PersonDetector instance and run concurrently, wrap
        # this call in MODEL_LOCK too, or give each camera its own detector
        # instance if you have the GPU/CPU headroom for it.
        with MODEL_LOCK:
            xyxy, confs, labels = self.detector.detect(frame)

        if len(xyxy) == 0:
            tracked_detections = sv.Detections.empty()
        else:
            detections = sv.Detections(
                xyxy=xyxy,
                confidence=confs,
                class_id=labels,
            )
            tracked_detections = self.byte_tracker.update_with_detections(detections)

        if len(tracked_detections) > 0 and tracked_detections.tracker_id is not None:
            print(f"[{camera_name}] Frame {self.frame_counter} track IDs: {tracked_detections.tracker_id}")

        tracks = []
        used_ids = set()
        current_centers = {}

        for idx in range(len(tracked_detections)):
            det_xyxy = tracked_detections.xyxy[idx]
            track_id = int(tracked_detections.tracker_id[idx]) if tracked_detections.tracker_id is not None else None
            if track_id is None:
                continue
            conf = tracked_detections.confidence[idx] if tracked_detections.confidence is not None else 1.0
            x1, y1, x2, y2 = det_xyxy.astype(int)
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            current_centers[track_id] = center

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            stable_id, stability = identity_gallery.get_stable_id_for_track(camera_name, track_id)
            if stable_id is not None:
                global_id = stable_id
                self.track_confidence[track_id] = stability
            else:
                global_id = identity_gallery.get_track_link(camera_name, track_id)

            # ---- PERFORMANCE: only re-run ReID/face extraction every N frames
            # once a track is already linked to an identity. New/unconfirmed
            # tracks still get extracted every frame since they need the
            # evidence to get confirmed in the first place.
            self.track_frames_seen[track_id] += 1
            already_linked = global_id is not None
            due_for_extraction = (
                not already_linked
                or REID_EVERY_N_FRAMES_CONFIRMED <= 1
                or self.track_frames_seen[track_id] % REID_EVERY_N_FRAMES_CONFIRMED == 0
            )

            if due_for_extraction:
                with MODEL_LOCK:
                    body_emb, _, _ = reid_ensemble.extract_embeddings(crop)
            else:
                body_emb = None

            # Face is NOT used for body/track identity matching (that stays
            # body-only, same as before, to avoid the false-match issues this
            # variant was tuned to avoid). It's only extracted so we can check
            # the person against the enrolled employee gallery for naming.
            if due_for_extraction:
                face_emb, face_quality = face_recognizer.extract_with_quality(crop)
            else:
                face_emb, face_quality = None, 0

            if global_id is None:
                pending_key = f"{camera_name}_{track_id}"
                matched_id = None
                confidence = 0

                if body_emb is not None:
                    if DEBUG_MATCHING:
                        print(f"[{camera_name}] Track {track_id}: searching body match")
                    matched_id, confidence = identity_gallery.find_by_body_infinite(
                        body_emb, camera_name, used_ids
                    )
                    if matched_id is not None:
                        time_since_seen = time.time() - identity_gallery.last_seen.get(matched_id, time.time())
                        last_cam = identity_gallery.last_camera.get(matched_id, None)
                        if time_since_seen > 5.0 or (last_cam is not None and last_cam != camera_name):
                            print(f"[{camera_name}] 🔄 REACQUIRED: Track {track_id} -> ID {matched_id} (gone {time_since_seen:.1f}s)")
                            identity_gallery.id_reacquisition_count[matched_id] += 1
                        identity_gallery.set_track_link(camera_name, track_id, matched_id)
                        identity_gallery.update_identity_infinite(
                            matched_id, body_emb=body_emb, face_emb=face_emb,
                            camera_name=camera_name, face_quality=face_quality
                        )
                        global_id = matched_id
                        used_ids.add(global_id)
                        tracks.append({
                            'track_id': track_id,
                            'global_id': global_id,
                            'bbox': (x1, y1, x2, y2),
                            'conf': conf,
                            'center': center,
                            'confirmed': True,
                            'match_type': 'reacquired' if (time_since_seen > 5.0 or (last_cam is not None and last_cam != camera_name)) else 'body_match',
                            'face_quality': face_quality,
                            'time_gone': time_since_seen if time_since_seen > 5.0 else 0,
                            'name': identity_gallery.get_display_name(global_id),
                        })
                        continue

                # Pending or new
                if pending_key in self.pending_identities:
                    pending = self.pending_identities[pending_key]
                    pending.add_observation(
                        body_emb=body_emb,
                        face_emb=face_emb,
                        match_id=matched_id,
                        match_score=confidence
                    )
                    if pending.is_confirmed():
                        best_id = pending.best_match_id
                        if best_id is None:
                            avg_body = pending.get_average_body_embedding()
                            if avg_body is not None:
                                best_id = identity_gallery.add_identity_infinite(
                                    body_emb=avg_body,
                                    face_emb=pending.get_average_face_embedding(),
                                    camera_name=camera_name,
                                    face_quality=face_quality
                                )
                            else:
                                best_id = identity_gallery.add_identity_infinite(
                                    camera_name=camera_name,
                                    face_quality=face_quality
                                )
                        if best_id is not None:
                            identity_gallery.set_track_link(camera_name, track_id, best_id)
                            global_id = best_id
                            avg_body = pending.get_average_body_embedding()
                            if avg_body is not None:
                                identity_gallery.update_identity_infinite(
                                    best_id,
                                    body_emb=avg_body,
                                    face_emb=pending.get_average_face_embedding(),
                                    camera_name=camera_name,
                                    face_quality=face_quality
                                )
                            print(f"[{camera_name}] ✅ CONFIRMED: Track {track_id} -> ID {global_id}")
                            del self.pending_identities[pending_key]
                        else:
                            global_id = self.temp_id_counter
                            self.temp_id_counter += 1
                    else:
                        global_id = self.temp_id_counter
                        self.temp_id_counter += 1
                else:
                    pending = InfinitePendingIdentity(
                        self.temp_id_counter,
                        body_emb=body_emb,
                        face_emb=face_emb,
                        camera_name=camera_name,
                        face_quality=face_quality
                    )
                    if matched_id is not None:
                        pending.add_observation(match_id=matched_id, match_score=confidence)
                    self.pending_identities[pending_key] = pending
                    global_id = self.temp_id_counter
                    self.temp_id_counter += 1
            else:
                if body_emb is not None or face_emb is not None:
                    identity_gallery.update_identity_infinite(
                        global_id,
                        body_emb=body_emb,
                        face_emb=face_emb,
                        camera_name=camera_name,
                        face_quality=face_quality
                    )
                else:
                    # Still refresh last_seen so the ID doesn't expire just
                    # because we skipped extraction on this particular frame.
                    identity_gallery.last_seen[global_id] = time.time()
                    identity_gallery.last_camera[global_id] = camera_name
                self.track_confidence[track_id] = min(1.0, self.track_confidence.get(track_id, 0) + 0.1)

            used_ids.add(global_id)
            is_confirmed = global_id < 10000
            tracks.append({
                'track_id': track_id,
                'global_id': global_id,
                'bbox': (x1, y1, x2, y2),
                'conf': conf,
                'center': center,
                'confirmed': is_confirmed,
                'face_quality': face_quality,
                'confidence': self.track_confidence.get(track_id, 0),
                'name': identity_gallery.get_display_name(global_id) if is_confirmed else UNKNOWN_LABEL,
            })

        if self.previous_tracks:
            for old_id in self.previous_tracks:
                if old_id not in current_centers:
                    pending_key = f"{camera_name}_{old_id}"
                    if pending_key in self.pending_identities:
                        del self.pending_identities[pending_key]
                    self.track_confidence[old_id] = max(0, self.track_confidence.get(old_id, 0) - 0.1)

        self.previous_tracks = {t['track_id']: {'center': t['center']} for t in tracks}
        return tracks

# ============================================================
# DISPLAY MANAGER
# ============================================================
class DisplayManager:
    def __init__(self, camera_names):
        self.camera_names = list(camera_names)
        self.frames = {}
        self.stats = {}
        self.last_update = {}
        self.running = True
        self.lock = threading.Lock()
        self.window_name = "Cross‑Camera Tracking - Named Employees"

    def update_frame(self, cam_name, frame, detection_count=0, fps=0, pending_count=0, reacquired_count=0):
        with self.lock:
            self.frames[cam_name] = frame.copy()
            self.stats[cam_name] = {
                'detections': detection_count,
                'fps': fps,
                'pending': pending_count,
                'reacquired': reacquired_count
            }
            self.last_update[cam_name] = time.time()

    def display_loop(self):
        print("\n🎥 Starting display... Press 'q' to quit")
        print("🟢 Green = Recognized employee | 🟠 Orange = Unknown/Visitor | 🔴 Red = Pending")
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            num_cams = max(1, len(self.camera_names))
            cv2.resizeWindow(self.window_name, DISPLAY_WIDTH * num_cams, DISPLAY_HEIGHT + BOTTOM_BAR_HEIGHT)
        except Exception as e:
            print(f"⚠️ Display error: {e}")
            return

        while self.running:
            with self.lock:
                frames_snapshot = dict(self.frames)
                stats_snapshot = dict(self.stats)
                last_update_snapshot = dict(self.last_update)

            display = self.create_vertical_display(frames_snapshot, stats_snapshot, last_update_snapshot)
            cv2.imshow(self.window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
                break

        cv2.destroyAllWindows()

    def create_vertical_display(self, frames_snapshot, stats_snapshot, last_update_snapshot):
        num_cams = max(1, len(self.camera_names))
        total_width = DISPLAY_WIDTH * num_cams
        total_height = DISPLAY_HEIGHT + BOTTOM_BAR_HEIGHT
        display = np.zeros((total_height, total_width, 3), dtype=np.uint8)
        now = time.time()

        for idx, cam_name in enumerate(self.camera_names):
            x_start = idx * DISPLAY_WIDTH
            frame = frames_snapshot.get(cam_name)
            last_seen = last_update_snapshot.get(cam_name)
            is_stale = (last_seen is None) or (now - last_seen > 5.0)

            if frame is not None and frame.size > 0:
                try:
                    resized = self._letterbox(frame, DISPLAY_WIDTH, DISPLAY_HEIGHT)
                    display[0:DISPLAY_HEIGHT, x_start:x_start + DISPLAY_WIDTH] = resized
                except Exception:
                    cv2.putText(display, f"Error: {cam_name}",
                                (x_start + 10, DISPLAY_HEIGHT // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                cv2.putText(display, f"Connecting... {cam_name}",
                            (x_start + DISPLAY_WIDTH // 2 - 150, DISPLAY_HEIGHT // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

            if frame is not None and is_stale:
                cv2.putText(display, "STALLED - reconnecting...",
                            (x_start + 10, DISPLAY_HEIGHT - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.rectangle(display, (x_start + 5, 5), (x_start + 200, 35), (0, 0, 0), -1)
            cv2.putText(display, f"Cam: {cam_name}", (x_start + 10, 28),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            stats = stats_snapshot.get(cam_name)
            if stats:
                det_text = f"Detections: {stats.get('detections', 0)}"
                fps_text = f"FPS: {stats.get('fps', 0):.1f}"
                pending_text = f"Pending: {stats.get('pending', 0)}"
                reacquired_text = f"Reacquired: {stats.get('reacquired', 0)}"
                cv2.rectangle(display,
                              (x_start + DISPLAY_WIDTH - 180, 5),
                              (x_start + DISPLAY_WIDTH - 5, 95),
                              (0, 0, 0), -1)
                cv2.putText(display, det_text, (x_start + DISPLAY_WIDTH - 175, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(display, fps_text, (x_start + DISPLAY_WIDTH - 175, 48),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                cv2.putText(display, pending_text, (x_start + DISPLAY_WIDTH - 175, 68),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                cv2.putText(display, reacquired_text, (x_start + DISPLAY_WIDTH - 175, 88),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

            if idx < num_cams - 1:
                cv2.line(display, (x_start + DISPLAY_WIDTH, 0), (x_start + DISPLAY_WIDTH, DISPLAY_HEIGHT),
                         (255, 255, 255), 2)

        total_text = f"Total People: {identity_gallery.person_count}  |  🔗 Strict Cross‑Camera ON  |  Press 'q' to quit"
        cv2.rectangle(display, (0, DISPLAY_HEIGHT), (total_width, total_height), (0, 0, 0), -1)
        cv2.putText(display, total_text, (10, total_height - 8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return display

    @staticmethod
    def _letterbox(frame, target_w, target_h):
        h, w = frame.shape[:2]
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h))
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        x_off = (target_w - new_w) // 2
        y_off = (target_h - new_h) // 2
        canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
        return canvas

    def stop(self):
        self.running = False

# ============================================================
# FRAME GRABBER — decouples capture from processing
# ============================================================
# Dedicated capture-only thread per camera. Just like a plain "video display"
# script, it does nothing but cap.read() in a loop and always holds only the
# single most recent frame (older ones are overwritten/dropped). The heavy
# Faster R-CNN + ReID pipeline pulls from here whenever it's ready, so a slow
# AI frame never causes the live camera read to fall behind or stall.
class FrameGrabber:
    def __init__(self, cam_name, video_path):
        self.cam_name = cam_name
        self.video_path = video_path
        self.cap = None
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_frame_id = 0
        self.running = True
        self.fps = 25
        self.width = 0
        self.height = 0

        self._connect()
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _connect(self):
        for attempt in range(3):
            self.cap = cv2.VideoCapture(self.video_path)
            if isinstance(self.video_path, str) and self.video_path.startswith(('rtsp://', 'http://')):
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
            if self.cap.isOpened():
                break
            print(f"[{self.cam_name}] Connection attempt {attempt + 1} failed")
            time.sleep(2)

        if self.cap and self.cap.isOpened():
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if self.width == 0 or self.height == 0:
                ret, frame = self.cap.read()
                if ret:
                    self.height, self.width = frame.shape[:2]
                    with self.lock:
                        self.latest_frame = frame
                        self.latest_frame_id += 1
                else:
                    self.width, self.height = 1280, 720
            print(f"[{self.cam_name}] Stream: {self.width}x{self.height} @ {self.fps:.2f} fps")

    def _capture_loop(self):
        reconnect_attempts = 0
        while self.running:
            if not self.cap or not self.cap.isOpened():
                time.sleep(1)
                self._connect()
                continue

            ret, frame = self.cap.read()

            if not ret:
                print(f"[{self.cam_name}] Reconnecting...")
                self.cap.release()
                time.sleep(2)
                self._connect()
                if self.cap and self.cap.isOpened():
                    reconnect_attempts = 0
                else:
                    reconnect_attempts += 1
                    if reconnect_attempts > 5:
                        print(f"[{self.cam_name}] Max reconnect attempts, giving up")
                        self.running = False
                continue

            reconnect_attempts = 0
            with self.lock:
                self.latest_frame = frame
                self.latest_frame_id += 1

    def get_latest(self):
        """Returns (frame_copy, frame_id) or (None, 0). Never blocks on AI work."""
        with self.lock:
            if self.latest_frame is None:
                return None, 0
            return self.latest_frame.copy(), self.latest_frame_id

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()

# ============================================================
# CAMERA WORKER — now a pure PROCESSING loop.
# Capture is handled entirely by FrameGrabber in its own thread.
# ============================================================
def process_camera(cam_name, grabber, display_manager, identity_gallery,
                    tracker, face_recognizer, reid_ensemble):
    print(f"[{cam_name}] Processing started...")

    out_path = os.path.join(OUTPUT_DIR, f"{cam_name}_tracked.mp4")
    out = None  # created lazily once we know real width/height

    frame_count = 0
    last_frame_id_seen = 0
    last_display_update = time.time()
    last_save_time = time.time()
    last_fps_time = time.time()
    fps_counter = 0
    measured_fps = grabber.fps or 25

    while display_manager.running and grabber.running:
        frame, frame_id = grabber.get_latest()

        if frame is None:
            time.sleep(0.01)
            continue

        # Skip if this exact frame was already processed (AI slower than
        # camera right now) — avoids wasted work / duplicate output frames.
        if frame_id == last_frame_id_seen:
            time.sleep(0.005)
            continue
        last_frame_id_seen = frame_id
        frame_count += 1

        # Optional: only run the AI pipeline on every Nth *captured* frame.
        # Faster R-CNN is heavy, so this is your main knob if things are
        # still too slow after decoupling capture/processing.
        if PROCESS_EVERY_N_CAPTURED_FRAMES > 1 and (frame_count % PROCESS_EVERY_N_CAPTURED_FRAMES != 0):
            display_manager.update_frame(cam_name, frame, 0, measured_fps, 0, 0)
            continue

        if out is None:
            h, w = frame.shape[:2]
            out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), grabber.fps or 25, (w, h))

        try:
            tracks = tracker.track_frame(
                frame, cam_name, identity_gallery,
                face_recognizer, reid_ensemble
            )
        except Exception as e:
            print(f"[{cam_name}] Tracking error: {e}")
            tracks = []

        fps_counter += 1
        if time.time() - last_fps_time >= 1.0:
            measured_fps = fps_counter / (time.time() - last_fps_time)
            fps_counter = 0
            last_fps_time = time.time()

        pending_count = sum(1 for t in tracks if not t.get('confirmed', False))
        reacquired_count = sum(1 for t in tracks if t.get('match_type') == 'reacquired')

        for track in tracks:
            global_id = track['global_id']
            x1, y1, x2, y2 = track['bbox']
            confirmed = track.get('confirmed', False)
            match_type = track.get('match_type', 'unknown')
            face_quality = track.get('face_quality', 0)
            confidence = track.get('confidence', 0)
            time_gone = track.get('time_gone', 0)
            name = track.get('name', UNKNOWN_LABEL)
            is_recognized = confirmed and name != UNKNOWN_LABEL

            # Color: a recognized employee is always green, regardless of
            # whether this frame's match_type is body_match/reacquired/etc.
            if is_recognized:
                color = (0, 255, 0)      # Green - recognized employee
                thickness = 3
            elif not confirmed:
                color = (0, 0, 255)      # Red - still pending / not confirmed
                thickness = 2
            else:
                color = (0, 140, 255)    # Orange - confirmed but NOT in employee gallery
                thickness = 3

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            if is_recognized:
                label = name
                if match_type == 'reacquired' and time_gone > 0:
                    label += f" (back, {time_gone:.0f}s)"
            elif not confirmed:
                label = f"...{global_id}"
                if face_quality > 0:
                    label += f" Q:{face_quality:.2f}"
            else:
                label = f"{UNKNOWN_LABEL} (ID {global_id})"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        if time.time() - last_display_update > 0.033:
            display_manager.update_frame(cam_name, frame, len(tracks), measured_fps, pending_count, reacquired_count)
            last_display_update = time.time()

        if out is not None:
            out.write(frame)

        if time.time() - last_save_time > 30:
            identity_gallery.save_state(GALLERY_STATE_PATH)
            last_save_time = time.time()

    if out is not None:
        out.release()
    print(f"[{cam_name}] Processing stopped")

# ============================================================
# UTILITY FUNCTIONS
# ============================================================
def get_color(idx):
    colors = [
        (0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
        (0, 128, 255), (255, 0, 128), (128, 0, 255), (0, 255, 128)
    ]
    return colors[int(idx) % len(colors)]

def load_config():
    try:
        with open("config.yaml", 'r') as f:
            return yaml.safe_load(f)
    except Exception:
        return None

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🔄 CROSS‑CAMERA TRACKING (Faster R‑CNN + Supervision ByteTrack) — Named Employees")
    print("=" * 60)

    config = load_config()
    if not config:
        print("⚠️ No config.yaml found, using default settings")

    camera_sources = {}
    if config:
        if 'gating_camera' in config:
            cam = config['gating_camera']
            camera_sources[cam['id']] = cam['source']
        if 'perimeter_cameras' in config:
            for cam in config['perimeter_cameras'].get('cameras', []):
                camera_sources[cam['id']] = cam['source']

    if not camera_sources:
        camera_sources = {
            "cam1": 0,
            "cam2": 0,
        }
        print("⚠️ No cameras in config, using webcam (both cameras same source for demo)")

    print(f"📷 Cameras: {list(camera_sources.keys())}")
    print(f"💻 Device: {DEVICE}")
    print(f"⚡ ReID/face re-extraction every {REID_EVERY_N_FRAMES_CONFIRMED} frames once confirmed")
    print(f"🧑‍💼 Known employee match threshold: {KNOWN_EMPLOYEE_MATCH_THRESHOLD}")

    if not os.path.exists(KNOWN_EMPLOYEES_PATH):
        print(f"ℹ️  Tip: run enroll_employees.py first to build {KNOWN_EMPLOYEES_PATH}")
        print(f"    Without it, everyone will be labeled '{UNKNOWN_LABEL}'.\n")

    # Initialise
    detector = PersonDetector(conf_threshold=0.6)   # higher to avoid false positives
    face_recognizer = FaceRecognizer()
    reid_ensemble = ReIDEnsemble()
    identity_gallery = InfiniteMemoryGallery()

    # Trackers (share the single detector instance, same as before)
    trackers = {}
    for cam_name in camera_sources:
        tracker = InfiniteTracker(detector, conf_threshold=0.25)
        trackers[cam_name] = tracker

    # Display
    display_manager = DisplayManager(list(camera_sources.keys()))
    display_thread = threading.Thread(target=display_manager.display_loop, daemon=True)
    display_thread.start()

    # Start one FrameGrabber (capture-only thread) per camera
    grabbers = {}
    for cam_name, source in camera_sources.items():
        grabbers[cam_name] = FrameGrabber(cam_name, source)
        time.sleep(0.5)

    # Start one processing thread per camera, each pulling from its grabber
    threads = []
    for cam_name in camera_sources:
        t = threading.Thread(
            target=process_camera,
            args=(cam_name, grabbers[cam_name], display_manager, identity_gallery,
                  trackers[cam_name], face_recognizer, reid_ensemble),
            daemon=True
        )
        threads.append(t)
        t.start()

    try:
        while display_manager.running:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        display_manager.stop()

    finally:
        for g in grabbers.values():
            g.stop()
        identity_gallery.save_state(GALLERY_STATE_PATH)
        print(f"👤 Total unique people: {identity_gallery.person_count}")
        print("✅ Gallery saved")
        print("Done.")
