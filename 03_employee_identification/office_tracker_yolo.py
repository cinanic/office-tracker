import cv2
import torch
import numpy as np
import yaml
import os
import sys
import pickle
import threading
import time
from collections import deque, defaultdict
from ultralytics import YOLO
import torchreid
from insightface.app import FaceAnalysis
from scipy.spatial.distance import cosine
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION - VERY STRICT MATCHING
# ============================================================
OUTPUT_DIR = "outputs"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 360
BOTTOM_BAR_HEIGHT = 30

# ---- VERY STRICT MATCHING ----
FACE_MATCH_THRESHOLD = 0.65              # Very strict face matching
BODY_MATCH_THRESHOLD = 0.70              # Very strict body matching
CROSS_CAMERA_MATCH_THRESHOLD = 0.60      # Strict cross-camera
MIN_MATCH_THRESHOLD = 0.60               # Absolute minimum for ANY match
REQUIRED_SCORE_GAP = 0.15                # Required gap between best and second best

# ---- KNOWN EMPLOYEE (NAMED) RECOGNITION ----
KNOWN_EMPLOYEES_PATH = "known_employees.pkl"   # built by enroll_employees.py
# Buffalo_l / ArcFace cosine-similarity threshold for verification against a
# still reference photo. Real-world CCTV footage (compression, angle,
# distance) will score lower than a lab setting, so this is intentionally
# looser than FACE_MATCH_THRESHOLD above (which is for track-to-track
# matching within the live session). Tune this against your own footage:
# too high -> employees keep showing as "Unknown"; too low -> strangers get
# misnamed.
KNOWN_EMPLOYEE_MATCH_THRESHOLD = 0.50
UNKNOWN_LABEL = "Unknown"

# ---- LONG-TERM MEMORY ----
ID_RETENTION_SECONDS = 1800              # 30 MINUTES
TRACK_LINK_TIMEOUT = 600                

# ---- CONFIRMATION ----
MIN_CONFIRMATION_FRAMES = 8              # More frames needed
PENDING_EXPIRY_SECONDS = 5.0

# ---- PERFORMANCE ----
EXTRACT_EVERY_N_FRAMES_CONFIRMED = 2     # Extract more frequently

# ---- DEBUG ----
DEBUG_MATCHING = True

os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_LOCK = threading.Lock()
GALLERY_STATE_PATH = "gallery_state.pkl"


# ============================================================
# LONG-TERM MEMORY GALLERY - ULTRA STRICT
# ============================================================
class LongTermGallery:
    def __init__(self):
        self.lock = threading.RLock()
        self.next_id = 1
        self.person_count = 0
        self.identities = {}
        self.last_seen = {}
        self.first_seen = {}
        self.camera_tracks = defaultdict(dict)
        self.track_id_map = {}
        self.cross_camera_matches = defaultdict(set)
        self.permanent_ids = set()
        self.id_match_history = defaultdict(list)  # Track match history for each ID
        self.recent_matches = {}  # Prevent rapid ID switching

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
            self.permanent_ids = set(data.get("permanent_ids", []))
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
                "permanent_ids": list(self.permanent_ids),
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
                print(f"   - {name}: {len(embs)} reference photo(s)")
        except Exception as e:
            print(f"⚠️ Error loading known employees: {e}")

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

    def match_known_employee(self, face_emb, threshold=KNOWN_EMPLOYEE_MATCH_THRESHOLD):
        """
        Match a live face embedding against the enrolled employee reference
        photos. Returns (name, score) or (None, 0.0) if nobody clears the
        threshold. This is independent of the live-session gallery matching
        below — it's purely "does this face belong to someone we enrolled."
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
        Safe to call every frame a decent face crop is available — once a
        gid has a name, we only overwrite it if a new match is clearly
        stronger, so a single bad frame can't bump someone to "Unknown".
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
            # if current is a *different* name, keep the existing one unless
            # this is a very strong match — avoids flip-flopping identities
            elif score >= 0.65:
                if DEBUG_MATCHING:
                    print(f"⚠️ Re-labeling ID {gid}: {current} -> {name} (score {score:.2f})")
                self.name_by_gid[gid] = name

        return self.name_by_gid.get(gid)

    def get_display_name(self, gid):
        return self.name_by_gid.get(gid, UNKNOWN_LABEL)

    def _get_best_match_score(self, identity, body_emb=None, face_emb=None):
        """Get the best matching score for an identity with multiple verification"""
        scores = []
        
        # Body matching
        if body_emb is not None and 'body_embs' in identity:
            for stored_emb in identity['body_embs']:
                if stored_emb is None:
                    continue
                score = self._cosine_similarity(body_emb, stored_emb)
                scores.append(('body', score))
        
        # Face matching
        if face_emb is not None and 'face_embs' in identity:
            for stored_emb in identity['face_embs']:
                if stored_emb is None:
                    continue
                score = self._cosine_similarity(face_emb, stored_emb)
                scores.append(('face', score))
        
        if not scores:
            return 0.0
        
        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)
        best_score = scores[0][1]
        
        # If we have both body and face, require both to be good
        if len(scores) >= 2:
            # Get top body and face scores
            body_scores = [s[1] for s in scores if s[0] == 'body']
            face_scores = [s[1] for s in scores if s[0] == 'face']
            
            if body_scores and face_scores:
                avg_body = np.mean(body_scores[:3]) if len(body_scores) >= 3 else body_scores[0]
                avg_face = np.mean(face_scores[:3]) if len(face_scores) >= 3 else face_scores[0]
                
                # Require BOTH to be good if we have both
                if avg_body > 0.5 and avg_face > 0.5:
                    # Combined score with both modalities
                    best_score = (avg_body * 0.7 + avg_face * 0.3)
                elif avg_body > 0.7:
                    # Body is very good, use it
                    best_score = avg_body
                elif avg_face > 0.7:
                    # Face is very good, use it
                    best_score = avg_face
                else:
                    # Neither is great, use the best
                    best_score = max(avg_body, avg_face)
        
        return best_score

    def find_match_ultra_strict(self, body_emb=None, face_emb=None, current_camera=None, exclude_ids=None):
        """
        ULTRA STRICT matching - only matches when VERY confident
        """
        if body_emb is None and face_emb is None:
            return None, 0.0

        exclude_ids = exclude_ids or set()
        best_id = None
        best_score = 0.0
        second_score = 0.0
        all_scores = []
        now = time.time()

        with self.lock:
            # Collect all candidates with their scores
            for gid, identity in self.identities.items():
                if gid in exclude_ids:
                    continue

                # Check if recently seen
                if now - self.last_seen.get(gid, 0) > ID_RETENTION_SECONDS:
                    continue

                # Check if this ID has been recently matched to someone else (prevent switching)
                if gid in self.recent_matches:
                    recent_time = self.recent_matches[gid]
                    if now - recent_time < 3.0:  # 3 second cooldown
                        if DEBUG_MATCHING:
                            print(f"   ID {gid} on cooldown (matched {now - recent_time:.1f}s ago)")
                        continue

                # Get best score for this identity
                score = self._get_best_match_score(identity, body_emb, face_emb)
                
                if score > 0.3:  # Only consider if there's some similarity
                    all_scores.append((gid, score))
                    if score > best_score:
                        second_score = best_score
                        best_score = score
                        best_id = gid
                    elif score > second_score:
                        second_score = score

            # Sort by score descending
            all_scores.sort(key=lambda x: x[1], reverse=True)

            if DEBUG_MATCHING and all_scores:
                print(f"\n🔍 ULTRA STRICT matching results:")
                for i, (gid, score) in enumerate(all_scores[:5]):
                    permanent = "🔒" if self.is_id_permanent(gid) else ""
                    camera_views = self.identities[gid].get('camera_views', [])
                    print(f"   {i+1}. ID {gid} {permanent}: {score:.3f} (seen on: {camera_views})")

            # ----- ULTRA STRICT MATCHING CONDITIONS -----
            if best_id is not None:
                # Condition 1: Must be above minimum threshold
                if best_score < MIN_MATCH_THRESHOLD:
                    if DEBUG_MATCHING:
                        print(f"❌ Best score {best_score:.3f} below minimum {MIN_MATCH_THRESHOLD}")
                    return None, 0.0

                # Condition 2: Must have a significant margin over second best
                if len(all_scores) >= 2:
                    margin = all_scores[0][1] - all_scores[1][1]
                    if margin < REQUIRED_SCORE_GAP:
                        if DEBUG_MATCHING:
                            print(f"❌ Margin too small: {margin:.3f} (need {REQUIRED_SCORE_GAP})")
                        return None, 0.0

                # Condition 3: For cross-camera, require higher confidence
                if current_camera and current_camera not in self.cross_camera_matches.get(best_id, set()):
                    if best_score < CROSS_CAMERA_MATCH_THRESHOLD:
                        if DEBUG_MATCHING:
                            print(f"❌ New camera {current_camera}, score {best_score:.3f} below {CROSS_CAMERA_MATCH_THRESHOLD}")
                        return None, 0.0

                # Condition 4: Check if this ID has been recently assigned to someone else
                if best_id in self.recent_matches:
                    # Allow if it was the same person (track ID might have changed)
                    # But we're checking through the exclude_ids mechanism anyway
                    pass

                # Condition 5: Verify with multiple embeddings if available
                identity = self.identities.get(best_id)
                if identity and len(identity['body_embs']) >= 3:
                    # Check variance - if variance is high, be more careful
                    variance = self._calculate_embedding_variance(identity['body_embs'])
                    if variance > 0.25:  # High variance means less stable identity
                        if DEBUG_MATCHING:
                            print(f"⚠️ ID {best_id} has high variance: {variance:.3f}")
                        # Require even higher score
                        if best_score < 0.70:
                            return None, 0.0

                if DEBUG_MATCHING:
                    print(f"✅ ULTRA STRICT MATCH: ID {best_id} with score {best_score:.3f}")
                
                # Record this match to prevent rapid switching
                self.recent_matches[best_id] = time.time()
                if len(self.recent_matches) > 20:
                    # Clean up old entries
                    old_keys = [k for k, v in self.recent_matches.items() if now - v > 10]
                    for k in old_keys:
                        del self.recent_matches[k]
                
                return best_id, best_score

        return None, 0.0

    def _calculate_embedding_variance(self, embeddings):
        if len(embeddings) < 2:
            return 0.0
        
        emb_array = np.array([e for e in embeddings if e is not None])
        if len(emb_array) < 2:
            return 0.0
        
        # Calculate pairwise distances
        distances = []
        for i in range(len(emb_array)):
            for j in range(i + 1, len(emb_array)):
                dist = 1 - self._cosine_similarity(emb_array[i], emb_array[j])
                distances.append(dist)
        
        return np.mean(distances) if distances else 0.0

    def add_identity_ultra_strict(self, body_emb=None, face_emb=None, camera_name=None, face_quality=0):
        """Add a NEW identity with ULTRA STRICT duplicate prevention"""
        
        # FINAL DUPLICATE CHECK - very strict
        if body_emb is not None:
            for gid, identity in self.identities.items():
                if 'body_embs' in identity:
                    for stored_emb in identity['body_embs']:
                        if stored_emb is None:
                            continue
                        score = self._cosine_similarity(body_emb, stored_emb)
                        if score > 0.75:  # Very high threshold
                            print(f"⚠️ DUPLICATE BLOCKED: Person matches ID {gid} with score {score:.3f}")
                            # Update existing instead
                            self.update_identity_ultra_strict(gid, body_emb=body_emb, face_emb=face_emb, 
                                                            camera_name=camera_name, face_quality=face_quality)
                            return gid

        with self.lock:
            gid = self.next_id
            self.next_id += 1
            self.person_count += 1

            identity = {
                "face_embs": deque(maxlen=30),
                "body_embs": deque(maxlen=30),
                "camera_views": [camera_name] if camera_name else [],
                "created_at": time.time(),
                "last_updated": time.time(),
                "detection_count": 1,
                "face_quality": face_quality,
            }

            if body_emb is not None:
                identity["body_embs"].append(body_emb)
            if face_emb is not None:
                identity["face_embs"].append(face_emb)

            self.identities[gid] = identity
            self.last_seen[gid] = time.time()
            self.first_seen[gid] = time.time()
            
            if camera_name:
                self.cross_camera_matches[gid].add(camera_name)

            self.permanent_ids.add(gid)

        # Try to resolve this new person against the enrolled employee gallery
        name = self.assign_name_if_possible(gid, face_emb)
        if name and name != UNKNOWN_LABEL:
            print(f"👤 NEW PERMANENT PERSON ID {gid} -> recognized as '{name}'")
        else:
            print(f"👤 NEW PERMANENT PERSON ID {gid} -> {UNKNOWN_LABEL} (not in employee gallery)")
        return gid

    def update_identity_ultra_strict(self, gid, body_emb=None, face_emb=None, 
                                     camera_name=None, face_quality=None):
        """Update existing identity with verification"""
        with self.lock:
            if gid not in self.identities:
                return False

            identity = self.identities[gid]

            # Verify this is the same person before updating
            if body_emb is not None and identity["body_embs"]:
                # Check against recent embeddings
                recent_embs = list(identity["body_embs"])[-5:]  # Last 5 embeddings
                if recent_embs:
                    avg_sim = np.mean([self._cosine_similarity(body_emb, e) for e in recent_embs if e is not None])
                    if avg_sim < 0.40:  # Too different
                        print(f"⚠️ WARNING: Body embedding for ID {gid} has low similarity ({avg_sim:.3f})")
                        # Still update but with lower weight

            if body_emb is not None:
                identity["body_embs"].append(body_emb)

            if face_emb is not None:
                if face_quality is None or face_quality >= 0.30:
                    identity["face_embs"].append(face_emb)
                    # Fresh, decent-quality face crop -> good opportunity to
                    # (re)confirm this person's name against the employee gallery
                    self.assign_name_if_possible(gid, face_emb)

            if camera_name:
                if camera_name not in identity["camera_views"]:
                    identity["camera_views"].append(camera_name)
                self.cross_camera_matches[gid].add(camera_name)

            identity["last_updated"] = time.time()
            identity["detection_count"] += 1
            self.last_seen[gid] = time.time()

            return True

    def is_id_permanent(self, gid):
        return gid in self.permanent_ids

    def get_track_link(self, cam_name, local_id):
        with self.lock:
            if local_id in self.camera_tracks[cam_name]:
                gid = self.camera_tracks[cam_name][local_id]
                if gid in self.identities:
                    return gid
            return None

    def set_track_link(self, cam_name, local_id, global_id):
        with self.lock:
            self.camera_tracks[cam_name][local_id] = global_id
            self.track_id_map[local_id] = (global_id, time.time())
            if global_id in self.identities:
                self.cross_camera_matches[global_id].add(cam_name)

    def get_stable_id_for_track(self, cam_name, local_id):
        with self.lock:
            if local_id in self.track_id_map:
                gid, last_seen = self.track_id_map[local_id]
                if time.time() - last_seen < TRACK_LINK_TIMEOUT:
                    if gid in self.identities:
                        return gid
            return None


# ============================================================
# PENDING IDENTITY - ULTRA STRICT
# ============================================================
class PendingIdentity:
    def __init__(self, temp_id):
        self.temp_id = temp_id
        self.body_embs = deque(maxlen=20)
        self.face_embs = deque(maxlen=15)
        self.frame_count = 0
        self.best_match_id = None
        self.best_match_score = 0.0
        self.last_seen = time.time()
        self.confirmed = False
        self.consistent_matches = 0

    def add_observation(self, body_emb=None, face_emb=None, match_id=None, match_score=0):
        if body_emb is not None:
            self.body_embs.append(body_emb)
        if face_emb is not None:
            self.face_embs.append(face_emb)
        
        self.frame_count += 1
        self.last_seen = time.time()

        if match_id is not None:
            if match_score > self.best_match_score:
                self.best_match_score = match_score
                self.best_match_id = match_id
            
            if match_score > 0.60:
                self.consistent_matches += 1

    def is_confirmed(self):
        # Need more frames and consistency
        if self.frame_count >= 10:
            self.confirmed = True
            return True
        
        if self.frame_count >= 6 and self.consistent_matches >= 4:
            self.confirmed = True
            return True
        
        if self.frame_count >= 4 and self.best_match_score > 0.75:
            self.confirmed = True
            return True
        
        return False

    def get_average_body_embedding(self):
        if not self.body_embs:
            return None
        emb_array = np.array([e for e in self.body_embs if e is not None])
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

        self.min_face_size = 35
        self.min_det_score = 0.50

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
# FRAME GRABBER
# ============================================================
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
        with self.lock:
            if self.latest_frame is None:
                return None, 0
            return self.latest_frame.copy(), self.latest_frame_id

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()


# ============================================================
# LONG-TERM TRACKER - ULTRA STRICT
# ============================================================
class LongTermTracker:
    def __init__(self, model_path="yolo26n.pt", conf_threshold=0.25):
        print(f"Loading detection model: {model_path}")
        self.model = None
        try:
            self.model = YOLO(model_path)
            print("✅ Detection model loaded")
        except Exception:
            print(f"⚠️ Error loading {model_path}, trying yolo26n.pt")
            try:
                self.model = YOLO("yolo26n.pt")
                print("✅ YOLO26 loaded")
            except Exception:
                print("❌ No detection model available!")
                sys.exit(1)

        self.conf_threshold = conf_threshold
        self.previous_tracks = {}
        self.pending_identities = {}
        self.temp_id_counter = 10000
        self.frame_counter = 0
        self.assigned_in_frame = set()
        self.track_id_to_global = {}  # Track mapping to prevent switching

    def track_frame(self, frame, camera_name, identity_gallery, face_recognizer, reid_ensemble):
        """Track frame with ULTRA STRICT ID assignment"""
        
        self.frame_counter += 1
        self.assigned_in_frame.clear()

        # Clean up expired pending
        expired = [tid for tid, pend in self.pending_identities.items() if pend.is_expired()]
        for tid in expired:
            del self.pending_identities[tid]

        results = self.model.track(
            frame,
            persist=True,
            classes=[0],
            conf=self.conf_threshold,
            iou=0.5,
            device=DEVICE,
            verbose=False,
            tracker="bytetrack.yaml"
        )

        tracks = []
        current_centers = {}

        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes

            if boxes is not None and boxes.id is not None:
                ids = boxes.id.cpu().numpy().astype(int)
                xyxy = boxes.xyxy.cpu().numpy().astype(int)
                confs = boxes.conf.cpu().numpy() if boxes.conf is not None else [1.0] * len(ids)

                for (x1, y1, x2, y2), track_id, conf in zip(xyxy, ids, confs):
                    if conf < self.conf_threshold:
                        continue

                    center = ((x1 + x2) / 2, (y1 + y2) / 2)
                    current_centers[track_id] = center

                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue

                    # Extract embeddings
                    with MODEL_LOCK:
                        body_emb, _, _ = reid_ensemble.extract_embeddings(crop)
                    face_emb, face_quality = face_recognizer.extract_with_quality(crop)

                    # Try to get existing ID from track link
                    global_id = identity_gallery.get_stable_id_for_track(camera_name, track_id)
                    if global_id is None:
                        global_id = identity_gallery.get_track_link(camera_name, track_id)

                    # If we have an existing ID, keep it
                    if global_id is not None:
                        identity_gallery.last_seen[global_id] = time.time()
                        if body_emb is not None or face_emb is not None:
                            identity_gallery.update_identity_ultra_strict(
                                global_id, body_emb=body_emb, face_emb=face_emb,
                                camera_name=camera_name, face_quality=face_quality
                            )
                        self.assigned_in_frame.add(global_id)
                        self.track_id_to_global[track_id] = global_id
                        
                        tracks.append({
                            'track_id': track_id,
                            'global_id': global_id,
                            'bbox': (x1, y1, x2, y2),
                            'conf': conf,
                            'center': center,
                            'confirmed': True,
                            'match_type': 'existing',
                            'face_quality': face_quality,
                            'name': identity_gallery.get_display_name(global_id),
                        })
                        continue

                    # --- NO EXISTING ID - Try to find a match ---
                    pending_key = f"{camera_name}_{track_id}"
                    matched_id = None
                    match_score = 0.0

                    # Check if we have a pending match for this track
                    if pending_key in self.pending_identities:
                        pending = self.pending_identities[pending_key]
                        if pending.best_match_id is not None:
                            # Use the pending's best match
                            matched_id = pending.best_match_id
                            match_score = pending.best_match_score

                    # If no pending match, try to find one with ULTRA STRICT matching
                    if matched_id is None:
                        matched_id, match_score = identity_gallery.find_match_ultra_strict(
                            body_emb=body_emb,
                            face_emb=face_emb,
                            current_camera=camera_name,
                            exclude_ids=self.assigned_in_frame
                        )

                    # If we found a match with sufficient confidence
                    if matched_id is not None and match_score >= MIN_MATCH_THRESHOLD:
                        # Double-check this ID isn't already assigned in this frame
                        if matched_id not in self.assigned_in_frame:
                            # Verify the match is consistent with previous observations
                            if pending_key in self.pending_identities:
                                pending = self.pending_identities[pending_key]
                                # Check if this ID is consistent with what we've seen
                                if pending.best_match_id is not None and pending.best_match_id != matched_id:
                                    # The pending was leaning towards a different ID
                                    # Only switch if the new match is significantly better
                                    if match_score - pending.best_match_score < 0.10:
                                        if DEBUG_MATCHING:
                                            print(f"⚠️ Pending conflict: {pending.best_match_id} vs {matched_id}")
                                        matched_id = None
                            
                            if matched_id is not None:
                                print(f"[{camera_name}] ✅ ULTRA STRICT MATCH: Track {track_id} -> ID {matched_id} (score: {match_score:.3f})")
                                
                                identity_gallery.set_track_link(camera_name, track_id, matched_id)
                                identity_gallery.update_identity_ultra_strict(
                                    matched_id, body_emb=body_emb, face_emb=face_emb,
                                    camera_name=camera_name, face_quality=face_quality
                                )
                                identity_gallery.last_seen[matched_id] = time.time()
                                self.assigned_in_frame.add(matched_id)
                                self.track_id_to_global[track_id] = matched_id
                                
                                # Clear pending if it exists
                                if pending_key in self.pending_identities:
                                    del self.pending_identities[pending_key]
                                
                                tracks.append({
                                    'track_id': track_id,
                                    'global_id': matched_id,
                                    'bbox': (x1, y1, x2, y2),
                                    'conf': conf,
                                    'center': center,
                                    'confirmed': True,
                                    'match_type': 'matched',
                                    'face_quality': face_quality,
                                    'confidence': match_score,
                                    'name': identity_gallery.get_display_name(matched_id),
                                })
                                continue

                    # --- NO MATCH FOUND - Check pending or create new ---
                    if pending_key in self.pending_identities:
                        pending = self.pending_identities[pending_key]
                        pending.add_observation(
                            body_emb=body_emb,
                            face_emb=face_emb,
                            match_id=matched_id,
                            match_score=match_score
                        )

                        if pending.is_confirmed():
                            # Create NEW identity
                            avg_body = pending.get_average_body_embedding()
                            # use the best/most recent face embedding we collected
                            # while pending, so name-matching has something to work with
                            pending_face_emb = pending.face_embs[-1] if pending.face_embs else None
                            new_id = identity_gallery.add_identity_ultra_strict(
                                body_emb=avg_body,
                                face_emb=pending_face_emb,
                                camera_name=camera_name,
                                face_quality=face_quality
                            )
                            
                            if new_id is not None:
                                identity_gallery.set_track_link(camera_name, track_id, new_id)
                                identity_gallery.last_seen[new_id] = time.time()
                                self.assigned_in_frame.add(new_id)
                                self.track_id_to_global[track_id] = new_id
                                print(f"[{camera_name}] ✅ NEW ID: {new_id}")
                                
                                tracks.append({
                                    'track_id': track_id,
                                    'global_id': new_id,
                                    'bbox': (x1, y1, x2, y2),
                                    'conf': conf,
                                    'center': center,
                                    'confirmed': True,
                                    'match_type': 'new',
                                    'face_quality': face_quality,
                                    'name': identity_gallery.get_display_name(new_id),
                                })
                                del self.pending_identities[pending_key]
                            else:
                                # Fallback: temporary ID
                                temp_id = self.temp_id_counter
                                self.temp_id_counter += 1
                                tracks.append({
                                    'track_id': track_id,
                                    'global_id': temp_id,
                                    'bbox': (x1, y1, x2, y2),
                                    'conf': conf,
                                    'center': center,
                                    'confirmed': False,
                                    'match_type': 'pending',
                                    'face_quality': face_quality,
                                    'name': UNKNOWN_LABEL,
                                })
                        else:
                            # Still pending
                            tracks.append({
                                'track_id': track_id,
                                'global_id': pending.temp_id,
                                'bbox': (x1, y1, x2, y2),
                                'conf': conf,
                                'center': center,
                                'confirmed': False,
                                'match_type': 'pending',
                                'face_quality': face_quality,
                                'name': UNKNOWN_LABEL,
                            })
                    else:
                        # Create new pending
                        pending = PendingIdentity(self.temp_id_counter)
                        pending.add_observation(
                            body_emb=body_emb,
                            face_emb=face_emb,
                            match_id=matched_id,
                            match_score=match_score
                        )
                        self.pending_identities[pending_key] = pending
                        
                        tracks.append({
                            'track_id': track_id,
                            'global_id': pending.temp_id,
                            'bbox': (x1, y1, x2, y2),
                            'conf': conf,
                            'center': center,
                            'confirmed': False,
                            'match_type': 'pending_new',
                            'face_quality': face_quality,
                            'name': UNKNOWN_LABEL,
                        })
                        self.temp_id_counter += 1

        # Update previous tracks
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
        self.window_name = "Office Tracking - Named Employees"

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
        print("🔒 ULTRA STRICT matching - IDs are PERMANENT and NEVER reused")
        print("📋 IDs remembered for 30 MINUTES")
        print(f"🎯 Minimum match threshold: {MIN_MATCH_THRESHOLD}")
        print(f"🧑‍💼 Known employee match threshold: {KNOWN_EMPLOYEE_MATCH_THRESHOLD}")

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

            cv2.rectangle(display,
                          (x_start + 5, 5),
                          (x_start + 200, 35),
                          (0, 0, 0), -1)
            cv2.putText(display, f"Cam: {cam_name}",
                       (x_start + 10, 28),
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
                cv2.putText(display, det_text,
                           (x_start + DISPLAY_WIDTH - 175, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(display, fps_text,
                           (x_start + DISPLAY_WIDTH - 175, 48),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                cv2.putText(display, pending_text,
                           (x_start + DISPLAY_WIDTH - 175, 68),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                cv2.putText(display, reacquired_text,
                           (x_start + DISPLAY_WIDTH - 175, 88),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

            if idx < num_cams - 1:
                cv2.line(display, (x_start + DISPLAY_WIDTH, 0), (x_start + DISPLAY_WIDTH, DISPLAY_HEIGHT),
                         (255, 255, 255), 2)

        total_text = f"Total People: {identity_gallery.person_count}  |  🔒 ULTRA STRICT matching  |  Press 'q' to quit"
        cv2.rectangle(display, (0, DISPLAY_HEIGHT), (total_width, total_height), (0, 0, 0), -1)
        cv2.putText(display, total_text,
                   (10, total_height - 8),
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
# CAMERA WORKER
# ============================================================
def process_camera(cam_name, grabber, display_manager, identity_gallery,
                    tracker, face_recognizer, reid_ensemble):
    print(f"[{cam_name}] Processing started...")

    out_path = os.path.join(OUTPUT_DIR, f"{cam_name}_tracked.mp4")
    out = None

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

        if frame_id == last_frame_id_seen:
            time.sleep(0.005)
            continue
        last_frame_id_seen = frame_id
        frame_count += 1

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
        reacquired_count = sum(1 for t in tracks if t.get('match_type') in ['matched', 'new'])

        for track in tracks:
            global_id = track['global_id']
            x1, y1, x2, y2 = track['bbox']
            confirmed = track.get('confirmed', False)
            match_type = track.get('match_type', 'unknown')
            face_quality = track.get('face_quality', 0)
            confidence = track.get('confidence', 0)
            name = track.get('name', UNKNOWN_LABEL)
            is_recognized = confirmed and name != UNKNOWN_LABEL

            # Color: recognized employee always wins visually, regardless of
            # whether the track is 'existing'/'matched'/'new'
            if is_recognized:
                color = (0, 255, 0)      # Green - recognized employee
                thickness = 3
            elif not confirmed:
                color = (0, 0, 255)       # Red - still pending / not confirmed
                thickness = 2
            else:
                color = (0, 140, 255)     # Orange - confirmed but NOT in employee gallery
                thickness = 3

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # Label
            if is_recognized:
                label = name
                if match_type == 'matched' and confidence:
                    label += f" [{confidence:.0%}]"
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


def create_bytetrack_config():
    config = {
        "tracker_type": "bytetrack",
        "track_high_thresh": 0.25,
        "track_low_thresh": 0.1,
        "new_track_thresh": 0.3,
        "track_buffer": 90,
        "match_thresh": 0.8,
        "fuse_score": True
    }
    with open("bytetrack.yaml", "w") as f:
        yaml.dump(config, f)
    print("✅ ByteTrack config created")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🔄 OFFICE TRACKING - NAMED EMPLOYEE RECOGNITION")
    print("=" * 60)

    create_bytetrack_config()

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
        print("⚠️ No cameras in config, using webcam")

    print(f"📷 Cameras: {list(camera_sources.keys())}")
    print(f"💻 Device: {DEVICE}")
    print(f"🎯 Minimum match threshold: {MIN_MATCH_THRESHOLD}")
    print(f"📏 Required score gap: {REQUIRED_SCORE_GAP}")
    print(f"🧑‍💼 Known employee match threshold: {KNOWN_EMPLOYEE_MATCH_THRESHOLD}")
    print(f"🔒 ULTRA STRICT matching - IDs are PERMANENT and NEVER reused")
    print("=" * 60 + "\n")

    if not os.path.exists(KNOWN_EMPLOYEES_PATH):
        print(f"ℹ️  Tip: run enroll_employees.py first to build {KNOWN_EMPLOYEES_PATH}")
        print(f"    Without it, everyone will be labeled '{UNKNOWN_LABEL}'.\n")

    # Initialize
    face_recognizer = FaceRecognizer()
    reid_ensemble = ReIDEnsemble()
    identity_gallery = LongTermGallery()

    model_path = config.get('perimeter_cameras', {}).get('yolo_model_path', 'yolo26n.pt') if config else 'yolo26n.pt'
    conf_threshold = config.get('perimeter_cameras', {}).get('confidence_threshold', 0.25) if config else 0.25

    # Create trackers
    trackers = {}
    for cam_name in camera_sources:
        tracker = LongTermTracker(model_path=model_path, conf_threshold=conf_threshold)
        trackers[cam_name] = tracker

    # Display
    display_manager = DisplayManager(list(camera_sources.keys()))
    display_thread = threading.Thread(target=display_manager.display_loop, daemon=True)
    display_thread.start()

    # Start one FrameGrabber per camera
    grabbers = {}
    for cam_name, source in camera_sources.items():
        grabbers[cam_name] = FrameGrabber(cam_name, source)
        time.sleep(0.5)

    # Start one processing thread per camera
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
