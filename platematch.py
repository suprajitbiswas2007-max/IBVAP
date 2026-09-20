

import math
import os
import queue
import re
import threading
import time
import difflib
from collections import Counter, deque
from datetime import datetime
from pathlib import Path

import cv2
import easyocr
import mysql.connector
import numpy as np
import torch
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = Path(r"C:\Programming\SIH\runs\detect\indian_plate_detector-4\weights\best.pt")

_local_vehicle = BASE_DIR / "yolov8n.pt"
VEHICLE_MODEL_PATH = str(_local_vehicle) if _local_vehicle.exists() else "yolov8n.pt"

_env_video = os.environ.get("ALPR_VIDEO")
if _env_video is None:
    VIDEO_SOURCE = Path(r"C:\Programming\SIH\VID_20260905_140217112.mp4")
elif _env_video.isdigit():
    VIDEO_SOURCE = int(_env_video)       
else:
    VIDEO_SOURCE = _env_video               

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Plate model not found: {MODEL_PATH}\n"
        "  -> fix MODEL_PATH above, or set the ALPR_PLATE_MODEL environment variable."
    )

METERS_PER_PIXEL  = 0.015  
SPEED_EMA_ALPHA   = 0.30  
SPEED_DEADZONE_PX = 8       
SPEED_MAX_KMH     = 200    
SPEED_WINDOW      = 10   
# OCR / matching
FUZZY_THRESHOLD     = 0.72 
MAX_EDITS           = 1    
SQUARE_PLATE_ASPECT = 1.7   
MIN_OCR_CONF        = 0.12 
MAX_VOTES           = 8   
MAX_ATTEMPTS        = 6     
MAX_FAILED          = 8    
DEBUG     = os.environ.get("ALPR_DEBUG") == "1"
DEBUG_DIR = BASE_DIR / "debug_crops"

# IoU spatial memory
IOU_THRESH     = 0.40   
MEMORY_TTL     = 3.0     
STALE_SECONDS  = 30.0      

VEHICLE_CLASSES = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}

# BGR colours
COLOR_SCANNING = (0, 255, 255)   # yellow
COLOR_MATCHED  = (0, 255, 0)     # green
COLOR_UNKNOWN  = (0, 165, 255)   # orange (read OK, not in DB)
COLOR_VEHICLE  = (255, 200, 0)

# ─────────────────────────────────────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────────────────────────────────────
USE_CUDA = torch.cuda.is_available()
DEVICE   = 0 if USE_CUDA else "cpu"
print(f"[INFO] Inference device: {'CUDA' if USE_CUDA else 'CPU (slow!)'}")

plate_model = YOLO(str(MODEL_PATH))
plate_model.to("cuda" if USE_CUDA else "cpu")

vehicle_model = YOLO(VEHICLE_MODEL_PATH)
vehicle_model.to("cuda" if USE_CUDA else "cpu")

ocr_reader = easyocr.Reader(["en"], gpu=USE_CUDA)
ocr_lock   = threading.Lock()     


DB_CONFIG = dict(
    host=os.environ.get("ALPR_DB_HOST", "localhost"),
    user=os.environ.get("ALPR_DB_USER", "root"),
    password=os.environ.get("ALPR_DB_PASSWORD", "idkanymore"),
    database=os.environ.get("ALPR_DB_NAME", "ibvapplate"),
    connection_timeout=5,
)
DB_TABLE = "platess"   


def get_known_plates_from_db():
    """Return {CLEAN_PLATE: owner}. Empty dict if the DB is unreachable."""
    known = {}
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute(f"SELECT plate_number, plate_owner FROM {DB_TABLE}")
        for plate_number, plate_owner in cursor.fetchall():
            clean = "".join(c for c in plate_number if c.isalnum()).upper()
            known[clean] = plate_owner
        cursor.close()
        conn.close()
        print(f"[INFO] Loaded {len(known)} registered plates from DB")
    except Exception as e:
        print(f"[DB ERROR] Could not load registered plates: {e}")
    return known


known_plates = get_known_plates_from_db()   # read-only after startup -> no lock needed

# State(2 letters) + district(1-2 digits) + series(0-3 letters) + number(3-4 digits)
PLATE_RE = re.compile(r"[A-Z]{2}\d{1,2}[A-Z]{0,3}\d{3,4}")

# Digit -> the letter OCR most likely meant (used in LETTER positions)
NUM_TO_CHAR = {"0": "O", "1": "I", "2": "Z", "3": "B", "4": "A",
               "5": "S", "6": "G", "7": "T", "8": "B", "9": "P"}

CHAR_TO_NUM = {"O": "0", "Q": "0", "D": "0", "U": "0", "I": "1", "L": "1",
               "Z": "2", "S": "5", "G": "6", "B": "8"}

INDIAN_STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CH", "CG", "DD", "DL", "GA", "GJ", "HR",
    "HP", "JH", "JK", "KA", "KL", "LA", "LD", "MP", "MH", "MN", "ML", "MZ",
    "NL", "OD", "OR", "PB", "PY", "RJ", "SK", "TN", "TG", "TS", "TR", "UP",
    "UK", "UA", "WB", "DN", "CT",
}


def _fit(seg, d, s, k):
    """
    Force `seg` into the layout  LL + d digits + s letters + k digits.
    Only swaps look-alike characters (O<->0, B<->8 ...). Returns (fixed, n_swaps)
    or (None, 0) if it can't fit or the state code is not a real one.
    """
    kinds = "LL" + "D" * d + "L" * s + "D" * k
    out, swaps = [], 0
    for ch, kind in zip(seg, kinds):
        if kind == "L":
            if ch.isalpha():
                out.append(ch)
            else:
                out.append(NUM_TO_CHAR[ch]); swaps += 1
        else:
            if ch.isdigit():
                out.append(ch)
            elif ch in CHAR_TO_NUM:
                out.append(CHAR_TO_NUM[ch]); swaps += 1
            else:
                return None, 0
    fixed = "".join(out)
    if fixed[:2] not in INDIAN_STATE_CODES:
        return None, 0
    if int(fixed[2:2 + d]) == 0:          # district codes run 01-99; "0"/"00" never exist
        return None, 0
    return fixed, swaps


def normalize_plate(raw):
    """
    Regex-shaped clean-up. Strips stray characters at the start/end and fixes
    look-alike swaps by POSITION (letters where letters belong, digits where
    digits belong). Returns a plate string that fully matches PLATE_RE, or None.

    Fixes the old bug where series letters such as B/S/D/L were turned into digits:
    the number block is now always 3-4 chars, never "everything that looks numeric".
    """
    text = re.sub(r"[^A-Z0-9]", "", (raw or "").upper())
    if len(text) < 7:
        return None

    best = None   # (cost, -length, plate)
    for start in range(0, min(4, len(text)) + 1):   
        for trim_end in (0, 1, 2):                    
            seg = text[start: len(text) - trim_end]
            strip_cost = 0.75 * (start + trim_end)
            for d in (1, 2):
                for s in (0, 1, 2, 3):
                    for k in (3, 4):
                        if 2 + d + s + k != len(seg):
                            continue
                        fixed, swaps = _fit(seg, d, s, k)
                        if fixed is None or not PLATE_RE.fullmatch(fixed):
                            continue
                        # 1-digit districts are rare (Delhi "DL1C..." is the main case), so a
                        # 1-digit reading must beat a 2-digit one by a clear margin. This stops
                        # "JH0IEN0030" being parsed as district "0" + series "IEN".
                        prior = 1.5 if (d == 1 and fixed[:2] != "DL") else 0.0
                        cand = (swaps + strip_cost + prior, -len(fixed), fixed)
                        if best is None or cand < best:
                            best = cand
    return best[2] if best else None


ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def _read_line(img):
    """OCR one line of text. Returns (text, mean_confidence)."""
    if img is None or img.size == 0:
        return "", 0.0
    # At least 2x; more for tiny crops so the text ends up roughly 100 px tall.
    scale = min(4.0, max(2.0, 100.0 / max(img.shape[0], 1)))
    up = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    try:
        with ocr_lock:
            results = ocr_reader.readtext(up, detail=1, paragraph=False, allowlist=ALLOWLIST)
    except Exception as e:
        print(f"[WARN] EasyOCR: {e}")
        return "", 0.0
    if not results:
        return "", 0.0

    # Drop small-text boxes. The blue "IND" strip / hologram on HSRP plates is much
    # shorter than the registration characters and otherwise leaks junk such as "EJH01...".
    heights = [max(p[1] for p in r[0]) - min(p[1] for p in r[0]) for r in results]
    tallest = max(heights)
    if DEBUG:
        print("[OCR boxes] " + " | ".join(
            f"{r[1]!r} conf={r[2]:.2f} h={hgt:.0f}" for r, hgt in zip(results, heights)))
    keep = [r for r, hgt in zip(results, heights) if hgt >= 0.55 * tallest]
    keep.sort(key=lambda r: min(p[0] for p in r[0]))              # left -> right
    text = re.sub(r"[^A-Z0-9]", "", "".join(r[1] for r in keep).upper())
    conf = sum(r[2] for r in keep) / len(keep)
    return text, conf


def _read_two_line_plate(plate_img):
    
    h = plate_img.shape[0]
    mid = h // 2
    overlap = max(1, int(h * 0.04))
    top_text, top_conf = _read_line(plate_img[: mid + overlap])
    bot_text, bot_conf = _read_line(plate_img[max(0, mid - overlap):])
    if not top_text or not bot_text:
        return "", 0.0
    return top_text + bot_text, (top_conf + bot_conf) / 2.0


_debug_saved = 0


def _save_debug_crop(img, tag):
    global _debug_saved
    if _debug_saved >= 300:
        return
    try:
        DEBUG_DIR.mkdir(exist_ok=True)
        cv2.imwrite(str(DEBUG_DIR / f"track{tag}_{int(time.time() * 1000)}.png"), img)
        _debug_saved += 1
    except Exception as e:
        print(f"[WARN] could not save debug crop: {e}")


def match_plate(plate_crop, tag=""):
    """
    Crop -> (clean regex-valid plate or None, best raw OCR text).
    The raw text is used as a fallback for DB matching when it doesn't parse.
    """
    try:
        h, w = plate_crop.shape[:2]
        if h < 8 or w < 20:
            return None, ""
        aspect = w / max(h, 1)

        plate, raw_best = None, ""
        if aspect < SQUARE_PLATE_ASPECT:                       # stacked plate
            raw, conf = _read_two_line_plate(plate_crop)
            if conf >= MIN_OCR_CONF:
                raw_best, plate = raw, normalize_plate(raw)
        if plate is None:                                      # normal single-line plate
            raw, conf = _read_line(plate_crop)
            if conf >= MIN_OCR_CONF:
                raw_best, plate = raw, normalize_plate(raw)

        if DEBUG:
            print(f"[OCR] track={tag} size={w}x{h} aspect={aspect:.2f} raw={raw_best!r} -> {plate}")
            _save_debug_crop(plate_crop, tag)
        return plate, raw_best
    except Exception as e:
        print(f"[OCR ERROR] {e}")
        return None, ""


def _edit_distance(a, b, limit):
    """Levenshtein distance; returns limit+1 as soon as it is clear it will exceed `limit`."""
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        if min(cur) > limit:
            return limit + 1
        prev = cur
    return prev[-1]


def _substring_distance(pattern, text):
    """Smallest edit distance between `pattern` and ANY substring of `text` (Sellers).
    Lets a DB plate be found inside raw OCR text that has junk before/after it."""
    prev = [0] * (len(text) + 1)
    for i, cp in enumerate(pattern, 1):
        cur = [i]
        for j, ct in enumerate(text, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (cp != ct)))
        prev = cur
    return min(prev)


def fuzzy_match_known_plate(candidates, known, threshold=FUZZY_THRESHOLD, substring=False):
    """
    STRICT match of OCR text against registered plates. -> (plate, owner, ratio) or (None, None, 0.0)

    A plate is accepted only if it is within MAX_EDITS characters of a DB plate (one misread
    character). A loose ratio alone is not enough: on 10-character plates a ratio of 0.72 lets
    a *different* vehicle match (JH01EE0030 vs JH01EE0183 scores 0.80 with only the
    number block wrong). If two DB plates are equally close, the read is ambiguous -> no match.
    substring=True is used for raw OCR text that failed to parse (may contain IND-strip junk).
    """
    best = None                                   # (edits, -ratio, plate, owner)
    ambiguous = False
    for cand in candidates:
        if not cand or (substring and len(cand) < 6):
            continue
        if cand in known:
            return cand, known[cand], 1.0
        for db_plate, owner in known.items():
            if substring:
                edits = _substring_distance(db_plate, cand)
                ratio = 1.0 - edits / max(len(db_plate), 1)
            else:
                edits = _edit_distance(cand, db_plate, MAX_EDITS)
                ratio = difflib.SequenceMatcher(None, cand, db_plate).ratio()
            if edits > MAX_EDITS or ratio < threshold:
                continue
            key = (edits, -ratio, db_plate, owner)
            if best is None or key[:2] < best[:2]:
                best, ambiguous = key, False
            elif key[:2] == best[:2] and db_plate != best[2]:
                ambiguous = True
    if best is None or ambiguous:
        return None, None, 0.0
    if DEBUG:
        print(f"[MATCH] {candidates} ~ {best[2]} (edits={best[0]})")
    return best[2], best[3], -best[1]


def is_plate_quality_ok(crop, min_size=12, blur_threshold=1.0):
    h, w = crop.shape[:2]
    if h < min_size or w < min_size:
        return False
    aspect = w / max(h, 1)
    if aspect < 0.4 or aspect > 8.0:
        return False
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() >= blur_threshold


def pad_plate_crop(frame, x1, y1, x2, y2, pad_ratio=0.18):
    fh, fw = frame.shape[:2]
    px, py = int((x2 - x1) * pad_ratio), int((y2 - y1) * pad_ratio)
    return frame[max(0, y1 - py): min(fh, y2 + py), max(0, x1 - px): min(fw, x2 + px)]


# ─────────────────────────────────────────────────────────────────────────────
# SHARED STATE  (everything below is guarded by state_lock)
# ─────────────────────────────────────────────────────────────────────────────
state_lock = threading.Lock()

# plate_tracks[track_id] = {
#   "status":  "scanning" | "matched" | "unknown",
#   "label":   text shown on screen,
#   "done":    True once we stop OCR-ing this plate,
#   "votes":   recent OCR reads, "attempts": number of reads,
#   "box":     last (x1,y1,x2,y2), "last_seen": time of last detection }
# This dict doubles as the IoU "spatial memory".
plate_tracks = {}
pending_tracks = set()      # track IDs currently queued / being OCR-ed
live_alerts = []            # newest first, max 50

ocr_queue = queue.Queue(maxsize=16)
_worker_started = False


def get_live_alerts():
    with state_lock:
        return list(live_alerts)


def reset_state():
    with state_lock:
        plate_tracks.clear()


def prune_state(now):
    """Drop tracks that have not been seen for a long time (stops unbounded growth)."""
    with state_lock:
        for tid in [k for k, r in plate_tracks.items() if now - r["last_seen"] > STALE_SECONDS]:
            del plate_tracks[tid]


# ── IoU spatial memory ───────────────────────────────────────────────────────
def iou(a, b):
    """Intersection-over-Union of two (x1,y1,x2,y2) boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1)


def _find_inherited_locked(track_id, box, now, active_ids):
    """
    IoU TRACKING / SPATIAL MEMORY
    -----------------------------
    YOLO sometimes loses a plate and re-issues it under a new ID (2 -> 7). The
    plate hasn't moved, so if a *different* track that we already fully
    recognised was seen here a moment ago (IoU >= IOU_THRESH, within
    MEMORY_TTL seconds), the new ID inherits its label immediately instead of
    being re-scanned.

    The donor must be ABSENT from the current frame (`active_ids`). If its ID is still
    being detected, the two boxes are two different plates, not one plate with a new ID,
    so it must not lend its label. Caller must hold state_lock.
    """
    best_iou, best_tid, best = 0.0, None, None
    for tid, rec in plate_tracks.items():
        if tid == track_id or tid in active_ids or not rec["done"]:
            continue
        if now - rec["last_seen"] > MEMORY_TTL:
            continue
        score = iou(box, rec["box"])
        if score >= IOU_THRESH and score > best_iou:
            best_iou, best_tid, best = score, tid, rec
    return best_tid, best, best_iou


def update_plate_track(track_id, box, now, active_ids=frozenset()):
    """Main-thread call, once per plate detection per frame. -> (label, status, done)."""
    with state_lock:
        rec = plate_tracks.get(track_id)
        if rec is None:
            rec = {"status": "scanning", "label": "Scanning...", "done": False,
                   "votes": [], "attempts": 0, "failed": 0, "box": box, "last_seen": now}
            plate_tracks[track_id] = rec

        if not rec["done"]:
            donor_tid, donor, score = _find_inherited_locked(track_id, box, now, active_ids)
            if donor is not None:
                rec["status"], rec["label"], rec["done"] = donor["status"], donor["label"], True
                if DEBUG:
                    print(f"[IoU] track {track_id} inherits {donor['label']!r} "
                          f"from lost track {donor_tid} (IoU={score:.2f})")

        rec["box"], rec["last_seen"] = box, now
        return rec["label"], rec["status"], rec["done"]


def _claim_scan(track_id):
    """Atomically reserve a track for OCR so we never queue it twice."""
    with state_lock:
        if track_id in pending_tracks:
            return False
        pending_tracks.add(track_id)
        return True


def _release_scan(track_id):
    with state_lock:
        pending_tracks.discard(track_id)


def enqueue_scan(track_id, crop, box, now):
    if not _claim_scan(track_id):
        return
    try:
        ocr_queue.put_nowait((track_id, crop, box, now))
    except queue.Full:
        _release_scan(track_id)          # OCR is saturated; try again on a later frame


# ── Background OCR thread ────────────────────────────────────────────────────
def _commit_locked(rec, track_id, status, label):
    """Finalise a plate (caller holds state_lock). Returns alert text to print, or None."""
    rec["status"], rec["label"], rec["done"] = status, label, True
    key = f"plate_{track_id}"
    if any(a["id"] == key for a in live_alerts):
        return None
    ts = datetime.now().strftime("%H:%M:%S")
    live_alerts.insert(0, {"time": ts, "msg": f"VEHICLE DETECTED: {label}", "id": key})
    del live_alerts[50:]
    return f"[{ts}] VEHICLE DETECTED: {label}"


def _register_read(track_id, text, now):
    # 1) record the vote
    with state_lock:
        rec = plate_tracks.get(track_id)
        if rec is None or rec["done"]:
            return
        rec["votes"].append(text)
        rec["votes"] = rec["votes"][-MAX_VOTES:]
        rec["attempts"] += 1
        rec["failed"] = 0
        if rec["status"] == "unknown":            # was "Unreadable"; we have a real read now
            rec["status"], rec["label"] = "scanning", "Scanning..."
        top_text, top_count = Counter(rec["votes"]).most_common(1)[0]
        attempts = rec["attempts"]

    # 2) fuzzy DB match (outside the lock: it loops over every DB plate)
    db_plate, owner, _ = fuzzy_match_known_plate([text, top_text], known_plates)

    # 3) commit the result
    with state_lock:
        rec = plate_tracks.get(track_id)
        if rec is None or rec["done"]:
            return
        if db_plate:                                           # DB hit -> instant green
            alert = _commit_locked(rec, track_id, "matched", f"{owner}, {db_plate}")
        elif top_count >= 2 or attempts >= MAX_ATTEMPTS:       # stable read, not in DB
            alert = _commit_locked(rec, track_id, "unknown", f"Unknown, {top_text}")
        else:
            return                                             # keep scanning (yellow)
    if alert:
        print(alert)


def _register_unparsed(track_id, raw):
    """
    OCR text that doesn't fit the plate pattern. Still try it against the DB (this
    rescues plates where the IND strip or one bad character broke the parse), and
    count the failure so a hopeless plate is flagged instead of scanning silently forever.
    """
    db_plate = owner = None
    if len(raw) >= 6:
        db_plate, owner, _ = fuzzy_match_known_plate([raw], known_plates, substring=True)
    alert = None
    with state_lock:
        rec = plate_tracks.get(track_id)
        if rec is None or rec["done"]:
            return
        if db_plate:
            alert = _commit_locked(rec, track_id, "matched", f"{owner}, {db_plate}")
        else:
            rec["failed"] += 1
            if rec["failed"] >= MAX_FAILED and rec["status"] == "scanning":
                # done stays False, so we keep trying; the label just stops claiming progress
                rec["status"], rec["label"] = "unknown", "Unreadable"
    if alert:
        print(alert)


def _ocr_worker():
    while True:
        track_id, crop, _box, now = ocr_queue.get()
        try:
            plate, raw = match_plate(crop, tag=track_id)
            if plate:
                _register_read(track_id, plate, now)
            else:
                _register_unparsed(track_id, raw)
        except Exception as e:
            print(f"[WORKER ERROR] {e}")
        finally:
            _release_scan(track_id)            # always runs, so a track can never get stuck
            ocr_queue.task_done()


def start_ocr_worker():
    global _worker_started
    with state_lock:
        if _worker_started:
            return
        _worker_started = True
    threading.Thread(target=_ocr_worker, daemon=True, name="ocr-worker").start()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────
def _is_local_file(source):
    return isinstance(source, (str, Path)) and Path(source).is_file()


def process_stream(source=VIDEO_SOURCE):
    """Generator: yields one annotated BGR frame per input frame."""
    is_file = _is_local_file(source)
    if not is_file and isinstance(source, (str, Path)) and not re.match(r"^\w+://", str(source)):
        raise FileNotFoundError(f"Video file not found: {source}\n  -> fix VIDEO_SOURCE or set ALPR_VIDEO.")

    cap = cv2.VideoCapture(str(source) if isinstance(source, Path) else source)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video source: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"[INFO] Source: {source}  ({fps:.1f} fps, {'file' if is_file else 'live'} clock)")

    start_ocr_worker()
    reset_state()
    vehicle_model.predictor = None       # fresh trackers for this stream
    plate_model.predictor = None

    vehicle_history = {}                 # id -> deque[(cx, cy, t)]
    speed_cache     = {}                 # id -> smoothed km/h
    vehicle_seen    = {}                 # id -> last time seen
    t0, frame_idx = time.time(), 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # SPEED needs video time, not processing time. For a file, time = frame/fps;
            # using time.time() would measure how fast the GPU is, not how fast cars move.
            now = frame_idx / fps if is_file else time.time() - t0
            frame_idx += 1

            if frame.shape[1] > 1280:
                frame = cv2.resize(frame, (1280, int(frame.shape[0] * 1280 / frame.shape[1])),
                                   interpolation=cv2.INTER_AREA)
            if np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)) < 60:      # night boost
                frame = cv2.convertScaleAbs(frame, alpha=1.25, beta=45)

            # `frame` stays CLEAN (used for detection + OCR crops); we draw on `disp`.
            disp = frame.copy()

            # ── VEHICLES + SPEED ────────────────────────────────────────────
            v_res = vehicle_model.track(
                frame, persist=True, conf=0.15, imgsz=1280, verbose=False,
                classes=list(VEHICLE_CLASSES), device=DEVICE,
            )[0]

            if v_res.boxes is not None and v_res.boxes.id is not None:
                v_boxes = v_res.boxes.xyxy.cpu().numpy().astype(int)
                v_confs = v_res.boxes.conf.cpu().numpy()
                v_clss  = v_res.boxes.cls.cpu().numpy().astype(int)
                v_ids   = v_res.boxes.id.cpu().numpy().astype(int)

                for (bx1, by1, bx2, by2), v_conf, v_cls, v_id in zip(v_boxes, v_confs, v_clss, v_ids):
                    name = VEHICLE_CLASSES.get(v_cls, "Vehicle")
                    cx, cy = (bx1 + bx2) // 2, by2                     # bottom-centre = road contact point

                    hist = vehicle_history.setdefault(v_id, deque(maxlen=SPEED_WINDOW + 5))
                    hist.append((cx, cy, now))
                    vehicle_seen[v_id] = now

                    if len(hist) >= SPEED_WINDOW:
                        old_cx, old_cy, old_t = hist[-SPEED_WINDOW]
                        dt = now - old_t
                        if dt > 0:
                            dist_px = math.hypot(cx - old_cx, cy - old_cy)
                            if dist_px < SPEED_DEADZONE_PX:
                                target = 0.0
                            else:
                                target = min(dist_px * METERS_PER_PIXEL / dt * 3.6, SPEED_MAX_KMH)
                            prev = speed_cache.get(v_id)
                            speed_cache[v_id] = target if prev is None else (
                                SPEED_EMA_ALPHA * target + (1.0 - SPEED_EMA_ALPHA) * prev)

                    cv2.rectangle(disp, (bx1, by1), (bx2, by2), COLOR_VEHICLE, 2)
                    cv2.putText(disp, f"{name} [{v_conf:.2f}]", (bx1, max(by1 - 10, 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_VEHICLE, 2)
                    spd = int(speed_cache.get(v_id, 0.0))
                    if spd > 0:
                        cv2.putText(disp, f"{spd} KM/H", (bx1, by2 + 22),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # ── PLATES + OCR ────────────────────────────────────────────────
            p_res = plate_model.track(
                frame, persist=True, conf=0.30, imgsz=1280, verbose=False, device=DEVICE,
            )[0]

            if p_res.boxes is not None and p_res.boxes.id is not None:
                fh, fw = frame.shape[:2]
                p_boxes = p_res.boxes.xyxy.cpu().numpy().astype(int)
                p_ids   = p_res.boxes.id.cpu().numpy().astype(int)
                active_ids = {int(i) for i in p_ids}          # plate IDs detected in THIS frame

                for (x1, y1, x2, y2), track_id in zip(p_boxes, p_ids):
                    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(fw, x2), min(fh, y2)
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    box = (int(x1), int(y1), int(x2), int(y2))
                    track_id = int(track_id)

                    # Inherit a label via IoU if YOLO re-issued the ID, else start "Scanning..."
                    label, status, done = update_plate_track(track_id, box, now, active_ids)

                    if not done and is_plate_quality_ok(crop):
                        padded = pad_plate_crop(frame, x1, y1, x2, y2)
                        enqueue_scan(track_id, padded.copy(), box, now)

                    color = {"matched": COLOR_MATCHED, "unknown": COLOR_UNKNOWN}.get(status, COLOR_SCANNING)
                    cv2.rectangle(disp, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(disp, label, (x1, max(y1 - 10, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # ── housekeeping ────────────────────────────────────────────────
            if frame_idx % 100 == 0:
                prune_state(now)
                for vid in [k for k, ts in vehicle_seen.items() if now - ts > STALE_SECONDS]:
                    vehicle_history.pop(vid, None)
                    speed_cache.pop(vid, None)
                    vehicle_seen.pop(vid, None)

            yield disp
    finally:
        cap.release()


def generate_frames(source=VIDEO_SOURCE):
    """MJPEG generator for a Flask route:  Response(generate_frames(), mimetype=...)."""
    for disp in process_stream(source):
        ok, buf = cv2.imencode(".jpg", disp, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ok:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")


if __name__ == "__main__":
    try:
        for annotated in process_stream():
            cv2.imshow("ALPR", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cv2.destroyAllWindows()
