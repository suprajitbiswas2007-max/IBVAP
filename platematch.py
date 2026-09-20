import cv2
import numpy as np
import mysql.connector
from datetime import datetime
import threading
from collections import Counter
import re
import difflib
import easyocr
import time
import math
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────────────────────
# MODEL INITIALIZATION
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH = r"C:\Programming\SIH\runs\detect\indian_plate_detector-4\weights\best.pt"
plate_model = YOLO(MODEL_PATH)
plate_model.to('cuda')

vehicle_model = YOLO("yolov8n.pt")
vehicle_model.to('cuda')
vehicle_classes = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}

# ─────────────────────────────────────────────────────────────────────────────
# SPEED CALIBRATION — TUNE METERS_PER_PIXEL FOR YOUR CAMERA
# ─────────────────────────────────────────────────────────────────────────────
METERS_PER_PIXEL  = 0.015    # ← ADJUST to match your camera
SPEED_EMA_ALPHA   = 0.30     # New-reading weight (lower = smoother / laggier)
SPEED_DEADZONE_PX = 8        # Pixel displacement below this → treat as stationary
SPEED_MAX_KMH     = 200      # Hard cap — discard physically impossible readings

# ─────────────────────────────────────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────────────────────────────────────
def get_known_plates_from_db():
    known_plates_dict = {}
    try:
        conn = mysql.connector.connect(
            host="localhost", user="root", password="idkanymore", database="ibvapplate"
        )
        cursor = conn.cursor()
        cursor.execute("SELECT plate_number, plate_owner FROM platess")
        rows = cursor.fetchall()
        for plate_number, plate_owner in rows:
            clean_num = "".join(c for c in plate_number if c.isalnum()).upper()
            known_plates_dict[clean_num] = plate_owner
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"[DB ERROR] Could not load registered plates: {e}")
    return known_plates_dict

known_plates = get_known_plates_from_db()

# ─────────────────────────────────────────────────────────────────────────────
# OCR ENGINE SETUP
# ─────────────────────────────────────────────────────────────────────────────
ocr_reader = easyocr.Reader(['en'], gpu=True)
ocr_lock   = threading.Lock()

try:
    import pytesseract
    # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    TESSERACT_AVAILABLE = True
    print("[INFO] pytesseract detected — ensemble OCR enabled")
except ImportError:
    TESSERACT_AVAILABLE = False
    print("[INFO] pytesseract not found — EasyOCR only")

# ─────────────────────────────────────────────────────────────────────────────
# CHARACTER CORRECTION MAPS
# ─────────────────────────────────────────────────────────────────────────────
NUM_TO_CHAR = {
    '0': 'O', '1': 'I', '2': 'Z', '3': 'J',
    '4': 'A', '5': 'S', '6': 'G', '7': 'T', '8': 'B', '9': 'P',
}
CHAR_TO_NUM = {
    'O': '0', 'Q': '0', 'D': '0', 'U': '0', 'C': '0',
    'I': '1', 'L': '1',
    'Z': '2', 'J': '3', 'A': '4', 'S': '5', 'G': '6',
    'T': '7', 'B': '8', 'E': '8',
}
ZERO_LOOKALIKES = ['O', 'U', 'Q', 'D', 'C']

# ─────────────────────────────────────────────────────────────────────────────
# PLATE IMAGE PREPROCESSING — MULTI-PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def _upscale_gray(img, min_width=320):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    h, w = gray.shape
    if w < min_width:
        scale = min(5.0, min_width / max(w, 1))
        gray  = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return gray


def preprocess_plate_variants(plate_img):
    g        = _upscale_gray(plate_img)
    clahe_fn = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    cl       = clahe_fn.apply(g)

    _, otsu = cv2.threshold(cv2.GaussianBlur(cl, (3, 3), 0), 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inv = cv2.bitwise_not(otsu)

    adap = cv2.adaptiveThreshold(
        cv2.GaussianBlur(g, (5, 5), 0), 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 19, 9,
    )

    k     = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    morph = cv2.morphologyEx(cl, cv2.MORPH_CLOSE, k)
    _, morph_t = cv2.threshold(morph, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    blurred = cv2.GaussianBlur(g, (0, 0), 3)
    sharp   = cv2.addWeighted(g, 1.6, blurred, -0.6, 0)

    return [
        (cl,      "CLAHE"),
        (otsu,    "Otsu"),
        (inv,     "InvOtsu"),
        (adap,    "Adaptive"),
        (morph_t, "Morph"),
        (sharp,   "Sharp"),
    ]


def _to_bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img


# ─────────────────────────────────────────────────────────────────────────────
# OCR RUNNERS
# ─────────────────────────────────────────────────────────────────────────────

def _easyocr_on(img):
    try:
        return ocr_reader.readtext(
            _to_bgr(img),
            detail=1,
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            paragraph=False,
            mag_ratio=1.5,
            contrast_ths=0.1,
            adjust_contrast=0.5,
            text_threshold=0.55,
            low_text=0.3,
        )
    except Exception as e:
        print(f"[WARN] EasyOCR: {e}")
        return []


def _tesseract_on(img):
    if not TESSERACT_AVAILABLE:
        return None, 0.0
    try:
        cfg  = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        data = pytesseract.image_to_data(img, config=cfg, output_type=pytesseract.Output.DICT)
        texts, confs = [], []
        for t, c in zip(data['text'], data['conf']):
            t = t.strip()
            if t and int(c) > 40:
                texts.append(t)
                confs.append(int(c) / 100.0)
        if texts:
            return ''.join(texts).upper(), sum(confs) / len(confs)
    except Exception as e:
        print(f"[WARN] Tesseract: {e}")
    return None, 0.0


def _sort_ocr_results(results, img_shape):
    h, w = img_shape[:2]
    if w / max(h, 1) < 1.5:
        return sorted(results, key=lambda r: r[0][0][1])
    return sorted(results, key=lambda r: r[0][0][0])


def _read_split_plate(plate_img):
    """
    Robust 2-line plate parser for stacked commercial plates.
    Top line: State + District + Series (e.g., JH01E)
    Bottom line: 4-digit number (e.g., 0183)
    """
    h, w = plate_img.shape[:2]
    if w / max(h, 1) >= 1.8:  # Wide 1-line plate — skip split
        return None, 0.0

    mid = h // 2
    top_half = plate_img[:mid + int(h * 0.1), :]
    bottom_half = plate_img[mid - int(h * 0.1):, :]

    top_text, top_conf = "", 0.0
    bot_text, bot_conf = "", 0.0

    with ocr_lock:
        # Read Top Half
        for img_v, _ in preprocess_plate_variants(top_half):
            res = _easyocr_on(img_v)
            if res:
                t = "".join(x[1] for x in res).replace(" ", "").upper()
                c = sum(x[2] for x in res) / len(res)
                if len(t) >= 3 and c > top_conf:
                    top_text, top_conf = t, c

        # Read Bottom Half
        for img_v, _ in preprocess_plate_variants(bottom_half):
            res = _easyocr_on(img_v)
            if res:
                t = "".join(x[1] for x in res).replace(" ", "").upper()
                c = sum(x[2] for x in res) / len(res)
                if len(t) >= 2 and c > bot_conf:
                    bot_text, bot_conf = t, c

    if top_text and bot_text:
        bot_digits = "".join(CHAR_TO_NUM.get(c, c) for c in bot_text if c.isalnum())
        combined = f"{top_text}{bot_digits}"
        avg_conf = (top_conf + bot_conf) / 2.0
        print(f"[OCR] Structured Split-Read: '{top_text}' + '{bot_digits}' → '{combined}' ({avg_conf:.2f})")
        return combined, avg_conf

    return None, 0.0


_PLATE_RE = re.compile(r'^[A-Z]{2}\d{1,2}[A-Z]{0,3}\d{1,4}$')


def extract_plate_text(plate_crop):
    h, w = plate_crop.shape[:2]
    if h < 10 or w < 10:
        return None, 0.0

    candidates = []

    with ocr_lock:
        for img_v, name in preprocess_plate_variants(plate_crop):
            res = _easyocr_on(img_v)
            if res:
                res  = _sort_ocr_results(res, img_v.shape)
                text = ''.join(t for _, t, _ in res).replace(' ', '').upper()
                conf = sum(c for _, _, c in res) / len(res)
                if len(text) >= 4:
                    candidates.append((text, conf, f"easy_{name}"))

            t_text, t_conf = _tesseract_on(img_v)
            if t_text and len(t_text) >= 4:
                candidates.append((t_text, t_conf, f"tess_{name}"))

        split_text, split_conf = _read_split_plate(plate_crop)
        if split_text and len(split_text) >= 4:
            candidates.append((split_text, split_conf, "split"))

    if not candidates:
        return None, 0.0

    valid   = [(t, c) for t, c, _ in candidates if _PLATE_RE.match(t)]
    plausible = [(t, c) for t, c, _ in candidates if 6 <= len(t) <= 12]
    pool     = valid or plausible or [(t, c) for t, c, _ in candidates]

    return max(pool, key=lambda x: x[1])


# ─────────────────────────────────────────────────────────────────────────────
# PADDING HELPER
# ─────────────────────────────────────────────────────────────────────────────

def pad_plate_crop(crop, frame, x1, y1, x2, y2, pad_ratio=0.20):
    h, w   = crop.shape[:2]
    px, py = int(w * pad_ratio), int(h * pad_ratio)
    fh, fw = frame.shape[:2]
    return frame[
        max(0, y1 - py) : min(fh, y2 + py),
        max(0, x1 - px) : min(fw, x2 + px),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# PLATE TEXT POST-PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def _plate_position_types(n):
    types = ['L'] * n
    if n < 8:
        return types
    types[0] = types[1] = 'L'
    types[2] = types[3] = 'D'
    for i in range(4, n - 4):
        types[i] = 'L'
    for i in range(n - 4, n):
        types[i] = 'D'
    return types


def _window_grammar_score(sub):
    types = _plate_position_types(len(sub))
    score = 0
    for c, t in zip(sub, types):
        if t == 'L':
            score += 2 if c.isalpha() else (1 if c in NUM_TO_CHAR else 0)
        else:
            score += 2 if c.isdigit() else (1 if c in CHAR_TO_NUM else 0)
    return score


def trim_to_best_plate_window(text, sizes=(10, 9, 8)):
    if len(text) < min(sizes):
        return text
    best_sub, best_score = text, -1
    for size in sizes:
        if len(text) < size:
            continue
        for start in range(len(text) - size + 1):
            sub = text[start:start + size]
            s   = _window_grammar_score(sub)
            if s > best_score:
                best_score, best_sub = s, sub
    return best_sub


def correct_indian_plate_format(text):
    if not text or len(text) < 4:
        return text
    chars = list(text)

    for i in range(min(2, len(chars))):
        chars[i] = NUM_TO_CHAR.get(chars[i], chars[i])

    if len(chars) > 2:
        chars[2] = CHAR_TO_NUM.get(chars[2], chars[2])

    if len(chars) > 3:
        if ''.join(chars[:2]) != 'DL':
            chars[3] = CHAR_TO_NUM.get(chars[3], chars[3])

    n = len(chars)
    if n == 10:
        for i in range(4, 6):    chars[i] = NUM_TO_CHAR.get(chars[i], chars[i])
        for i in range(6, 10):  chars[i] = CHAR_TO_NUM.get(chars[i], chars[i])
    elif n == 9:
        chars[4] = NUM_TO_CHAR.get(chars[4], chars[4])
        for i in range(5, 9):   chars[i] = CHAR_TO_NUM.get(chars[i], chars[i])
    elif n == 8:
        for i in range(4, 8):   chars[i] = CHAR_TO_NUM.get(chars[i], chars[i])

    return ''.join(chars)


def generate_series_candidates(corrected_text, max_candidates=8):
    n = len(corrected_text)
    if n < 8:
        return [corrected_text]
    ambig = [i for i in range(4, n - 4) if corrected_text[i] == 'O'][:2]
    if not ambig:
        return [corrected_text]
    candidates = {corrected_text}
    for pos in ambig:
        new_set = set()
        for cand in candidates:
            c_list = list(cand)
            for letter in ZERO_LOOKALIKES:
                c_list[pos] = letter
                new_set.add(''.join(c_list))
        candidates |= new_set
        if len(candidates) >= max_candidates:
            break
    return list(candidates)[:max_candidates]


def fuzzy_match_known_plate(candidate_texts, known_plates_dict, threshold=0.80):
    best_plate, best_owner, best_ratio = None, None, 0.0
    for cand in candidate_texts:
        if cand in known_plates_dict:
            return cand, known_plates_dict[cand], 1.0
        for db_plate, owner in known_plates_dict.items():
            if abs(len(cand) - len(db_plate)) > 2:
                continue
            ratio = difflib.SequenceMatcher(None, cand, db_plate).ratio()
            if ratio > best_ratio:
                best_ratio, best_plate, best_owner = ratio, db_plate, owner
    if best_ratio >= threshold:
        return best_plate, best_owner, best_ratio
    return None, None, best_ratio


def match_plate(plate_crop):
    try:
        raw_text, confidence = extract_plate_text(plate_crop)
        print(f"[OCR DEBUG] Raw: '{raw_text}' | conf: {confidence:.2f}")

        if not raw_text or confidence < 0.15:
            return None

        trimmed = trim_to_best_plate_window(raw_text)

        if not re.match(r'^[A-Z0-9]{3,12}$', trimmed):
            return None

        return correct_indian_plate_format(trimmed)
    except Exception as e:
        print(f"[OCR ERROR] {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# PLATE QUALITY FILTER
# ─────────────────────────────────────────────────────────────────────────────

def is_plate_quality_ok(plate_crop, min_size=20, blur_threshold=8.0):
    h, w = plate_crop.shape[:2]
    if h < min_size or w < min_size:
        return False
    aspect = w / max(h, 1)
    if aspect < 0.7 or aspect > 6.0:
        return False
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY) if len(plate_crop.shape) == 3 else plate_crop
    return cv2.Laplacian(gray, cv2.CV_64F).var() >= blur_threshold


# ─────────────────────────────────────────────────────────────────────────────
# THREADING & CACHES
# ─────────────────────────────────────────────────────────────────────────────
match_lock            = threading.Lock()
matching_in_progress  = set()
match_semaphore       = threading.Semaphore(4)

plate_track_labels    = {}
track_frame_age       = {}
plate_vote_history    = {}
live_alerts           = []

MAX_OCR_ATTEMPTS = 12


def async_match_plate(track_id, plate_crop):
    if track_id in matching_in_progress:
        return
    matching_in_progress.add(track_id)

    def worker():
        with match_semaphore:
            read_text = match_plate(plate_crop)

        with match_lock:
            if read_text:
                history = plate_vote_history.setdefault(track_id, [])
                history.append(read_text)
                if len(history) > MAX_OCR_ATTEMPTS:
                    plate_vote_history[track_id] = history[-MAX_OCR_ATTEMPTS:]

                top_text, _ = Counter(plate_vote_history[track_id]).most_common(1)[0]
                candidates  = generate_series_candidates(top_text)
                db_plate, db_owner, sim = fuzzy_match_known_plate(candidates, known_plates)

                matched = f"{db_owner}, {db_plate}" if db_plate else f"Unknown, {top_text}"

                if plate_track_labels.get(track_id) != matched:
                    plate_track_labels[track_id] = matched
                    ts  = datetime.now().strftime("%H:%M:%S")
                    msg = f"VEHICLE DETECTED: {matched}"
                    if not any(a['id'] == f"plate_{track_id}" for a in live_alerts):
                        live_alerts.insert(0, {"time": ts, "msg": msg, "id": f"plate_{track_id}"})
                        if len(live_alerts) > 50:
                            live_alerts.pop()
                        print(f"[{ts}] ALERT: {msg}")

        matching_in_progress.discard(track_id)

    threading.Thread(target=worker, daemon=True).start()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN VIDEO LOOP
# ─────────────────────────────────────────────────────────────────────────────

def generate_frames():
    source = r"C:\Programming\SIH\VID_20260905_140217112.mp4"
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    vehicle_history = {}
    speed_cache     = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame.shape[1] > 1280:
            frame = cv2.resize(
                frame,
                (1280, int(frame.shape[0] * 1280 / frame.shape[1])),
                interpolation=cv2.INTER_AREA,
            )

        gray_check = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if np.mean(gray_check) < 60:
            frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=40)

        # ── VEHICLE DETECTION & SPEED ────────────────────────────────────────
        vehicle_results = vehicle_model.track(
            frame, persist=True, conf=0.15, imgsz=1280, verbose=False, device=0
        )[0]

        if len(vehicle_results.boxes) > 0 and vehicle_results.boxes.id is not None:
            v_boxes = vehicle_results.boxes.xyxy.cpu().numpy()
            v_confs = vehicle_results.boxes.conf.cpu().numpy()
            v_clss  = vehicle_results.boxes.cls.cpu().numpy().astype(int)
            v_ids   = vehicle_results.boxes.id.cpu().numpy().astype(int)
            now     = time.time()

            for v_box, v_conf, v_cls, v_id in zip(v_boxes, v_confs, v_clss, v_ids):
                if v_cls not in vehicle_classes:
                    continue

                v_box  = list(map(int, v_box))
                v_name = vehicle_classes[v_cls]

                cx = (v_box[0] + v_box[2]) // 2
                cy = v_box[3]

                if v_id not in vehicle_history:
                    vehicle_history[v_id] = []
                    speed_cache[v_id]     = 0.0

                vehicle_history[v_id].append((cx, cy, now))
                if len(vehicle_history[v_id]) > 15:
                    vehicle_history[v_id].pop(0)

                if len(vehicle_history[v_id]) >= 10:
                    old_cx, old_cy, old_time = vehicle_history[v_id][-10]
                    dist_px   = math.hypot(cx - old_cx, cy - old_cy)
                    time_diff = now - old_time

                    if time_diff > 0:
                        if dist_px < SPEED_DEADZONE_PX:
                            target_speed = 0.0
                        else:
                            dist_m       = dist_px * METERS_PER_PIXEL
                            speed_mps    = dist_m / time_diff
                            target_speed = min(speed_mps * 3.6, SPEED_MAX_KMH)

                        speed_cache[v_id] = (
                            SPEED_EMA_ALPHA * target_speed
                            + (1.0 - SPEED_EMA_ALPHA) * speed_cache[v_id]
                        )

                cv2.rectangle(frame, (v_box[0], v_box[1]), (v_box[2], v_box[3]), (255, 200, 0), 2)
                cv2.putText(
                    frame, f"{v_name} [{v_conf:.2f}]",
                    (v_box[0], max(v_box[1] - 10, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2,
                )

                spd = int(speed_cache[v_id])
                if spd > 0:
                    cv2.putText(
                        frame, f"{spd} KM/H",
                        (v_box[0], v_box[3] + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
                    )

        # ── PLATE DETECTION & OCR ────────────────────────────────────────────
        plate_results = plate_model.track(
            frame, persist=True, verbose=False, conf=0.45, imgsz=1280, device=0
        )[0]

        if len(plate_results.boxes) > 0 and plate_results.boxes.id is not None:
            p_boxes     = plate_results.boxes.xyxy.cpu().numpy()
            p_confs     = plate_results.boxes.conf.cpu().numpy()
            p_track_ids = plate_results.boxes.id.cpu().numpy().astype(int)

            for box, conf, track_id in zip(p_boxes, p_confs, p_track_ids):
                x1, y1, x2, y2 = map(int, box)
                plate_crop = frame[y1:y2, x1:x2]
                if plate_crop.size == 0:
                    continue

                if track_id not in plate_track_labels:
                    plate_track_labels[track_id] = "Scanning..."
                    track_frame_age[track_id]    = 0

                track_frame_age[track_id] += 1

                votes_so_far = len(plate_vote_history.get(track_id, []))
                if votes_so_far < MAX_OCR_ATTEMPTS and is_plate_quality_ok(plate_crop):
                    padded = pad_plate_crop(plate_crop, frame, x1, y1, x2, y2, pad_ratio=0.20)
                    async_match_plate(track_id, padded.copy())

                p_label  = plate_track_labels[track_id]
                is_known = "Scanning" not in p_label and "Unknown" not in p_label
                p_color  = (0, 255, 0) if is_known else (0, 165, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), p_color, 2)
                cv2.putText(frame, p_label, (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, p_color, 2)

        # ── ENCODE & YIELD ────────────────────────────────────────────────────
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()


if __name__ == '__main__':
    for _ in generate_frames():
        pass