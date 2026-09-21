import cv2
import numpy as np
import mysql.connector
import json
from deepface import DeepFace
from datetime import datetime
import threading
from collections import defaultdict, Counter
from ultralytics import YOLO

face_model = YOLO("yolov8n-face-lindevs.pt")
face_model.to('cuda')
person_model = YOLO("yolov8s.pt")
person_model.to('cuda')


restrictedzone = np.array([[213, 160], [427, 160], [427, 320], [213, 320]])

def isinsidezone(center, zone):
    return cv2.pointPolygonTest(zone, center, False) >= 0

def load_known_faces():
    try:
        conn = mysql.connector.connect(host="", user="", password="", database="") # ADD YOUR OWN DATABASE HERE
        cursor = conn.cursor()
        cursor.execute("SELECT name, encoding FROM faces")
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        faces = defaultdict(list)
        for name, enc in rows:
            faces[name].append(np.array(json.loads(enc)))
        return dict(faces)
    except Exception as e:
        print(f"[DB ERROR] Could not load faces: {e}")
        return {}

known_faces = load_known_faces()

def cosine_distance(a, b):
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0: return 1.0
    return 1 - np.dot(a / norm_a, b / norm_b)

def pad_face_crop(crop, frame, x1, y1, x2, y2, pad_ratio=0.25):
    h, w = crop.shape[:2]
    pad_x, pad_y = int(w * pad_ratio), int(h * pad_ratio)
    fh, fw = frame.shape[:2]
    return frame[max(0, y1 - pad_y):min(fh, y2 + pad_y), max(0, x1 - pad_x):min(fw, x2 + pad_x)]

def is_lower_face_obscured(face_crop):
    h, w = face_crop.shape[:2]
    if h < 40 or w < 40: return False
    
    lower_half = face_crop[int(h * 0.5):h, :]
    gray_lower = cv2.cvtColor(lower_half, cv2.COLOR_BGR2GRAY)
    contrast_score = np.std(gray_lower)
    
    return contrast_score < 12.2

def preprocess_crop(crop):
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)

def match_face(face_crop):
    try:
        h, w = face_crop.shape[:2]
        if h < 40 or w < 40: return None, 0.0
        face_crop = preprocess_crop(face_crop)
        result = DeepFace.represent(face_crop, model_name="ArcFace", enforce_detection=False, detector_backend="opencv", align=True)
        if not result: return None, 0.0
        embedding = np.array(result[0]["embedding"])
        
        best_match, best_distance = "Unknown", float("inf")
        for name, embeddings_list in known_faces.items():
            for known_embedding in embeddings_list:
                distance = cosine_distance(embedding, known_embedding)
                if distance < best_distance:
                    best_distance, best_match = distance, name
        return (best_match if best_distance < 0.40 else "Unknown"), best_distance
    except Exception:
        return None, 0.0

# THREADING & CACHED OVERLAYS
match_lock = threading.Lock()
matching_in_progress = set()
match_semaphore = threading.Semaphore(4)

track_labels = {}
track_frame_count = defaultdict(int)
person_no_face_frames = {}
track_vote_history = defaultdict(list)
live_alerts = []
logged_unknown_appearances = set()

def async_match_face(track_id, person_crop):
    if track_id in matching_in_progress: return
    matching_in_progress.add(track_id)
    
    def worker():
        with match_semaphore:
            matched_name, distance = match_face(person_crop)
        with match_lock:
            if matched_name:
                track_vote_history[track_id].append(matched_name)
                if len(track_vote_history[track_id]) > 7:
                    track_vote_history[track_id] = track_vote_history[track_id][-7:]
                
                name_votes = [v for v in track_vote_history[track_id] if v != "Unknown"]
                if name_votes:
                    top_name, top_count = Counter(name_votes).most_common(1)[0]
                    if top_count >= 1:
                        track_labels[track_id] = top_name
                else:
                    # Confirm "Unknown" only after accumulated recognition attempts
                    if len(track_vote_history[track_id]) >= 2:
                        track_labels[track_id] = "Unknown"
            
            # Log single instance of unknown person appearance once confirmed
            if track_labels.get(track_id) == "Unknown" and track_id not in logged_unknown_appearances:
                logged_unknown_appearances.add(track_id)
                alert_msg = f"UNKNOWN SUBJECT DETECTED (ID {track_id})"
                timestamp = datetime.now().strftime("%H:%M:%S")
                if not any(a['id'] == f"unknown_{track_id}" for a in live_alerts):
                    live_alerts.insert(0, {"time": timestamp, "msg": alert_msg, "id": f"unknown_{track_id}"})
                    if len(live_alerts) > 50: live_alerts.pop()
                    print(f"[{timestamp}] ALERT: {alert_msg}")

        matching_in_progress.discard(track_id)
    threading.Thread(target=worker, daemon=True).start()

def generate_frames():
    source = 0 #THE WEBCAM SOURCE or ADD YOUR OWN SOURCE
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        if frame.shape[1] > 1280:
            frame = cv2.resize(frame, (1280, int(frame.shape[0] * (1280 / frame.shape[1]))), interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.polylines(frame, [restrictedzone], isClosed=True, color=[0, 0, 255], thickness=2)
        if np.mean(gray) < 60: frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=40)

        small_frame = cv2.resize(frame, (int(frame.shape[1] * 0.75), int(frame.shape[0] * 0.75)))
        scale_x, scale_y = frame.shape[1] / small_frame.shape[1], frame.shape[0] / small_frame.shape[0]

        # 1. TRACK PERSONS
        person_results = person_model.track(small_frame, persist=True, tracker="bytetrack.yaml", verbose=False, device=0)[0]
        person_data = []
        
        if person_results.boxes.id is not None:
            for box, p_tid in zip(person_results.boxes, person_results.boxes.id):
                if int(box.cls[0]) == 0 and float(box.conf[0]) > 0.4:
                    px1, py1, px2, py2 = map(int, box.xyxy[0])
                    person_data.append({
                        'id': int(p_tid), 
                        'box': (px1, py1, px2, py2),
                        'scaled_box': (int(px1*scale_x), int(py1*scale_y), int(px2*scale_x), int(py2*scale_y))
                    })
                    cv2.rectangle(frame, (int(px1*scale_x), int(py1*scale_y)), (int(px2*scale_x), int(py2*scale_y)), (255, 255, 0), 2)

        # 2. TRACK FACES
        face_results = face_model.track(small_frame, persist=True, tracker="bytetrack.yaml", verbose=False, device=0)[0]
        face_centers = []
        
        if face_results.boxes.id is not None:
            for box, tid in zip(face_results.boxes, face_results.boxes.id):
                if float(box.conf[0]) < 0.4: 
                    continue
                track_id = int(tid)
                fx1, fy1, fx2, fy2 = map(int, box.xyxy[0])
                
                face_centers.append(((fx1 + fx2) // 2, (fy1 + fy2) // 2))
                fx1, fy1, fx2, fy2 = int(fx1*scale_x), int(fy1*scale_y), int(fx2*scale_x), int(fy2*scale_y)
                
                # Initialize state to 'Scanning...' instead of 'Unknown'
                if track_id not in track_labels: 
                    track_labels[track_id] = "Scanning..."
                
                track_frame_count[track_id] += 1
                raw_crop = frame[fy1:fy2, fx1:fx2]
                
                # Check recognition status
                if track_labels[track_id] in ["Scanning...", "Unknown", "OBSCURED"]:
                    if raw_crop.size > 0 and is_lower_face_obscured(raw_crop):
                        track_labels[track_id] = "OBSCURED"
                    else:
                        if track_labels[track_id] == "OBSCURED":
                            track_labels[track_id] = "Scanning..." 
                        
                        # Request asynchronous matching if votes are still needed
                        if len(track_vote_history[track_id]) < 5:
                            async_match_face(track_id, pad_face_crop(raw_crop, frame, fx1, fy1, fx2, fy2, pad_ratio=0.25).copy())

                label = track_labels[track_id]
                cx, cy = int((fx1 + fx2) / 2), int((fy1 + fy2) / 2)
                in_zone = (
                    isinsidezone((cx, cy), restrictedzone) or    
                    isinsidezone((fx1, fy1), restrictedzone) or   
                    isinsidezone((fx2, fy1), restrictedzone) or 
                    isinsidezone((fx1, fy2), restrictedzone) or    
                    isinsidezone((fx2, fy2), restrictedzone)      
                )
                
                # Determine display status & colors
                if label == "OBSCURED":
                    status, color = ("MASK/COVER DETECTED", (0, 0, 255))
                    alert_msg = f"SECURITY ALERT: Obscured Face (ID {track_id})"
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    if not any(a['id'] == f"obscured_{track_id}" for a in live_alerts):
                        live_alerts.insert(0, {"time": timestamp, "msg": alert_msg, "id": f"obscured_{track_id}"})
                elif label == "Scanning...":
                    status, color = ("Verifying Identity...", (255, 255, 0))
                else:
                    color = (0, 165, 255) if label == "Unknown" else (0, 255, 0)
                    status = ""
                    if in_zone:
                        if label != "Unknown":
                            status, color = ("Access Granted", (0, 255, 0))
                        else:
                            status, color = ("INTRUSION DETECTED", (0, 0, 255))
                            alert_msg = f"SECURITY BREACH: Unauthorized Subject (ID {track_id})"
                            timestamp = datetime.now().strftime("%H:%M:%S")
                            if not any(a['id'] == track_id and a['time'] == timestamp for a in live_alerts):
                                live_alerts.insert(0, {"time": timestamp, "msg": alert_msg, "id": track_id})

                cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), color, 2)
                cv2.putText(frame, f"{label}", (fx1, fy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                if status:
                    cv2.putText(frame, status, (fx1, fy2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 3. CHECK OBSCURED / COVERED PERSONS
        for p in person_data:
            p_id = p['id']
            px1, py1, px2, py2 = p['box']
            spx1, spy1, spx2, spy2 = p['scaled_box']
            
            has_face = False
            for (fcx, fcy) in face_centers:
                if px1 <= fcx <= px2 and py1 <= fcy <= (py1 + int((py2 - py1) * 0.6)):
                    has_face = True
                    break
                    
            if not has_face:
                person_no_face_frames[p_id] = person_no_face_frames.get(p_id, 0) + 1
                if person_no_face_frames[p_id] == 15:
                    alert_msg = f"SUSPICIOUS: Obscured/Covered Face (Person ID {p_id})"
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    live_alerts.insert(0, {"time": timestamp, "msg": alert_msg, "id": f"obscured_{p_id}"})
                    print(f"[{timestamp}] ALERT: {alert_msg}")
                
                if person_no_face_frames[p_id] > 10:
                    cv2.putText(frame, "FACE OBSCURED", (spx1, spy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.rectangle(frame, (spx1, spy1), (spx2, spy2), (0, 0, 255), 2)
            else:
                person_no_face_frames[p_id] = 0

        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()

if __name__ == '__main__':
    generate_frames()
