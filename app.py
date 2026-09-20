from cryptography.fernet import Fernet
import os

from flask import Flask, render_template, request, redirect, url_for, session, Response, jsonify
from facematching import generate_frames as generate_face_frames, live_alerts as face_alerts
from platematch import generate_frames as generate_plate_frames, live_alerts as plate_alerts

app = Flask(__name__)
app.secret_key = 'ibvap_tactical_key_2026'

if not os.path.exists("master.key"):
    with open("master.key", "wb") as key_file:
        key_file.write(Fernet.generate_key())

with open("master.key", "rb") as key_file:
    cipher_suite = Fernet(key_file.read())

def secure_log_event(event_string):
    """Encrypts an alert and writes it to the local disk"""
    encrypted_text = cipher_suite.encrypt(event_string.encode('utf-8'))
    with open("encrypted_surveillance.log", "ab") as log_file:
        log_file.write(encrypted_text + b"\n")

@app.route('/')
def index():
    if 'logged_in' in session: return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    valid_users = ['OF-0012', 'OF-0013', 'OF-0014', 'OF-0015', 'OF-0016']
    if request.form['username'] in valid_users and request.form['password'] == 'ibvap123':
        session['logged_in'] = True
        return redirect(url_for('dashboard'))
    return "ACCESS DENIED: INVALID CLEARANCE KEY", 401

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    if 'logged_in' not in session: return redirect(url_for('index'))
    return render_template('dashboard.html')

@app.route('/stream/faces')
def stream_faces():
    if 'logged_in' not in session: return "Unauthorized", 401
    return Response(generate_face_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stream/plates')
def stream_plates():
    if 'logged_in' not in session: return "Unauthorized", 401
    return Response(generate_plate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_alerts')
def get_alerts():
    if 'logged_in' not in session: return jsonify([])
    
    combined_alerts = face_alerts + plate_alerts
    combined_alerts.sort(key=lambda x: x['time'], reverse=True)
    if combined_alerts:
        latest = combined_alerts[0]
        log_msg = f"[{latest['time']}] {latest['msg']}"
      
        secure_log_event(log_msg) 
    
    return jsonify(combined_alerts[:50])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, ssl_context='adhoc')