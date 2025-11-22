from flask import Flask, render_template, request, redirect, url_for, flash
import sqlite3
import os
import dlib
import cv2
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "your_secret_key"
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load Dlib models (download .dat files beforehand)
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")


# ---------------------- Database Init ----------------------
def init_db():
    conn = sqlite3.connect("attendance.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS students (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id TEXT,
                    name TEXT,
                    age INTEGER,
                    location TEXT,
                    image_path TEXT,
                    face_encoding BLOB
                )''')
    conn.commit()
    conn.close()


# ---------------------- Helper: Encode Face ----------------------
def encode_face(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dets = detector(img_rgb)
    if len(dets) == 0:
        return None
    shape = sp(img_rgb, dets[0])
    face_descriptor = facerec.compute_face_descriptor(img_rgb, shape)
    return np.array(face_descriptor, dtype=np.float32)


# ---------------------- Register Student ----------------------
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        student_id = request.form['student_id']
        name = request.form['name']
        age = request.form['age']
        location = request.form['location']
        file = request.files['image']

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            encoding = encode_face(filepath)
            if encoding is None:
                flash("❌ No face detected in image. Try another photo.", "danger")
                os.remove(filepath)
                return redirect(url_for('register'))

            conn = sqlite3.connect("attendance.db")
            c = conn.cursor()
            c.execute("INSERT INTO students (student_id, name, age, location, image_path, face_encoding) VALUES (?, ?, ?, ?, ?, ?)",
                      (student_id, name, age, location, filepath, encoding.tobytes()))
            conn.commit()
            conn.close()

            flash("✅ Student registered successfully!", "success")
            return redirect(url_for('register'))

    return render_template('register.html')


# ---------------------- View Students ----------------------
@app.route('/students')
def students():
    conn = sqlite3.connect("attendance.db")
    c = conn.cursor()
    c.execute("SELECT id, student_id, name, age, location, image_path FROM students")
    data = c.fetchall()
    conn.close()
    return render_template('students.html', data=data)


# ---------------------- Attendance (Face Recognition) ----------------------
@app.route('/attendance')
def attendance():
    conn = sqlite3.connect("attendance.db")
    c = conn.cursor()
    c.execute("SELECT name, face_encoding FROM students")
    data = c.fetchall()
    conn.close()

    known_names = []
    known_encodings = []

    for name, enc in data:
        known_names.append(name)
        known_encodings.append(np.frombuffer(enc, dtype=np.float32))

    cap = cv2.VideoCapture(0)
    recognized = set()

    flash("Press 'Q' to stop camera after recognition starts!", "info")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        dets = detector(rgb)

        for det in dets:
            shape = sp(rgb, det)
            face_desc = facerec.compute_face_descriptor(rgb, shape)
            face_enc = np.array(face_desc, dtype=np.float32)

            distances = np.linalg.norm(known_encodings - face_enc, axis=1)
            min_distance = np.min(distances)

            if min_distance < 0.6:
                idx = np.argmin(distances)
                name = known_names[idx]
                recognized.add(name)
                color = (0, 255, 0)
                label = name
            else:
                color = (0, 0, 255)
                label = "Unknown"

            cv2.rectangle(frame, (det.left(), det.top()), (det.right(), det.bottom()), color, 2)
            cv2.putText(frame, label, (det.left(), det.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Attendance System - Press Q to Exit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return render_template("attendance.html", recognized=recognized)


# ---------------------- Main ----------------------
if __name__ == '__main__':
    if not os.path.exists('attendance.db'):
        init_db()
    os.makedirs('static/uploads', exist_ok=True)
    app.run(debug=True)
