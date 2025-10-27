import os, base64
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO
from werkzeug.utils import secure_filename
from face_knn import train, predict, predict_image_from_array
from database import init_db, add_person, get_person
import cv2
import numpy as np

BASE = os.path.dirname(__file__)
UPLOAD_DIR = os.path.join(BASE, "uploads")
TRAIN_DIR = os.path.join(BASE, "train")
TEST_DIR = os.path.join(BASE, "test")
MODEL_PATH = os.path.join(BASE, "trained_knn_model.clf")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")
init_db()

@app.route("/upload", methods=["POST"])
def upload_images():
    name = request.form.get("name")
    person_id = request.form.get("id")
    age = request.form.get("age")
    location = request.form.get("location")
    files = request.files.getlist("images")
    if not person_id or not name:
        return jsonify({"error":"Missing id or name"}), 400
    folder_name = f"{person_id}_{name}"
    person_folder = os.path.join(TRAIN_DIR, folder_name)
    os.makedirs(person_folder, exist_ok=True)
    for f in files:
        filename = secure_filename(f.filename)
        f.save(os.path.join(person_folder, filename))
    add_person(person_id, name, age or None, location or None)
    try: train(TRAIN_DIR, MODEL_PATH)
    except Exception as e: return jsonify({"status":"partial","message":f"Uploaded but training failed: {e}"}),200
    socketio.emit("new_person", {"id": person_id,"name":name,"age":age,"location":location,"message":"Person added and model trained"})
    return jsonify({"status":"success","message":"Person added and model trained."}),200

@app.route("/predict", methods=["POST"])
def predict_image():
    if 'image' not in request.files:
        return jsonify({"error":"No file"}),400
    file = request.files['image']
    filename = secure_filename(file.filename)
    path = os.path.join(TEST_DIR, filename)
    file.save(path)
    results = predict(path, MODEL_PATH)
    if not results: return jsonify({"status":"no_face"}),200
    pid,_ = results[0]
    if pid=="unknown": return jsonify({"status":"unknown"}),200
    person_id = str(pid).split("_")[0]
    row = get_person(person_id)
    if row:
        person = {"id":row[0],"name":row[1],"age":row[2],"location":row[3]}
        socketio.emit("person_found", {"id": person["id"], "name": person["name"], "age": person["age"], "location": person["location"], "message": f"{person['name']} detected!"})
        return jsonify({"status":"found","person":person}),200
    return jsonify({"status":"not_found"}),200

@app.route("/predict_video", methods=["POST"])
def predict_video():
    if 'video' not in request.files: return jsonify({"error":"No video"}),400
    file = request.files['video']
    video_path = os.path.join(TEST_DIR, secure_filename(file.filename))
    file.save(video_path)
    cap = cv2.VideoCapture(video_path)
    results_summary=[]
    while True:
        ret, frame = cap.read()
        if not ret: break
        try:
            faces = predict_image_from_array(frame, MODEL_PATH)
            for pid,_ in faces:
                if pid != "unknown":
                    person_id = str(pid).split("_")[0]
                    row = get_person(person_id)
                    if row:
                        person = {"id": row[0], "name": row[1], "age": row[2], "location": row[3]}
                        socketio.emit("person_found", {"id": person["id"], "name": person["name"], "age": person["age"], "location": person["location"], "message": f"{person['name']} detected!"})
                        results_summary.append(person)
        except: continue
    cap.release()
    return jsonify({"status":"done","found":results_summary})

@socketio.on("webcam_frame")
def handle_webcam_frame(data):
    img_bytes = base64.b64decode(data.split(",")[1])
    npimg = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    try:
        faces = predict_image_from_array(frame, MODEL_PATH)
        for pid,_ in faces:
            if pid != "unknown":
                person_id = str(pid).split("_")[0]
                row = get_person(person_id)
                if row:
                    person = {"id": row[0], "name": row[1], "age": row[2], "location": row[3]}
                    socketio.emit("person_found", {"id": person["id"], "name": person["name"], "age": person["age"], "location": person["location"], "message": f"{person['name']} detected!"})
    except: pass

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
