# videoupload.py
from flask import Blueprint, request, jsonify, send_from_directory
import os
import sqlite3
from datetime import datetime
from werkzeug.utils import secure_filename

# Blueprint তৈরি
video_bp = Blueprint('video_bp', __name__)

UPLOAD_FOLDER = 'static/videos'
DB_NAME = 'database.db'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Database initialize
def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS videos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            upload_time TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# ফাইল অনুমতি যাচাই
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# 📤 Upload route
@video_bp.route('/upload_videos', methods=['POST'])
def upload_videos():
    if 'videos' not in request.files:
        return jsonify({'status': 'error', 'message': 'No videos found'}), 400

    files = request.files.getlist('videos')
    saved_files = []

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            conn = sqlite3.connect(DB_NAME)
            c = conn.cursor()
            c.execute("INSERT INTO videos (filename, upload_time) VALUES (?, ?)",
                      (filename, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            conn.commit()
            conn.close()

            saved_files.append(filename)

    return jsonify({'status': 'success', 'files': saved_files})


# 🎥 ভিডিও সার্ভ করার রুট
@video_bp.route('/videos/<filename>')
def serve_video(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)
