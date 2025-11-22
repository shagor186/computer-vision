import os
import sqlite3
from flask import request, jsonify
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
UPLOAD_FOLDER = 'static/train'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def submit_form():
    person_id = request.form['id']
    name = request.form['name']
    age = request.form['age']
    location = request.form['location']
    files = request.files.getlist('images')

    # DB তে save বা update
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO persons (id, name, age, location) VALUES (?, ?, ?, ?)",
              (person_id, name, age, location))
    conn.commit()
    conn.close()

    # ID অনুযায়ী ফোল্ডার তৈরি
    person_folder = os.path.join(UPLOAD_FOLDER, person_id)
    if not os.path.exists(person_folder):
        os.makedirs(person_folder)

    # ইমেজ save
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(person_folder, filename))

    return jsonify({'message': 'Submitted successfully!'})
