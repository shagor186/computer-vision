# from flask import Flask
# from flask_cors import CORS
# import sqlite3
# from upload_form import submit_form

# app = Flask(__name__)
# CORS(app)  # React থেকে request access করতে হবে
# app.config['UPLOAD_FOLDER'] = 'static/train'

# # SQLite ডাটাবেস তৈরি (main app এ)
# conn = sqlite3.connect('database.db')
# c = conn.cursor()
# c.execute('''
# CREATE TABLE IF NOT EXISTS persons (
#     id TEXT PRIMARY KEY,
#     name TEXT,
#     age INTEGER,
#     location TEXT
# )
# ''')
# conn.commit()
# conn.close()

# # Route
# app.add_url_rule('/submit', view_func=submit_form, methods=['POST'])

# if __name__ == '__main__':
#     app.run(port=5500, debug=True)



# webcam
# from flask import Flask, jsonify, request, send_from_directory
# from flask_cors import CORS
# from webcam import init_db, save_snapshot, SNAPSHOT_DIR

# app = Flask(__name__)
# CORS(app)

# init_db()  # database তৈরি

# @app.route('/')
# def home():
#     return "Flask Webcam API Running!"

# @app.route('/save_snapshot', methods=['POST'])
# def handle_save_snapshot():
#     data = request.json
#     image = data.get("image")
#     if not image:
#         return jsonify({"status": "error", "message": "No image provided"}), 400
    
#     result = save_snapshot(image)
#     return jsonify(result)

# @app.route('/snapshots/<filename>')
# def get_snapshot(filename):
#     return send_from_directory(SNAPSHOT_DIR, filename)

# if __name__ == "__main__":
#     app.run(port=5500, debug=True)





# app.py
# from flask import Flask
# from flask_cors import CORS
# from videoupload import video_bp

# app = Flask(__name__)
# CORS(app)

# # Blueprint রেজিস্টার
# app.register_blueprint(video_bp)

# @app.route('/')
# def home():
#     return "Flask Video Upload API Running with Blueprint!"

# if __name__ == "__main__":
#     app.run(port=5500, debug=True)





from flask import Flask, request, jsonify
import os, sqlite3
from werkzeug.utils import secure_filename
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Folder setup
UPLOAD_FOLDER = "static/images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Database setup
conn = sqlite3.connect('database.db')
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS people (
    id TEXT PRIMARY KEY,
    name TEXT,
    age INTEGER,
    location TEXT,
    image TEXT
)''')
conn.commit()
conn.close()

# ---------- API ROUTES ---------- #

@app.route('/upload', methods=['POST'])
def upload():
    data = request.form
    file = request.files['image']

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute("INSERT INTO people (id, name, age, location, image) VALUES (?, ?, ?, ?, ?)", 
                   (data['id'], data['name'], data['age'], data['location'], filename))
    conn.commit()
    conn.close()

    return jsonify({"message": "Form submitted successfully!"})

@app.route('/people', methods=['GET'])
def get_people():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM people")
    rows = cursor.fetchall()
    conn.close()
    
    people = []
    for r in rows:
        people.append({
            "id": r[0],
            "name": r[1],
            "age": r[2],
            "location": r[3],
            "image": r[4]
        })
    return jsonify(people)

if __name__ == '__main__':
    app.run(port=5500, debug=True)
