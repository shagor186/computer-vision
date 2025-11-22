import os
import base64
import sqlite3
from datetime import datetime

DB_NAME = 'database.db'
SNAPSHOT_DIR = 'static/snapshots'
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            filename TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_snapshot(image_base64):
    try:
        img_data = image_base64.split(',')[1]
        img_bytes = base64.b64decode(img_data)
        filename = datetime.now().strftime("%Y%m%d%H%M%S") + ".png"
        filepath = os.path.join(SNAPSHOT_DIR, filename)
        
        with open(filepath, 'wb') as f:
            f.write(img_bytes)
        
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("INSERT INTO snapshots (timestamp, filename) VALUES (?, ?)",
                  (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), filename))
        conn.commit()
        conn.close()
        
        return {"status": "success", "filename": filename}
    except Exception as e:
        return {"status": "error", "message": str(e)}
