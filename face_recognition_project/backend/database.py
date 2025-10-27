import sqlite3, os
DB_PATH = os.path.join(os.path.dirname(__file__), "instance", "face_rec.db")

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS persons (
        id TEXT PRIMARY KEY,
        name TEXT,
        age INTEGER,
        location TEXT
    )''')
    conn.commit()
    conn.close()

def add_person(person_id, name, age, location):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("INSERT OR REPLACE INTO persons VALUES (?, ?, ?, ?)", (person_id, name, age, location))
    conn.commit()
    conn.close()

def get_person(person_id):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT * FROM persons WHERE id=?", (person_id,))
    row = cur.fetchone()
    conn.close()
    return row
