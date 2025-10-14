import sqlite3
import pandas as pd

# Database file (auto-created in project folder)
DB_FILE = "face_recognition.db"


def create_database():
    """Create SQLite database and persons table"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS persons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            age INTEGER,
            location TEXT
        )
    ''')
    conn.commit()
    conn.close()


def insert_person(name, age, location):
    """Insert a new person"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO persons (name, age, location) VALUES (?, ?, ?)", (name, age, location))
        conn.commit()
        print(f"Inserted: {name}, Age {age}, Location {location}")
    except sqlite3.IntegrityError:
        print(f"Person '{name}' already exists.")
    conn.close()


def fetch_all_persons():
    """Return all persons as Pandas DataFrame"""
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM persons", conn)
    conn.close()
    return df


if __name__ == "__main__":
    create_database()

    insert_person("ronaldo", 40, "Funchal, Portugal")
    insert_person("sinner", 24, "San Candido, Italy")
    insert_person("cummins", 32, "Westmead, Australia")
    insert_person("shelton", 22, "Georgia, United States")

    print(fetch_all_persons())