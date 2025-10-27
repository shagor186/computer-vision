import sqlite3

conn = sqlite3.connect('images.db')
cursor = conn.cursor()

cursor.execute("SELECT id, name, user_id, age, location, filename FROM images")
rows = cursor.fetchall()

if rows:
    for r in rows:
        print(r)
else:
    print("No records found.")

conn.close()
