import os
import sqlite3
import uuid
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SESSION_SECRET', 'dev-secret-key')
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def get_db_connection():
    conn = sqlite3.connect('images.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            user_id TEXT NOT NULL,
            age INTEGER NOT NULL,
            location TEXT NOT NULL,
            filename TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    if 'csrf_token' not in session:
        session['csrf_token'] = str(uuid.uuid4())
    conn = get_db_connection()
    images = conn.execute('SELECT * FROM images ORDER BY id DESC').fetchall()
    conn.close()
    return render_template('index.html', images=images)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)
        
        file = request.files['image']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
        
        if file and file.filename and allowed_file(file.filename):
            name = request.form.get('name', '').strip()
            user_id = request.form.get('user_id', '').strip()
            age = request.form.get('age', '').strip()
            location = request.form.get('location', '').strip()
            
            if not name or not user_id or not age or not location:
                flash('All fields are required', 'error')
                return redirect(request.url)
            
            try:
                age_int = int(age)
                if age_int < 0:
                    flash('Age must be a positive number', 'error')
                    return redirect(request.url)
            except ValueError:
                flash('Age must be a valid number', 'error')
                return redirect(request.url)
            
            filename = secure_filename(file.filename)
            ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
            unique_filename = f"{uuid.uuid4()}.{ext}"
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], unique_filename))
            
            conn = get_db_connection()
            conn.execute('INSERT INTO images (name, user_id, age, location, filename) VALUES (?, ?, ?, ?, ?)',
                        (name, user_id, age_int, location, unique_filename))
            conn.commit()
            conn.close()
            
            flash('Image uploaded successfully!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid file type. Allowed types: png, jpg, jpeg, gif, webp', 'error')
    
    return render_template('upload.html')

@app.route('/edit/<int:id>', methods=['GET', 'POST'])
def edit(id):
    conn = get_db_connection()
    image = conn.execute('SELECT * FROM images WHERE id = ?', (id,)).fetchone()
    
    if not image:
        conn.close()
        flash('Image not found', 'error')
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        user_id = request.form.get('user_id', '').strip()
        age = request.form.get('age', '').strip()
        location = request.form.get('location', '').strip()
        
        if not name or not user_id or not age or not location:
            flash('All fields are required', 'error')
            return render_template('edit.html', image=image)
        
        try:
            age_int = int(age)
            if age_int < 0:
                flash('Age must be a positive number', 'error')
                return render_template('edit.html', image=image)
        except ValueError:
            flash('Age must be a valid number', 'error')
            return render_template('edit.html', image=image)
        
        conn.execute('UPDATE images SET name = ?, user_id = ?, age = ?, location = ? WHERE id = ?',
                    (name, user_id, age_int, location, id))
        conn.commit()
        conn.close()
        
        flash('Image information updated successfully!', 'success')
        return redirect(url_for('index'))
    
    conn.close()
    return render_template('edit.html', image=image)

@app.route('/delete/<int:id>', methods=['POST'])
def delete(id):
    csrf_token = request.form.get('csrf_token')
    if csrf_token != session.get('csrf_token'):
        flash('Invalid request', 'error')
        return redirect(url_for('index'))
    
    conn = get_db_connection()
    image = conn.execute('SELECT * FROM images WHERE id = ?', (id,)).fetchone()
    
    if image:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], image['filename'])
        if os.path.exists(filepath):
            os.remove(filepath)
        
        conn.execute('DELETE FROM images WHERE id = ?', (id,))
        conn.commit()
        flash('Image deleted successfully!', 'success')
    else:
        flash('Image not found', 'error')
    
    conn.close()
    return redirect(url_for('index'))

if __name__ == '__main__':
    init_db()
    app.run(port=5000, debug=True)
