import os
from werkzeug.utils import secure_filename

def allowed_file(filename, allowed_extensions={"png","jpg","jpeg"}):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in allowed_extensions

def save_person_images(person_id, files, base_path="static/train_data"):
    person_folder = os.path.join(base_path, str(person_id))
    os.makedirs(person_folder, exist_ok=True)
    saved_files = []
    for file in files:
        filename = secure_filename(file.filename)
        file.save(os.path.join(person_folder, filename))
        saved_files.append(filename)
    return saved_files
