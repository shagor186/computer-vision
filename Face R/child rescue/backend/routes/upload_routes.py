import os
from flask import Blueprint, request, current_app
from werkzeug.utils import secure_filename

upload_bp = Blueprint("upload", __name__, url_prefix="/upload")

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in {"png", "jpg", "jpeg", "mp4"}

@upload_bp.route("/file", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return {"msg": "No file part"}, 400
    file = request.files["file"]
    if file.filename == "":
        return {"msg": "No selected file"}, 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        upload_folder = os.path.join(current_app.root_path, "static/uploads/images")
        os.makedirs(upload_folder, exist_ok=True)
        file.save(os.path.join(upload_folder, filename))
        return {"msg": "File uploaded"}, 200
