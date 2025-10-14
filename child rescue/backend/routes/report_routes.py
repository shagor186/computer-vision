from flask import Blueprint, request, jsonify
from extensions import db
from models.person_model import Person
from utils.helpers import save_person_images

report_bp = Blueprint("report", __name__, url_prefix="/report")

@report_bp.route("/add", methods=["POST"])
def add_person():
    # data part
    data = request.form
    name = data.get("name")
    age = data.get("age")
    gender = data.get("gender")
    location = data.get("location")

    # create person in DB
    person = Person(name=name, age=age, gender=gender, location=location)
    db.session.add(person)
    db.session.commit()

    # handle multiple images upload
    files = request.files.getlist("images")
    saved_files = save_person_images(person.id, files)

    # save filenames to DB column
    person.images = ",".join(saved_files)
    db.session.commit()

    return jsonify({"msg":"Person added with images","person_id": person.id})
