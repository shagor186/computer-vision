from flask import Blueprint, request, jsonify

face_bp = Blueprint("face", __name__, url_prefix="/face")

@face_bp.route("/recognize", methods=["POST"])
def recognize_face():
    # Placeholder for face recognition logic
    return jsonify({"matches": ["person1", "person2"]})
