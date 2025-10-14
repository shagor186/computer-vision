from flask import Flask, jsonify
from flask_cors import CORS
from config import Config
from extensions import db, jwt

app = Flask(__name__)
app.config.from_object(Config)

CORS(app)
db.init_app(app)
jwt.init_app(app)

@app.route('/')
def home():
    return jsonify({"message": "Flask Backend Running"})

# Import Blueprints after db init
from routes.auth_routes import auth_bp
from routes.report_routes import report_bp
from routes.face_routes import face_bp
from routes.upload_routes import upload_bp

app.register_blueprint(auth_bp)
app.register_blueprint(report_bp)
app.register_blueprint(face_bp)
app.register_blueprint(upload_bp)

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
