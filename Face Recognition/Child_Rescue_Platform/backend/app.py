# from flask import Flask, request, jsonify
# from flask_sqlalchemy import SQLAlchemy
# from flask_cors import CORS
# from flask_mail import Mail, Message
# from itsdangerous import URLSafeTimedSerializer
# from werkzeug.security import generate_password_hash, check_password_hash
# import os
# import re
# import jwt
# import datetime
# from functools import wraps

# app = Flask(__name__)
# CORS(app)

# # Database Configuration
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# app.config['SECRET_KEY'] = 'your-secret-key-2024'
# app.config['JWT_SECRET_KEY'] = 'your-jwt-secret-key-2024'

# # Email Configuration for Password Reset
# app.config['MAIL_SERVER'] = 'smtp.gmail.com'
# app.config['MAIL_PORT'] = 587
# app.config['MAIL_USE_TLS'] = True
# app.config['MAIL_USERNAME'] = 'your-email@gmail.com'
# app.config['MAIL_PASSWORD'] = 'your-app-password'
# app.config['MAIL_DEFAULT_SENDER'] = 'your-email@gmail.com'

# db = SQLAlchemy(app)
# mail = Mail(app)
# serializer = URLSafeTimedSerializer(app.config['SECRET_KEY'])

# # User Model
# class User(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     email = db.Column(db.String(120), unique=True, nullable=False)
#     password = db.Column(db.String(200), nullable=False)
#     created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

#     def to_dict(self):
#         return {
#             'id': self.id,
#             'email': self.email,
#             'created_at': self.created_at.isoformat()
#         }

# # Email validation function
# def is_valid_email(email):
#     pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
#     return re.match(pattern, email) is not None

# # JWT Token Required Decorator
# def token_required(f):
#     @wraps(f)
#     def decorated(*args, **kwargs):
#         token = request.headers.get('Authorization')
        
#         if not token:
#             return jsonify({'error': 'Token is missing'}), 401
        
#         try:
#             if token.startswith('Bearer '):
#                 token = token[7:]
#             data = jwt.decode(token, app.config['JWT_SECRET_KEY'], algorithms=['HS256'])
#             current_user = User.query.get(data['user_id'])
            
#             if not current_user:
#                 return jsonify({'error': 'User not found'}), 401
                
#         except jwt.ExpiredSignatureError:
#             return jsonify({'error': 'Token has expired'}), 401
#         except jwt.InvalidTokenError:
#             return jsonify({'error': 'Invalid token'}), 401
#         except Exception as e:
#             return jsonify({'error': 'Token verification failed'}), 401
        
#         return f(current_user, *args, **kwargs)
    
#     return decorated

# # Generate JWT Token
# def generate_token(user_id, email):
#     token = jwt.encode({
#         'user_id': user_id,
#         'email': email,
#         'exp': datetime.datetime.utcnow() + datetime.timedelta(days=7)
#     }, app.config['JWT_SECRET_KEY'], algorithm='HS256')
    
#     return token

# # Routes

# @app.route('/api/health', methods=['GET'])
# def health_check():
#     return jsonify({'message': 'Server is running', 'status': 'healthy'}), 200

# @app.route('/api/signup', methods=['POST'])
# def signup():
#     try:
#         data = request.get_json()
        
#         if not data:
#             return jsonify({'error': 'No data provided'}), 400
            
#         email = data.get('email', '').strip().lower()
#         password = data.get('password', '')

#         # Validation
#         if not email or not password:
#             return jsonify({'error': 'Email and password are required'}), 400
        
#         if not is_valid_email(email):
#             return jsonify({'error': 'Invalid email format'}), 400
        
#         if len(password) < 6:
#             return jsonify({'error': 'Password must be at least 6 characters'}), 400

#         # Check if user already exists
#         existing_user = User.query.filter_by(email=email).first()
#         if existing_user:
#             return jsonify({'error': 'User with this email already exists'}), 400

#         # Create new user
#         hashed_password = generate_password_hash(password)
#         new_user = User(email=email, password=hashed_password)
        
#         db.session.add(new_user)
#         db.session.commit()

#         # Generate token for auto-login
#         token = generate_token(new_user.id, new_user.email)

#         return jsonify({
#             'message': 'User created successfully', 
#             'user': new_user.to_dict(),
#             'token': token
#         }), 201

#     except Exception as e:
#         db.session.rollback()
#         return jsonify({'error': 'Internal server error'}), 500

# @app.route('/api/signin', methods=['POST'])
# def signin():
#     try:
#         data = request.get_json()
        
#         if not data:
#             return jsonify({'error': 'No data provided'}), 400
            
#         email = data.get('email', '').strip().lower()
#         password = data.get('password', '')

#         if not email or not password:
#             return jsonify({'error': 'Email and password are required'}), 400

#         user = User.query.filter_by(email=email).first()

#         if user and check_password_hash(user.password, password):
#             # Generate JWT token
#             token = generate_token(user.id, user.email)
            
#             return jsonify({
#                 'message': 'Login successful', 
#                 'user': user.to_dict(),
#                 'token': token
#             }), 200
#         else:
#             return jsonify({'error': 'Invalid email or password'}), 401

#     except Exception as e:
#         return jsonify({'error': 'Internal server error'}), 500

# @app.route('/api/forgot-password', methods=['POST'])
# def forgot_password():
#     try:
#         data = request.get_json()
        
#         if not data:
#             return jsonify({'error': 'No data provided'}), 400
            
#         email = data.get('email', '').strip().lower()

#         if not email:
#             return jsonify({'error': 'Email is required'}), 400

#         user = User.query.filter_by(email=email).first()
#         if not user:
#             # Don't reveal if email exists or not for security
#             return jsonify({'message': 'If the email exists, a reset link has been sent'}), 200

#         # Generate reset token
#         token = serializer.dumps(email, salt='password-reset-salt')
        
#         # Create reset link (in production, use your frontend URL)
#         reset_link = f"http://localhost:3000/reset-password/{token}"
        
#         # Send email (commented for now - configure email first)
#         try:
#             msg = Message(
#                 'Password Reset Request - Auth System',
#                 recipients=[email],
#                 html=f'''
#                 <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
#                     <h2 style="color: #2563eb;">Password Reset Request</h2>
#                     <p>Hello,</p>
#                     <p>You requested to reset your password. Click the button below to reset it:</p>
#                     <div style="text-align: center; margin: 30px 0;">
#                         <a href="{reset_link}" style="background-color: #2563eb; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px; display: inline-block;">
#                             Reset Password
#                         </a>
#                     </div>
#                     <p>If you didn't request this, please ignore this email.</p>
#                     <p>This link will expire in 1 hour.</p>
#                     <hr style="margin: 20px 0;">
#                     <p style="color: #6b7280; font-size: 14px;">
#                         Auth System Team
#                     </p>
#                 </div>
#                 '''
#             )
#             mail.send(msg)
#         except Exception as email_error:
#             print(f"Email error: {email_error}")
#             # Don't fail the request if email fails
#             pass

#         return jsonify({'message': 'If the email exists, a reset link has been sent'}), 200

#     except Exception as e:
#         return jsonify({'error': 'Internal server error'}), 500

# @app.route('/api/reset-password/<token>', methods=['POST'])
# def reset_password(token):
#     try:
#         data = request.get_json()
        
#         if not data:
#             return jsonify({'error': 'No data provided'}), 400
            
#         new_password = data.get('password', '')

#         if not new_password:
#             return jsonify({'error': 'Password is required'}), 400

#         if len(new_password) < 6:
#             return jsonify({'error': 'Password must be at least 6 characters'}), 400

#         # Verify token
#         email = serializer.loads(token, salt='password-reset-salt', max_age=3600)  # 1 hour expiry
        
#         user = User.query.filter_by(email=email).first()
#         if not user:
#             return jsonify({'error': 'Invalid token'}), 400

#         # Update password
#         user.password = generate_password_hash(new_password)
#         db.session.commit()

#         return jsonify({'message': 'Password updated successfully'}), 200

#     except Exception as e:
#         return jsonify({'error': 'Invalid or expired token'}), 400

# @app.route('/api/verify-token', methods=['POST'])
# @token_required
# def verify_token(current_user):
#     return jsonify({
#         'message': 'Token is valid',
#         'user': current_user.to_dict()
#     }), 200

# @app.route('/api/user/profile', methods=['GET'])
# @token_required
# def get_user_profile(current_user):
#     return jsonify({
#         'user': current_user.to_dict()
#     }), 200

# # Create tables
# with app.app_context():
#     db.create_all()
#     print("Database tables created successfully!")

# if __name__ == '__main__':
#     app.run(debug=True, port=5000)





# ..............................................................


from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from face_trainer import run_pipeline, load_model_and_encoder
from face_predictor import predict_image, predict_webcam_frame, predict_video_frame

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
DATASET_FOLDER = 'dataset'
MODEL_PATH = 'svm_face_model.pkl'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATASET_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return jsonify({"message": "Face Recognition API Running", "status": "success"})

@app.route('/api/train-model', methods=['POST'])
def train_model():
    """
    ট্রেনিং ডাটাসেট থেকে মডেল ট্রেন করে
    """
    try:
        train_dir = os.path.join(DATASET_FOLDER, 'train')
        
        if not os.path.exists(train_dir):
            return jsonify({
                "status": "error", 
                "message": "Training directory not found"
            }), 400
        
        # মডেল ট্রেনিং
        result = run_pipeline(train_dir, plot_samples=False)
        
        if result:
            return jsonify({
                "status": "success",
                "message": "Model trained successfully"
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Model training failed"
            }), 500
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Training error: {str(e)}"
        }), 500

@app.route('/api/add-person', methods=['POST'])
def add_person():
    """
    নতুন ব্যক্তির ছবি আপলোড করে ডাটাসেটে যোগ করে
    """
    try:
        if 'personName' not in request.form:
            return jsonify({
                "status": "error",
                "message": "Person name is required"
            }), 400
            
        if 'images' not in request.files:
            return jsonify({
                "status": "error", 
                "message": "No images uploaded"
            }), 400

        person_name = request.form['personName']
        images = request.files.getlist('images')
        
        # ব্যক্তির ফোল্ডার তৈরি
        person_dir = os.path.join(DATASET_FOLDER, 'train', person_name)
        os.makedirs(person_dir, exist_ok=True)
        
        saved_count = 0
        for image in images:
            if image.filename:
                # ফাইল সেভ করো
                filename = f"{person_name}_{saved_count + 1}.jpg"
                filepath = os.path.join(person_dir, filename)
                image.save(filepath)
                saved_count += 1
        
        return jsonify({
            "status": "success",
            "message": f"Added {saved_count} images for {person_name}"
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error adding person: {str(e)}"
        }), 500

@app.route('/api/predict-image', methods=['POST'])
def predict_image_api():
    """
    ছবি থেকে face recognition করে
    """
    try:
        if 'image' not in request.files:
            return jsonify({
                "status": "error",
                "message": "No image file"
            }), 400
            
        image_file = request.files['image']
        
        if image_file.filename == '':
            return jsonify({
                "status": "error", 
                "message": "No selected file"
            }), 400
        
        # আপলোড করা ছবি সেভ করো
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
        image_file.save(filepath)
        
        # prediction করো
        result = predict_image(filepath)
        
        if result:
            name, confidence = result
            return jsonify({
                "status": "success",
                "prediction": name,
                "confidence": float(confidence)
            })
        else:
            return jsonify({
                "status": "error",
                "message": "No face detected"
            }), 400
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Prediction error: {str(e)}"
        }), 500

@app.route('/api/model-status', methods=['GET'])
def model_status():
    """
    মডেলের status check করে
    """
    try:
        if os.path.exists(MODEL_PATH):
            return jsonify({
                "status": "success",
                "model_exists": True,
                "message": "Model is ready"
            })
        else:
            return jsonify({
                "status": "success", 
                "model_exists": False,
                "message": "Model not trained yet"
            })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == '__main__':
    print("🚀 Starting Face Recognition Server...")
    print("📁 Dataset folder:", DATASET_FOLDER)
    print("📁 Upload folder:", UPLOAD_FOLDER)
    app.run(debug=True, host='0.0.0.0', port=5000)