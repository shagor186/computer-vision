import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
import pickle
from PIL import Image
import os

# Global variables
detector = MTCNN()
embedder = FaceNet()
TARGET_SIZE = (160, 160)
MODEL_PATH = 'svm_face_model.pkl'

# Load model once
def load_face_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            data = pickle.load(f)
        return data['model'], data['encoder']
    return None, None

model, encoder = load_face_model()

def get_face_embedding(face_image):
    image = face_image.astype("float32")
    image = np.expand_dims(image, axis=0)
    embedding = embedder.embeddings(image)
    return embedding[0]

def predict_image(image_path):
    """
    Single image থেকে prediction করে
    """
    global model, encoder
    
    if model is None:
        print("❌ Model not loaded")
        return None
    
    try:
        # Image load করো
        img = Image.open(image_path).convert("RGB")
        img_np = np.array(img)
        
        # Face detection
        results = detector.detect_faces(img_np)
        if not results:
            print("❌ No face detected")
            return None
        
        # প্রথম face নাও
        x, y, width, height = results[0]['box']
        x, y = abs(x), abs(y)
        x2, y2 = x + width, y + height
        
        face = img_np[y:y2, x:x2]
        if face.size == 0:
            return None
        
        # Face resize এবং embedding করো
        face_resized = cv2.resize(face, TARGET_SIZE)
        embedding = get_face_embedding(face_resized).reshape(1, -1)
        
        # Prediction করো
        prediction = model.predict(embedding)
        name = encoder.inverse_transform(prediction)[0]
        confidence = np.max(model.predict_proba(embedding))
        
        print(f"✅ Predicted: {name} (Confidence: {confidence:.3f})")
        return name, confidence
        
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        return None

def predict_webcam_frame(frame):
    """
    Webcam frame থেকে real-time prediction করে
    """
    global model, encoder
    
    if model is None:
        return frame, []
    
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(rgb_frame)
        
        predictions = []
        
        for result in results:
            x, y, width, height = result['box']
            x, y = abs(x), abs(y)
            x2, y2 = x + width, y + height
            
            face = rgb_frame[y:y2, x:x2]
            if face.size == 0:
                continue
            
            try:
                face_resized = cv2.resize(face, TARGET_SIZE)
                embedding = get_face_embedding(face_resized).reshape(1, -1)
                
                prediction = model.predict(embedding)
                name = encoder.inverse_transform(prediction)[0]
                confidence = np.max(model.predict_proba(embedding))
                
                # Draw rectangle and label
                cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
                label = f"{name} ({confidence:.2f})"
                cv2.putText(frame, label, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                predictions.append({
                    'name': name,
                    'confidence': float(confidence),
                    'bbox': [x, y, x2, y2]
                })
                
            except Exception as e:
                continue
                
        return frame, predictions
        
    except Exception as e:
        print(f"Webcam prediction error: {str(e)}")
        return frame, []