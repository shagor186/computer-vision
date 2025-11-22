import os
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
from PIL import Image
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings("ignore")

# Configuration
TARGET_SIZE = (160, 160)
ALLOWED_EXT = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
SAVED_MODEL_PATH = 'svm_face_model.pkl'

class FaceDatasetLoader:
    def __init__(self, directory, target_size=TARGET_SIZE):
        self.directory = directory
        self.target_size = target_size
        self.detector = MTCNN()
        self.faces = []
        self.labels = []

    def extract_face(self, image_path):
        try:
            img = Image.open(image_path).convert("RGB")
            img_np = np.array(img)
            
            # Face detection
            results = self.detector.detect_faces(img_np)
            if not results:
                return None
                
            x, y, width, height = results[0]['box']
            x, y = abs(x), abs(y)
            x2, y2 = x + width, y + height
            
            face = img_np[y:y2, x:x2]
            if face.size == 0:
                return None
                
            face_resized = cv2.resize(face, self.target_size)
            return face_resized
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return None

    def load_dataset(self):
        for person_name in os.listdir(self.directory):
            person_dir = os.path.join(self.directory, person_name)
            if not os.path.isdir(person_dir):
                continue
                
            print(f"Loading faces for: {person_name}")
            count = 0
            
            for filename in os.listdir(person_dir):
                if filename.lower().endswith(ALLOWED_EXT):
                    image_path = os.path.join(person_dir, filename)
                    face = self.extract_face(image_path)
                    
                    if face is not None:
                        self.faces.append(face)
                        self.labels.append(person_name)
                        count += 1
            
            print(f"✅ Loaded {count} faces for {person_name}")
        
        return np.array(self.faces), np.array(self.labels)

# FaceNet embedder
embedder = FaceNet()

def get_embedding(face_image):
    image = face_image.astype('float32')
    image = np.expand_dims(image, axis=0)
    embedding = embedder.embeddings(image)
    return embedding[0]

def train_face_model(faces, labels):
    print("Generating face embeddings...")
    embeddings = np.array([get_embedding(face) for face in faces])
    
    # Label encoding
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
    )
    
    # Train SVM model
    model = SVC(kernel='linear', probability=True, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"✅ Model trained successfully!")
    print(f"📊 Test Accuracy: {accuracy:.4f}")
    print("\n📈 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))
    
    return model, encoder

def save_model(model, encoder, filepath=SAVED_MODEL_PATH):
    with open(filepath, 'wb') as f:
        pickle.dump({'model': model, 'encoder': encoder}, f)
    print(f"💾 Model saved to {filepath}")

def run_pipeline(dataset_path, plot_samples=True):
    """
    সম্পূর্ণ training pipeline run করে
    """
    print("🚀 Starting face recognition training...")
    
    # ডাটা লোড করো
    loader = FaceDatasetLoader(dataset_path)
    faces, labels = loader.load_dataset()
    
    if len(faces) == 0:
        print("❌ No faces found in dataset")
        return False
    
    print(f"📊 Loaded {len(faces)} faces for {len(np.unique(labels))} persons")
    
    # মডেল ট্রেন করো
    model, encoder = train_face_model(faces, labels)
    
    # মডেল সেভ করো
    save_model(model, encoder)
    
    return True