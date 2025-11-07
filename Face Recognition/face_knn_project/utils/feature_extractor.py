import os
import cv2
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# CPU-only mode
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
print("Using device:", "GPU" if tf.config.list_physical_devices('GPU') else "CPU")

# MobileNetV2 base model
base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

def extract_features(img_array, model=base_model, target_size=(224, 224)):
    """
    Image array থেকে 1280-dimension feature বের করবে
    """
    img = cv2.resize(img_array, target_size)
    x = np.expand_dims(img, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    return features.flatten()

def prepare_dataset(train_dir):
    """
    Train folder থেকে সব ছবি feature + label তৈরি করবে
    """
    X, y = [], []
    for person_name in os.listdir(train_dir):
        person_path = os.path.join(train_dir, person_name)
        if not os.path.isdir(person_path):
            continue
        for img_file in os.listdir(person_path):
            img_path = os.path.join(person_path, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            features = extract_features(img_rgb)
            X.append(features)
            y.append(person_name)  # folder name as label: "1", "2", etc.
    return np.array(X), y

def save_knn_model(model, path="models/knn_model.clf"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)

def load_knn_model(path="models/knn_model.clf"):
    with open(path, "rb") as f:
        return pickle.load(f)
