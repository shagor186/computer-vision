import os, math, pickle
from PIL import Image, ImageDraw
from sklearn import neighbors
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

def train(train_dir, model_save_path="trained_knn_model.clf", n_neighbors=None, knn_algo='ball_tree'):
    X, y = [], []
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)
            if len(face_bounding_boxes) != 1:
                print(f"Skipping {img_path}: found {len(face_bounding_boxes)} faces.")
                continue
            X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
            y.append(class_dir)
    if not X:
        raise ValueError("No training data found in train_dir.")
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)
    with open(model_save_path, 'wb') as f:
        pickle.dump(knn_clf, f)
    print(f"Model trained and saved to {model_save_path}")
    return knn_clf

def predict(image_path, model_path="trained_knn_model.clf", distance_threshold=0.6):
    if not os.path.isfile(model_path):
        raise FileNotFoundError("Model file not found: " + model_path)
    with open(model_path, 'rb') as f:
        knn_clf = pickle.load(f)
    X_img = face_recognition.load_image_file(image_path)
    X_face_locations = face_recognition.face_locations(X_img)
    if len(X_face_locations) == 0:
        return []
    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]
    return [
        (pred, loc) if rec else ("unknown", loc)
        for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)
    ]

def predict_image_from_array(frame_array, model_path="trained_knn_model.clf"):
    import numpy as np
    import cv2
    X_face_locations = face_recognition.face_locations(frame_array)
    if len(X_face_locations) == 0:
        return []
    faces_encodings = face_recognition.face_encodings(frame_array, known_face_locations=X_face_locations)
    with open(model_path, 'rb') as f:
        knn_clf = pickle.load(f)
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= 0.6 for i in range(len(X_face_locations))]
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]
