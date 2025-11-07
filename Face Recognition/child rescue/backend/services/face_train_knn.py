import os
import pickle
from sklearn import neighbors
import face_recognition

TRAIN_DIR = "static/train_data"
MODEL_SAVE_PATH = "static/trained_knn_model.clf"
KNN_NEIGHBORS = 3

def train_knn_model():
    X, y = [], []
    for person_name in os.listdir(TRAIN_DIR):
        person_dir = os.path.join(TRAIN_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            image = face_recognition.load_image_file(img_path)
            boxes = face_recognition.face_locations(image)
            if len(boxes) != 1:
                continue
            enc = face_recognition.face_encodings(image, known_face_locations=boxes)[0]
            X.append(enc)
            y.append(person_name)

    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS, algorithm='ball_tree', weights='distance')
    knn_clf.fit(X, y)

    with open(MODEL_SAVE_PATH, 'wb') as f:
        pickle.dump(knn_clf, f)
    print("KNN model trained and saved!")
