import pickle
import face_recognition

MODEL_PATH = "static/trained_knn_model.clf"
THRESHOLD = 0.5

with open(MODEL_PATH, 'rb') as f:
    knn_clf = pickle.load(f)

def predict_face(image_path):
    image = face_recognition.load_image_file(image_path)
    boxes = face_recognition.face_locations(image)
    encodings = face_recognition.face_encodings(image, boxes)
    results = []

    for enc, box in zip(encodings, boxes):
        distances, indices = knn_clf.kneighbors([enc], n_neighbors=1)
        if distances[0][0] <= THRESHOLD:
            name = knn_clf.predict([enc])[0]
        else:
            name = "unknown"
        results.append((name, box))
    return results
