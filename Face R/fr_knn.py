import cv2
import dlib
import numpy as np
import os
import sqlite3
from sklearn.neighbors import KNeighborsClassifier
import joblib  # saving/loading the KNN model

# Paths to dlib models
predictor_path = r"shape_predictor_68_face_landmarks.dat"
face_rec_model_path = r"dlib_face_recognition_resnet_model_v1.dat"

# Load models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
face_rec_model = dlib.face_recognition_model_v1(face_rec_model_path)

# Database file
DB_FILE = "face_recognition.db"

def fetch_person_info(name):
    """Fetch person details by name"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM persons WHERE name = ?", (name,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return {"id": row[0], "name": row[1], "age": row[2], "location": row[3]}
    return None


reference_folder = "images"
reference_encodings, reference_names = [], []

for file_name in os.listdir(reference_folder):
    if file_name.endswith((".jpg", ".png", ".jpeg")):
        img_path = os.path.join(reference_folder, file_name)
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = detector(img_rgb)
        if len(faces) == 0:
            continue
        shape = predictor(img_rgb, faces[0])
        encoding = np.array(face_rec_model.compute_face_descriptor(img_rgb, shape))
        reference_encodings.append(encoding)
        reference_names.append(os.path.splitext(file_name)[0])

print(f"Loaded {len(reference_encodings)} reference faces.")

# train KNN classifier
if reference_encodings:
    knn_clf = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
    knn_clf.fit(reference_encodings, reference_names)
    joblib.dump(knn_clf, "knn_face_model.pkl")  # Save model
else:
    raise ValueError("No reference faces found!")

# eal-time face recognition using CNN-KNN
cap = cv2.VideoCapture(0)
knn_clf = joblib.load("knn_face_model.pkl")  # Load trained KNN

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector(img_rgb)

    for face in faces:
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        shape = predictor(img_rgb, face)
        encoding = np.array(face_rec_model.compute_face_descriptor(img_rgb, shape)).reshape(1, -1)

        # KNN prediction
        pred_name = knn_clf.predict(encoding)[0]
        distances, indices = knn_clf.kneighbors(encoding)
        min_dist = distances[0][0]

        if min_dist < 0.6:
            info = fetch_person_info(pred_name)
            if info:
                label = f"{info['name']} | Age: {info['age']} | {info['location']}"
            else:
                label = pred_name
            color = (0, 255, 0)
        else:
            label = "Unknown"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Hybrid CNN-KNN Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()