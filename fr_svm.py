import cv2
import dlib
import numpy as np
import os
import sqlite3
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import joblib  # saving/loading the SVM model

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

# Prepare reference embeddings
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

# train SVM classifier
if reference_encodings:
    # Encode labels as numbers
    le = LabelEncoder()
    labels_encoded = le.fit_transform(reference_names)

    # Train SVM classifier
    svm_clf = SVC(kernel='linear', probability=True)
    svm_clf.fit(reference_encodings, labels_encoded)

    # Save the models
    joblib.dump(svm_clf, "svm_face_model.pkl")
    joblib.dump(le, "label_encoder.pkl")
else:
    raise ValueError("No reference faces found!")

# Real-time face recognition using CNN-SVM
cap = cv2.VideoCapture(0)
svm_clf = joblib.load("svm_face_model.pkl")
le = joblib.load("label_encoder.pkl")

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

        # --- SVM prediction ---
        probs = svm_clf.predict_proba(encoding)[0]
        max_prob_index = np.argmax(probs)
        max_prob = probs[max_prob_index]
        pred_name = le.inverse_transform([max_prob_index])[0]

        if max_prob > 0.6:  # confidence threshold
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

    cv2.imshow("Hybrid CNN-SVM Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()