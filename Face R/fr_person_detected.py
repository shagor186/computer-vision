import cv2
import dlib
import numpy as np
import os
import sqlite3

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

# Load reference images
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

# Webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector(img_rgb)

    for face in faces:
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        shape = predictor(img_rgb, face)
        encoding = np.array(face_rec_model.compute_face_descriptor(img_rgb, shape))

        distances = np.linalg.norm(reference_encodings - encoding, axis=1)
        min_dist_index = np.argmin(distances)

        if distances[min_dist_index] < 0.6:
            person_name = reference_names[min_dist_index]
            info = fetch_person_info(person_name)
            if info:
                label = f"{info['name']} | Age: {info['age']} | {info['location']}"
                color = (0, 255, 0)
                # Print recognized person info in terminal
                print(f"Found person: ID={info['id']}, Name={info['name']}, Age={info['age']}, Location={info['location']}")
            else:
                label = person_name
                color = (0, 255, 0)
                print(f"Found person: {person_name} (not in DB)")
        else:
            label = "Unknown"
            color = (0, 0, 255)
            print("Unknown person detected")

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Face Recognition System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()