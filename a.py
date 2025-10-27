import os
import cv2
import dlib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import joblib

# =========================
# Force CPU usage
# =========================
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# =========================
# 1. Load Dlib Face Detector
# =========================
detector = dlib.get_frontal_face_detector()

# =========================
# 2. Load MobileNetV2 Model for Embeddings
# =========================
base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
embedding_model = Model(inputs=base_model.input, outputs=base_model.output)

# =========================
# 3. Prepare Dataset
# =========================
dataset_path = r'dataset_fixed/train'  # raw string for spaces
X = []  # embeddings
y = []  # labels

for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_folder):
        continue

    for img_name in os.listdir(person_folder):
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        img_path = os.path.join(person_folder, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARNING] Cannot read {img_path}")
            continue

        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = detector(rgb_img)
        if len(faces) == 0:
            print(f"[WARNING] No face detected in {img_path}")
            continue

        # Take the first face
        face = faces[0]
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        face_img = rgb_img[max(0, y1):y2, max(0, x1):x2]
        if face_img.size == 0:
            continue
        face_img = cv2.resize(face_img, (224, 224))
        face_array = image.img_to_array(face_img)
        face_array = np.expand_dims(face_array, axis=0)
        face_array = preprocess_input(face_array)

        embedding = embedding_model.predict(face_array)[0]
        X.append(embedding)
        y.append(person_name)
        print(f"[INFO] Processed {person_name}/{img_name}")

print("Total embeddings:", len(X))

# =========================
# 4. Train KNN Classifier
# =========================
if len(X) == 0:
    raise ValueError("No embeddings found. Check dataset path and images!")

knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn.fit(X, y)

# Save the trained KNN model
joblib.dump(knn, 'knn_face_model.pkl')
print("KNN Model trained and saved!")

# =========================
# 5. Prediction Function
# =========================
def predict_faces(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Cannot read {image_path}")
        return

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = detector(rgb_img)
    if len(faces) == 0:
        print("[INFO] No faces detected in the test image.")
        return

    for face in faces:
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        face_img = rgb_img[max(0, y1):y2, max(0, x1):x2]
        face_img = cv2.resize(face_img, (224, 224))
        face_array = image.img_to_array(face_img)
        face_array = np.expand_dims(face_array, axis=0)
        face_array = preprocess_input(face_array)

        embedding = embedding_model.predict(face_array)[0].reshape(1, -1)
        prediction = knn.predict(embedding)[0]
        distance = knn.kneighbors(embedding, n_neighbors=1)[0][0][0]

        if distance < 0.6:  # threshold
            color = (0, 255, 0)  # Green for known
            label = prediction
        else:
            color = (0, 0, 255)  # Red for unknown
            label = "Unknown"

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Face Recognition", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# =========================
# 6. Test Prediction
# =========================
predict_faces(r'dataset_fixed/test/p.jpg')
