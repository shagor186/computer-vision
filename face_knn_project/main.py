import os
import cv2
import face_recognition
from sklearn.neighbors import KNeighborsClassifier
from utils.feature_extractor import prepare_dataset, save_knn_model, load_knn_model, extract_features

# ==============================
# TRAIN KNN
# ==============================
train_dir = "data/train"
print("Preparing dataset...")
X, y = prepare_dataset(train_dir)
print("Features shape:", X.shape)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)
save_knn_model(knn)
print("KNN model trained and saved at models/knn_model.clf")

# ==============================
# TEST IMAGES RECOGNITION
# ==============================
knn = load_knn_model("models/knn_model.clf")
test_dir = "data/test"

for img_name in os.listdir(test_dir):
    img_path = os.path.join(test_dir, img_name)
    frame = cv2.imread(img_path)
    if frame is None:
        print(f"Cannot read image {img_name}")
        continue

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)

    for (top, right, bottom, left) in face_locations:
        face_image = rgb_frame[top:bottom, left:right]
        features = extract_features(face_image).reshape(1, -1)
        prediction = knn.predict(features)[0]

        # Draw rectangle + label
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, prediction, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)
    cv2.waitKey(0)  # press any key for next image

cv2.destroyAllWindows()
