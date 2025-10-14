import os
import math
import pickle
import cv2
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

    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        print(f"Chose n_neighbors automatically: {n_neighbors}")

    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    with open(model_save_path, 'wb') as f:
        pickle.dump(knn_clf, f)
    print(f"✅ Model trained and saved as {model_save_path}")
    return knn_clf


def predict(image, knn_clf=None, model_path="trained_knn_model.clf", distance_threshold=0.6):
    """Recognize faces from an image array or file path."""
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # Load image if path is given
    if isinstance(image, str):
        X_img = face_recognition.load_image_file(image)
    else:
        X_img = image

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


def show_predictions(image_path, predictions):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for name, (top, right, bottom, left) in predictions:
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 255, 0), width=3)
        draw.text((left + 6, bottom - 20), name, fill=(0, 255, 0))
    image.show()


def webcam_recognition(model_path="trained_knn_model.clf"):
    """Run face recognition in real-time using webcam."""
    print("🎥 Starting webcam recognition...")
    with open(model_path, 'rb') as f:
        knn_clf = pickle.load(f)

    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("⚠️ Failed to capture frame.")
            break

        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]  # BGR → RGB

        predictions = predict(rgb_small_frame, knn_clf=knn_clf)

        # Draw results
        for name, (top, right, bottom, left) in predictions:
            # Scale back up face locations since we processed at 1/4 scale
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            color = (0, 255, 0) if name != "unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        cv2.imshow("KNN Face Recognition", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    print("👋 Webcam closed.")


if __name__ == "__main__":
    # Step 1: Train the model (only once)
    print("Training KNN classifier...")
    model = train("data/train")
    print("Training complete!\n")

    # Step 2: Test on static images (optional)
    test_dir = "data/test"
    for file in os.listdir(test_dir):
        full_path = os.path.join(test_dir, file)
        print(f"Predicting for {file}...")
        preds = predict(full_path)
        for name, (top, right, bottom, left) in preds:
            print(f"Found {name} at ({left}, {top})")
        show_predictions(full_path, preds)

    # Step 3: Run live recognition from webcam
    webcam_recognition()