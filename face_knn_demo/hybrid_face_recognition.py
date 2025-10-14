import os
import math
import pickle
from PIL import Image, ImageDraw
from sklearn import neighbors
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

# ==================== TRAIN ====================
def train(train_dir, model_save_path="trained_knn_model.clf", n_neighbors=None, knn_algo='ball_tree', use_cnn=True):
    """
    Train KNN classifier on face embeddings extracted from images.
    """
    X, y = [], []

    for class_dir in os.listdir(train_dir):
        person_dir = os.path.join(train_dir, class_dir)
        if not os.path.isdir(person_dir):
            continue

        for img_path in image_files_in_folder(person_dir):
            image = face_recognition.load_image_file(img_path)
            # Use CNN for better accuracy (CPU may be slow)
            model_type = "cnn" if use_cnn else "hog"
            face_bounding_boxes = face_recognition.face_locations(image, model=model_type)

            if len(face_bounding_boxes) != 1:
                print(f"Skipping {img_path}: found {len(face_bounding_boxes)} faces.")
                continue

            # Extract face embedding (128-D)
            encoding = face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0]
            X.append(encoding)
            y.append(class_dir)

    if len(X) == 0:
        raise Exception("No valid training images found!")

    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        print(f"Chose n_neighbors automatically: {n_neighbors}")

    # Train KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    # Save model
    with open(model_save_path, 'wb') as f:
        pickle.dump(knn_clf, f)

    print(f"✅ Model trained and saved as {model_save_path}")
    return knn_clf

# ==================== PREDICT ====================
def predict(image_path, knn_clf=None, model_path="trained_knn_model.clf", distance_threshold=0.6, use_cnn=True):
    """
    Predict faces in an image using trained KNN classifier.
    """
    if not os.path.isfile(image_path):
        raise Exception(f"Invalid image path: {image_path}")

    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    X_img = face_recognition.load_image_file(image_path)
    model_type = "cnn" if use_cnn else "hog"
    X_face_locations = face_recognition.face_locations(X_img, model=model_type)

    if len(X_face_locations) == 0:
        return []

    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    return [(pred, loc) if rec else ("unknown", loc)
            for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]

# ==================== SHOW ====================
def show_predictions(image_path, predictions):
    """
    Display image with green boxes and names.
    """
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    for name, (top, right, bottom, left) in predictions:
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 255, 0), width=3)
        draw.text((left + 6, bottom - 20), name, fill=(0, 255, 0))

    image.show()

# ==================== MAIN ====================
if __name__ == "__main__":
    # Folder paths
    TRAIN_DIR = os.path.join(os.getcwd(), "data/train")
    TEST_DIR = os.path.join(os.getcwd(), "data/test")

    print("Training Hybrid CNN+KNN classifier...")
    model = train(TRAIN_DIR, use_cnn=True)
    print("Training complete!\n")

    for file in os.listdir(TEST_DIR):
        full_path = os.path.join(TEST_DIR, file)
        print(f"Predicting for {file}...")
        preds = predict(full_path, use_cnn=True)
        for name, (top, right, bottom, left) in preds:
            print(f"🧠 Found {name} at ({left}, {top})")
        show_predictions(full_path, preds)
