import os
import pickle
from PIL import Image, ImageDraw
from sklearn import svm
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

# TRAIN
def train(train_dir, model_save_path="trained_svm_model.clf", kernel='linear', C=1.0):
    X, y = [], []

    # Loop through each person in train_dir
    for class_dir in os.listdir(train_dir):
        person_dir = os.path.join(train_dir, class_dir)
        if not os.path.isdir(person_dir):
            continue

        for img_path in image_files_in_folder(person_dir):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)

            if len(face_bounding_boxes) != 1:
                print(f"Skipping {img_path}: found {len(face_bounding_boxes)} faces.")
                continue

            # Add face encoding and label
            X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
            y.append(class_dir)

    print(f"✅ Training SVM classifier with {len(X)} samples...")

    # Train SVM
    svm_clf = svm.SVC(kernel=kernel, C=C, probability=True)
    svm_clf.fit(X, y)

    # Save model
    with open(model_save_path, 'wb') as f:
        pickle.dump(svm_clf, f)
    print(f"✅ Model trained and saved as {model_save_path}")

    return svm_clf

# PREDICT
def predict(image_path, model_path="trained_svm_model.clf", confidence_threshold=0.4):
    if not os.path.isfile(image_path):
        raise Exception(f"Invalid image path: {image_path}")

    with open(model_path, 'rb') as f:
        svm_clf = pickle.load(f)

    X_img = face_recognition.load_image_file(image_path)
    # Use cnn for better detection
    X_face_locations = face_recognition.face_locations(X_img, model="cnn")

    if len(X_face_locations) == 0:
        return []

    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

    predictions = []
    for encoding, loc in zip(faces_encodings, X_face_locations):
        probs = svm_clf.predict_proba([encoding])[0]
        best_class = svm_clf.classes_[probs.argmax()]
        confidence = probs.max()

        if confidence >= confidence_threshold:
            predictions.append((best_class, loc))
        else:
            predictions.append(("unknown", loc))

    return predictions


# SHOW
def show_predictions(image_path, predictions):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    for name, (top, right, bottom, left) in predictions:
        # Green rectangle
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 255, 0), width=3)
        # Green text
        draw.text((left + 6, bottom - 20), name, fill=(0, 255, 0))

    image.show()

# MAIN
if __name__ == "__main__":
    print("Training SVM classifier...")
    model = train("data/train")
    print("Training complete!\n")

    test_dir = "data/test"
    for file in os.listdir(test_dir):
        full_path = os.path.join(test_dir, file)
        print(f"Predicting for {file}...")
        preds = predict(full_path)
        for name, (top, right, bottom, left) in preds:
            print(f"Found {name} at ({left}, {top})")
        show_predictions(full_path, preds)
