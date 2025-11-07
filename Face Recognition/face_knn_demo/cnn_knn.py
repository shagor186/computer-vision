import os
import pickle
import cv2
import numpy as np
from sklearn import neighbors
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# -------------------------------
# CPU-only setup
# -------------------------------
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.keras.backend.clear_session()

# -------------------------------
# Load pre-trained CNN (MobileNetV2)
# -------------------------------
print("Loading MobileNetV2 model...")
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    pooling='avg',
    input_shape=(224, 224, 3)
)
print("✅ MobileNetV2 model loaded successfully!")


# -------------------------------
# CNN Feature Extraction
# -------------------------------
def extract_cnn_features(img):
    """
    Extract features from image using MobileNetV2
    """
    # Convert to RGB if grayscale
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize to MobileNetV2 input size
    img = cv2.resize(img, (224, 224))

    # Normalize pixel values
    img = img.astype(np.float32) / 255.0

    # Expand dimensions for batch
    x = np.expand_dims(img, axis=0)

    # Extract features
    features = base_model.predict(x, verbose=0)
    return features.flatten()


# -------------------------------
# Face Detection
# -------------------------------
def detect_face(img_path):
    """
    Detect face in image using Haar Cascade
    """
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ Could not read image: {img_path}")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    if face_cascade.empty():
        print("❌ Haar cascade not loaded properly")
        return None

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    if len(faces) == 0:
        print(f"❌ No face detected in: {img_path}")
        return None

    # Take the first face
    (x, y, w, h) = faces[0]
    face_img = img[y:y + h, x:x + w]

    print(f"✅ Face detected in: {os.path.basename(img_path)}")
    return face_img


# -------------------------------
# Train CNN+KNN
# -------------------------------
def train(train_dir, model_save_path="mobilenet_knn_model.clf", n_neighbors=None):
    """
    Train KNN classifier with MobileNetV2 features
    """
    X_train, y_train = [], []

    # Check if training directory exists
    if not os.path.exists(train_dir):
        raise Exception(f"❌ Training directory {train_dir} does not exist")

    print(f"📁 Scanning training directory: {train_dir}")

    person_count = 0
    image_count = 0

    for person_name in os.listdir(train_dir):
        person_folder = os.path.join(train_dir, person_name)
        if not os.path.isdir(person_folder):
            continue

        person_count += 1
        print(f"👤 Processing: {person_name}")

        person_images = 0
        for img_name in os.listdir(person_folder):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                img_path = os.path.join(person_folder, img_name)
                face = detect_face(img_path)

                if face is not None:
                    try:
                        features = extract_cnn_features(face)
                        X_train.append(features)
                        y_train.append(person_name)
                        person_images += 1
                        image_count += 1
                    except Exception as e:
                        print(f"❌ Error processing {img_name}: {e}")
                        continue

        print(f"   ✅ Added {person_images} images for {person_name}")

    if len(X_train) == 0:
        raise Exception("❌ No training data found. Check your dataset and face detection.")

    print(f"\n📊 Training Summary:")
    print(f"   People: {person_count}")
    print(f"   Total Images: {image_count}")
    print(f"   Feature Dimension: {len(X_train[0])}")

    # Calculate optimal n_neighbors
    if n_neighbors is None:
        n_neighbors = max(1, int(round(np.sqrt(len(X_train)))))

    print(f"🤖 Training KNN with n_neighbors = {n_neighbors}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train KNN
    knn_clf = neighbors.KNeighborsClassifier(
        n_neighbors=n_neighbors,
        algorithm='ball_tree',
        weights='distance',
        metric='euclidean'
    )

    knn_clf.fit(X_train_scaled, y_train)

    # Calculate average distance for threshold
    distances = []
    for features in X_train_scaled:
        dist, _ = knn_clf.kneighbors([features], n_neighbors=1)
        distances.append(dist[0][0])

    avg_distance = np.mean(distances)
    std_distance = np.std(distances)

    print(f"📏 Distance Statistics:")
    print(f"   Min Distance: {np.min(distances):.4f}")
    print(f"   Max Distance: {np.max(distances):.4f}")
    print(f"   Average Distance: {avg_distance:.4f}")
    print(f"   Std Distance: {std_distance:.4f}")
    print(f"   Recommended Threshold: {avg_distance + 2 * std_distance:.4f}")

    # Save the model
    with open(model_save_path, 'wb') as f:
        pickle.dump({
            'knn_model': knn_clf,
            'scaler': scaler,
            'avg_distance': avg_distance,
            'std_distance': std_distance,
            'X_train': X_train,  # For debugging
            'y_train': y_train  # For debugging
        }, f)

    print(f"✅ MobileNet+KNN model trained and saved as {model_save_path}")
    print(f"✅ Training samples: {len(X_train)}")

    return knn_clf, scaler, avg_distance, std_distance


# -------------------------------
# Prediction
# -------------------------------
def predict(img_path, model_path="mobilenet_knn_model.clf", distance_threshold=None):
    """
    Predict person from image
    """
    if not os.path.isfile(model_path):
        raise Exception(f"❌ Model file {model_path} not found. Train the model first.")

    if not os.path.isfile(img_path):
        raise Exception(f"❌ Invalid image path: {img_path}")

    print(f"🔍 Predicting: {os.path.basename(img_path)}")

    # Load trained model
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    knn_clf = model_data['knn_model']
    scaler = model_data['scaler']
    avg_distance = model_data['avg_distance']
    std_distance = model_data['std_distance']

    # Use auto threshold if not provided
    if distance_threshold is None:
        distance_threshold = avg_distance + 2 * std_distance
        print(f"   Auto Threshold: {distance_threshold:.4f}")

    # Detect face
    face = detect_face(img_path)
    if face is None:
        return "No Face Detected", 0.0

    # Extract features
    features = extract_cnn_features(face)

    # Transform features
    features_scaled = scaler.transform([features])[0]

    # Get distances and predictions
    distances, indices = knn_clf.kneighbors([features_scaled], n_neighbors=1)
    prediction = knn_clf.predict([features_scaled])[0]
    distance_value = distances[0][0]

    # Calculate confidence (0 to 1 scale)
    max_reasonable_distance = max(avg_distance + 3 * std_distance, distance_threshold + 5)
    if max_reasonable_distance > 0:
        confidence = max(0, 1 - (distance_value / max_reasonable_distance))
    else:
        confidence = 0.0

    print(f"   Distance: {distance_value:.4f}")
    print(f"   Confidence: {confidence:.4f}")
    print(f"   Threshold: {distance_threshold:.4f}")

    if distance_value <= distance_threshold:
        return prediction, confidence
    else:
        return "Unknown", confidence


# -------------------------------
# Manual Threshold Adjustment
# -------------------------------
def find_optimal_threshold(model_path="mobilenet_knn_model.clf"):
    """
    Find optimal threshold by analyzing training distances
    """
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    distances = []
    X_train = model_data['X_train']
    scaler = model_data['scaler']
    knn_clf = model_data['knn_model']

    X_train_scaled = scaler.transform(X_train)

    for features in X_train_scaled:
        dist, _ = knn_clf.kneighbors([features], n_neighbors=1)
        distances.append(dist[0][0])

    distances = np.array(distances)

    print(f"\n🎯 Optimal Threshold Analysis:")
    print(f"   Min distance: {np.min(distances):.4f}")
    print(f"   Max distance: {np.max(distances):.4f}")
    print(f"   Mean distance: {np.mean(distances):.4f}")
    print(f"   Std distance: {np.std(distances):.4f}")
    print(f"   25th percentile: {np.percentile(distances, 25):.4f}")
    print(f"   50th percentile: {np.percentile(distances, 50):.4f}")
    print(f"   75th percentile: {np.percentile(distances, 75):.4f}")
    print(f"   95th percentile: {np.percentile(distances, 95):.4f}")

    # Recommended thresholds
    threshold_1 = np.percentile(distances, 95)  # 95% of training data within this
    threshold_2 = np.mean(distances) + 2 * np.std(distances)

    print(f"\n💡 Recommended Thresholds:")
    print(f"   Conservative: {threshold_1:.4f} (95th percentile)")
    print(f"   Moderate: {threshold_2:.4f} (mean + 2*std)")

    return threshold_1, threshold_2


# -------------------------------
# Batch Prediction
# -------------------------------
def predict_batch(test_dir, model_path="mobilenet_knn_model.clf", distance_threshold=None):
    """
    Predict multiple images in a directory
    """
    results = []

    if not os.path.exists(test_dir):
        print(f"❌ Test directory {test_dir} not found")
        return results

    # Load model for threshold info
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    if distance_threshold is None:
        distance_threshold = model_data['avg_distance'] + 2 * model_data['std_distance']

    print(f"🎯 Using distance threshold: {distance_threshold:.4f}")

    for file in os.listdir(test_dir):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            full_path = os.path.join(test_dir, file)
            pred_name, confidence = predict(full_path, model_path, distance_threshold)
            results.append((file, pred_name, confidence))

    return results


# -------------------------------
# Model Information
# -------------------------------
def model_info():
    """
    Display model information
    """
    print("\n📋 Model Information:")
    print(f"   Base Model: MobileNetV2")
    print(f"   Input Shape: {base_model.input_shape}")
    print(f"   Output Shape: {base_model.output_shape}")
    print(f"   Feature Dimension: {base_model.output_shape[1]}")


# -------------------------------
# Main Execution
# -------------------------------
if __name__ == "__main__":
    # Directories
    train_dir = "data/train"
    test_dir = "data/test"

    # Create directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    print("🚀 Starting MobileNetV2 + KNN Face Recognition")
    print("=" * 50)

    # Display model info
    model_info()

    # Check if training data exists
    if not os.listdir(train_dir):
        print(f"\n⚠️  No training data found in {train_dir}")
        print("Please add folders with person names containing images.")
    else:
        # Train the model
        print(f"\n🎯 Training Model...")
        try:
            model, scaler, avg_dist, std_dist = train(train_dir, "mobilenet_knn_model.clf")
            print("✅ Training completed successfully!")

            # Find optimal threshold
            threshold_1, threshold_2 = find_optimal_threshold()

            # Test the model with different thresholds
            print(f"\n🧪 Testing Model...")
            if os.path.exists(test_dir) and os.listdir(test_dir):

                # Test with moderate threshold first
                print(f"\n🔹 Testing with Moderate Threshold ({threshold_2:.4f})")
                results = predict_batch(test_dir, "mobilenet_knn_model.clf", threshold_2)

                print(f"\n📊 Test Results (Moderate Threshold):")
                print("-" * 50)
                print(f"{'Image':20} {'Prediction':15} {'Confidence':12} {'Status':10}")
                print("-" * 50)
                for file, pred, conf in results:
                    status = "✅ Known" if pred != "Unknown" and pred != "No Face Detected" else "❌ Unknown"
                    print(f"{file:20} {pred:15} {conf:12.4f} {status:10}")

                # If too many unknowns, try conservative threshold
                unknown_count = sum(1 for _, pred, _ in results if pred == "Unknown")
                if unknown_count > len(results) * 0.5:  # If more than 50% unknown
                    print(f"\n🔹 Too many unknowns. Trying Conservative Threshold ({threshold_1:.4f})")
                    results = predict_batch(test_dir, "mobilenet_knn_model.clf", threshold_1)

                    print(f"\n📊 Test Results (Conservative Threshold):")
                    print("-" * 50)
                    print(f"{'Image':20} {'Prediction':15} {'Confidence':12} {'Status':10}")
                    print("-" * 50)
                    for file, pred, conf in results:
                        status = "✅ Known" if pred != "Unknown" and pred != "No Face Detected" else "❌ Unknown"
                        print(f"{file:20} {pred:15} {conf:12.4f} {status:10}")

                # Calculate statistics
                known_count = sum(1 for _, pred, _ in results if pred != "Unknown" and pred != "No Face Detected")
                unknown_count = sum(1 for _, pred, _ in results if pred == "Unknown")
                no_face_count = sum(1 for _, pred, _ in results if pred == "No Face Detected")
                total_count = len(results)

                print(f"\n📈 Final Statistics:")
                print(f"   Known: {known_count}/{total_count} ({known_count / total_count * 100:.1f}%)")
                print(f"   Unknown: {unknown_count}/{total_count} ({unknown_count / total_count * 100:.1f}%)")
                print(f"   No Face: {no_face_count}/{total_count} ({no_face_count / total_count * 100:.1f}%)")

            else:
                print(f"⚠️  No test images found in {test_dir}")

        except Exception as e:
            print(f"❌ Error during training: {e}")
            import traceback

            traceback.print_exc()

    print("\n🎉 Program completed!")