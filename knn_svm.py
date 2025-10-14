import cv2, dlib, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

def get_embedding(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        shape = predictor(gray, face)
        return np.array(face_rec_model.compute_face_descriptor(img, shape))
    return None


from sklearn.neighbors import KNeighborsClassifier

# Example dataset
X = []  # embeddings
y = []  # labels

# ধরুন আমাদের dataset-এ person1, person2 এর ছবি আছে
for i in range(1,6):
    emb = get_embedding(f"img/js{i}.jpg")
    if emb is not None:
        X.append(emb); y.append("person1")

for i in range(1,6):
    emb = get_embedding(f"img/js_r{i}.jpeg")
    if emb is not None:
        X.append(emb); y.append("person2")

X = np.array(X); y = np.array(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train KNN
knn = KNeighborsClassifier(n_neighbors=3, metric="euclidean")
knn.fit(X_train, y_train)

# Predict
y_pred = knn.predict(X_test)
print("KNN Accuracy:", accuracy_score(y_test, y_pred))
