import cv2, dlib
import numpy as np
from numpy.linalg import norm

# Load dlib models
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

# Same child at two different ages
embed_young = get_embedding("img/cr_mu.jpg")
embed_old = get_embedding("img/cr_juv.jpg")

# Similarity
euclidean_dist = norm(embed_young - embed_old)
print("Euclidean Distance (Age 5 vs Age 15):", euclidean_dist)