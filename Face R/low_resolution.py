# import cv2
#
# img = cv2.imread("img/cr7_low_r.jpeg", 0)  # grayscale
# equ = cv2.equalizeHist(img)
#
# cv2.imshow("Original", img)
# cv2.imshow("Histogram Equalized", equ)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# import cv2
#
# # সুপার রেজোলিউশন মডেল লোড করুন
# sr = cv2.dnn_superres.DnnSuperResImpl_create()
# sr.readModel("EDSR_x4.pb")
# sr.setModel("edsr", 4)
#
# # লো রেজোলিউশন ইমেজ লোড করুন
# img = cv2.imread("img/cr7_low_r.jpeg")
# result = sr.upsample(img)
#
# # ইমেজ প্রদর্শন করুন
# cv2.imshow("Low Resolution", img)
# cv2.imshow("Enhanced Resolution", result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# import dlib
# import cv2
#
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
#
# img = cv2.imread("img/cr7_low_r.jpeg")
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# faces = detector(gray)
#
# for face in faces:
#     landmarks = predictor(gray, face)
#
#     # চোখ এবং নাকের পয়েন্ট বের করি
#     left_eye = (landmarks.part(36).x, landmarks.part(36).y)
#     right_eye = (landmarks.part(45).x, landmarks.part(45).y)
#     nose = (landmarks.part(30).x, landmarks.part(30).y)
#
#     # ছোট bounding box crop করি (চোখ+নাক অঞ্চল)
#     x1 = min(left_eye[0], right_eye[0]) - 20
#     y1 = min(left_eye[1], right_eye[1]) - 20
#     x2 = max(left_eye[0], right_eye[0]) + 20
#     y2 = nose[1] + 30
#
#     cropped = img[y1:y2, x1:x2]
#     cv2.imshow("Cropped Region (Eyes + Nose)", cropped)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#
# import numpy as np
# from numpy.linalg import norm
# import dlib, cv2
#
# # Load dlib models
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
#
# def get_embedding(img):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     faces = detector(gray)
#     for face in faces:
#         shape = predictor(gray, face)
#         return np.array(face_rec_model.compute_face_descriptor(img, shape))
#     return None
#
# # Load images
# lowres = cv2.imread("img/cr7_low_r.jpeg")
# enhanced = cv2.imread("img/cr.jpg")
#
# # Verify images
# if lowres is None:
#     raise FileNotFoundError("Low-res image not found")
# if enhanced is None:
#     raise FileNotFoundError("Enhanced image not found")
#
# # Get embeddings
# embed_low = get_embedding(lowres)
# embed_high = get_embedding(enhanced)
#
# # Verify face detection
# if embed_low is None:
#     raise ValueError("No face detected in low-res image")
# if embed_high is None:
#     raise ValueError("No face detected in enhanced image")
#
# # Similarity check
# euclidean_dist = norm(embed_low - embed_high)
# print("Euclidean Distance (Before vs After Enhancement):", euclidean_dist)


import cv2
import dlib
import numpy as np
from numpy.linalg import norm

# Load dlib models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

def get_embedding(img, upsample=1):
    """
    Returns 128D face embedding or None if no face detected.
    upsample: number of times to upsample image for detection
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, upsample)  # upsample improves detection for small faces
    if len(faces) == 0:
        return None
    shape = predictor(gray, faces[0])
    return np.array(face_rec_model.compute_face_descriptor(img, shape))

# Load images
lowres = cv2.imread("img/cr7_low_r.jpeg")
enhanced = cv2.imread("img/cr.jpg")

if lowres is None:
    raise FileNotFoundError("Low-res image not found")
if enhanced is None:
    raise FileNotFoundError("Enhanced image not found")

# Option 1: Use upsample=1 to improve detection
embed_low = get_embedding(lowres, upsample=1)
embed_high = get_embedding(enhanced, upsample=1)

# Option 2: If still None, apply super-resolution first
if embed_low is None:
    from cv2 import dnn_superres
    sr = dnn_superres.DnnSuperResImpl_create()
    sr.readModel("EDSR_x4.pb")
    sr.setModel("edsr", 4)
    lowres_up = sr.upsample(lowres)
    embed_low = get_embedding(lowres_up, upsample=1)

# Verify face detection
if embed_low is None:
    raise ValueError("No face detected in low-res image even after SR")
if embed_high is None:
    raise ValueError("No face detected in enhanced image")

# Compute Euclidean distance
euclidean_dist = norm(embed_low - embed_high)
print("Euclidean Distance (Before vs After Enhancement):", euclidean_dist)
