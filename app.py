import cv2
import dlib

# Load dlib's frontal face detector
detector = dlib.get_frontal_face_detector()
#
# # Read image
# img = cv2.imread("pat_i.jpg")
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# # Detect faces
# faces = detector(gray)
#
# # Draw rectangle around faces
# for face in faces:
#     x, y, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
#     cv2.rectangle(img, (x, y), (x2, y2), (0, 255, 0), 2)
#
# cv2.imshow("Face Detection", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# Load landmark predictor (68 points)
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
#
# img = cv2.imread("pat_i.jpg")
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# faces = detector(gray)
#
# for face in faces:
#     landmarks = predictor(gray, face)
#     for n in range(0, 68):
#         x, y = landmarks.part(n).x, landmarks.part(n).y
#         cv2.circle(img, (x, y), 2, (0, 0, 255), -1)
#
# cv2.imshow("Facial Landmarks", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



import numpy as np

# Load dlib face recognition model (128D embeddings)
face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

img = cv2.imread("pat_i.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = detector(gray)

for face in faces:
    shape = predictor(gray, face)
    face_embedding = np.array(face_rec_model.compute_face_descriptor(img, shape))
    print("128D Face Embedding:\n", face_embedding)
    print("Vector Length:", len(face_embedding))
