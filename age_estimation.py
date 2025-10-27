# from deepface import DeepFace
# import cv2
#
# # ছবি লোড
# img_path = "images/ronaldo.jpg"
# img = cv2.imread(img_path)
# cv2.imshow("Input Image", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# # Age estimation
# analysis = DeepFace.analyze(img_path, actions=['age'])
# print("Estimated Age:", analysis['age'])

#
# import cv2
# from deepface import DeepFace
#
# cap = cv2.VideoCapture(0)
#
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     # Analyze age
#     result = DeepFace.analyze(frame, actions=['age'], enforce_detection=False)
#     age = int(result['age'])
#
#     # Draw
#     cv2.putText(frame, f"Age: {age}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
#     cv2.imshow("Webcam Age Estimation", frame)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()




import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0)

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # optional: resize for speed
    small_frame = cv2.resize(frame, (320, 240))

    # analyze every 5th frame
    if frame_count % 5 == 0:
        result = DeepFace.analyze(small_frame, actions=['age'], enforce_detection=False)
        age = int(result[0]['age'])  # first face only
    frame_count += 1

    # draw
    cv2.putText(frame, f"Age: {age}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Webcam Age Estimation", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
