import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # GPU ব্যবহার বন্ধ করে CPU-তে চালাবে

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model('mnist_cnn_gpu.h5')

# Open webcam
cap = cv2.VideoCapture(0)
print("Press 'c' to capture digit, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Show live webcam feed
    cv2.imshow('Webcam - MNIST Capture', frame)

    key = cv2.waitKey(1)

    if key & 0xFF == ord('c'):
        # Capture frame and convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (28, 28))
        img = resized.reshape(1, 28, 28, 1) / 255.0

        # Predict digit
        pred = model.predict(img)
        digit = np.argmax(pred)
        print("Predicted Digit:", digit)

        # Show captured digit
        plt.imshow(resized, cmap='gray')
        plt.show()

    elif key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
