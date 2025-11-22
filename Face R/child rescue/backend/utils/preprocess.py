from PIL import Image
import face_recognition
import numpy as np

def preprocess_image(image_path, target_size=(160,160)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img)/255.0
    return img_array

def extract_face_encoding(image_path):
    image = face_recognition.load_image_file(image_path)
    boxes = face_recognition.face_locations(image)
    if len(boxes) != 1:
        return None
    enc = face_recognition.face_encodings(image, known_face_locations=boxes)[0]
    return enc
