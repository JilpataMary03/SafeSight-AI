
import cv2
import os
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

interpreter = tf.lite.Interpreter(model_path="MobileFaceNet.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess(img_array):
    img = Image.fromarray(img_array).convert("RGB").resize((112, 112))
    arr = np.array(img).astype(np.float32)
    arr = (arr - 127.5) / 128.0
    return arr

def get_embedding(frame):
    img = preprocess(frame)
    batch_input = np.stack([img, img], axis=0)  # Shape: (2, 112, 112, 3)

    interpreter.set_tensor(input_details[0]['index'], batch_input)
    interpreter.invoke()
    embeddings = interpreter.get_tensor(output_details[0]['index'])  # Shape: (2, 192)

    mean_embedding = np.mean(embeddings, axis=0)
    final_embeddings=mean_embedding / np.linalg.norm(mean_embedding)
    return final_embeddings


def get_embedding_for_match(frame):
    img = preprocess(frame)
    batch_input = np.stack([img, img], axis=0)  # Shape: (2, 112, 112, 3)

    interpreter.set_tensor(input_details[0]['index'], batch_input)
    interpreter.invoke()
    embeddings = interpreter.get_tensor(output_details[0]['index'])  # Shape: (2, 192)

    mean_embedding = np.mean(embeddings, axis=0)
    final_embeddings=mean_embedding / np.linalg.norm(mean_embedding)
    return final_embeddings

def load_database(employee_dir="workers"):
    db = {}
    for person in os.listdir(employee_dir):
        person_path = os.path.join(employee_dir, person)
        if os.path.isdir(person_path):
            embeddings = []
            for file in os.listdir(person_path):
                if file.lower().endswith(('.jpg', '.png')):
                    img_path = os.path.join(person_path, file)
                    img = cv2.imread(img_path)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = face_detector.detectMultiScale(gray, 1.3, 5)
                    if len(faces) == 0:
                        continue
                    x, y, w, h = faces[0]
                    face = img[y:y+h, x:x+w]
                    face = cv2.resize(face, (160, 160))
                    embedding = get_embedding(face)
                    embeddings.append(embedding)
            if embeddings:
                db[person] = np.mean(embeddings, axis=0)
    return db

def match_embedding(input_embedding, db, threshold=0.5):
    max_score = -1
    identity = "unknown"
    for name, db_emb in db.items():
        score = cosine_similarity([input_embedding], [db_emb])[0][0]
        if score > max_score and score > threshold:
            max_score = score
            identity = name
    return identity
 