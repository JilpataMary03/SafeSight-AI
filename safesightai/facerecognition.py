import os
import io
import cv2
import base64
import numpy as np
import sqlite3
from tempfile import NamedTemporaryFile
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse
from google.cloud import vision
from google.cloud.vision_v1 import types
from fastapi.middleware.cors import CORSMiddleware
 
# Import your TFLite Face Embedding pipeline
from face_recog import (
    face_detector,
    get_embedding,
    load_database,
    match_embedding
)
 
# ============================================================
# 🧠 FASTAPI SETUP
# ============================================================
app = FastAPI(
    title="PPE Detection + Face Validation API",
    description="Detect Helmet, Vest, Shoes using GCV + Face Validation and Registration via MobileFaceNet TFLite",
    version="2.0.0"
)
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ✅ Allow all origins (for testing)
    allow_credentials=True,
    allow_methods=["*"],  # ✅ Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # ✅ Allow all headers
)
 
DB_PATH = "ppe_faces.db"
 
# ============================================================
# 🧱 DATABASE SETUP
# ============================================================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS persons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            face_encoding BLOB
        )
    """)
    conn.commit()
    conn.close()
 
def add_person(name, face_encoding):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO persons (name, face_encoding) VALUES (?, ?)",
              (name, face_encoding.tobytes()))
    conn.commit()
    conn.close()
 
def get_all_known_faces():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT name, face_encoding FROM persons")
    rows = c.fetchall()
    conn.close()
    known_faces = []
    for name, encoding_blob in rows:
        encoding = np.frombuffer(encoding_blob, dtype=np.float64)
        known_faces.append((name, encoding))
    return known_faces
 
def crop_to_base64(crop_img):
    _, buffer = cv2.imencode('.jpg', crop_img)
    return base64.b64encode(buffer).decode('utf-8')
 
init_db()
 
# ============================================================
# ⚙️ PPE DETECTION USING GOOGLE CLOUD VISION
# ============================================================
VEST_KEYWORDS = ["safety vest", "high-visibility", "reflective clothing", "vest"]
SHOE_OBJECTS = ["shoe", "footwear", "boot", "safety shoe"]
CONFIDENCE_THRESHOLD_VEST = 0.6
CONFIDENCE_THRESHOLD_SHOE = 0.5
 
 
def detect_ppe(image_path):
    client = vision.ImageAnnotatorClient()
 
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()
    image = types.Image(content=content)
 
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Unable to open image.")
    h, w, _ = img.shape
 
    face_response = client.face_detection(image=image)
    label_response = client.label_detection(image=image)
    object_response = client.object_localization(image=image)
 
    faces = face_response.face_annotations
    labels = label_response.label_annotations
    objects = object_response.localized_object_annotations
 
    helmet_detected = any(face.headwear_likelihood >= vision.Likelihood.LIKELY for face in faces)
    vest_detected = any(
        any(keyword in label.description.lower() for keyword in VEST_KEYWORDS)
        and label.score >= CONFIDENCE_THRESHOLD_VEST
        for label in labels
    )
    shoe_detected = any(
        any(keyword in obj.name.lower() for keyword in SHOE_OBJECTS)
        and obj.score >= CONFIDENCE_THRESHOLD_SHOE
        for obj in objects
    )
 
    known_faces = get_all_known_faces()
    recognized_persons = []
 
    for face in faces:
        vertices = [(v.x, v.y) for v in face.bounding_poly.vertices]
        if len(vertices) == 4:
            x1, y1 = int(vertices[0][0]), int(vertices[0][1])
            x2, y2 = int(vertices[2][0]), int(vertices[2][1])
            face_crop = img[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue
 
            # Use TFLite-based embedding
            embedding = get_embedding(cv2.resize(face_crop, (160, 160)))
            db = load_database("Employees")
            identity = match_embedding(embedding, db)
 
            recognized_persons.append({
                "name": identity,
                "face_image": crop_to_base64(face_crop)
            })
 
    return {
        "helmet": helmet_detected,
        "vest": vest_detected,
        "shoes": shoe_detected,
        "persons": recognized_persons
    }
 
# ============================================================
# 🧾 API ENDPOINT — PPE Detection
# ============================================================
@app.post("/detect_ppe")
async def detect_ppe_api(file: UploadFile = File(...)):
    try:
        with NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
 
        results = detect_ppe(tmp_path)
        os.remove(tmp_path)
 
        return JSONResponse(content=results, status_code=200)
 
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
 
 
# ============================================================
# 🧍 FACE VALIDATION ENDPOINT (MobileFaceNet)
# ============================================================
@app.post("/validate-face")
async def validate_face(request: Request):
    try:
        data = await request.json()
        base64_image = data.get("image")
        if not base64_image:
            return JSONResponse({"status": "error", "message": "No image provided"}, status_code=400)
 
        image_bytes = base64.b64decode(base64_image)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
 
        if frame is None:
            return JSONResponse({"status": "error", "message": "Invalid image format"}, status_code=400)
 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
 
        if len(faces) == 0:
            return JSONResponse({"status": "error", "message": "No face detected"}, status_code=404)
        if len(faces) > 1:
            return JSONResponse({"status": "error", "message": "Multiple faces detected"}, status_code=400)
 
        (x, y, w, h) = faces[0]
        face_area = w * h
        frame_area = frame.shape[0] * frame.shape[1]
        face_ratio = face_area / frame_area
 
        if face_ratio < 0.05:
            return JSONResponse({"status": "error", "message": "Face too small or too far"}, status_code=400)
 
        brightness = np.mean(gray[y:y+h, x:x+w])
        if brightness < 60:
            return JSONResponse({"status": "error", "message": "Image too dark"}, status_code=400)
        elif brightness > 200:
            return JSONResponse({"status": "error", "message": "Image too bright"}, status_code=400)
 
        lap_var = cv2.Laplacian(gray[y:y+h, x:x+w], cv2.CV_64F).var()
        if lap_var < 80:
            return JSONResponse({"status": "error", "message": "Image blurry"}, status_code=400)
 
        return JSONResponse({
            "status": "success",
            "message": "Image is suitable for registration",
            "details": {
                "brightness": round(brightness, 2),
                "sharpness": round(lap_var, 2),
                "face_ratio": round(face_ratio, 3)
            }
        })
 
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
 
 
# ============================================================
# 👤 LOCAL FACE REGISTRATION (Extract & Save Embedding)
# ============================================================
@app.post("/local-register")
async def local_register(request: Request):
    try:
        data = await request.json()
        base64_image = data.get("image")
        name = data.get("name", "Unknown")
 
        if not base64_image:
            return JSONResponse({"status": "error", "message": "No image provided"}, status_code=400)
 
        image_bytes = base64.b64decode(base64_image)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
 
        if frame is None:
            return JSONResponse({"status": "error", "message": "Invalid image"}, status_code=400)
 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
 
        if len(faces) == 0:
            return JSONResponse({"status": "error", "message": "No face detected"}, status_code=404)
 
        (x, y, w, h) = faces[0]
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (160, 160))
 
        embedding = get_embedding(face)
        #add_person(name, embedding)
       
        return JSONResponse({"status": "Success", "embedding": embedding.tolist()})
        # return JSONResponse({"status": "Success", "name": name, "embedding": embedding.tolist()})
 
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
 
 
@app.post("/recognize")
async def rec(request: Request):
    data = await request.json()
    base64_image = data.get("embedding")
    print(base64_image)
    return JSONResponse({"status": "Success", "embedding": base64_image}, status_code=200)
 
@app.get("/")
def root():
    return {
        "message": "Welcome to PPE Detection + Face Validation API",
        "usage": [
            "POST /detect_ppe → Detect PPE & Recognize faces",
            "POST /validate-face → Validate face quality before registration",
            "POST /local-register → Register face locally (TFLite embedding)"
        ]
    }