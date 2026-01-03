import cv2
import numpy as np
import json
import time
from insightface.app import FaceAnalysis
from scipy.spatial.distance import cosine

# ================= CONFIG =================
DB_PATH = "faces.json"
RECOGNITION_THRESHOLD = 0.5
FACE_MIN_AREA = 500

# ================= LOAD MODEL =================
app = FaceAnalysis(name="buffalo_s")
app.prepare(ctx_id=0, det_size=(320, 320))

# ================= LOAD FACE DATABASE =================
with open(DB_PATH, "r") as f:
    raw_db = json.load(f)

db_embeddings = {
    name: np.array(emb, dtype=np.float32)
    for name, emb in raw_db.items()
}

print(f"✅ Loaded {len(db_embeddings)} registered faces")

# ================= FACE MATCH FUNCTION =================
def recognize(embedding):
    best_name = None
    best_dist = 1.0

    for name, db_emb in db_embeddings.items():
        dist = cosine(embedding, db_emb)
        if dist < best_dist and dist < RECOGNITION_THRESHOLD:
            best_dist = dist
            best_name = name

    return best_name, best_dist

# ================= ROI SELECTION =================
roi_start = roi_end = None
roi_defined = False
drawing = False

def mouse(event, x, y, flags, param):
    global roi_start, roi_end, roi_defined, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        roi_start = (x, y)
        roi_end = (x, y)
        drawing = True
        roi_defined = False

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        roi_end = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        roi_end = (x, y)
        drawing = False
        roi_defined = True

# ================= CAMERA =================
cap = cv2.VideoCapture(0)
cv2.namedWindow("Face Recognition")
cv2.setMouseCallback("Face Recognition", mouse)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()

    # Draw ROI
    if roi_start and roi_end:
        cv2.rectangle(display, roi_start, roi_end, (255, 0, 0), 2)

    if not roi_defined:
        cv2.putText(display, "Draw ROI with mouse",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2)
        cv2.imshow("Face Recognition", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    x1, y1 = roi_start
    x2, y2 = roi_end
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])

    roi = frame[y1:y2, x1:x2]
    faces = app.get(roi)

    for face in faces:
        fx1, fy1, fx2, fy2 = face.bbox.astype(int)

        if (fx2 - fx1) * (fy2 - fy1) < FACE_MIN_AREA:
            continue

        fx1 += x1
        fx2 += x1
        fy1 += y1
        fy2 += y1

        name, dist = recognize(face.embedding)

        if name:
            label = f"{name} ({dist:.2f})"
            color = (0, 255, 0)
        else:
            label = "UNKNOWN"
            color = (0, 0, 255)

        cv2.rectangle(display, (fx1, fy1), (fx2, fy2), color, 2)
        cv2.putText(display, label,
                    (fx1, fy1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2)

    cv2.imshow("Face Recognition", display)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
