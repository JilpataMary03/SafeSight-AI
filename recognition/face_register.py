import cv2
import json
import os
import numpy as np
from insightface.app import FaceAnalysis

# ======================================================
# CONFIG
# ======================================================
DB_PATH = "faces.json"
FACE_MIN_AREA = 500
SAMPLES_PER_PERSON = 5   # number of frames to capture

# ======================================================
# LOAD MODEL
# ======================================================
app = FaceAnalysis(name="buffalo_s")
app.prepare(ctx_id=0, det_size=(320, 320))

# ======================================================
# LOAD / INIT DATABASE
# ======================================================
if os.path.exists(DB_PATH):
    with open(DB_PATH, "r") as f:
        db = json.load(f)
else:
    db = {}

print(f"📂 Database loaded | {len(db)} persons registered")

# ======================================================
# CAMERA
# ======================================================
cap = cv2.VideoCapture(0)

print("\n🚀 Instructions:")
print(" - Enter person name")
print(" - Look at camera")
print(" - Press 's' to start capture")
print(" - Press 'q' to quit\n")

name = input("📝 Enter person name: ").strip()
if name == "":
    raise ValueError("Name cannot be empty")

embeddings = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = app.get(frame)

    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        area = (x2 - x1) * (y2 - y1)

        if area < FACE_MIN_AREA:
            continue

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"Samples: {len(embeddings)}/{SAMPLES_PER_PERSON}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            embeddings.append(face.embedding)
            print(f"📸 Captured sample {len(embeddings)}")

            if len(embeddings) >= SAMPLES_PER_PERSON:
                avg_embedding = np.mean(embeddings, axis=0)

                db[name] = avg_embedding.tolist()

                with open(DB_PATH, "w") as f:
                    json.dump(db, f, indent=4)

                print(f"\n✅ Registration completed for: {name}")
                print(f"🧠 Embedding size: {len(avg_embedding)}")
                cap.release()
                cv2.destroyAllWindows()
                exit(0)

    cv2.imshow("Face Registration", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
