# import cv2
# import time
# import numpy as np
# import requests
# from face_recog import get_embedding
 
# # === CONFIG ===
# RTSP_URL = "rtsp://admin:admin%40123@172.16.0.98:554/ch01/1"
# API_URL = "http://172.16.0.69:5000/api/attendance/match"
# ROI = (200, 10, 800, 400)
# FACE_SIZE_THRESHOLD = 100
# FACE_CONF_THRESHOLD = 0.9
# FACE_LOST_TIMEOUT = 3.0
 
# # === DNN MODEL ===
# net = cv2.dnn.readNetFromCaffe(
#     "models/deploy.prototxt",
#     "models/res10_300x300_ssd_iter_140000.caffemodel"
# )
 
# tracked_faces = {}
# next_face_id = 0
 
 
# def iou(boxA, boxB):
#     xA = max(boxA[0], boxB[0])
#     yA = max(boxA[1], boxB[1])
#     xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
#     yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
#     inter = max(0, xB - xA) * max(0, yB - yA)
#     union = (boxA[2] * boxA[3]) + (boxB[2] * boxB[3]) - inter
#     return inter / union if union > 0 else 0
 
 
# def match_existing_face(new_box):
#     for face_id, info in tracked_faces.items():
#         if iou(new_box, info["box"]) > 0.3:
#             return face_id
#     return None
 
 
# def send_to_api(embedding):
#     try:
#         payload = {"embedding": embedding.tolist()}
#         r = requests.post(API_URL, json=payload, timeout=3)
#         print("API:", r.json())
#     except Exception as e:
#         print("API error:", e)
 
 
# def main():
#     global next_face_id
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("❌ Failed to open RTSP stream")
#         return
 
#     print("✅ Stream started — press 'q' to quit")
 
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             time.sleep(0.5)
#             continue
 
#         x, y, w, h = ROI
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         roi_frame = frame[y:y + h, x:x + w]
 
#         blob = cv2.dnn.blobFromImage(roi_frame, 1.0, (300, 300),
#                                      (104.0, 177.0, 123.0), False, False)
#         net.setInput(blob)
#         detections = net.forward()
 
#         current_time = time.time()
#         seen_ids = []
 
#         for i in range(detections.shape[2]):
#             confidence = detections[0, 0, i, 2]
#             if confidence < FACE_CONF_THRESHOLD:
#                 continue
 
#             box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#             (fx, fy, fx2, fy2) = box.astype("int")
#             fw, fh = fx2 - fx, fy2 - fy
#             print(f"FACE WIDTH: {fw} AND FACE HEIGHT: {fh}")
#             if fw < FACE_SIZE_THRESHOLD or fh < FACE_SIZE_THRESHOLD:
#                 continue
 
#             full_box = (fx + x, fy + y, fw, fh)
#             face_id = match_existing_face(full_box)
 
#             if face_id is None:
#                 face_id = next_face_id
#                 next_face_id += 1
 
#                 face_crop = frame[full_box[1]:full_box[1]+full_box[3],
#                                   full_box[0]:full_box[0]+full_box[2]]
#                 if face_crop.size > 0:
#                     embedding = get_embedding(face_crop)
#                     send_to_api(embedding)
#                     print(f"✅ Sent embedding for Face ID {face_id}")
 
#             tracked_faces[face_id] = {"last_seen": current_time, "box": full_box}
#             seen_ids.append(face_id)
 
#             cv2.rectangle(frame, (full_box[0], full_box[1]),
#                           (full_box[0]+full_box[2], full_box[1]+full_box[3]),
#                           (255, 0, 0), 2)
#             cv2.putText(frame, f"ID:{face_id}", (full_box[0], full_box[1]-10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
 
#         for fid in list(tracked_faces.keys()):
#             if current_time - tracked_faces[fid]["last_seen"] > FACE_LOST_TIMEOUT:
#                 del tracked_faces[fid]
 
#         cv2.imshow("Fast RTSP Face ROI", frame)
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
 
#     cap.release()
#     cv2.destroyAllWindows()
 
 
# if __name__ == "__main__":
#     main()










# import cv2
# import time
# import numpy as np
# import requests
# from face_recog import get_embedding

# # === CONFIG ===
# CAMERA_INDEX = 0  # Use 0 for laptop webcam
# API_URL = "http://172.16.0.56:5000/api/attendance/match"
# FACE_SIZE_THRESHOLD = 100
# FACE_CONF_THRESHOLD = 0.9
# FACE_LOST_TIMEOUT = 3.0

# # === DNN MODEL ===
# net = cv2.dnn.readNetFromCaffe(
#     "models/deploy.prototxt",
#     "models/res10_300x300_ssd_iter_140000.caffemodel"
# )

# tracked_faces = {}
# next_face_id = 0


# def iou(boxA, boxB):
#     xA = max(boxA[0], boxB[0])
#     yA = max(boxA[1], boxB[1])
#     xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
#     yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
#     inter = max(0, xB - xA) * max(0, yB - yA)
#     union = (boxA[2] * boxA[3]) + (boxB[2] * boxB[3]) - inter
#     return inter / union if union > 0 else 0


# def match_existing_face(new_box):
#     for face_id, info in tracked_faces.items():
#         if iou(new_box, info["box"]) > 0.3:
#             return face_id
#     return None


# def send_to_api(embedding):
#     try:
#         payload = {"embedding": embedding.tolist()}
#         r = requests.post(API_URL, json=payload, timeout=3)
#         print("API:", r.json())
#     except Exception as e:
#         print("API error:", e)


# def main():
#     global next_face_id
#     cap = cv2.VideoCapture(CAMERA_INDEX)
#     if not cap.isOpened():
#         print("❌ Failed to open webcam")
#         return

#     print("✅ Webcam started — press 'q' to quit")

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             time.sleep(0.5)
#             continue

#         (h, w) = frame.shape[:2]
#         blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
#                                      1.0, (300, 300),
#                                      (104.0, 177.0, 123.0))
#         net.setInput(blob)
#         detections = net.forward()

#         current_time = time.time()
#         seen_ids = []

#         for i in range(0, detections.shape[2]):
#             confidence = detections[0, 0, i, 2]
#             if confidence < FACE_CONF_THRESHOLD:
#                 continue

#             # Get box coordinates, scaled to frame size
#             box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#             (x1, y1, x2, y2) = box.astype("int")

#             # Clip to frame bounds
#             x1, y1 = max(0, x1), max(0, y1)
#             x2, y2 = min(w - 1, x2), min(h - 1, y2)

#             fw, fh = x2 - x1, y2 - y1
#             if fw < FACE_SIZE_THRESHOLD or fh < FACE_SIZE_THRESHOLD:
#                 continue

#             full_box = (x1, y1, fw, fh)
#             face_id = match_existing_face(full_box)

#             if face_id is None:
#                 face_id = next_face_id
#                 next_face_id += 1

#                 face_crop = frame[y1:y2, x1:x2]
#                 if face_crop.size > 0:
#                     embedding = get_embedding(face_crop)
#                     send_to_api(embedding)
#                     print(f"✅ Sent embedding for Face ID {face_id}")

#             tracked_faces[face_id] = {"last_seen": current_time, "box": full_box}
#             seen_ids.append(face_id)

#             # Draw face box and ID
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(frame, f"ID:{face_id}", (x1, y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

#         # Remove faces not seen recently
#         for fid in list(tracked_faces.keys()):
#             if current_time - tracked_faces[fid]["last_seen"] > FACE_LOST_TIMEOUT:
#                 del tracked_faces[fid]

#         cv2.imshow("Webcam Face Detection", frame)
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break

#     cap.release()
#     cv2.destroyAllWindows()


# if __name__ == "__main__":
#     main()














###################################################################################
#  the below code using UDP
###################################################################################







# import cv2
# import time
# import numpy as np
# import requests
# from face_recog import get_embedding

# # === CONFIG ===
# CAMERA_INDEX = 0
# RTSP_URL = "rtsp://admin:admin%40123@172.16.0.98:554/ch01/1"
# API_URL = "http://172.16.0.42:5000/api/attendance/match"
# FACE_SIZE_THRESHOLD = 100
# FACE_CONF_THRESHOLD = 0.9
# FACE_LOST_TIMEOUT = 3.0  # seconds — after this, ID resets
# MAX_ATTEMPTS_PER_FACE = 4  # max API tries per ID

# # === DNN MODEL ===
# net = cv2.dnn.readNetFromCaffe(
#     "models/deploy.prototxt",
#     "models/res10_300x300_ssd_iter_140000.caffemodel"
# )

# tracked_faces = {}
# next_face_id = 0


# def iou(boxA, boxB):
#     """Intersection-over-Union for tracking continuity."""
#     xA = max(boxA[0], boxB[0])
#     yA = max(boxA[1], boxB[1])
#     xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
#     yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
#     inter = max(0, xB - xA) * max(0, yB - yA)
#     union = (boxA[2] * boxA[3]) + (boxB[2] * boxB[3]) - inter
#     return inter / union if union > 0 else 0


# def match_existing_face(new_box):
#     """Return existing face_id if IoU > 0.3."""
#     for face_id, info in tracked_faces.items():
#         if iou(new_box, info["box"]) > 0.3:
#             return face_id
#     return None


# def send_to_api(embedding):
#     try:
#         payload = {"embedding": embedding.tolist()}
#         r = requests.post(API_URL, json=payload, timeout=3)
#         data = r.json()
#         print("API:", data)
#         return data
#     except Exception as e:
#         print("API error:", e)
#         return None


# def try_recognize_face(face_crop, face_id):
#     """Try to recognize a face, respecting max attempts."""
#     face_info = tracked_faces[face_id]

#     # Skip if already matched or permanently unmatched
#     if face_info.get("matched") or face_info.get("unmatched_final"):
#         return

#     attempts = face_info.get("attempts", 0)
#     if attempts >= MAX_ATTEMPTS_PER_FACE:
#         face_info["unmatched_final"] = True
#         print(f"🚫 Face ID {face_id}: Reached max attempts ({MAX_ATTEMPTS_PER_FACE}). Stopping API calls.")
#         return

#     # Perform recognition attempt
#     embedding = get_embedding(face_crop)
#     response = send_to_api(embedding)
#     face_info["attempts"] = attempts + 1

#     if response and response.get("status") == "success":
#         face_info["matched"] = True
#         print(f"✅ Face ID {face_id} matched on attempt {face_info['attempts']}")
#     elif face_info["attempts"] >= MAX_ATTEMPTS_PER_FACE:
#         face_info["unmatched_final"] = True
#         print(f"❌ Face ID {face_id} not matched after {MAX_ATTEMPTS_PER_FACE} attempts.")


# def main():
#     global next_face_id
#     cap = cv2.VideoCapture(RTSP_URL) 
#     if not cap.isOpened():
#         print("❌ Failed to open webcam")
#         return

#     print("✅ Webcam started — press 'q' to quit")

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             time.sleep(0.3)
#             continue

#         (h, w) = frame.shape[:2]
#         blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
#                                      1.0, (300, 300),
#                                      (104.0, 177.0, 123.0))
#         net.setInput(blob)
#         detections = net.forward()

#         current_time = time.time()
#         seen_ids = []

#         # Process detections
#         for i in range(0, detections.shape[2]):
#             confidence = detections[0, 0, i, 2]
#             if confidence < FACE_CONF_THRESHOLD:
#                 continue

#             box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#             (x1, y1, x2, y2) = box.astype("int")
#             x1, y1 = max(0, x1), max(0, y1)
#             x2, y2 = min(w - 1, x2), min(h - 1, y2)
#             fw, fh = x2 - x1, y2 - y1

#             if fw < FACE_SIZE_THRESHOLD or fh < FACE_SIZE_THRESHOLD:
#                 continue

#             full_box = (x1, y1, fw, fh)
#             face_id = match_existing_face(full_box)

#             if face_id is None:
#                 face_id = next_face_id
#                 next_face_id += 1
#                 tracked_faces[face_id] = {
#                     "last_seen": current_time,
#                     "box": full_box,
#                     "matched": False,
#                     "attempts": 0,
#                 }

#             tracked_faces[face_id]["last_seen"] = current_time
#             tracked_faces[face_id]["box"] = full_box
#             seen_ids.append(face_id)

#             face_crop = frame[y1:y2, x1:x2]
#             if face_crop.size > 0:
#                 try_recognize_face(face_crop, face_id)

#             # Color coding
#             info = tracked_faces[face_id]
#             if info.get("matched"):
#                 color = (0, 255, 0)  # green = matched
#             elif info.get("unmatched_final"):
#                 color = (0, 0, 255)  # red = permanently unmatched
#             else:
#                 color = (0, 255, 255)  # yellow = pending

#             cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#             cv2.putText(frame, f"ID:{face_id}", (x1, y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

#         # Remove faces that left the frame
#         for fid in list(tracked_faces.keys()):
#             if current_time - tracked_faces[fid]["last_seen"] > FACE_LOST_TIMEOUT:
#                 print(f"🧹 Face ID {fid} left the frame — resetting.")
#                 del tracked_faces[fid]

#         cv2.imshow("Webcam Face Detection", frame)
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break

#     cap.release()
#     cv2.destroyAllWindows()


# if __name__ == "__main__":
#     main()
















########################################################################
# the below code using TCP ,   this code runs but still get notifying h.264 error
########################################################################












# import os
# import cv2
# import time
# import numpy as np
# import tensorflow as tf



# # cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)

# # ===============================================================
# # ✅ 1. Environment setup for stable RTSP decoding
# # ===============================================================
# # Use TCP (not UDP) for RTSP
# os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# # Mute FFmpeg decoding warnings
# os.environ["FFMPEG_LOG_LEVEL"] = "fatal"

# # ===============================================================
# # ✅ 2. Load your TFLite face model (adjust path if needed)
# # ===============================================================
# print("Loading TFLite model...")
# interpreter = tf.lite.Interpreter(model_path="MobileFaceNet.tflite")
# interpreter.allocate_tensors()
# print("✅ Model loaded successfully")

# # ===============================================================
# # ✅ 3. Camera source (RTSP or local webcam)
# # ===============================================================
# # RTSP URL
# RTSP_URL = "rtsp://admin:admin%40123@172.16.0.98:554/ch01/1"

# # Set use_rtsp = True if you want to use IP camera, else webcam
# use_rtsp = False

# def open_camera():
#     """Open RTSP or webcam with buffering and retry logic."""
#     cap = None
#     if use_rtsp:
#         print(f"🎥 Connecting to RTSP stream: {RTSP_URL}")
#         cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
#         cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
#     else:
#         print("🎥 Opening local webcam...")
#         cap = cv2.VideoCapture(0)
#     return cap

# # ===============================================================
# # ✅ 4. Helper: get embedding (stub – replace with your logic)
# # ===============================================================
# # def get_embedding(image):
# #     """Placeholder for your face embedding extraction function."""
# #     # TODO: Replace this with actual preprocessing + inference
# #     resized = cv2.resize(image, (112, 112))
# #     input_data = np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)
# #     input_details = interpreter.get_input_details()
# #     output_details = interpreter.get_output_details()
# #     interpreter.set_tensor(input_details[0]['index'], input_data)
# #     interpreter.invoke()
# #     embedding = interpreter.get_tensor(output_details[0]['index'])
# #     return embedding.flatten()

# # ===============================================================
# # ✅ 5. Main streaming + face detection loop
# # ===============================================================
# def main():
#     cap = open_camera()
#     if not cap or not cap.isOpened():
#         print("❌ Failed to open camera stream.")
#         return

#     print("✅ Webcam started — press 'q' to quit")

#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#     last_frame_time = time.time()

#     while True:
#         ret, frame = cap.read()

#         # Auto-reconnect if stream drops
#         if not ret:
#             print("⚠️ Frame read failed — reconnecting in 3 seconds...")
#             time.sleep(3)
#             cap.release()
#             cap = open_camera()
#             continue

#         # Optional resize to reduce CPU load
#         frame = cv2.resize(frame, (640, 360))

#         # Face detection
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(gray, 1.3, 5)

#         for (x, y, w, h) in faces:
#             face = frame[y:y+h, x:x+w]
#             embedding = get_embedding(face)
#             cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#             cv2.putText(frame, "Face", (x, y - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#         cv2.imshow("Live Attendance", frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#         last_frame_time = time.time()

#     cap.release()
#     cv2.destroyAllWindows()

# # ===============================================================
# # ✅ 6. Run main
# # ===============================================================
# if __name__ == "__main__":
#     try:
#         main()
#     except KeyboardInterrupt:
#         print("\n🛑 Interrupted by user.")
#     except Exception as e:
#         print(f"❌ Unexpected error: {e}")




















##############################################################################
#this will works without h264 error
##############################################################################

# import os
# import cv2
# import time
# import numpy as np
# import tensorflow as tf

# from face_recog import get_embedding

# # ===============================================================
# # ✅ 1. Environment setup for stable RTSP decoding and silent logs
# # ===============================================================
# os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
#     "rtsp_transport;tcp|fflags;nobuffer|max_delay;5000000|loglevel;quiet"
# )
# os.environ["FFMPEG_LOG_LEVEL"] = "quiet"

# print("✅ Logging muted — using TCP mode for RTSP stability")

# # ===============================================================
# # ✅ 2. Load your TFLite face model
# # ===============================================================
# print("Loading TFLite model...")
# interpreter = tf.lite.Interpreter(model_path="MobileFaceNet.tflite")
# interpreter.allocate_tensors()
# print("✅ Model loaded successfully")

# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()
# expected_shape = input_details[0]["shape"]
# print(f"📐 Model expects input shape: {expected_shape}")

# # ===============================================================
# # ✅ 3. Camera source (RTSP or local webcam)
# # ===============================================================
# RTSP_URL = "rtsp://admin:admin%40123@172.16.0.98:554/ch01/1"
# use_rtsp = False

# def open_camera():
#     if use_rtsp:
#         print(f"🎥 Connecting to RTSP stream: {RTSP_URL}")
#         cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
#         cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
#     else:
#         print("🎥 Opening local webcam...")
#         cap = cv2.VideoCapture(0)
#     return cap

# # ===============================================================
# # ✅ 4. Face embedding function (auto-reshape logic)
# # ===============================================================
# # def get_embedding(image):
# #     """Generate embedding from detected face using TFLite model."""
# #     resized = cv2.resize(image, (112, 112))
# #     input_data = resized.astype(np.float32) / 255.0

# #     # Add batch dimension (1,112,112,3)
# #     input_data = np.expand_dims(input_data, axis=0)

# #     # If model expects (2,112,112,3), pad with a blank second image
# #     expected_shape = input_details[0]["shape"]
# #     if input_data.shape != tuple(expected_shape):
# #         if expected_shape[0] == 2:
# #             blank = np.zeros_like(input_data)
# #             input_data = np.concatenate([input_data, blank], axis=0)
# #         else:
# #             input_data = np.reshape(input_data, expected_shape)

# #     # Feed into the interpreter
# #     interpreter.set_tensor(input_details[0]["index"], input_data)
# #     interpreter.invoke()
# #     embedding = interpreter.get_tensor(output_details[0]["index"])

# #     # Some models output both embeddings (2,128); we take the first
# #     if embedding.ndim == 2 and embedding.shape[0] > 1:
# #         embedding = embedding[0]

# #     return embedding.flatten()


# # ===============================================================
# # ✅ 5. Main loop for live attendance
# # ===============================================================
# def main():
#     cap = open_camera()
#     if not cap or not cap.isOpened():
#         print("❌ Failed to open camera stream.")
#         return

#     print("✅ Webcam started — press 'q' to quit")
#     face_cascade = cv2.CascadeClassifier(
#         cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
#     )

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("⚠️ Frame read failed — reconnecting in 3 seconds...")
#             time.sleep(3)
#             cap.release()
#             cap = open_camera()
#             continue

#         frame = cv2.resize(frame, (640, 360))
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(gray, 1.3, 5)

#         for (x, y, w, h) in faces:
#             face = frame[y:y+h, x:x+w]
#             embedding = get_embedding(face)
#             if embedding is not None:
#                 print("Got Embedding")
#             cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#             cv2.putText(frame, "Face", (x, y - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#         cv2.imshow("Live Attendance", frame)
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# # ===============================================================
# # ✅ 6. Entry point
# # ===============================================================
# if __name__ == "__main__":
#     try:
#         main()
#     except KeyboardInterrupt:
#         print("\n🛑 Interrupted by user.")
#     except Exception as e:
#         print(f"❌ Unexpected error: {e}")
























# import os
# import cv2
# import time
# import numpy as np
# import requests
# import threading

# from face_recog import get_embedding

# # ===============================================================
# # ✅ 1. Environment setup for stable RTSP decoding and silent logs
# # ===============================================================
# os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
#     "rtsp_transport;tcp|fflags;nobuffer|max_delay;5000000|loglevel;quiet"
# )
# os.environ["FFMPEG_LOG_LEVEL"] = "quiet"

# print("✅ Logging muted — using TCP mode for RTSP stability")

# # ===============================================================
# # ✅ 2. (Optional) TFLite model was loaded earlier in your file.
# #    We will continue to use face_recog.get_embedding(...) which
# #    you already import. If you want to use the local interpreter
# #    version, integrate that into get_embedding or replace calls.
# # ===============================================================
# # ===============================================================
# # ✅ 3. Camera source (RTSP or local webcam)
# # ===============================================================
# RTSP_URL = "rtsp://admin:admin%40123@172.16.0.98:554/ch01/1"
# use_rtsp = False

# def open_camera():
#     if use_rtsp:
#         print(f"🎥 Connecting to RTSP stream: {RTSP_URL}")
#         cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
#         # small buffer to keep latency low
#         cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
#     else:
#         print("🎥 Opening local webcam...")
#         cap = cv2.VideoCapture(0)
#     return cap

# # ===============================================================
# # ✅ 4. Config: thresholds, timeouts and API
# # ===============================================================
# API_URL = "http://172.16.0.56:5000/api/attendance/match"  # same as your original
# FACE_SIZE_THRESHOLD = 100    # min face width/height to consider
# FACE_LOST_TIMEOUT = 3.0      # seconds until a face id is removed
# IOU_MATCH_THRESHOLD = 0.3    # IoU threshold to consider same face
# MAX_API_ATTEMPTS = 4         # stop making API calls after this many attempts per face
# API_COOLDOWN = 1.0           # seconds between API calls for same face id
# REQUEST_TIMEOUT = 3          # requests.post timeout in seconds

# # ===============================================================
# # ✅ 5. Tracking structures (thread-safe updates)
# # ===============================================================
# tracked_faces = {}  # face_id -> {box, last_seen, attempts, matched, last_api}
# next_face_id = 0
# tracked_lock = threading.Lock()


# # ===============================================================
# # ✅ ROI definition (X, Y, WIDTH, HEIGHT)
# # ===============================================================
# ROI = (100, 0, 440, 260)  # (x, y, width, height)



# # ===============================================================
# # ✅ 6. Utility: IoU for box matching
# # ===============================================================
# def iou(boxA, boxB):
#     # boxes are (x, y, w, h)
#     xA = max(boxA[0], boxB[0])
#     yA = max(boxA[1], boxB[1])
#     xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
#     yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
#     inter_w = max(0, xB - xA)
#     inter_h = max(0, yB - yA)
#     inter = inter_w * inter_h
#     union = (boxA[2] * boxA[3]) + (boxB[2] * boxB[3]) - inter
#     return inter / union if union > 0 else 0.0

# def match_existing_face(new_box):
#     """
#     Return face_id if IoU with any tracked_faces > threshold, else None.
#     Uses tracked_lock for safe access.
#     """
#     with tracked_lock:
#         for face_id, info in tracked_faces.items():
#             if iou(new_box, info["box"]) > IOU_MATCH_THRESHOLD:
#                 return face_id
#     return None

# # ===============================================================
# # ✅ 7. API sending logic (runs in thread to avoid blocking loop)
# #    - respects attempts count, cooldown and marks matched if API says so
# # ===============================================================
# def send_to_api_async(embedding, face_id):
#     """
#     Called in background thread. Updates tracked_faces entries.
#     """
#     try:
#         payload = {"embedding": embedding.tolist()}
#     except Exception:
#         # embedding might be numpy array already; convert robustly
#         payload = {"embedding": np.array(embedding).flatten().tolist()}

#     # mark attempt and set last_api before making call to avoid races
#     with tracked_lock:
#         info = tracked_faces.get(face_id)
#         if info is None:
#             # face disappeared before we could send
#             return
#         # check whether we should still attempt
#         if info.get("matched"):
#             return
#         if info.get("attempts", 0) >= MAX_API_ATTEMPTS:
#             return
#         now = time.time()
#         if now - info.get("last_api", 0) < API_COOLDOWN:
#             # too soon, skip this call (main loop will try again later)
#             return
#         # update pre-call
#         info["attempts"] = info.get("attempts", 0) + 1
#         info["last_api"] = now

#     # perform POST (may raise; we catch)
#     try:
#         r = requests.post(API_URL, json=payload, timeout=REQUEST_TIMEOUT)
#         try:
#             resp = r.json()
#         except Exception:
#             resp = {"raw_text": r.text}

#         # Heuristic: if API returns a 'matched' boolean or 'status'/'success', use it.
#         matched = False
#         if isinstance(resp, dict):
#             # common possibilities: {'matched': True}, {'status': 'ok', 'matched': True}
#             if resp.get("matched") is True:
#                 matched = True
#             elif resp.get("status") in ("ok", "matched", "found"):
#                 # tolerate different API designs
#                 matched = True
#             # you can expand these heuristics depending on your API contract
#         # update tracked_faces state
#         with tracked_lock:
#             if face_id in tracked_faces:
#                 tracked_faces[face_id]["matched"] = matched or tracked_faces[face_id].get("matched", False)
#                 tracked_faces[face_id]["last_api_response"] = resp
#         print(f"API response for face {face_id}: {resp}")
#     except Exception as e:
#         # network / timeout / other error: we already incremented attempts earlier,
#         # allow main loop to try again until attempts >= MAX_API_ATTEMPTS
#         print(f"API error for face {face_id}: {e}")

# # ===============================================================
# # ✅ 8. Main loop for live attendance (detection + tracking + API)
# # ===============================================================
# def main():
#     global next_face_id
#     cap = open_camera()
#     if not cap or not cap.isOpened():
#         print("❌ Failed to open camera stream.")
#         return

#     print("✅ Camera started — press 'q' to quit")

#     face_cascade = cv2.CascadeClassifier(
#         cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
#     )

#     try:
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 print("⚠️ Frame read failed — reconnecting in 3 seconds...")
#                 time.sleep(3)
#                 cap.release()
#                 cap = open_camera()
#                 continue

#             # resize for speed/consistency
#             frame = cv2.resize(frame, (640, 360))
#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#             # detect faces
#             faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
#             current_time = time.time()
#             seen_ids = []
            
#             # Draw ROI rectangle
#             (xr, yr, rw, rh) = ROI
#             cv2.rectangle(frame, (xr, yr), (xr + rw, yr + rh), (255, 0, 0), 2)
#             cv2.putText(frame, "ROI", (xr + 5, yr + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

#             for (x, y, w, h) in faces:
#                 # ignore tiny detections
#                 if w < FACE_SIZE_THRESHOLD or h < FACE_SIZE_THRESHOLD:
#                     continue
                
#                 # ✅ Skip faces that are *outside* the ROI box
#                 if not (xr < x and y > yr and (x + w) < (xr + rw) and (y + h) < (yr + rh)):
#                     continue

#                 full_box = (int(x), int(y), int(w), int(h))
#                 face_id = match_existing_face(full_box)

#                 if face_id is None:
#                     # assign a new id
#                     with tracked_lock:
#                         face_id = next_face_id
#                         next_face_id += 1
#                         # initialize tracking record
#                         tracked_faces[face_id] = {
#                             "box": full_box,
#                             "last_seen": current_time,
#                             "attempts": 0,
#                             "matched": False,
#                             "last_api": 0,
#                             "last_api_response": None,
#                         }
#                 else:
#                     # update box/time for existing face
#                     with tracked_lock:
#                         if face_id in tracked_faces:
#                             tracked_faces[face_id]["box"] = full_box
#                             tracked_faces[face_id]["last_seen"] = current_time

#                 seen_ids.append(face_id)

#                 # If this face is not matched and hasn't exhausted attempts, try to send embedding
#                 with tracked_lock:
#                     info = tracked_faces.get(face_id)
#                     should_call = False
#                     if info and not info.get("matched", False) and info.get("attempts", 0) < MAX_API_ATTEMPTS:
#                         # respect cooldown
#                         if time.time() - info.get("last_api", 0) >= API_COOLDOWN:
#                             should_call = True

#                 if should_call:
#                     # extract face crop, compute embedding and spawn background request
#                     x1, y1, fw, fh = full_box
#                     x2, y2 = x1 + fw, y1 + fh
#                     face_crop = frame[y1:y2, x1:x2]
#                     if face_crop.size == 0:
#                         # weird crop; skip
#                         continue

#                     try:
#                         embedding = get_embedding(face_crop)
#                         if embedding is None:
#                             # Skip if embedding extraction failed; do not increment attempts here
#                             print(f"⚠️ embedding extraction returned None for face {face_id}")
#                         else:
#                             # spawn a thread to send to API so main loop isn't blocked
#                             t = threading.Thread(target=send_to_api_async, args=(embedding, face_id), daemon=True)
#                             t.start()
#                             print(f"➡️ Sent embedding (async) for face {face_id}")
#                     except Exception as e:
#                         print(f"⚠️ Error creating embedding for face {face_id}: {e}")

#                 # draw rectangle + ID + status
#                 label = f"ID:{face_id}"
#                 with tracked_lock:
#                     info = tracked_faces.get(face_id, {})
#                     if info.get("matched"):
#                         label += " ✓"
#                     else:
#                         attempts = info.get("attempts", 0)
#                         label += f" {attempts}/{MAX_API_ATTEMPTS}"

#                 cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#                 cv2.putText(frame, label, (x, y - 8),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

#             # Remove faces not seen recently
#             with tracked_lock:
#                 for fid in list(tracked_faces.keys()):
#                     if current_time - tracked_faces[fid]["last_seen"] > FACE_LOST_TIMEOUT:
#                         # optional: log final API response for record
#                         resp = tracked_faces[fid].get("last_api_response")
#                         print(f"🔁 Removing face id {fid} (last_api_resp={resp})")
#                         del tracked_faces[fid]

#             cv2.imshow("Live Attendance", frame)
#             if cv2.waitKey(1) & 0xFF == ord("q"):
#                 break

#     finally:
#         cap.release()
#         cv2.destroyAllWindows()

# # ===============================================================
# # ✅ 9. Entry point
# # ===============================================================
# if __name__ == "__main__":
#     try:
#         main()
#     except KeyboardInterrupt:
#         print("\n🛑 Interrupted by user.")
#     except Exception as e:
#         print(f"❌ Unexpected error: {e}")




















# import os
# import cv2
# import numpy as np
# import tensorflow as tf
# import time
# from collections import defaultdict
# from face_recog import get_embedding

# # ===============================================================
# # ✅ 1. Silent FFmpeg setup
# # ===============================================================
# os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
#     "rtsp_transport;tcp|fflags;nobuffer|max_delay;5000000|loglevel;quiet"
# )
# os.environ["FFMPEG_LOG_LEVEL"] = "quiet"

# print("✅ Logging muted — using TCP mode for RTSP stability")

# # ===============================================================
# # ✅ 2. Load TFLite face embedding model
# # ===============================================================
# print("Loading TFLite model...")
# interpreter = tf.lite.Interpreter(model_path="MobileFaceNet.tflite")
# interpreter.allocate_tensors()
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()
# print("✅ Model loaded successfully")

# # ===============================================================
# # ✅ 3. Load YuNet (accurate face detector)
# # ===============================================================
# yunet_model = "face_detection_yunet_2023mar.onnx"
# if not os.path.exists(yunet_model):
#     print(f"⚠️ Model '{yunet_model}' not found — download from:")
#     print("👉 https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet")
#     exit(1)

# face_detector = cv2.FaceDetectorYN_create(
#     yunet_model, "", (320, 320), 0.9, 0.3, 5000
# )

# # ===============================================================
# # ✅ 4. Camera setup
# # ===============================================================
# RTSP_URL = "rtsp://admin:admin%40123@172.16.0.98:554/ch01/1"
# use_rtsp = False

# def open_camera():
#     if use_rtsp:
#         print(f"🎥 Connecting to RTSP stream: {RTSP_URL}")
#         cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
#     else:
#         print("🎥 Opening local webcam...")
#         cap = cv2.VideoCapture(0)
#     return cap

# # ===============================================================
# # ✅ 5. Simple stable tracker (centroid + bounding box memory)
# # ===============================================================
# class CentroidTracker:
#     def __init__(self, max_distance=60, max_disappeared=10):
#         self.next_id = 0
#         self.objects = {}       # id → centroid
#         self.boxes = {}         # id → last bounding box
#         self.disappeared = {}   # id → frames since last seen
#         self.max_distance = max_distance
#         self.max_disappeared = max_disappeared

#     def register(self, centroid, box):
#         self.objects[self.next_id] = centroid
#         self.boxes[self.next_id] = box
#         self.disappeared[self.next_id] = 0
#         self.next_id += 1

#     def deregister(self, object_id):
#         self.objects.pop(object_id, None)
#         self.boxes.pop(object_id, None)
#         self.disappeared.pop(object_id, None)

#     def update(self, rects):
#         if len(rects) == 0:
#             for object_id in list(self.disappeared.keys()):
#                 self.disappeared[object_id] += 1
#                 if self.disappeared[object_id] > self.max_disappeared:
#                     self.deregister(object_id)
#             return self.boxes

#         input_centroids = np.array(
#             [(int(x + w / 2), int(y + h / 2)) for (x, y, w, h) in rects]
#         )

#         if len(self.objects) == 0:
#             for i, centroid in enumerate(input_centroids):
#                 self.register(centroid, rects[i])
#         else:
#             object_ids = list(self.objects.keys())
#             object_centroids = list(self.objects.values())

#             D = np.linalg.norm(
#                 np.array(object_centroids)[:, np.newaxis] - input_centroids[np.newaxis, :],
#                 axis=2,
#             )
#             rows = D.min(axis=1).argsort()
#             cols = D.argmin(axis=1)[rows]

#             used_rows = set()
#             used_cols = set()

#             for (row, col) in zip(rows, cols):
#                 if row in used_rows or col in used_cols:
#                     continue
#                 if D[row, col] > self.max_distance:
#                     continue
#                 object_id = object_ids[row]
#                 self.objects[object_id] = input_centroids[col]
#                 self.boxes[object_id] = rects[col]
#                 self.disappeared[object_id] = 0
#                 used_rows.add(row)
#                 used_cols.add(col)

#             # Register new objects
#             for col in range(len(input_centroids)):
#                 if col not in used_cols:
#                     self.register(input_centroids[col], rects[col])

#             # Increase disappeared count
#             for row in range(len(object_centroids)):
#                 if row not in used_rows:
#                     object_id = object_ids[row]
#                     self.disappeared[object_id] += 1
#                     if self.disappeared[object_id] > self.max_disappeared:
#                         self.deregister(object_id)

#         return self.boxes

# # ===============================================================
# # ✅ 6. Main loop (YuNet + tracking + retry logic)
# # ===============================================================
# def main():
#     cap = open_camera()
#     if not cap.isOpened():
#         print("❌ Failed to open camera stream.")
#         return

#     tracker = CentroidTracker()
#     attempts = defaultdict(int)
#     matched = set()

#     print("✅ Camera started — press 'q' to quit")

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("⚠️ Frame read failed — reconnecting...")
#             time.sleep(3)
#             cap.release()
#             cap = open_camera()
#             continue

#         h, w = frame.shape[:2]
#         face_detector.setInputSize((w, h))
#         _, faces = face_detector.detect(frame)
#         rects = []

#         if faces is not None:
#             faces = faces.astype(int)
#             for (x, y, box_w, box_h, score) in faces[:, :5]:
#                 if score >= 0.85:
#                     rects.append((x, y, box_w, box_h))

#         tracked_faces = tracker.update(rects)

#         for object_id, (x, y, bw, bh) in tracked_faces.items():
#             face_roi = frame[y:y+bh, x:x+bw]

#             # Recognition logic with retry limit
#             if object_id not in matched:
#                 if attempts[object_id] < 4:
#                     embedding = get_embedding(face_roi)
#                     # Replace with your API / matching logic
#                     is_match = np.random.rand() > 0.7
#                     if is_match:
#                         print(f"✅ Face {object_id} matched on attempt {attempts[object_id]+1}")
#                         matched.add(object_id)
#                     else:
#                         print(f"❌ Face {object_id} not matched (try {attempts[object_id]+1})")
#                     attempts[object_id] += 1

#             # Draw one clean bounding box per tracked face
#             color = (0, 255, 0) if object_id in matched else (0, 0, 255)
#             label = f"ID:{object_id} ✅" if object_id in matched else f"ID:{object_id} ({attempts[object_id]}/4)"
#             cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
#             cv2.putText(frame, label, (x, y - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

#         cv2.imshow("Face Detection + Stable Tracking", frame)
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# # ===============================================================
# # ✅ 7. Run
# # ===============================================================
# if __name__ == "__main__":
#     try:
#         main()
#     except KeyboardInterrupt:
#         print("\n🛑 Interrupted by user.")
#     except Exception as e:
#         print(f"❌ Error: {e}")










# import os
# import cv2
# import numpy as np
# import tensorflow as tf
# import time
# import requests
# from collections import defaultdict
# from face_recog import get_embedding

# # ===============================================================
# # ✅ 1. Environment setup
# # ===============================================================
# os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
#     "rtsp_transport;tcp|fflags;nobuffer|max_delay;5000000|loglevel;quiet"
# )
# os.environ["FFMPEG_LOG_LEVEL"] = "quiet"
# print("✅ Logging muted — using TCP mode for RTSP stability")

# # ===============================================================
# # ✅ 2. Face embedding model
# # ===============================================================
# print("Loading TFLite model...")
# interpreter = tf.lite.Interpreter(model_path="MobileFaceNet.tflite")
# interpreter.allocate_tensors()
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()
# print("✅ Model loaded successfully")

# # ===============================================================
# # ✅ 3. Face detection model (YuNet)
# # ===============================================================
# yunet_model = "face_detection_yunet_2023mar.onnx"
# if not os.path.exists(yunet_model):
#     print(f"⚠️ Model '{yunet_model}' not found — download from:")
#     print("👉 https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet")
#     exit(1)

# face_detector = cv2.FaceDetectorYN_create(
#     yunet_model, "", (320, 320), 0.9, 0.3, 5000
# )

# # ===============================================================
# # ✅ 4. Camera setup
# # ===============================================================
# RTSP_URL = "rtsp://admin:admin%40123@172.16.0.98:554/ch01/1"
# use_rtsp = True

# def open_camera():
#     if use_rtsp:
#         print(f"🎥 Connecting to RTSP stream: {RTSP_URL}")
#         cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
#     else:
#         print("🎥 Opening local webcam...")
#         cap = cv2.VideoCapture(0)
#     return cap

# # ===============================================================
# # ✅ 5. Simple centroid tracker
# # ===============================================================
# class CentroidTracker:
#     def __init__(self, max_distance=60, max_disappeared=10):
#         self.next_id = 0
#         self.objects = {}
#         self.boxes = {}
#         self.disappeared = {}
#         self.max_distance = max_distance
#         self.max_disappeared = max_disappeared

#     def register(self, centroid, box):
#         self.objects[self.next_id] = centroid
#         self.boxes[self.next_id] = box
#         self.disappeared[self.next_id] = 0
#         self.next_id += 1

#     def deregister(self, object_id):
#         self.objects.pop(object_id, None)
#         self.boxes.pop(object_id, None)
#         self.disappeared.pop(object_id, None)

#     def update(self, rects):
#         if len(rects) == 0:
#             for object_id in list(self.disappeared.keys()):
#                 self.disappeared[object_id] += 1
#                 if self.disappeared[object_id] > self.max_disappeared:
#                     self.deregister(object_id)
#             return self.boxes

#         input_centroids = np.array(
#             [(int(x + w / 2), int(y + h / 2)) for (x, y, w, h) in rects]
#         )

#         if len(self.objects) == 0:
#             for i, centroid in enumerate(input_centroids):
#                 self.register(centroid, rects[i])
#         else:
#             object_ids = list(self.objects.keys())
#             object_centroids = list(self.objects.values())

#             D = np.linalg.norm(
#                 np.array(object_centroids)[:, np.newaxis] - input_centroids[np.newaxis, :],
#                 axis=2,
#             )
#             rows = D.min(axis=1).argsort()
#             cols = D.argmin(axis=1)[rows]

#             used_rows, used_cols = set(), set()
#             for (row, col) in zip(rows, cols):
#                 if row in used_rows or col in used_cols:
#                     continue
#                 if D[row, col] > self.max_distance:
#                     continue
#                 object_id = object_ids[row]
#                 self.objects[object_id] = input_centroids[col]
#                 self.boxes[object_id] = rects[col]
#                 self.disappeared[object_id] = 0
#                 used_rows.add(row)
#                 used_cols.add(col)

#             for col in range(len(input_centroids)):
#                 if col not in used_cols:
#                     self.register(input_centroids[col], rects[col])

#             for row in range(len(object_centroids)):
#                 if row not in used_rows:
#                     object_id = object_ids[row]
#                     self.disappeared[object_id] += 1
#                     if self.disappeared[object_id] > self.max_disappeared:
#                         self.deregister(object_id)

#         return self.boxes

# # ===============================================================
# # ✅ 6. API function for face matching
# # ===============================================================
# API_URL = "http://172.16.0.56:5000/api/attendance/match"

# def send_to_api(embedding):
#     try:
#         payload = {"embedding": embedding.tolist()}
#         response = requests.post(API_URL, json=payload, timeout=5)
#         if response.status_code == 200:
#             data = response.json()
#             # Assuming API returns: {"matched": true, "name": "John"}
#             return data.get("status", False), data.get("name", "Unknown")
#         else:
#             print(f"⚠️ API error: {response.status_code}")
#             return False, None
#     except Exception as e:
#         print(f"❌ API call failed: {e}")
#         return False, None

# # ===============================================================
# # ✅ 7. Main loop
# # ===============================================================
# def main():
#     cap = open_camera()
#     if not cap.isOpened():
#         print("❌ Failed to open camera stream.")
#         return

#     tracker = CentroidTracker()
#     attempts = defaultdict(int)
#     matched = {}
#     print("✅ Camera started — press 'q' to quit")

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("⚠️ Frame read failed — reconnecting...")
#             time.sleep(3)
#             cap.release()
#             cap = open_camera()
#             continue

#         h, w = frame.shape[:2]
#         face_detector.setInputSize((w, h))
#         _, faces = face_detector.detect(frame)
#         rects = []

#         if faces is not None:
#             faces = faces.astype(int)
#             for (x, y, bw, bh, score) in faces[:, :5]:
#                 if score >= 0.85:
#                     rects.append((x, y, bw, bh))

#         tracked_faces = tracker.update(rects)

#         for object_id, (x, y, bw, bh) in tracked_faces.items():
#             face_roi = frame[y:y+bh, x:x+bw]

#             if object_id not in matched:
#                 if attempts[object_id] < 4:
#                     embedding = get_embedding(face_roi)
#                     success, name = send_to_api(embedding)
#                     if success == "success":
#                         print(f"✅ Face {object_id} matched: {name}")
#                         matched[object_id] = name
#                     else:
#                         print(f"❌ Face {object_id} not matched (attempt {attempts[object_id]+1})")
#                     attempts[object_id] += 1

#             color = (0, 255, 0) if object_id in matched else (0, 0, 255)
#             label = (
#                 f"{matched[object_id]}"
#                 if object_id in matched
#                 else f"ID:{object_id} ({attempts[object_id]}/4)"
#             )
#             cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
#             cv2.putText(frame, label, (x, y - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

#         cv2.imshow("Face Detection + API Matching", frame)
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# # ===============================================================
# # ✅ 8. Run
# # ===============================================================
# if __name__ == "__main__":
#     try:
#         main()
#     except KeyboardInterrupt:
#         print("\n🛑 Interrupted by user.")
#     except Exception as e:
#         print(f"❌ Error: {e}")

















# import os
# import cv2
# import numpy as np
# import tensorflow as tf
# import time
# import requests
# from collections import defaultdict
# from face_recog import get_embedding

# # ===============================================================
# # ✅ 1. Environment setup
# # ===============================================================
# os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
#     "rtsp_transport;tcp|fflags;nobuffer|max_delay;5000000|loglevel;quiet"
# )
# os.environ["FFMPEG_LOG_LEVEL"] = "quiet"
# print("✅ Logging muted — using TCP mode for RTSP stability")

# # ===============================================================
# # ✅ 2. Face embedding model
# # ===============================================================
# print("Loading TFLite model...")
# interpreter = tf.lite.Interpreter(model_path="MobileFaceNet.tflite")
# interpreter.allocate_tensors()
# print("✅ Model loaded successfully")

# # ===============================================================
# # ✅ 3. Face detection model (YuNet)
# # ===============================================================
# yunet_model = "face_detection_yunet_2023mar.onnx"
# if not os.path.exists(yunet_model):
#     print(f"⚠️ Missing '{yunet_model}'. Download from:")
#     print("👉 https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet")
#     exit(1)

# face_detector = cv2.FaceDetectorYN_create(
#     yunet_model, "", (320, 320), 0.9, 0.3, 5000
# )

# # ===============================================================
# # ✅ 4. Camera setup
# # ===============================================================
# RTSP_URL = "rtsp://admin:admin%40123@172.16.0.98:554/ch01/1"
# use_rtsp = True

# def open_camera():
#     if use_rtsp:
#         print(f"🎥 Connecting to RTSP stream: {RTSP_URL}")
#         cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
#         cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer
#     else:
#         print("🎥 Opening local webcam...")
#         cap = cv2.VideoCapture(0)
#     return cap

# # ===============================================================
# # ✅ 5. Simple centroid tracker
# # ===============================================================
# class CentroidTracker:
#     def __init__(self, max_distance=60, max_disappeared=10):
#         self.next_id = 0
#         self.objects = {}
#         self.boxes = {}
#         self.disappeared = {}
#         self.max_distance = max_distance
#         self.max_disappeared = max_disappeared

#     def register(self, centroid, box):
#         self.objects[self.next_id] = centroid
#         self.boxes[self.next_id] = box
#         self.disappeared[self.next_id] = 0
#         self.next_id += 1

#     def deregister(self, object_id):
#         self.objects.pop(object_id, None)
#         self.boxes.pop(object_id, None)
#         self.disappeared.pop(object_id, None)

#     def update(self, rects):
#         if len(rects) == 0:
#             for object_id in list(self.disappeared.keys()):
#                 self.disappeared[object_id] += 1
#                 if self.disappeared[object_id] > self.max_disappeared:
#                     self.deregister(object_id)
#             return self.boxes

#         input_centroids = np.array(
#             [(int(x + w / 2), int(y + h / 2)) for (x, y, w, h) in rects]
#         )

#         if len(self.objects) == 0:
#             for i, centroid in enumerate(input_centroids):
#                 self.register(centroid, rects[i])
#         else:
#             object_ids = list(self.objects.keys())
#             object_centroids = list(self.objects.values())
#             D = np.linalg.norm(
#                 np.array(object_centroids)[:, np.newaxis] - input_centroids[np.newaxis, :],
#                 axis=2,
#             )
#             rows = D.min(axis=1).argsort()
#             cols = D.argmin(axis=1)[rows]
#             used_rows, used_cols = set(), set()
#             for (row, col) in zip(rows, cols):
#                 if row in used_rows or col in used_cols:
#                     continue
#                 if D[row, col] > self.max_distance:
#                     continue
#                 object_id = object_ids[row]
#                 self.objects[object_id] = input_centroids[col]
#                 self.boxes[object_id] = rects[col]
#                 self.disappeared[object_id] = 0
#                 used_rows.add(row)
#                 used_cols.add(col)
#             for col in range(len(input_centroids)):
#                 if col not in used_cols:
#                     self.register(input_centroids[col], rects[col])
#             for row in range(len(object_centroids)):
#                 if row not in used_rows:
#                     object_id = object_ids[row]
#                     self.disappeared[object_id] += 1
#                     if self.disappeared[object_id] > self.max_disappeared:
#                         self.deregister(object_id)
#         return self.boxes

# # ===============================================================
# # ✅ 6. API for matching
# # ===============================================================
# API_URL = "http://172.16.0.56:5000/api/attendance/match"

# def send_to_api(embedding):
#     try:
#         payload = {"embedding": embedding.tolist()}
#         response = requests.post(API_URL, json=payload, timeout=5)
#         if response.status_code == 200:
#             data = response.json()
#             return data.get("status", False), data.get("name", "Unknown")
#         else:
#             print(f"⚠️ API error: {response.status_code}")
#             return False, None
#     except Exception as e:
#         print(f"❌ API call failed: {e}")
#         return False, None

# # ===============================================================
# # ✅ 7. Main loop
# # ===============================================================
# def main():
#     cap = open_camera()
#     if not cap.isOpened():
#         print("❌ Failed to open camera stream.")
#         return

#     tracker = CentroidTracker()
#     attempts = defaultdict(int)
#     matched = {}
#     last_api_call = defaultdict(float)  # ID → last attempt timestamp
#     frame_skip = 2                      # process every 2nd frame
#     api_gap = 1.5                       # minimum 1.5 seconds between API calls

#     print("✅ Camera started — press 'q' to quit")

#     frame_count = 0
#     while True:
#         # Always grab the latest frame (avoid lag)
#         cap.grab()
#         ret, frame = cap.read()
#         if not ret:
#             print("⚠️ Frame read failed — reconnecting...")
#             time.sleep(3)
#             cap.release()
#             cap = open_camera()
#             continue

#         frame_count += 1
#         if frame_count % frame_skip != 0:
#             continue  # skip frames for performance

#         h, w = frame.shape[:2]
#         face_detector.setInputSize((w, h))
#         _, faces = face_detector.detect(frame)
#         rects = []

#         if faces is not None:
#             faces = faces.astype(int)
#             for (x, y, bw, bh, score) in faces[:, :5]:
#                 if score >= 0.85:
#                     rects.append((x, y, bw, bh))

#         tracked_faces = tracker.update(rects)

#         for object_id, (x, y, bw, bh) in tracked_faces.items():
#             face_roi = frame[y:y+bh, x:x+bw]

#             now = time.time()
#             if object_id not in matched:
#                 # Check retry delay
#                 if attempts[object_id] < 4 and now - last_api_call[object_id] > api_gap:
#                     embedding = get_embedding(face_roi)
#                     success, name = send_to_api(embedding)
#                     last_api_call[object_id] = now  # update timestamp

#                     if success == "success":
#                         print(f"✅ Face {object_id} matched: {name}")
#                         matched[object_id] = name
#                     else:
#                         print(f"❌ Face {object_id} not matched (attempt {attempts[object_id]+1})")
#                     attempts[object_id] += 1

#             color = (0, 255, 0) if object_id in matched else (0, 0, 255)
#             label = (
#                 f"{matched[object_id]}"
#                 if object_id in matched
#                 else f"ID:{object_id} ({attempts[object_id]}/4)"
#             )
#             cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
#             cv2.putText(frame, label, (x, y - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

#         cv2.imshow("Face Detection + API Matching", frame)
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# # ===============================================================
# # ✅ 8. Run
# # ===============================================================
# if __name__ == "__main__":
#     try:
#         main()
#     except KeyboardInterrupt:
#         print("\n🛑 Interrupted by user.")
#     except Exception as e:
#         print(f"❌ Error: {e}")




















import os
import cv2
import numpy as np
import tensorflow as tf
import time
import requests
from collections import defaultdict
from face_recog import get_embedding

# ===============================================================
# ✅ 1. Environment setup
# ===============================================================
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|fflags;nobuffer|max_delay;5000000|loglevel;quiet"
)
os.environ["FFMPEG_LOG_LEVEL"] = "quiet"
print("✅ Logging muted — using TCP mode for RTSP stability")

# ===============================================================
# ✅ 2. Face embedding model
# ===============================================================
print("Loading TFLite model...")
interpreter = tf.lite.Interpreter(model_path="MobileFaceNet.tflite")
interpreter.allocate_tensors()
print("✅ Model loaded successfully")

# ===============================================================
# ✅ 3. Face detection model (YuNet)
# ===============================================================
yunet_model = "face_detection_yunet_2023mar.onnx"
if not os.path.exists(yunet_model):
    print(f"⚠️ Missing '{yunet_model}'. Download from:")
    print("👉 https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet")
    exit(1)

face_detector = cv2.FaceDetectorYN_create(
    yunet_model, "", (320, 320), 0.9, 0.3, 5000
)

# ===============================================================
# ✅ 4. Camera setup
# ===============================================================
RTSP_URL = "rtsp://admin:Admin%40123@192.168.1.116:554/ch01/0"
use_rtsp = True

def open_camera():
    if use_rtsp:
        print(f"🎥 Connecting to RTSP stream: {RTSP_URL}")
        cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    else:
        print("🎥 Opening local webcam...")
        cap = cv2.VideoCapture(0)
    return cap

# ===============================================================
# ✅ 5. Centroid Tracker
# ===============================================================
class CentroidTracker:
    def __init__(self, max_distance=60, max_disappeared=10):
        self.next_id = 0
        self.objects = {}
        self.boxes = {}
        self.disappeared = {}
        self.max_distance = max_distance
        self.max_disappeared = max_disappeared

    def register(self, centroid, box):
        self.objects[self.next_id] = centroid
        self.boxes[self.next_id] = box
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def deregister(self, object_id):
        self.objects.pop(object_id, None)
        self.boxes.pop(object_id, None)
        self.disappeared.pop(object_id, None)

    def update(self, rects):
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.boxes

        input_centroids = np.array(
            [(int(x + w / 2), int(y + h / 2)) for (x, y, w, h) in rects]
        )

        if len(self.objects) == 0:
            for i, centroid in enumerate(input_centroids):
                self.register(centroid, rects[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            D = np.linalg.norm(
                np.array(object_centroids)[:, np.newaxis] - input_centroids[np.newaxis, :],
                axis=2,
            )
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            used_rows, used_cols = set(), set()
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if D[row, col] > self.max_distance:
                    continue
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.boxes[object_id] = rects[col]
                self.disappeared[object_id] = 0
                used_rows.add(row)
                used_cols.add(col)
            for col in range(len(input_centroids)):
                if col not in used_cols:
                    self.register(input_centroids[col], rects[col])
            for row in range(len(object_centroids)):
                if row not in used_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
        return self.boxes

# ===============================================================
# ✅ 6. API for matching
# ===============================================================
API_URL = "http://192.168.1.21:5000/api/attendance/match"

def send_to_api(embedding):
    try:
        payload = {"embedding": embedding.tolist()}
        response = requests.post(API_URL, json=payload, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get("status", False), data.get("name", "Unknown")
        else:
            print(f"⚠️ API error: {response.status_code}")
            return False, None
    except Exception as e:
        print(f"❌ API call failed: {e}")
        return False, None

# ===============================================================
# ✅ 7. Mouse-based ROI selection
# ===============================================================
ROI = None
drawing = False
ix, iy = -1, -1

def draw_roi(event, x, y, flags, param):
    global ix, iy, ROI, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        temp_frame = param.copy()
        cv2.rectangle(temp_frame, (ix, iy), (x, y), (255, 0, 0), 2)
        cv2.imshow("Draw ROI", temp_frame)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        ROI = (min(ix, x), min(iy, y), max(ix, x), max(iy, y))
        print(f"🎯 ROI selected: {ROI}")

def select_roi(cap):
    print("🟦 Draw ROI using mouse — press and release left button.")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        display = frame.copy()
        cv2.imshow("Draw ROI", display)
        cv2.setMouseCallback("Draw ROI", draw_roi, display)
        if ROI is not None:
            cv2.destroyWindow("Draw ROI")
            return ROI
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("❌ ROI selection cancelled.")
            exit(0)

# ===============================================================
# ✅ 8. Main Loop with ROI filtering
# ===============================================================
def main():
    cap = open_camera()
    if not cap.isOpened():
        print("❌ Failed to open camera stream.")
        return

    # Step 1: Let user draw ROI
    roi_box = select_roi(cap)

    tracker = CentroidTracker()
    attempts = defaultdict(int)
    matched = {}
    last_api_call = defaultdict(float)
    frame_skip = 2
    api_gap = 1.5
    
    MIN_FACE_SIZE = 95

    print("✅ ROI locked. Starting recognition inside that region only.")
    print("✅ Press 'q' to quit.")

    frame_count = 0
    fps_timer = time.time()
    fps = 0

    while True:
        cap.grab()
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Frame read failed — reconnecting...")
            time.sleep(3)
            cap.release()
            cap = open_camera()
            continue

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        (rx1, ry1, rx2, ry2) = roi_box
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 0, 0), 2)
        roi_frame = frame[ry1:ry2, rx1:rx2].copy()

        # Detect faces only inside ROI
        face_detector.setInputSize((roi_frame.shape[1], roi_frame.shape[0]))
        _, faces = face_detector.detect(roi_frame)
        rects = []

        if faces is not None:
            faces = faces.astype(int)
            for (x, y, bw, bh, score) in faces[:, :5]:
                if score >= 0.45:
                    rects.append((x + rx1, y + ry1, bw, bh))

        tracked_faces = tracker.update(rects)

        # Process each face
        for object_id, (x, y, bw, bh) in tracked_faces.items():
            cx, cy = x + bw // 2, y + bh // 2
            if not (rx1 <= cx <= rx2 and ry1 <= cy <= ry2):
                continue
            
            print(f"Facewidth: {bw} Faceheight: {bh}")
            
            if bw < MIN_FACE_SIZE or bh < MIN_FACE_SIZE:
                continue

            face_roi = frame[y:y+bh, x:x+bw]
            now = time.time()
            if object_id not in matched:
                if attempts[object_id] < 4 and now - last_api_call[object_id] > api_gap:
                    embedding = get_embedding(face_roi)
                    success, name = send_to_api(embedding)
                    last_api_call[object_id] = now
                    if success == "success":
                        print(f"✅ Face {object_id} matched: {name}")
                        matched[object_id] = name
                    else:
                        print(f"❌ Face {object_id} not matched (attempt {attempts[object_id]+1})")
                    attempts[object_id] += 1

            color = (0, 255, 0) if object_id in matched else (0, 0, 255)
            label = (
                f"{matched[object_id]}"
                if object_id in matched
                else f"ID:{object_id} ({attempts[object_id]}/4)"
            )
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # FPS display
        if frame_count % 10 == 0:
            now = time.time()
            fps = 10 / (now - fps_timer)
            fps_timer = now

        # cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("ROI-based Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# ===============================================================
# ✅ 9. Run
# ===============================================================
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user.")
    except Exception as e:
        print(f"❌ Error: {e}")
