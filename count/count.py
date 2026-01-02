
import cv2
import json
import time
import uuid
import threading
import os
from datetime import datetime
from ultralytics import YOLO
import paho.mqtt.client as mqtt

# =====================================================
# CONFIG
# =====================================================
SOURCE = "people.mp4"  #your input video or rtsp
RESIZE_W, RESIZE_H = 960, 540
CONF_THRES = 0.3
INFER_IMGSZ = 960

SITE_ID = "site_001"
CAMERA_ID = "cam_001"
EDGE_DEVICE_ID = "edge_device_001"
MODEL_VERSION = "person_detect_v1.8"

MAX_CAPACITY = 15
COUNT_INTERVAL = 30          # seconds
STABLE_TIME = 2.0            # entry/exit stability

# ---------------- MQTT ----------------
MQTT_BROKER = "localhost"     # or "127.0.0.1"
MQTT_PORT = 1883
MQTT_USERNAME = None          # no auth for local Mosquitto
MQTT_PASSWORD = None

COUNT_TOPIC = f"site/{SITE_ID}/camera/{CAMERA_ID}/count"
EVENT_TOPIC = f"site/{SITE_ID}/camera/{CAMERA_ID}/events/entry_exit"
ALERT_TOPIC = f"site/{SITE_ID}/camera/{CAMERA_ID}/alert/capacity"

# =====================================================
# SHARED STATE (THREAD SAFE)
# =====================================================
lock = threading.Lock()
shared = {
    "count": 0,
    "prev_count": 0,
    "boxes": [],
    "conf": [],
    "entries_1min": 0,
    "exits_1min": 0,
    "frame_copy": None
}

# =====================================================
# MQTT INIT
# =====================================================
def init_mqtt():
    client = mqtt.Client(client_id=f"safesight_{uuid.uuid4()}", protocol=mqtt.MQTTv311)
    client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_start()
    print("✅ Connected to MQTT broker")
    return client

mqtt_client = init_mqtt()

# =====================================================
# LOAD YOLO MODEL
# =====================================================
model = YOLO("yolov8n.pt")

# =====================================================
# THREAD 1 — PEOPLE COUNT (EVERY 30s)
# =====================================================
def people_count_thread():
    while True:
        time.sleep(COUNT_INTERVAL)

        with lock:
            total = shared["count"]
            prev = shared["prev_count"]
            entries = shared["entries_1min"]
            exits = shared["exits_1min"]
            confs = shared["conf"][:]

            shared["entries_1min"] = 0
            shared["exits_1min"] = 0
            shared["prev_count"] = total

        change = total - prev
        reason = "person_entered" if change > 0 else "person_exited" if change < 0 else "no_change"
        occupancy = round((total / MAX_CAPACITY) * 100, 2)

        status = "normal" if occupancy < 80 else "warning" if occupancy < 100 else "critical"

        payload = {
            "message_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "site_id": SITE_ID,
            "camera_id": CAMERA_ID,
            "message_type": "people_count_update",
            "count_data": {
                "total_count": total,
                "previous_count": prev,
                "count_change": change,
                "max_capacity": MAX_CAPACITY,
                "occupancy_percentage": occupancy,
                "status": status,
                "count_change_reason": reason,
                "entry_exit_tracking": {
                    "entries_last_minute": entries,
                    "exits_last_minute": exits,
                    "net_change_last_minute": entries - exits
                }
            },
            "confidence_metrics": {
                "detection_confidence_avg": round(sum(confs) / len(confs), 2) if confs else 0
            },
            "metadata": {
                "model_version": MODEL_VERSION,
                "edge_device_id": EDGE_DEVICE_ID
            }
        }

        mqtt_client.publish(COUNT_TOPIC, json.dumps(payload), qos=1)
        print("📤 COUNT SENT")

# =====================================================
# THREAD 2 — ENTRY / EXIT EVENTS
# =====================================================
def entry_exit_thread():
    last_count = 0
    pending = None
    start_ts = 0
    last_event_time = 0
    COOLDOWN = 2.5

    while True:
        time.sleep(0.1)

        with lock:
            count = shared["count"]
            boxes = shared["boxes"][:]
            confs = shared["conf"][:]

        now = time.time()

        if pending is None:
            if count > last_count:
                pending = "entry"
                start_ts = now
            elif count < last_count:
                pending = "exit"
                start_ts = now
        else:
            if now - start_ts >= STABLE_TIME and now - last_event_time >= COOLDOWN:
                payload = {
                    "message_id": str(uuid.uuid4()),
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "site_id": SITE_ID,
                    "camera_id": CAMERA_ID,
                    "event_type": f"person_{pending}",
                    "direction": "entering" if pending == "entry" else "exiting",
                    "confidence": round(sum(confs) / len(confs), 2) if confs else 0
                }

                mqtt_client.publish(EVENT_TOPIC, json.dumps(payload), qos=1)
                print(f"🚶 {pending.upper()} EVENT SENT")

                with lock:
                    if pending == "entry":
                        shared["entries_1min"] += 1
                    else:
                        shared["exits_1min"] += 1

                last_count = count
                pending = None
                last_event_time = now

# =====================================================
# THREAD 3 — CAPACITY ALERT
# =====================================================
def capacity_alert_thread():
    over_start = None
    last_alert_time = 0

    while True:
        time.sleep(1)

        with lock:
            count = shared["count"]
            frame_snapshot = shared.get("frame_copy", None)

        now = time.time()

        if count > MAX_CAPACITY:
            if over_start is None:
                over_start = now

            if now - last_alert_time >= 30:
                snapshot_path = None
                if frame_snapshot is not None:
                    os.makedirs("capacity_snapshots", exist_ok=True)
                    snapshot_path = f"capacity_snapshots/capacity_{int(time.time())}.jpg"
                    cv2.imwrite(snapshot_path, frame_snapshot)

                payload = {
                    "message_id": str(uuid.uuid4()),
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "site_id": SITE_ID,
                    "camera_id": CAMERA_ID,
                    "alert_type": "capacity_exceeded",
                    "severity": "high",
                    "violation_details": {
                        "current_count": count,
                        "max_capacity": MAX_CAPACITY,
                        "overcapacity_by": count - MAX_CAPACITY,
                        "occupancy_percentage": round((count / MAX_CAPACITY) * 100, 2)
                    }
                }

                mqtt_client.publish(ALERT_TOPIC, json.dumps(payload), qos=1)
                print("🚨 CAPACITY ALERT SENT")
                last_alert_time = now
        else:
            over_start = None
            last_alert_time = 0

# =====================================================
# START THREADS
# =====================================================
threading.Thread(target=people_count_thread, daemon=True).start()
threading.Thread(target=entry_exit_thread, daemon=True).start()
threading.Thread(target=capacity_alert_thread, daemon=True).start()

# =====================================================
# MAIN VIDEO LOOP
# =====================================================
cap = cv2.VideoCapture(SOURCE)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (RESIZE_W, RESIZE_H))
    results = model(frame, conf=CONF_THRES, imgsz=INFER_IMGSZ, verbose=False)

    boxes = []
    confs = []

    for r in results:
        for b in r.boxes:
            if int(b.cls[0]) == 0:  # person
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                boxes.append((x1, y1, x2, y2))
                confs.append(float(b.conf[0]))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    count = len(boxes)
    occupancy = (count / MAX_CAPACITY) * 100

    if occupancy < 80:
        color = (0, 255, 0)
    elif occupancy < 100:
        color = (0, 165, 255)
    else:
        color = (0, 0, 255)

    # 🔥 ON-SCREEN COUNT DISPLAY
    cv2.putText(
        frame,
        f"People: {count} / {MAX_CAPACITY}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        color,
        2,
        cv2.LINE_AA
    )

    with lock:
        shared["count"] = count
        shared["boxes"] = boxes
        shared["conf"] = confs
        shared["frame_copy"] = frame.copy()

    cv2.imshow("SAFESIGHT AI", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()





#   "C:\Program Files\mosquitto\mosquitto.exe" -v
