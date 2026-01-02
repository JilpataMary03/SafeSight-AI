
import cv2
import numpy as np
import time
import os
from collections import deque

# ---------------------------------------------------
# CONFIG – tune these for your site
# ---------------------------------------------------
ANALYSIS_WIDTH = 320

# --- SHAKE (OPTICAL FLOW BASED) ---
SHAKE_MOTION_THRESHOLD = 2.0     # pixels
SHAKE_DIRECTION_STD = 0.5        # radians

# --- BLACK COVER ---
BLACK_BRIGHTNESS_THRESHOLD = 40
BLACK_DARK_RATIO_THRESHOLD = 0.8
BLACK_VARIANCE_THRESHOLD = 100

# --- BLUR ---
BLUR_THRESHOLD = 60

# --- FOG / DUST ---
FOG_CONTRAST_THRESHOLD = 25
FOG_SAT_THRESHOLD = 40

CONSEC_FRAMES_TAMPER = 5
PRINT_EVERY_N_FRAMES = 10
SNAPSHOT_COOLDOWN = 10

# ---------------------------------------------------
# SNAPSHOT FOLDER
# ---------------------------------------------------
SNAPSHOT_DIR = "snapshots"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# Counters
shake_count = 0
black_count = 0
blur_count = 0
fog_count = 0

# Snapshot cooldown
last_snap = {
    "shake": 0,
    "black": 0,
    "blur": 0,
    "fog": 0
}

# ---------------------------------------------------
# FUNCTIONS
# ---------------------------------------------------
def resize_for_analysis(frame):
    h, w = frame.shape[:2]
    scale = ANALYSIS_WIDTH / float(w)
    return cv2.resize(frame, (ANALYSIS_WIDTH, int(h * scale)))


def detect_camera_shake(prev_gray, curr_gray):
    p0 = cv2.goodFeaturesToTrack(
        prev_gray, maxCorners=100,
        qualityLevel=0.3, minDistance=7, blockSize=7
    )

    if p0 is None:
        return 0.0, 0.0

    p1, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None)
    if p1 is None:
        return 0.0, 0.0

    good_old = p0[status == 1]
    good_new = p1[status == 1]

    motion = good_new - good_old
    magnitudes = np.linalg.norm(motion, axis=1)
    angles = np.arctan2(motion[:, 1], motion[:, 0])

    return float(np.mean(magnitudes)), float(np.std(angles))


def detect_black_cover(gray):
    brightness = float(np.mean(gray))
    variance = float(np.var(gray))
    dark_ratio = np.sum(gray < 30) / gray.size

    covered = (
        brightness < BLACK_BRIGHTNESS_THRESHOLD and
        dark_ratio > BLACK_DARK_RATIO_THRESHOLD and
        variance < BLACK_VARIANCE_THRESHOLD
    )

    return covered, brightness, dark_ratio, variance


def detect_blur(gray):
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm < BLUR_THRESHOLD, float(fm)


def detect_fog(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    contrast = float(gray.std())
    saturation = float(np.mean(hsv[:, :, 1]))

    fog = contrast < FOG_CONTRAST_THRESHOLD and saturation < FOG_SAT_THRESHOLD
    return fog, contrast, saturation


def save_snapshot(prefix, frame):
    now = time.time()
    if now - last_snap[prefix] >= SNAPSHOT_COOLDOWN:
        cv2.imwrite(f"{SNAPSHOT_DIR}/{prefix}_{int(now)}.jpg", frame)
        last_snap[prefix] = now


# ---------------------------------------------------
# MAIN
# ---------------------------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Camera not opened")
    exit(1)

cv2.namedWindow("CAMERA TAMPER DETECTION", cv2.WINDOW_AUTOSIZE)

ret, frame = cap.read()
if not ret:
    print("Error: Cannot read camera")
    exit(1)

small = resize_for_analysis(frame)
prev_gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

frame_idx = 0
t_last = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1

    small = resize_for_analysis(frame)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    # -------- BLACK COVER --------
    black_now, brightness, dark_ratio, variance = detect_black_cover(gray)
    black_count = black_count + 1 if black_now else 0
    black = black_count >= CONSEC_FRAMES_TAMPER

    if black:
        cv2.putText(frame, "TAMPER: BLACK / COVERED", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        save_snapshot("black", frame)

    # -------- SHAKE --------
    mean_motion, angle_std = detect_camera_shake(prev_gray, gray)

    if not black and mean_motion > SHAKE_MOTION_THRESHOLD and angle_std < SHAKE_DIRECTION_STD:
        shake_count += 1
    else:
        shake_count = 0

    shake = shake_count >= CONSEC_FRAMES_TAMPER

    if shake:
        cv2.putText(frame, "TAMPER: CAMERA SHAKING", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        save_snapshot("shake", frame)

    # -------- BLUR --------
    blur_now, blur_val = detect_blur(gray)
    blur_count = blur_count + 1 if blur_now else 0
    blur = blur_count >= CONSEC_FRAMES_TAMPER

    if blur:
        cv2.putText(frame, "TAMPER: BLURRY", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        save_snapshot("blur", frame)

    # -------- FOG --------
    fog_now, contrast, saturation = detect_fog(small)
    fog_count = fog_count + 1 if fog_now else 0
    fog = fog_count >= CONSEC_FRAMES_TAMPER

    if fog:
        cv2.putText(frame, "TAMPER: FOG / DUST", (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        save_snapshot("fog", frame)

    # -------- METRICS --------
    if frame_idx % PRINT_EVERY_N_FRAMES == 0:
        fps = PRINT_EVERY_N_FRAMES / (time.time() - t_last)
        t_last = time.time()

        print(
            f"FPS:{fps:.1f} | "
            f"Motion:{mean_motion:.2f} | AngleStd:{angle_std:.2f} | "
            f"Bright:{brightness:.1f} | DarkRatio:{dark_ratio:.2f} | Var:{variance:.1f} | "
            f"Blur:{blur_val:.1f} | Contrast:{contrast:.1f} | Sat:{saturation:.1f}"
        )

    prev_gray = gray
    cv2.imshow("CAMERA TAMPER DETECTION", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
