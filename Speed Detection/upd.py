import cv2
import dlib
import time
import math
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8s.pt')

vehicle_classes = ['car', 'motorcycle', 'bus', 'truck']

WIDTH = 1280
HEIGHT = 720

def estimateSpeed(location1, location2):
    d_pixels = math.sqrt((location2[0] - location1[0])**2 + (location2[1] - location1[1])**2)
    ppm = 8.8  # pixels per meter (adjust to your camera setup)
    d_meters = d_pixels / ppm
    fps = 18  # frames per second of your video/camera
    speed = d_meters * fps * 3.6  # convert m/s to km/h
    return speed

def detect_vehicles(image):
    results = model(image)[0]
    boxes = []
    for result in results.boxes:
        cls_id = int(result.cls.cpu().numpy())
        conf = float(result.conf.cpu().numpy())
        class_name = model.names[cls_id]
        if class_name in vehicle_classes and conf > 0.3:
            x1, y1, x2, y2 = map(int, result.xyxy.cpu().numpy()[0])
            boxes.append((x1, y1, x2 - x1, y2 - y1))
    return boxes

def trackVehicles():
    video = cv2.VideoCapture("rtsp://admin:Admin%40123@172.16.0.221:554/ch01/1")

    carTracker = {}
    carLocation1 = {}
    carLocation2 = {}
    speed = [None] * 1000
    currentCarID = 0
    frameCounter = 0

    while True:
        ret, frame = video.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, (WIDTH, HEIGHT))
        resultImage = frame.copy()
        frameCounter += 1

        # Update existing trackers
        carIDsToDelete = []
        for carID in list(carTracker.keys()):
            trackingQuality = carTracker[carID].update(frame)
            if trackingQuality < 7:
                carIDsToDelete.append(carID)

        for carID in carIDsToDelete:
            carTracker.pop(carID, None)
            carLocation1.pop(carID, None)
            carLocation2.pop(carID, None)
            speed[carID] = None

        # Detect vehicles every 10 frames
        if frameCounter % 10 == 0:
            boxes = detect_vehicles(frame)
            
            for (x, y, w, h) in boxes:
                x_bar = x + 0.5 * w
                y_bar = y + 0.5 * h
                
                matchedCarID = None
                for carID in carTracker.keys():
                    pos = carTracker[carID].get_position()
                    t_x = int(pos.left())
                    t_y = int(pos.top())
                    t_w = int(pos.width())
                    t_h = int(pos.height())
                    t_x_bar = t_x + 0.5 * t_w
                    t_y_bar = t_y + 0.5 * t_h

                    # Check if detected box overlaps with tracker box center
                    if (t_x <= x_bar <= t_x + t_w) and (t_y <= y_bar <= t_y + t_h) and \
                       (x <= t_x_bar <= x + w) and (y <= t_y_bar <= y + h):
                        matchedCarID = carID
                        break
                
                # If no match, create a new tracker
                if matchedCarID is None:
                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(x, y, x + w, y + h)
                    tracker.start_track(frame, rect)
                    carTracker[currentCarID] = tracker
                    carLocation1[currentCarID] = [x, y, w, h]
                    speed[currentCarID] = None
                    currentCarID += 1

        # Update tracked objects and estimate speed
        for carID in carTracker.keys():
            pos = carTracker[carID].get_position()
            t_x = int(pos.left())
            t_y = int(pos.top())
            t_w = int(pos.width())
            t_h = int(pos.height())

            cv2.rectangle(resultImage, (t_x, t_y), (t_x + t_w, t_y + t_h), (0, 255, 0), 3)
            carLocation2[carID] = [t_x, t_y, t_w, t_h]

        for i in carLocation1.keys():
            if frameCounter % 1 == 0:
                x1, y1, w1, h1 = carLocation1[i]
                x2, y2, w2, h2 = carLocation2[i]
                
                carLocation1[i] = [x2, y2, w2, h2]

                if [x1, y1, w1, h1] != [x2, y2, w2, h2]:
                    if (speed[i] is None or speed[i] == 0) and y1 >= 275 and y1 <= 285:
                        speed[i] = estimateSpeed([x1, y1, w1, h1], [x2, y2, w2, h2])

                    if speed[i] is not None and y1 >= 180:
                        cv2.putText(resultImage, f"{int(speed[i])} km/h", (x1 + w1 // 2, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        cv2.imshow("Vehicle Detection and Speed Estimation", resultImage)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    trackVehicles()
