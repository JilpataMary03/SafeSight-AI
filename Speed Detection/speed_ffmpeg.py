import cv2
import dlib
import time
import math
import subprocess
import os

carCascade = cv2.CascadeClassifier('myhaar.xml')
video_path = "input.mp4"
video = cv2.VideoCapture(video_path)

WIDTH = 1280
HEIGHT = 720
FPS = 18

# Create HLS output directory
hls_dir = "./static"
os.makedirs(hls_dir, exist_ok=True)

# FFmpeg HLS streaming command
ffmpeg_cmd = [
    "ffmpeg",
    "-y",
    "-f", "rawvideo",
    "-vcodec", "rawvideo",
    "-pix_fmt", "bgr24",
    "-s", f"{WIDTH}x{HEIGHT}",
    "-r", str(FPS),
    "-i", "-",
    "-c:v", "libx264",
    "-preset", "ultrafast",
    "-f", "hls",
    "-hls_time", "2",
    "-hls_list_size", "4",
    "-hls_flags", "delete_segments",
    os.path.join(hls_dir, "stream.m3u8")
]

# Start FFmpeg subprocess
ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

def estimateSpeed(location1, location2):
    d_pixels = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
    ppm = 8.8  # pixels per meter
    d_meters = d_pixels / ppm
    speed = d_meters * FPS * 3.6
    return speed

def trackMultipleObjects():
    rectangleColor = (0, 255, 0)
    frameCounter = 0
    currentCarID = 0
    
    carTracker = {}
    carLocation1 = {}
    carLocation2 = {}
    speed = [None] * 1000

    while True:
        ret, image = video.read()
        if not ret:
            break

        image = cv2.resize(image, (WIDTH, HEIGHT))
        resultImage = image.copy()
        frameCounter += 1

        # Remove low-quality trackers
        carIDtoDelete = [carID for carID in carTracker if carTracker[carID].update(image) < 7]
        for carID in carIDtoDelete:
            carTracker.pop(carID, None)
            carLocation1.pop(carID, None)
            carLocation2.pop(carID, None)

        # Detect new cars every 10 frames
        if frameCounter % 10 == 0:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cars = carCascade.detectMultiScale(gray, 1.1, 13, 18, (24, 24))

            for (_x, _y, _w, _h) in cars:
                x, y, w, h = int(_x), int(_y), int(_w), int(_h)
                x_bar = x + 0.5 * w
                y_bar = y + 0.5 * h

                matchCarID = None
                for carID in carTracker:
                    trackedPosition = carTracker[carID].get_position()
                    t_x = int(trackedPosition.left())
                    t_y = int(trackedPosition.top())
                    t_w = int(trackedPosition.width())
                    t_h = int(trackedPosition.height())

                    t_x_bar = t_x + 0.5 * t_w
                    t_y_bar = t_y + 0.5 * t_h

                    if (t_x <= x_bar <= (t_x + t_w)) and (t_y <= y_bar <= (t_y + t_h)) \
                            and (x <= t_x_bar <= (x + w)) and (y <= t_y_bar <= (y + h)):
                        matchCarID = carID

                if matchCarID is None:
                    tracker = dlib.correlation_tracker()
                    tracker.start_track(image, dlib.rectangle(x, y, x + w, y + h))
                    carTracker[currentCarID] = tracker
                    carLocation1[currentCarID] = [x, y, w, h]
                    currentCarID += 1

        for carID in carTracker:
            trackedPosition = carTracker[carID].get_position()
            t_x, t_y = int(trackedPosition.left()), int(trackedPosition.top())
            t_w, t_h = int(trackedPosition.width()), int(trackedPosition.height())
            cv2.rectangle(resultImage, (t_x, t_y), (t_x + t_w, t_y + t_h), rectangleColor, 4)
            carLocation2[carID] = [t_x, t_y, t_w, t_h]

        # Speed calculation
        for i in list(carLocation1.keys()):
            if frameCounter % 1 == 0 and i in carLocation2:
                [x1, y1, w1, h1] = carLocation1[i]
                [x2, y2, w2, h2] = carLocation2[i]
                carLocation1[i] = [x2, y2, w2, h2]

                if [x1, y1, w1, h1] != [x2, y2, w2, h2]:
                    if (speed[i] is None or speed[i] == 0) and 275 <= y1 <= 285:
                        speed[i] = estimateSpeed([x1, y1, w1, h1], [x2, y2, w2, h2])

                    if speed[i] is not None and y1 >= 180:
                        cv2.putText(resultImage, str(int(speed[i])) + " km/h",
                                    (int(x1 + w1 / 2), int(y1 - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        # Write annotated frame to FFmpeg
        try:
            ffmpeg_proc.stdin.write(resultImage.tobytes())
        except (BrokenPipeError, IOError):
            print("❌ FFmpeg pipe broken.")
            break

    video.release()
    ffmpeg_proc.stdin.close()
    ffmpeg_proc.wait()

if __name__ == '__main__':
    trackMultipleObjects()
