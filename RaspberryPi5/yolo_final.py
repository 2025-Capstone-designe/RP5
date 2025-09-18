from flask import Flask, Response, request
from flask_cors import CORS
import subprocess, shlex, cv2, numpy as np
from ultralytics import YOLO
from threading import Thread, Lock, Event
from collections import deque
import socket
import signal
import sys
import time

# Load YOLO model
model = YOLO('best.pt')  # Fine-tuned model

# Flask app setup
app = Flask(__name__)
CORS(app)
# Global variables
buffer = b""
frame_idx = 0
last_position = (0, 0)
process_every_n_frames = 10

# Camera state
current_camera = 0
process = None
camera_lock = Lock()

# Frame buffer (up to 1800 frames = 1 minute at 30fps)
frame_buffer = deque(maxlen=1800)
buffer_lock = Lock()
buffer_ready = Event()
is_buffering = True

# UDP settings
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Start camera process
def start_camera_process(camera_index):
    global process
    cmd = f'libcamera-vid --inline --nopreview -t 0 --codec mjpeg --width 640 --height 480 --framerate 30 -o - --camera {camera_index}'
    process = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

# Read frames, run YOLO, send via UDP, and save to buffer
def read_frames():
    global buffer, frame_idx, last_position, process, is_buffering

    while True:
        with camera_lock:
            if process is None:
                continue
            try:
                buffer += process.stdout.read(4096)
            except Exception:
                continue

        a = buffer.find(b'\xff\xd8')
        b_idx = buffer.find(b'\xff\xd9')

        if a != -1 and b_idx != -1:
            jpg = buffer[a:b_idx+2]
            buffer = buffer[b_idx+2:]

            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue

            if frame_idx % process_every_n_frames == 0:
                results = model.track(frame, persist=True, classes=[0])
                boxes = results[0].boxes
                if boxes is not None and boxes.xywh is not None and len(boxes) > 0:
                    centers = boxes.xywh.cpu().numpy()[:, :2]
                    x, y = map(int, centers[0])
                    last_position = (x, y)

                message = f"{last_position[0]},{last_position[1]}"
                print(f"[YOLO] Object center: {message}")
                try:
                    udp_socket.sendto(message.encode(), (UDP_IP, UDP_PORT))
                except Exception as e:
                    print(f"[UDP] Send error: {e}")

            with buffer_lock:
                frame_buffer.append(frame.copy())
                if len(frame_buffer) >= 300:
                    buffer_ready.set()
                    is_buffering = False
                elif len(frame_buffer) <= 30:
                    buffer_ready.clear()
                    is_buffering = True

            frame_idx += 1

# Frame generator for MJPEG streaming - 30fps, drop old frames if over 900
def gen_frames():
    frame_interval = 1.0 / 30.0  # 30fps
    frame_count = 0

    buffer_ready.wait()
    last_frame_time = time.time()
    start_cycle_time = time.time()  # Start of 3-minute cycle

    while True:
        buffer_ready.wait()

        # Check if 3 minutes (180 seconds) have passed
        if time.time() - start_cycle_time >= 180:
            print("[INFO] 3 minutes passed. Clearing frame buffer.")
            with buffer_lock:
                frame_buffer.clear()
                buffer_ready.clear()  # Wait until frames are collected again
                is_buffering = True
            start_cycle_time = time.time()  # Reset cycle timer
            continue

        with buffer_lock:
            if not frame_buffer:
                continue

            if len(frame_buffer) >= 900:
                latest_frame_idx = frame_count + len(frame_buffer) - 1
                delay = latest_frame_idx - frame_count
                if delay > 900:
                    discard_count = delay - 900
                    for _ in range(discard_count):
                        if frame_buffer:
                            frame_buffer.popleft()

            if not frame_buffer:
                continue

            frame = frame_buffer.popleft()

        now = time.time()
        elapsed = now - last_frame_time
        sleep_time = frame_interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

        last_frame_time = time.time()
        frame_count += 1

        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')


@app.route('/')
def index():
    return '<html><body><h1>Live Stream</h1><img src="/video_feed"></body></html>'

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


current_mode = 'day'

@app.route('/switch_camera_day', methods=['POST'])
def switch_camera_day():
    global current_camera, buffer, current_mode

    if current_mode == 'day':
        print("[INFO] Already in day mode. Ignored.")
        return "Already in day mode", 200

    with camera_lock:
        current_camera = 0
        current_mode = 'day'
        print(f"[INFO] Switching to day mode. Camera {current_camera}.")

        if process:
            process.terminate()
            process.wait()

        buffer = b""
        start_camera_process(current_camera)

    return "Switched to day mode", 200

@app.route('/switch_camera_night', methods=['POST'])
def switch_camera_night():
    global current_camera, buffer, current_mode

    if current_mode == 'night':
        print("[INFO] Already in night mode. Ignored.")
        return "Already in night mode", 200

    with camera_lock:
        current_camera = 1
        current_mode = 'night'
        print(f"[INFO] Switching to night mode. Camera {current_camera}.")

        if process:
            process.terminate()
            process.wait()

        buffer = b""
        start_camera_process(current_camera)

    return "Switched to night mode", 200

# Graceful shutdown
def cleanup_and_exit(signum=None, frame=None):
    print("\n[INFO] Shutting down...")
    if process:
        process.terminate()
        process.wait()
    udp_socket.close()
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup_and_exit)   # Ctrl+C
signal.signal(signal.SIGTERM, cleanup_and_exit)  # kill

# Run app
if __name__ == '__main__':
    start_camera_process(current_camera)
    t1 = Thread(target=read_frames, daemon=True)
    t1.start()
    app.run(host='0.0.0.0', port=5005, debug=False, ssl_context=('cert.pem', 'key.pem'))
