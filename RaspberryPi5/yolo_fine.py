from flask import Flask, Response, request
import subprocess, shlex, cv2, numpy as np
from ultralytics import YOLO
from threading import Thread, Lock, Event
from collections import deque
import socket
import signal
import sys
import time

# YOLO 모델 로드
model = YOLO('best.pt')  # 파인튜닝된 모델

# Flask 앱 설정
app = Flask(__name__)

# 전역 변수
buffer = b""
frame_idx = 0
last_position = (0, 0)
process_every_n_frames = 15

# 카메라 상태
current_camera = 0
process = None
camera_lock = Lock()

# 프레임 버퍼
frame_buffer = deque(maxlen=300)
buffer_lock = Lock()
buffer_ready = Event()
is_buffering = True

# UDP 설정
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# 카메라 프로세스 시작
def start_camera_process(camera_index):
    global process
    cmd = f'libcamera-vid --inline --nopreview -t 0 --codec mjpeg --width 640 --height 480 --framerate 30 -o - --camera {camera_index}'
    process = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

# 프레임 읽기 및 YOLO 추론 + UDP 전송 + 버퍼 저장
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

# 프레임 생성기 (스트리밍)
def gen_frames():
    while True:
        buffer_ready.wait()  # 최소 300프레임 모일 때까지 대기

        with buffer_lock:
            if not frame_buffer:
                continue
            frame = frame_buffer.popleft()

        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

        time.sleep(1 / 30)  # 30fps 고정 속도

@app.route('/')
def index():
    return '<html><body><h1>Live Stream</h1><img src="/video_feed"></body></html>'

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/switch_camera', methods=['POST'])
def switch_camera():
    global current_camera, buffer

    with camera_lock:
        current_camera = 1 - current_camera
        print(f"[INFO] Switching to camera {current_camera}")

        if process:
            process.terminate()
            process.wait()

        buffer = b""
        start_camera_process(current_camera)

    return f"Switched to camera {current_camera}", 200

# 안전한 종료 처리
def cleanup_and_exit(signum=None, frame=None):
    print("\n[INFO] Shutting down...")
    if process:
        process.terminate()
        process.wait()
    udp_socket.close()
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup_and_exit)  # Ctrl+C
signal.signal(signal.SIGTERM, cleanup_and_exit) # kill

# 실행
if __name__ == '__main__':
    start_camera_process(current_camera)
    t1 = Thread(target=read_frames, daemon=True)
    t1.start()
    app.run(host='0.0.0.0', port=5005, debug=False)
