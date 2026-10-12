import socket
import struct
import pickle
import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

HOST = "0.0.0.0"
PORT = 9999

CONTROL_PORT = 10000

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(1)

print("接続待機中...")
conn, addr = server_socket.accept()
print("接続:", addr)

control_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
control_socket.connect((addr[0], CONTROL_PORT))

data = b""
payload_size = struct.calcsize("Q")

model_path = "face_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

landmarker = FaceLandmarker.create_from_options(options)

show_facemesh = False

try:
    while True:
        # サイズ情報を受信
        while len(data) < payload_size:
            packet = conn.recv(4096)
            if not packet:
                break
            data += packet

        if len(data) < payload_size:
            break

        packed_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack("Q", packed_size)[0]

        # 画像データ本体を受信
        while len(data) < msg_size:
            packet = conn.recv(4096)
            if not packet:
                break
            data += packet

        frame_data = data[:msg_size]
        data = data[msg_size:]

        encoded = pickle.loads(frame_data)

        frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

        if frame is None:
            continue

        display_frame = frame.copy()

        if show_facemesh:

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(time.perf_counter() * 1000)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.face_landmarks:
                h, w, _ = display_frame.shape

                for face_landmarks in result.face_landmarks:
                    for lm in face_landmarks:
                        x = int(lm.x * w)
                        y = int(lm.y * h)
                        cv2.circle(display_frame, (x, y), 1, (0, 255, 0), -1)

        cv2.imshow("Raspberry Pi Camera", display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord(' '):
            control_socket.sendall(b"TOGGLE_IR")

        elif key == ord('f'):
            show_facemesh = not show_facemesh
            print("FaceMesh ON" if show_facemesh else "FaceMesh OFF")

finally:
    conn.close()
    server_socket.close()
    control_socket.close()
    cv2.destroyAllWindows()
    landmarker.close()
