import socket
import struct
import pickle
import cv2
from picamera2 import Picamera2
import time

# ===== 接続先設定 =====
LAPTOP_IP = "192.168.210.62"   # ← ノートPCのIPアドレスに書き換える
PORT = 9999

# ===== カメラ初期化 =====
picam2 = Picamera2()
config = picam2.create_video_configuration(
    main={"size": (1280, 800), "format": "RGB888"}
)
picam2.configure(config)
picam2.start()
time.sleep(2)

# ===== ソケット接続 =====
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((LAPTOP_IP, PORT))

try:
    while True:
        frame = picam2.capture_array()

        # OpenCV用にBGRへ変換
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # JPEG圧縮（画質80）
        result, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not result:
            continue

        data = pickle.dumps(encoded)
        size = len(data)

        # 先頭にデータサイズを付けて送る
        client_socket.sendall(struct.pack("Q", size) + data)

except KeyboardInterrupt:
    print("終了します")

finally:
    client_socket.close()
    picam2.stop()