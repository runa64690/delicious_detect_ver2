import socket
import struct
import pickle
import cv2
from picamera2 import Picamera2
import time
import RPi.GPIO as GPIO
import threading

# ===== 接続先設定 =====
LAPTOP_IP = "192.168.0.161"   # ← ノートPCのIPアドレスに書き換える
PORT = 9999

CONTROL_PORT = 10000

IR_PIN = 23
PWM_FREQ = 1000
IR_DUTY_ON = 20
IR_DUTY_OFF = 0

ir_on = False

# GPIO初期化
GPIO.setmode(GPIO.BCM)
GPIO.setup(IR_PIN, GPIO.OUT)

ir_pwm = GPIO.PWM(IR_PIN, PWM_FREQ)
ir_pwm.start(IR_DUTY_OFF)

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

# 制御命令を受ける関数
def control_server():
    global ir_on

    control_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    control_socket.bind(("0.0.0.0", CONTROL_PORT))
    control_socket.listen(1)

    print("制御接続待機中...")
    control_conn, control_addr = control_socket.accept()
    print("制御接続:", control_addr)

    try:
        while True:
            data = control_conn.recv(1024)
            if not data:
                break

            command = data.decode().strip()

            if command == "TOGGLE_IR":
                if ir_on:
                    ir_pwm.ChangeDutyCycle(IR_DUTY_OFF)
                    ir_on = False
                    print("IR OFF")
                else:
                    ir_pwm.ChangeDutyCycle(IR_DUTY_ON)
                    ir_on = True
                    print("IR ON")
    finally:
        control_conn.close()
        control_socket.close()

control_thread = threading.Thread(target=control_server, daemon=True)
control_thread.start()

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

    try:
        ir_pwm.ChangeDutyCycle(0)
        ir_pwm.stop()
    except Exception:
        pass

    GPIO.cleanup()