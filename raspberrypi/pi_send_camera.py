import socket
import struct
import pickle
import cv2
from picamera2 import Picamera2
import time
import RPi.GPIO as GPIO
import threading
import colorsys
import board
import neopixel

# ===== 接続先設定 =====
LAPTOP_IP = "10.144.98.97"   # ← ノートPCのIPアドレスに書き換える
PORT = 9999

CONTROL_PORT = 10000

IR_PIN = 23
PWM_FREQ = 1000
IR_DUTY_ON = 20
IR_DUTY_OFF = 0

# ===== フルカラーLEDリング設定 =====
PIXEL_PIN = board.D18
NUM_PIXELS = 12
PIXEL_BRIGHTNESS = 0.15
PIXEL_ORDER = neopixel.GRB
RESULT_LED_SECONDS = 3.0
RAINBOW_INTERVAL_SECONDS = 0.05

COLOR_OFF = (0, 0, 0)
COLOR_FACE_DETECTED = (0, 255, 0)
COLOR_WATER = (0, 0, 255)
COLOR_SPARKLING = (255, 0, 0)

ir_on = False

# GPIO初期化
GPIO.setmode(GPIO.BCM)
GPIO.setup(IR_PIN, GPIO.OUT)

ir_pwm = GPIO.PWM(IR_PIN, PWM_FREQ)
ir_pwm.start(IR_DUTY_OFF)

pixels = neopixel.NeoPixel(
    PIXEL_PIN,
    NUM_PIXELS,
    brightness=PIXEL_BRIGHTNESS,
    auto_write=False,
    pixel_order=PIXEL_ORDER,
)
pixels.fill(COLOR_OFF)
pixels.show()


class LedRingController:
    """状態の優先順位を保ちながらLEDリングを1スレッドで駆動する。"""

    def __init__(self, pixel_device):
        self.pixels = pixel_device
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.face_detected = False
        self.mode = "face"
        self.result_color = COLOR_OFF
        self.result_until = 0.0
        self.rainbow_offset = 0
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def set_face_detected(self, detected):
        with self.lock:
            self.face_detected = bool(detected)

    def set_processing(self):
        with self.lock:
            self.mode = "processing"
            self.rainbow_offset = 0

    def set_result(self, color):
        with self.lock:
            self.mode = "result"
            self.result_color = color
            self.result_until = time.monotonic() + RESULT_LED_SECONDS

    def return_to_face_state(self):
        with self.lock:
            self.mode = "face"

    def turn_off(self):
        with self.lock:
            self.face_detected = False
            self.mode = "face"

    @staticmethod
    def rainbow_color(position):
        hue = (position % 256) / 256.0
        red, green, blue = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        return int(red * 255), int(green * 255), int(blue * 255)

    def _show_solid(self, color):
        self.pixels.fill(color)
        self.pixels.show()

    def _show_rainbow(self, offset):
        for index in range(NUM_PIXELS):
            position = offset + int(index * 256 / NUM_PIXELS)
            self.pixels[index] = self.rainbow_color(position)
        self.pixels.show()

    def _run(self):
        last_solid_color = None
        while not self.stop_event.is_set():
            with self.lock:
                if self.mode == "result" and time.monotonic() >= self.result_until:
                    self.mode = "face"
                mode = self.mode
                face_detected = self.face_detected
                result_color = self.result_color
                rainbow_offset = self.rainbow_offset
                if mode == "processing":
                    self.rainbow_offset = (self.rainbow_offset + 5) % 256

            if mode == "processing":
                self._show_rainbow(rainbow_offset)
                last_solid_color = None
            else:
                color = (
                    result_color
                    if mode == "result"
                    else COLOR_FACE_DETECTED if face_detected else COLOR_OFF
                )
                if color != last_solid_color:
                    self._show_solid(color)
                    last_solid_color = color
            self.stop_event.wait(RAINBOW_INTERVAL_SECONDS)

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=1.0)
        self._show_solid(COLOR_OFF)


led_controller = LedRingController(pixels)
led_controller.start()

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

KNOWN_COMMANDS = {
    "TOGGLE_IR",
    "LED_FACE_ON",
    "LED_FACE_OFF",
    "LED_PROCESSING",
    "LED_RESULT_WATER",
    "LED_RESULT_SPARKLING",
    "LED_IDLE",
    "LED_OFF",
}


def handle_control_command(command):
    global ir_on

    if command == "TOGGLE_IR":
        if ir_on:
            ir_pwm.ChangeDutyCycle(IR_DUTY_OFF)
            ir_on = False
            print("IR OFF")
        else:
            ir_pwm.ChangeDutyCycle(IR_DUTY_ON)
            ir_on = True
            print("IR ON")
    elif command == "LED_FACE_ON":
        led_controller.set_face_detected(True)
    elif command == "LED_FACE_OFF":
        led_controller.set_face_detected(False)
    elif command == "LED_PROCESSING":
        led_controller.set_processing()
    elif command == "LED_RESULT_WATER":
        led_controller.set_result(COLOR_WATER)
    elif command == "LED_RESULT_SPARKLING":
        led_controller.set_result(COLOR_SPARKLING)
    elif command == "LED_IDLE":
        led_controller.return_to_face_state()
    elif command == "LED_OFF":
        led_controller.turn_off()
    else:
        print("不明な制御命令:", command)


# 制御命令を受ける関数。PCが再起動しても再接続を受け付ける。
def control_server():

    control_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    control_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    control_socket.bind(("0.0.0.0", CONTROL_PORT))
    control_socket.listen(1)

    try:
        while True:
            print("制御接続待機中...")
            control_conn, control_addr = control_socket.accept()
            print("制御接続:", control_addr)
            command_buffer = ""
            try:
                while True:
                    data = control_conn.recv(1024)
                    if not data:
                        break
                    command_buffer += data.decode("utf-8")
                    while "\n" in command_buffer:
                        command, command_buffer = command_buffer.split("\n", 1)
                        command = command.strip()
                        if command:
                            handle_control_command(command)

                    # pc_receive_camera.pyとの後方互換性。
                    if command_buffer.strip() in KNOWN_COMMANDS:
                        handle_control_command(command_buffer.strip())
                        command_buffer = ""
            except (ConnectionError, OSError) as exc:
                print("制御接続エラー:", exc)
            finally:
                control_conn.close()
                led_controller.turn_off()
    finally:
        control_socket.close()

control_thread = threading.Thread(target=control_server, daemon=True)
control_thread.start()

try:
    while True:
        frame = picam2.capture_array()

        # OpenCV用にBGRへ変換
        #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

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
except (BrokenPipeError, ConnectionResetError, OSError) as exc:
    print("PCとの映像接続が切断されました:", exc)

finally:
    client_socket.close()
    picam2.stop()

    try:
        led_controller.stop()
    except Exception:
        pass

    try:
        ir_pwm.ChangeDutyCycle(0)
        ir_pwm.stop()
    except Exception:
        pass

    GPIO.cleanup()
