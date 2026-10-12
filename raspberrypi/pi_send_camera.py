"""Pi camera sender and acknowledged LED/IR control; reconnects after PC restart."""
import argparse
import json
import pickle
import socket
import struct
import threading
import time

try:
    from .led_control import KNOWN_COMMANDS, LedRingController, handle_message
except ImportError:
    from led_control import KNOWN_COMMANDS, LedRingController, handle_message

PORT = 9999
CONTROL_PORT = 10000
IR_PIN = 23
PWM_FREQ = 1000
IR_DUTY_ON = 20
NUM_PIXELS = 12
PIXEL_BRIGHTNESS = 0.15


def control_server(controller, port, stop_event, ready, startup_errors):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind(("0.0.0.0", port))
            server.listen(1)
            server.settimeout(0.5)
            ready.set()
            while not stop_event.is_set():
                try:
                    connection, address = server.accept()
                except socket.timeout:
                    continue
                print("制御接続:", address)
                with connection:
                    connection.settimeout(0.5)
                    buffer = b""
                    acknowledged_protocol = False
                    last_message = time.monotonic()
                    try:
                        while not stop_event.is_set():
                            try:
                                data = connection.recv(4096)
                            except socket.timeout:
                                if acknowledged_protocol and time.monotonic() - last_message > 4:
                                    break
                                continue
                            if not data:
                                break
                            buffer += data
                            if len(buffer) > 16384:
                                raise ValueError("制御命令が長すぎます")
                            while b"\n" in buffer:
                                line, buffer = buffer.split(b"\n", 1)
                                text = line.decode("utf-8").strip()
                                if not text:
                                    continue
                                response = handle_message(text, controller)
                                if response is not None:
                                    acknowledged_protocol = True
                                    connection.sendall((json.dumps(response) + "\n").encode())
                                last_message = time.monotonic()
                            if buffer.decode("utf-8").strip() in KNOWN_COMMANDS:
                                handle_message(buffer.decode("utf-8").strip(), controller)
                                buffer = b""
                    except (OSError, ValueError) as exc:
                        print("制御接続エラー:", exc)
                    finally:
                        controller.off()
    except Exception as exc:
        startup_errors.append(exc)
        ready.set()
        stop_event.set()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pc-host", required=True, help="PCのIPアドレスまたはホスト名")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--control-port", type=int, default=CONTROL_PORT)
    args = parser.parse_args()
    import cv2
    import board
    import neopixel
    import RPi.GPIO as GPIO
    from picamera2 import Picamera2

    stop = threading.Event()
    controller = camera = pwm = None
    control_thread = None
    try:
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(IR_PIN, GPIO.OUT)
        pwm = GPIO.PWM(IR_PIN, PWM_FREQ)
        pwm.start(0)
        pixels = neopixel.NeoPixel(board.D18, NUM_PIXELS, brightness=PIXEL_BRIGHTNESS,
                                  auto_write=False, pixel_order=neopixel.GRB)
        controller = LedRingController(pixels, lambda on: pwm.ChangeDutyCycle(IR_DUTY_ON if on else 0))
        controller.off()
        controller.start()
        ready, errors = threading.Event(), []
        control_thread = threading.Thread(target=control_server,
            args=(controller, args.control_port, stop, ready, errors), daemon=True)
        control_thread.start()
        if not ready.wait(timeout=5):
            raise RuntimeError("制御サーバーの起動がタイムアウトしました")
        if errors:
            raise errors[0]
        camera = Picamera2()
        camera.configure(camera.create_video_configuration(main={"size": (1280, 800), "format": "RGB888"}))
        camera.start()
        time.sleep(2)
        while not stop.is_set():
            try:
                with socket.create_connection((args.pc_host, args.port), timeout=3) as connection:
                    print("映像接続:", args.pc_host, args.port)
                    connection.settimeout(3)
                    while not stop.is_set():
                        frame = camera.capture_array()
                        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        if ok:
                            payload = pickle.dumps(encoded)
                            connection.sendall(struct.pack("Q", len(payload)) + payload)
            except OSError as exc:
                print("PCへ再接続します:", exc)
                stop.wait(1)
    except KeyboardInterrupt:
        print("終了します")
    finally:
        stop.set()
        if control_thread:
            control_thread.join(timeout=2)
        if camera:
            camera.stop()
        if controller:
            controller.stop()
        if pwm:
            pwm.ChangeDutyCycle(0)
            pwm.stop()
        GPIO.cleanup()


if __name__ == "__main__":
    main()
