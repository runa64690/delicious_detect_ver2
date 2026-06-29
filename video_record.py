import socket
import struct
import pickle
import cv2
import time
import os
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

HOST = "0.0.0.0"
PORT = 9999

SAVE_DIR = "video_data"
os.makedirs(SAVE_DIR, exist_ok=True)

TARGET_FPS = 15.0
RECORD_SECONDS = 10.0
COUNTDOWN_SECONDS = 3
MODEL_PATH = "face_landmarker.task"

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False
)

landmarker = vision.FaceLandmarker.create_from_options(options)

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(1)

print("接続待機中...")
conn, addr = server_socket.accept()
print("接続:", addr)

data = b""
payload_size = struct.calcsize("Q")

recording = False
countdown = False

video_writer = None
record_start_time = None
countdown_start_time = None

video_count = 1
frame_index = 0

rec_total_frames = 0
rec_facemesh_ok_frames = 0

while os.path.exists(os.path.join(SAVE_DIR, f"video_{video_count:03d}.mp4")):
    video_count += 1


def get_next_filename():
    global video_count

    while os.path.exists(os.path.join(SAVE_DIR, f"video_{video_count:03d}.mp4")):
        video_count += 1

    return os.path.join(SAVE_DIR, f"video_{video_count:03d}.mp4")


def start_recording(frame):
    global recording
    global video_writer
    global record_start_time
    global rec_total_frames
    global rec_facemesh_ok_frames

    h, w, _ = frame.shape

    filename = get_next_filename()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    video_writer = cv2.VideoWriter(
        filename,
        fourcc,
        TARGET_FPS,
        (w, h)
    )

    recording = True
    record_start_time = time.time()

    rec_total_frames = 0
    rec_facemesh_ok_frames = 0

    print(f"録画開始: {filename}")


def stop_recording():
    global recording
    global video_writer
    global record_start_time
    global video_count

    recording = False
    record_start_time = None

    if video_writer is not None:
        video_writer.release()
        video_writer = None

    if rec_total_frames > 0:
        success_rate = rec_facemesh_ok_frames / rec_total_frames * 100
    else:
        success_rate = 0

    print(f"録画終了: video_{video_count:03d}.mp4")
    print(f"FaceMesh成功率: {success_rate:.1f}%")

    if success_rate >= 90:
        print("判定: 使用可能そうです")
    elif success_rate >= 70:
        print("判定: 要確認です")
    else:
        print("判定: 撮り直し推奨です")

    video_count += 1


try:
    while True:
        while len(data) < payload_size:
            packet = conn.recv(4096)
            if not packet:
                raise ConnectionError("接続が切断されました")
            data += packet

        packed_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack("Q", packed_size)[0]

        while len(data) < msg_size:
            packet = conn.recv(4096)
            if not packet:
                raise ConnectionError("接続が切断されました")
            data += packet

        frame_data = data[:msg_size]
        data = data[msg_size:]

        encoded = pickle.loads(frame_data)
        frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

        if frame is None:
            continue

        frame_index += 1

        display_frame = frame.copy()
        h, w, _ = frame.shape

        # =========================
        # FaceMesh / FaceLandmarker検出
        # =========================
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp_ms = int(frame_index * (1000 / TARGET_FPS))
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        facemesh_ok = False

        if result.face_landmarks:
            facemesh_ok = True

            for face_landmarks in result.face_landmarks:
                for lm in face_landmarks:
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    cv2.circle(display_frame, (x, y), 1, (0, 255, 0), -1)

            cv2.putText(
                display_frame,
                "FaceMesh: OK",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )
        else:
            cv2.putText(
                display_frame,
                "FaceMesh: LOST",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        # =========================
        # カウントダウン中
        # =========================
        if countdown and not recording:
            elapsed_countdown = time.time() - countdown_start_time
            remaining_countdown = COUNTDOWN_SECONDS - elapsed_countdown

            if remaining_countdown > 0:
                count_text = str(int(remaining_countdown) + 1)

                cv2.putText(
                    display_frame,
                    count_text,
                    (w // 2 - 40, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    3.0,
                    (0, 0, 255),
                    5
                )

                cv2.putText(
                    display_frame,
                    "Get ready to drink",
                    (w // 2 - 220, h // 2 + 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2
                )

            else:
                countdown = False
                start_recording(frame)

                cv2.putText(
                    display_frame,
                    "START!",
                    (w // 2 - 120, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2.0,
                    (0, 0, 255),
                    4
                )

        # =========================
        # 録画中
        # =========================
        if recording:
            video_writer.write(frame)

            rec_total_frames += 1
            if facemesh_ok:
                rec_facemesh_ok_frames += 1

            elapsed_time = time.time() - record_start_time
            remaining_time = RECORD_SECONDS - elapsed_time

            success_rate = rec_facemesh_ok_frames / rec_total_frames * 100

            cv2.putText(
                display_frame,
                f"REC remaining: {max(remaining_time, 0):.1f}s",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2
            )

            cv2.putText(
                display_frame,
                f"FaceMesh success: {success_rate:.1f}%",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2
            )

            if not facemesh_ok:
                cv2.putText(
                    display_frame,
                    "FACE LOST - CHECK POSITION",
                    (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

            if elapsed_time >= RECORD_SECONDS:
                stop_recording()

        cv2.putText(
            display_frame,
            "SPACE: countdown & record / q: quit",
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow("Raspberry Pi Camera", display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord(" "):
            if not recording and not countdown:
                countdown = True
                countdown_start_time = time.time()
                print("カウントダウン開始")

finally:
    if video_writer is not None:
        video_writer.release()

    landmarker.close()
    conn.close()
    server_socket.close()
    cv2.destroyAllWindows()