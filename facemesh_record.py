import socket
import struct
import pickle
import cv2
import time
import csv
import os
import math
from datetime import datetime

import mediapipe as mp


# =========================
# 通信設定
# =========================
HOST = "0.0.0.0"
PORT = 9999


# =========================
# CSV保存設定
# =========================
DATA_DIR = "./data"
CSV_PATH = os.path.join(DATA_DIR, "facemesh_landmarks.csv")


# =========================
# FaceMesh設定
# =========================
MODEL_PATH = "face_landmarker.task"

# 鼻根を原点にする
NOSE_LANDMARK_ID = 168

# 眼間距離の計算に使うランドマーク
LEFT_EYE_ID = 33
RIGHT_EYE_ID = 263

# 記録する主要ランドマーク
TARGET_LANDMARKS = [168, 61, 291, 13, 14, 216, 436]

LANDMARK_NAMES = {
    168: "nose_root",
    61: "left_mouth_corner",
    291: "right_mouth_corner",
    13: "upper_lip",
    14: "lower_lip",
    216: "left_cheek",
    436: "right_cheek",
}


# =========================
# CSV準備
# =========================
os.makedirs(DATA_DIR, exist_ok=True)

csv_exists = os.path.exists(CSV_PATH)

csv_file = open(CSV_PATH, "a", newline="", encoding="utf-8")
csv_writer = csv.writer(csv_file)

if not csv_exists:
    csv_writer.writerow([
        "session_id",
        "session_start_datetime",
        "recording_id",
        "recording_start_datetime",
        "recording_elapsed_sec",
        "record_datetime",

        "frame_index",
        "timestamp_ms",
        "landmark_id",
        "landmark_name",

        "x_abs_px",
        "y_abs_px",
        "z_abs",

        "x_nose_origin_px",
        "y_nose_origin_px",
        "z_nose_origin",

        "x_norm_by_eye",
        "y_norm_by_eye",
        "z_norm_by_eye",

        "eye_distance_px",
        "image_width",
        "image_height"
    ])


# 実行ごとのID
session_start_dt = datetime.now()
session_id = session_start_dt.strftime("%Y%m%d_%H%M%S")
session_start_datetime = session_start_dt.strftime("%Y-%m-%d %H:%M:%S")


# =========================
# MediaPipe FaceLandmarker準備
# =========================
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

landmarker = FaceLandmarker.create_from_options(options)


# =========================
# ソケット準備
# =========================
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(1)

print("接続待機中...")
conn, addr = server_socket.accept()
print("接続:", addr)

data = b""
payload_size = struct.calcsize("Q")


# =========================
# 状態管理
# =========================
show_facemesh = False
recording = False
frame_index = 0

# rキーで記録を開始するたびに増やす
recording_id = 0
recording_start_time = None
recording_start_datetime = ""


print("操作方法:")
print("  f : FaceMesh表示 ON/OFF")
print("  r : CSV記録 開始/停止")
print("  q : 終了")
print("CSV保存先:", CSV_PATH)
print("session_id:", session_id)


try:
    while True:
        # =========================
        # データサイズを受信
        # =========================
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

        # =========================
        # 画像データ本体を受信
        # =========================
        while len(data) < msg_size:
            packet = conn.recv(4096)
            if not packet:
                break
            data += packet

        if len(data) < msg_size:
            break

        frame_data = data[:msg_size]
        data = data[msg_size:]

        encoded = pickle.loads(frame_data)
        frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

        if frame is None:
            continue

        frame_index += 1
        display_frame = frame.copy()
        h, w, _ = frame.shape

        # 記録中、またはFaceMesh表示中だけ推論する
        need_facemesh = recording or show_facemesh

        if need_facemesh:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            timestamp_ms = int(time.perf_counter() * 1000)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.face_landmarks:
                face_landmarks = result.face_landmarks[0]

                nose = face_landmarks[NOSE_LANDMARK_ID]
                left_eye = face_landmarks[LEFT_EYE_ID]
                right_eye = face_landmarks[RIGHT_EYE_ID]

                nose_x_px = nose.x * w
                nose_y_px = nose.y * h

                left_eye_x_px = left_eye.x * w
                left_eye_y_px = left_eye.y * h
                right_eye_x_px = right_eye.x * w
                right_eye_y_px = right_eye.y * h

                eye_distance_px = math.sqrt(
                    (right_eye_x_px - left_eye_x_px) ** 2
                    + (right_eye_y_px - left_eye_y_px) ** 2
                )

                # 眼間距離が極端に小さい場合はスキップ
                if eye_distance_px > 1e-6:

                    # =========================
                    # CSV記録
                    # =========================
                    if recording:
                        record_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

                        if recording_start_time is not None:
                            recording_elapsed_sec = time.perf_counter() - recording_start_time
                        else:
                            recording_elapsed_sec = 0.0

                        for landmark_id in TARGET_LANDMARKS:
                            lm = face_landmarks[landmark_id]

                            x_abs_px = lm.x * w
                            y_abs_px = lm.y * h
                            z_abs = lm.z

                            x_nose_origin_px = x_abs_px - nose_x_px
                            y_nose_origin_px = y_abs_px - nose_y_px
                            z_nose_origin = z_abs - nose.z

                            x_norm_by_eye = x_nose_origin_px / eye_distance_px
                            y_norm_by_eye = y_nose_origin_px / eye_distance_px
                            z_norm_by_eye = z_nose_origin / eye_distance_px

                            csv_writer.writerow([
                                session_id,
                                session_start_datetime,
                                recording_id,
                                recording_start_datetime,
                                recording_elapsed_sec,
                                record_datetime,

                                frame_index,
                                timestamp_ms,
                                landmark_id,
                                LANDMARK_NAMES.get(landmark_id, ""),

                                x_abs_px,
                                y_abs_px,
                                z_abs,

                                x_nose_origin_px,
                                y_nose_origin_px,
                                z_nose_origin,

                                x_norm_by_eye,
                                y_norm_by_eye,
                                z_norm_by_eye,

                                eye_distance_px,
                                w,
                                h
                            ])

                        csv_file.flush()

                    # =========================
                    # FaceMesh表示
                    # =========================
                    if show_facemesh:
                        for landmark_id in TARGET_LANDMARKS:
                            lm = face_landmarks[landmark_id]
                            x = int(lm.x * w)
                            y = int(lm.y * h)

                            cv2.circle(display_frame, (x, y), 4, (0, 255, 0), -1)
                            cv2.putText(
                                display_frame,
                                str(landmark_id),
                                (x + 5, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.4,
                                (0, 255, 0),
                                1
                            )

                        # 鼻根を強調
                        cv2.circle(
                            display_frame,
                            (int(nose_x_px), int(nose_y_px)),
                            6,
                            (0, 0, 255),
                            -1
                        )

        # =========================
        # 状態表示
        # =========================
        status_text = []

        if show_facemesh:
            status_text.append("FaceMesh: ON")
        else:
            status_text.append("FaceMesh: OFF")

        if recording:
            status_text.append(f"REC: ON id={recording_id}")
        else:
            status_text.append("REC: OFF")

        cv2.putText(
            display_frame,
            " | ".join(status_text),
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255) if recording else (255, 255, 255),
            2
        )

        cv2.imshow("Raspberry Pi Camera", display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("f"):
            show_facemesh = not show_facemesh
            print("FaceMesh表示 ON" if show_facemesh else "FaceMesh表示 OFF")

        elif key == ord("r"):
            if not recording:
                # 記録開始
                recording = True
                recording_id += 1
                recording_start_time = time.perf_counter()
                recording_start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

                print("CSV記録 開始")
                print("  recording_id:", recording_id)
                print("  recording_start_datetime:", recording_start_datetime)

            else:
                # 記録停止
                recording = False

                if recording_start_time is not None:
                    elapsed = time.perf_counter() - recording_start_time
                else:
                    elapsed = 0.0

                print("CSV記録 停止")
                print("  recording_id:", recording_id)
                print("  elapsed_sec:", elapsed)

                recording_start_time = None
                recording_start_datetime = ""


except KeyboardInterrupt:
    print("終了します")


finally:
    csv_file.close()
    conn.close()
    server_socket.close()
    cv2.destroyAllWindows()
    landmarker.close()