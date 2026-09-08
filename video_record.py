import socket
import struct
import pickle
import cv2
import time
import os
import csv
import math
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

HOST = "0.0.0.0"
PORT = 50000

SAVE_DIR = "video_data"
CSV_DIR = "facemesh_csv"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)

TARGET_FPS = 15.0
RECORD_SECONDS = 10.0
MODEL_PATH = "face_landmarker.task"

# FaceLandmarkerで取得されるランドマーク数
# 環境によって468や478の場合があります。
# 基本は478で問題ないことが多いです。
NUM_LANDMARKS = 478

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    min_face_detection_confidence=0.3,
    min_face_presence_confidence=0.3,
    min_tracking_confidence=0.3
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

video_writer = None
csv_file = None
csv_writer = None

record_start_time = None

video_count = 1
frame_index = 0
rec_frame_index = 0

rec_total_frames = 0
rec_facemesh_ok_frames = 0

current_video_path = None
current_csv_path = None


while os.path.exists(os.path.join(SAVE_DIR, f"video_{video_count:03d}.mp4")):
    video_count += 1


def get_next_paths():
    global video_count

    while os.path.exists(os.path.join(SAVE_DIR, f"video_{video_count:03d}.mp4")):
        video_count += 1

    video_name = f"video_{video_count:03d}"
    video_path = os.path.join(SAVE_DIR, video_name + ".mp4")
    csv_path = os.path.join(CSV_DIR, video_name + "_facemesh.csv")

    return video_path, csv_path


def create_csv_header():
    header = [
        "video_name",
        "rec_frame",
        "global_frame",
        "timestamp_ms",
        "elapsed_sec",
        "facemesh_ok"
    ]

    for i in range(NUM_LANDMARKS):
        header += [
            f"lm{i}_x_norm",
            f"lm{i}_y_norm",
            f"lm{i}_z_norm",
            f"lm{i}_x_px",
            f"lm{i}_y_px"
        ]

    return header


def write_facemesh_csv_row(
    landmarks,
    video_name,
    rec_frame,
    global_frame,
    timestamp_ms,
    elapsed_sec,
    frame_w,
    frame_h
):
    """
    録画中の1フレーム分のFaceMeshをCSVに保存する。
    landmarks が None の場合は、FaceMesh検出失敗として空欄を保存する。
    """

    global csv_writer

    facemesh_ok = landmarks is not None

    row = [
        video_name,
        rec_frame,
        global_frame,
        timestamp_ms,
        elapsed_sec,
        int(facemesh_ok)
    ]

    for i in range(NUM_LANDMARKS):
        if facemesh_ok and i < len(landmarks):
            lm = landmarks[i]

            x_norm = lm.x
            y_norm = lm.y
            z_norm = lm.z

            x_px = int(lm.x * frame_w)
            y_px = int(lm.y * frame_h)

            row += [
                x_norm,
                y_norm,
                z_norm,
                x_px,
                y_px
            ]
        else:
            row += ["", "", "", "", ""]

    csv_writer.writerow(row)


def start_recording(frame):
    global recording
    global video_writer
    global csv_file
    global csv_writer
    global record_start_time
    global rec_total_frames
    global rec_facemesh_ok_frames
    global rec_frame_index
    global current_video_path
    global current_csv_path

    h, w, _ = frame.shape

    current_video_path, current_csv_path = get_next_paths()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    video_writer = cv2.VideoWriter(
        current_video_path,
        fourcc,
        TARGET_FPS,
        (w, h)
    )

    csv_file = open(
        current_csv_path,
        "w",
        newline="",
        encoding="utf-8"
    )

    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(create_csv_header())

    recording = True
    record_start_time = time.time()

    rec_total_frames = 0
    rec_facemesh_ok_frames = 0
    rec_frame_index = 0

    print(f"録画開始: {current_video_path}")
    print(f"FaceMesh CSV保存開始: {current_csv_path}")


def stop_recording():
    global recording
    global video_writer
    global csv_file
    global csv_writer
    global record_start_time
    global video_count
    global current_video_path
    global current_csv_path

    recording = False
    record_start_time = None

    if video_writer is not None:
        video_writer.release()
        video_writer = None

    if csv_file is not None:
        csv_file.close()
        csv_file = None
        csv_writer = None

    if rec_total_frames > 0:
        success_rate = rec_facemesh_ok_frames / rec_total_frames * 100
    else:
        success_rate = 0

    print(f"録画終了: {os.path.basename(current_video_path)}")
    print(f"FaceMesh CSV保存終了: {os.path.basename(current_csv_path)}")
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
        # =========================
        # Raspberry Piからフレーム受信
        # =========================
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

        # VIDEOモードではtimestamp_msが単調増加している必要があります
        timestamp_ms = int(frame_index * (1000 / TARGET_FPS))

        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        facemesh_ok = False
        landmarks_to_save = None

        if result.face_landmarks:
            facemesh_ok = True
            landmarks_to_save = result.face_landmarks[0]

            for lm in landmarks_to_save:
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
        # 録画中
        # =========================
        if recording:
            video_writer.write(frame)

            rec_frame_index += 1
            rec_total_frames += 1

            if facemesh_ok:
                rec_facemesh_ok_frames += 1

            elapsed_time = time.time() - record_start_time
            remaining_time = RECORD_SECONDS - elapsed_time

            video_name = os.path.splitext(os.path.basename(current_video_path))[0]

            # ここで、録画と同時にFaceMesh座標をCSVへ保存
            write_facemesh_csv_row(
                landmarks=landmarks_to_save,
                video_name=video_name,
                rec_frame=rec_frame_index,
                global_frame=frame_index,
                timestamp_ms=timestamp_ms,
                elapsed_sec=elapsed_time,
                frame_w=w,
                frame_h=h
            )

            success_rate = rec_facemesh_ok_frames / rec_total_frames * 100

            # 10秒カウントダウン表示
            countdown_number = max(0, math.ceil(remaining_time))

            cv2.putText(
                display_frame,
                str(countdown_number),
                (w // 2 - 40, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                3.0,
                (0, 0, 255),
                6
            )

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

            # 残り3秒になったら、カップから口を離す指示を表示
            if remaining_time <= 3.0:
                cv2.putText(
                    display_frame,
                    "Release cup from mouth",
                    (w // 2 - 330, h // 2 + 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 0, 255),
                    3
                )

                cv2.putText(
                    display_frame,
                    "Cup off",
                    (w // 2 - 100, h // 2 + 130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    3
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

        # =========================
        # 録画していないときの案内
        # =========================
        if not recording:
            cv2.putText(
                display_frame,
                "Press ENTER to start 10s recording",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

        cv2.putText(
            display_frame,
            "ENTER: start recording / q: quit",
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

        # Enterキーで即録画開始
        # 環境によってEnterが13または10になることがあるため両方対応
        elif key == 13 or key == 10:
            if not recording:
                start_recording(frame)
                print("Enterが押されたため録画開始")

finally:
    if video_writer is not None:
        video_writer.release()

    if csv_file is not None:
        csv_file.close()

    landmarker.close()
    conn.close()
    server_socket.close()
    cv2.destroyAllWindows()