import cv2
import mediapipe as mp
import os
import csv

VIDEO_DIR = "video_data"
CSV_DIR = "facemesh_csv"
os.makedirs(CSV_DIR, exist_ok=True)

MODEL_PATH = "face_landmarker.task"

video_files = sorted([
    f for f in os.listdir(VIDEO_DIR)
    if f.endswith(".mp4")
])

if len(video_files) == 0:
    print("動画がありません")
    exit()

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.3,
    min_face_presence_confidence=0.3,
    min_tracking_confidence=0.3
)


def get_taste_score():
    while True:
        score = input("おいしさ評価を1〜5で入力してください: ")
        if score in ["1", "2", "3", "4", "5"]:
            return int(score)
        print("1〜5の整数で入力してください")


def save_facemesh_csv(video_path, csv_path, taste_score):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("動画を開けませんでした:", video_path)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 15.0

    landmarker = FaceLandmarker.create_from_options(options)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_index = 0
    detected_count = 0

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        header = [
            "frame_index",
            "time_sec",
            "detected",
            "taste_score"
        ]

        for i in range(478):
            header += [
                f"landmark_{i}_x",
                f"landmark_{i}_y",
                f"landmark_{i}_z"
            ]

        writer.writerow(header)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            timestamp_ms = int(frame_index * 1000 / fps)

            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            time_sec = frame_index / fps

            if result.face_landmarks:
                detected_count += 1

                row = [
                    frame_index,
                    time_sec,
                    1,
                    taste_score
                ]

                face_landmarks = result.face_landmarks[0]

                for lm in face_landmarks:
                    row += [lm.x, lm.y, lm.z]

            else:
                row = [
                    frame_index,
                    time_sec,
                    0,
                    taste_score
                ]

                for _ in range(478):
                    row += ["", "", ""]

            writer.writerow(row)

            frame_index += 1

            if frame_index % 30 == 0:
                print(f"{frame_index}/{total_frames} フレーム処理中...")

    cap.release()
    landmarker.close()

    if frame_index > 0:
        success_rate = detected_count / frame_index * 100
    else:
        success_rate = 0

    print("CSV保存完了:", csv_path)
    print(f"FaceMesh成功率: {success_rate:.1f}%")


for video_file in video_files:
    video_path = os.path.join(VIDEO_DIR, video_file)

    print("--------------------------------")
    print("処理対象:", video_file)

    taste_score = get_taste_score()

    base_name = os.path.splitext(video_file)[0]
    csv_path = os.path.join(
        CSV_DIR,
        f"{base_name}_score{taste_score}_facemesh.csv"
    )

    save_facemesh_csv(video_path, csv_path, taste_score)