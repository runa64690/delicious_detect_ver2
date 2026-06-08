import cv2
import mediapipe as mp
import os
import csv
import time

VIDEO_DIR = "video_data"
CSV_DIR = "facemesh_csv"
os.makedirs(CSV_DIR, exist_ok=True)

video_files = sorted([
    f for f in os.listdir(VIDEO_DIR)
    if f.endswith(".mp4")
])

if len(video_files) == 0:
    print("動画がありません")
    exit()

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


def get_taste_score():
    while True:
        score = input("おいしさ評価を1〜5で入力してください: ")
        if score in ["1", "2", "3", "4", "5"]:
            return int(score)
        print("1〜5の整数で入力してください")


def save_facemesh_csv(video_path, csv_path, taste_score, drink_start_frame):
    print(f"CSV保存開始: {csv_path}")
    print(f"おいしさ評価: {taste_score}")
    print(f"飲用開始フレーム: {drink_start_frame}")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("動画を開けませんでした")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    csv_landmarker = FaceLandmarker.create_from_options(options)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        header = [
            "frame_index",
            "time_sec",
            "detected",
            "taste_score",
            "drink_start_frame"
        ]

        for i in range(478):
            header += [
                f"landmark_{i}_x",
                f"landmark_{i}_y",
                f"landmark_{i}_z"
            ]

        writer.writerow(header)

        frame_index = 0

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

            result = csv_landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            time_sec = frame_index / fps

            if result.face_landmarks:
                row = [
                    frame_index,
                    time_sec,
                    1,
                    taste_score,
                    drink_start_frame
                ]

                face_landmarks = result.face_landmarks[0]

                for lm in face_landmarks:
                    row += [lm.x, lm.y, lm.z]

            else:
                row = [
                    frame_index,
                    time_sec,
                    0,
                    taste_score,
                    drink_start_frame
                ]

                for _ in range(478):
                    row += ["", "", ""]

            writer.writerow(row)
            frame_index += 1

    cap.release()
    csv_landmarker.close()

    print("CSV保存完了")


def draw_facemesh(frame, result):
    display_frame = frame.copy()

    if result.face_landmarks:
        h, w, _ = display_frame.shape

        for face_landmarks in result.face_landmarks:
            for lm in face_landmarks:
                x = int(lm.x * w)
                y = int(lm.y * h)

                cv2.circle(
                    display_frame,
                    (x, y),
                    1,
                    (0, 255, 0),
                    -1
                )

    return display_frame


current_index = 0

while True:
    video_path = os.path.join(VIDEO_DIR, video_files[current_index])
    print(f"表示中: {video_files[current_index]}")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("動画を開けませんでした:", video_path)
        current_index = (current_index + 1) % len(video_files)
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_index = 0
    paused = True
    drink_start_frame = None

    landmarker = FaceLandmarker.create_from_options(options)

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()

        if not ret:
            frame_index = max(0, total_frames - 1)
            continue

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

        display_frame = draw_facemesh(frame, result)

        cv2.putText(
            display_frame,
            f"{current_index + 1}/{len(video_files)} : {video_files[current_index]}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2
        )

        cv2.putText(
            display_frame,
            f"frame: {frame_index}/{total_frames - 1}  time: {frame_index / fps:.2f}s",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )

        if drink_start_frame is None:
            start_text = "drink_start_frame: not set"
            color = (0, 0, 255)
        else:
            start_text = f"drink_start_frame: {drink_start_frame}"
            color = (255, 0, 255)

        cv2.putText(
            display_frame,
            start_text,
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2
        )

        cv2.putText(
            display_frame,
            "a:prev d:next space:play/pause s:set start Enter:save n:next b:back q:quit",
            (20, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            2
        )

        cv2.imshow("Frame Check FaceMesh UI", display_frame)

        if paused:
            key = cv2.waitKey(0) & 0xFF
        else:
            key = cv2.waitKey(int(1000 / fps)) & 0xFF

        if key == ord("q"):
            cap.release()
            landmarker.close()
            cv2.destroyAllWindows()
            exit()

        elif key == ord("d"):
            paused = True
            frame_index = min(frame_index + 1, total_frames - 1)

        elif key == ord("a"):
            paused = True
            frame_index = max(frame_index - 1, 0)

        elif key == ord(" "):
            paused = not paused

        elif key == ord("s"):
            drink_start_frame = frame_index
            print(f"飲用開始フレームを設定: {drink_start_frame}")

        elif key == ord("n"):
            current_index = (current_index + 1) % len(video_files)
            break

        elif key == ord("b"):
            current_index = (current_index - 1) % len(video_files)
            break

        elif key == 13:
            if drink_start_frame is None:
                print("先に s キーで飲用開始フレームを設定してください")
                continue

            taste_score = get_taste_score()

            base_name = os.path.splitext(video_files[current_index])[0]
            csv_path = os.path.join(
                CSV_DIR,
                f"{base_name}_score{taste_score}_start{drink_start_frame}_facemesh.csv"
            )

            save_facemesh_csv(
                video_path,
                csv_path,
                taste_score,
                drink_start_frame
            )

        else:
            if not paused:
                frame_index += 1
                if frame_index >= total_frames:
                    frame_index = total_frames - 1
                    paused = True

    cap.release()
    landmarker.close()