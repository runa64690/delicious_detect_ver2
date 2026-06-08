import cv2
import os
import csv
import pandas as pd

VIDEO_DIR = "video_data"
CSV_DIR = "facemesh_csv"

video_files = sorted([
    f for f in os.listdir(VIDEO_DIR)
    if f.endswith(".mp4")
])

if len(video_files) == 0:
    print("動画がありません")
    exit()


def find_existing_csv(video_name):
    base_name = os.path.splitext(video_name)[0]

    candidates = [
        f for f in os.listdir(CSV_DIR)
        if f.endswith(".csv") and base_name in f
    ]

    if len(candidates) == 0:
        return None

    if len(candidates) == 1:
        return os.path.join(CSV_DIR, candidates[0])

    print("候補CSVが複数あります:")
    for i, name in enumerate(candidates):
        print(f"{i}: {name}")

    while True:
        idx = input("使うCSV番号を入力してください: ")
        if idx.isdigit() and int(idx) < len(candidates):
            return os.path.join(CSV_DIR, candidates[int(idx)])
        print("正しい番号を入力してください")


def add_drink_start_frame_to_csv(csv_path, drink_start_frame):
    df = pd.read_csv(csv_path)

    df["drink_start_frame"] = drink_start_frame

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"追記完了: {csv_path}")
    print(f"drink_start_frame = {drink_start_frame}")


current_index = 0

while True:
    video_name = video_files[current_index]
    video_path = os.path.join(VIDEO_DIR, video_name)

    print(f"表示中: {video_name}")

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

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()

        if not ret:
            frame_index = max(0, total_frames - 1)
            continue

        display_frame = frame.copy()

        cv2.putText(
            display_frame,
            f"{current_index + 1}/{len(video_files)} : {video_name}",
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
            "a:prev d:next space:play/pause s:set start Enter:add to CSV n:next b:back q:quit",
            (20, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            2
        )

        cv2.imshow("Frame Check UI", display_frame)

        if paused:
            key = cv2.waitKey(0) & 0xFF
        else:
            key = cv2.waitKey(int(1000 / fps)) & 0xFF

        if key == ord("q"):
            cap.release()
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

        elif key == 13:
            if drink_start_frame is None:
                print("先に s キーで飲用開始フレームを設定してください")
                continue

            csv_path = find_existing_csv(video_name)

            if csv_path is None:
                print("対応するCSVが見つかりませんでした")
                continue

            add_drink_start_frame_to_csv(csv_path, drink_start_frame)

        elif key == ord("n"):
            current_index = (current_index + 1) % len(video_files)
            break

        elif key == ord("b"):
            current_index = (current_index - 1) % len(video_files)
            break

        if not paused:
            frame_index += 1
            if frame_index >= total_frames:
                frame_index = total_frames - 1
                paused = True

    cap.release()