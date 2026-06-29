import cv2
import os
import csv

VIDEO_DIR = "video_data"
CSV_DIR = "facemesh_csv"

video_files = sorted([
    f for f in os.listdir(VIDEO_DIR)
    if f.endswith(".mp4")
])

if len(video_files) == 0:
    print("動画がありません")
    exit()


def find_csv_for_video(video_file):
    base_name = os.path.splitext(video_file)[0]

    csv_files = sorted([
        f for f in os.listdir(CSV_DIR)
        if f.startswith(base_name) and f.endswith(".csv")
    ])

    if len(csv_files) == 0:
        return None

    return os.path.join(CSV_DIR, csv_files[0])


def load_facemesh_csv(csv_path):
    data = {}

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            frame_index = int(row["frame_index"])
            detected = int(row["detected"])

            landmarks = []

            if detected == 1:
                for i in range(478):
                    x = row[f"landmark_{i}_x"]
                    y = row[f"landmark_{i}_y"]

                    if x != "" and y != "":
                        landmarks.append((float(x), float(y)))

            data[frame_index] = {
                "detected": detected,
                "landmarks": landmarks
            }

    return data


def draw_csv_landmarks(frame, frame_data):
    display = frame.copy()
    h, w, _ = display.shape

    if frame_data is None or frame_data["detected"] == 0:
        cv2.putText(
            display,
            "FaceMesh: LOST",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )
        return display

    for x_norm, y_norm in frame_data["landmarks"]:
        x = int(x_norm * w)
        y = int(y_norm * h)

        cv2.circle(
            display,
            (x, y),
            1,
            (0, 255, 0),
            -1
        )

    cv2.putText(
        display,
        "FaceMesh: OK",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    return display


current_index = 0

while True:
    video_file = video_files[current_index]
    video_path = os.path.join(VIDEO_DIR, video_file)
    csv_path = find_csv_for_video(video_file)

    print("--------------------------------")
    print("表示中:", video_file)

    if csv_path is None:
        print("対応するCSVがありません:", video_file)
        current_index = (current_index + 1) % len(video_files)
        continue

    print("使用CSV:", csv_path)

    facemesh_data = load_facemesh_csv(csv_path)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("動画を開けませんでした:", video_path)
        current_index = (current_index + 1) % len(video_files)
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 15.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_index = 0
    paused = True

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()

        if not ret:
            frame_index = max(0, total_frames - 1)
            continue

        frame_data = facemesh_data.get(frame_index)
        display_frame = draw_csv_landmarks(frame, frame_data)

        cv2.putText(
            display_frame,
            f"{current_index + 1}/{len(video_files)} : {video_file}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2
        )

        cv2.putText(
            display_frame,
            f"frame: {frame_index}/{total_frames - 1}  time: {frame_index / fps:.2f}s",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )

        cv2.putText(
            display_frame,
            "a:prev d:next space:play/pause n:next b:back q:quit",
            (20, 155),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            2
        )

        cv2.imshow("Check FaceMesh from CSV", display_frame)

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

        elif key == ord("n"):
            current_index = (current_index + 1) % len(video_files)
            break

        elif key == ord("b"):
            current_index = (current_index - 1) % len(video_files)
            break

        else:
            if not paused:
                frame_index += 1
                if frame_index >= total_frames:
                    frame_index = total_frames - 1
                    paused = True

    cap.release()