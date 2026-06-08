import os
import pandas as pd
import matplotlib.pyplot as plt
import math

INPUT_DIR = "baseline_delta_csv"

TARGET_LANDMARKS = {
    "left_cheek_216": 216,
    "right_cheek_436": 436,
    "left_mouth_corner_61": 61,
    "right_mouth_corner_291": 291,
    "upper_lip_13": 13,
    "lower_lip_14": 14,
}

BATCH_SIZE = 10

csv_files = sorted([
    f for f in os.listdir(INPUT_DIR)
    if f.endswith(".csv")
])

if len(csv_files) == 0:
    print("CSVファイルがありません")
    exit()

num_batches = math.ceil(len(csv_files) / BATCH_SIZE)

for batch_idx in range(num_batches):

    start = batch_idx * BATCH_SIZE
    end = min(start + BATCH_SIZE, len(csv_files))

    batch_files = csv_files[start:end]

    print(f"表示中: {start+1}〜{end} / {len(csv_files)}")

    fig, axes = plt.subplots(
        len(batch_files),
        1,
        figsize=(14, 3 * len(batch_files)),
        sharex=True
    )

    if len(batch_files) == 1:
        axes = [axes]

    fig.suptitle(
        f"Facial Movement Batch {batch_idx+1}",
        fontsize=18
    )

    for ax, csv_file in zip(axes, batch_files):

        path = os.path.join(INPUT_DIR, csv_file)

        df = pd.read_csv(path)

        drink_start_frame = int(
            df["drink_start_frame"].dropna().iloc[0]
        )

        for landmark_name, lm_id in TARGET_LANDMARKS.items():

            col = f"landmark_{lm_id}_y_delta"

            if col not in df.columns:
                continue

            # 上方向を正にする
            y_up = -df[col]

            ax.plot(
                df["frame_index"],
                y_up,
                label=landmark_name
            )

        ax.axvline(
            drink_start_frame,
            linestyle="--"
        )

        score = "?"

        if "taste_score" in df.columns:
            score = df["taste_score"].iloc[0]

        ax.set_title(
            f"{csv_file}   score={score}"
        )

        ax.grid(True)

        ax.legend(
            loc="upper right",
            fontsize=7
        )

    axes[-1].set_xlabel("frame")

    plt.tight_layout()
    plt.show()