import os
import pandas as pd
import numpy as np

INPUT_CSV_DIR = "facemesh_csv"
OUTPUT_CSV_DIR = "normalized_facemesh_csv"
os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)

NUM_LANDMARKS = 478

# ローカル座標化の基準
NOSE_ID = 168
LEFT_EYE_ID = 33
RIGHT_EYE_ID = 263


def get_col(lm_id, axis):
    return f"landmark_{lm_id}_{axis}"


def normalize_csv(input_path, output_path):
    print(f"処理中: {input_path}")

    df = pd.read_csv(input_path)

    required_cols = [
        get_col(NOSE_ID, "x"),
        get_col(NOSE_ID, "y"),
        get_col(LEFT_EYE_ID, "x"),
        get_col(LEFT_EYE_ID, "y"),
        get_col(RIGHT_EYE_ID, "x"),
        get_col(RIGHT_EYE_ID, "y"),
    ]

    for col in required_cols:
        if col not in df.columns:
            print(f"必要な列がありません: {col}")
            return

    # detected == 1 の行だけ計算対象
    detected_mask = df["detected"] == 1

    nose_x = df[get_col(NOSE_ID, "x")]
    nose_y = df[get_col(NOSE_ID, "y")]

    left_eye_x = df[get_col(LEFT_EYE_ID, "x")]
    left_eye_y = df[get_col(LEFT_EYE_ID, "y")]
    right_eye_x = df[get_col(RIGHT_EYE_ID, "x")]
    right_eye_y = df[get_col(RIGHT_EYE_ID, "y")]

    eye_dist = np.sqrt(
        (right_eye_x - left_eye_x) ** 2 +
        (right_eye_y - left_eye_y) ** 2
    )

    # 0除算防止
    eye_dist = eye_dist.replace(0, np.nan)

    out_df = pd.DataFrame()

    # メタ情報を残す
    meta_cols = [
        "frame_index",
        "time_sec",
        "detected",
        "taste_score",
        "drink_start_frame"
    ]

    for col in meta_cols:
        if col in df.columns:
            out_df[col] = df[col]

    out_df["eye_dist"] = eye_dist

    for i in range(NUM_LANDMARKS):
        x_col = get_col(i, "x")
        y_col = get_col(i, "y")
        z_col = get_col(i, "z")

        if x_col not in df.columns:
            continue

        out_df[f"landmark_{i}_x_local"] = np.nan
        out_df[f"landmark_{i}_y_local"] = np.nan
        out_df[f"landmark_{i}_z_local"] = np.nan

        out_df.loc[detected_mask, f"landmark_{i}_x_local"] = (
            df.loc[detected_mask, x_col] - nose_x.loc[detected_mask]
        ) / eye_dist.loc[detected_mask]

        out_df.loc[detected_mask, f"landmark_{i}_y_local"] = (
            df.loc[detected_mask, y_col] - nose_y.loc[detected_mask]
        ) / eye_dist.loc[detected_mask]

        # zも同じスケールで正規化
        if z_col in df.columns:
            out_df.loc[detected_mask, f"landmark_{i}_z_local"] = (
                df.loc[detected_mask, z_col] / eye_dist.loc[detected_mask]
            )

    out_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"保存完了: {output_path}")


csv_files = sorted([
    f for f in os.listdir(INPUT_CSV_DIR)
    if f.endswith(".csv")
])

if len(csv_files) == 0:
    print("CSVファイルがありません")
    exit()

for csv_file in csv_files:
    input_path = os.path.join(INPUT_CSV_DIR, csv_file)

    base_name = os.path.splitext(csv_file)[0]
    output_path = os.path.join(
        OUTPUT_CSV_DIR,
        f"{base_name}_normalized.csv"
    )

    normalize_csv(input_path, output_path)

print("全CSVの正規化が完了しました")