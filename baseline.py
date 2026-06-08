import os
import pandas as pd
import numpy as np

INPUT_DIR = "normalized_facemesh_csv"
OUTPUT_DIR = "baseline_delta_csv"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_LANDMARKS = 478
BASELINE_OFFSET = -1


def make_delta_csv(input_path, output_path):
    print(f"処理中: {input_path}")

    df = pd.read_csv(input_path)

    if "drink_start_frame" not in df.columns:
        print("drink_start_frame列がありません")
        return

    drink_start_frame = int(df["drink_start_frame"].dropna().iloc[0])
    baseline_frame = drink_start_frame + BASELINE_OFFSET

    if baseline_frame < 0 or baseline_frame >= len(df):
        print(f"baseline_frameが範囲外です: {baseline_frame}")
        return

    baseline_row = df[df["frame_index"] == baseline_frame]

    if len(baseline_row) == 0:
        print(f"baseline_frameが見つかりません: {baseline_frame}")
        return

    baseline_row = baseline_row.iloc[0]

    out_df = pd.DataFrame()

    meta_cols = [
        "frame_index",
        "time_sec",
        "detected",
        "taste_score",
        "drink_start_frame",
        "eye_dist"
    ]

    for col in meta_cols:
        if col in df.columns:
            out_df[col] = df[col]

    out_df["baseline_frame"] = baseline_frame

    for i in range(NUM_LANDMARKS):
        for axis in ["x", "y", "z"]:
            col = f"landmark_{i}_{axis}_local"

            if col not in df.columns:
                continue

            delta_col = f"landmark_{i}_{axis}_delta"

            out_df[delta_col] = df[col] - baseline_row[col]

    out_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"保存完了: {output_path}")


csv_files = sorted([
    f for f in os.listdir(INPUT_DIR)
    if f.endswith(".csv")
])

if len(csv_files) == 0:
    print("正規化済みCSVがありません")
    exit()

for csv_file in csv_files:
    input_path = os.path.join(INPUT_DIR, csv_file)

    base_name = os.path.splitext(csv_file)[0]
    output_path = os.path.join(
        OUTPUT_DIR,
        f"{base_name}_delta.csv"
    )

    make_delta_csv(input_path, output_path)

print("baseline差分CSVの作成が完了しました")