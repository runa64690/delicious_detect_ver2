import os
import csv
import re

# =========================
# 設定
# =========================

# 入力CSVが入っているフォルダ
INPUT_DIR_NAME = "facemesh_csv"

# 変換後CSVを保存するフォルダ
OUTPUT_DIR_NAME = "nose_origin_facemesh_csv"

# 鼻の根として使うランドマーク番号
# これまでの処理で基準点として使っていた鼻付近の点
NOSE_ROOT_ID = 168


# =========================
# パス設定
# =========================

def get_base_dir():
    """
    このプログラムが置かれている場所を取得する。
    Jupyterなどで __file__ が使えない場合は、現在の作業フォルダを使う。
    """
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        return os.getcwd()


BASE_DIR = get_base_dir()
INPUT_DIR = os.path.join(BASE_DIR, INPUT_DIR_NAME)
OUTPUT_DIR = os.path.join(BASE_DIR, OUTPUT_DIR_NAME)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# 補助関数
# =========================

def to_float(value):
    """
    CSVから読んだ値をfloatに変換する。
    空欄や変換できない値の場合は None を返す。
    """
    if value is None:
        return None

    value = str(value).strip()

    if value == "":
        return None

    try:
        return float(value)
    except ValueError:
        return None


def detect_landmark_ids(header):
    """
    CSVヘッダーからランドマーク番号を自動検出する。
    例:
        lm0_x_norm
        lm1_x_norm
        ...
    から 0, 1, ... を取り出す。
    """
    landmark_ids = []

    pattern = re.compile(r"lm(\d+)_x_norm")

    for col in header:
        match = pattern.fullmatch(col)
        if match:
            landmark_ids.append(int(match.group(1)))

    landmark_ids.sort()
    return landmark_ids


def make_output_header(input_header, landmark_ids):
    """
    変換後CSVのヘッダーを作る。

    元のメタ情報:
        video_name, rec_frame, global_frame, timestamp_ms, elapsed_sec, facemesh_ok

    追加情報:
        nose_x_norm, nose_y_norm, nose_z_norm, nose_x_px, nose_y_px

    各ランドマークの変換後座標:
        lm0_x_rel_norm, lm0_y_rel_norm, lm0_z_rel_norm, lm0_x_rel_px, lm0_y_rel_px
        ...
    """

    meta_cols = [
        "video_name",
        "rec_frame",
        "global_frame",
        "timestamp_ms",
        "elapsed_sec",
        "facemesh_ok"
    ]

    output_header = []

    for col in meta_cols:
        if col in input_header:
            output_header.append(col)

    output_header += [
        "nose_origin_landmark_id",
        "nose_x_norm",
        "nose_y_norm",
        "nose_z_norm",
        "nose_x_px",
        "nose_y_px"
    ]

    for lm_id in landmark_ids:
        output_header += [
            f"lm{lm_id}_x_rel_norm",
            f"lm{lm_id}_y_rel_norm",
            f"lm{lm_id}_z_rel_norm",
            f"lm{lm_id}_x_rel_px",
            f"lm{lm_id}_y_rel_px"
        ]

    return output_header


def convert_one_csv(input_path, output_path):
    """
    1つのCSVファイルを読み込み、
    鼻の根を原点にした座標へ変換して保存する。
    """

    with open(input_path, "r", newline="", encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)
        input_header = reader.fieldnames

        if input_header is None:
            print(f"スキップ: ヘッダーがありません: {input_path}")
            return

        landmark_ids = detect_landmark_ids(input_header)

        if not landmark_ids:
            print(f"スキップ: ランドマーク列が見つかりません: {input_path}")
            return

        required_nose_cols = [
            f"lm{NOSE_ROOT_ID}_x_norm",
            f"lm{NOSE_ROOT_ID}_y_norm",
            f"lm{NOSE_ROOT_ID}_z_norm",
            f"lm{NOSE_ROOT_ID}_x_px",
            f"lm{NOSE_ROOT_ID}_y_px"
        ]

        missing_cols = [col for col in required_nose_cols if col not in input_header]

        if missing_cols:
            print(f"スキップ: 鼻の根ランドマーク列がありません: {input_path}")
            print("不足列:", missing_cols)
            return

        output_header = make_output_header(input_header, landmark_ids)

        with open(output_path, "w", newline="", encoding="utf-8") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=output_header)
            writer.writeheader()

            total_rows = 0
            converted_rows = 0

            for row in reader:
                total_rows += 1

                out_row = {}

                # 元のメタ情報をコピー
                for col in [
                    "video_name",
                    "rec_frame",
                    "global_frame",
                    "timestamp_ms",
                    "elapsed_sec",
                    "facemesh_ok"
                ]:
                    if col in output_header:
                        out_row[col] = row.get(col, "")

                # 鼻の根の座標を取得
                nose_x_norm = to_float(row.get(f"lm{NOSE_ROOT_ID}_x_norm"))
                nose_y_norm = to_float(row.get(f"lm{NOSE_ROOT_ID}_y_norm"))
                nose_z_norm = to_float(row.get(f"lm{NOSE_ROOT_ID}_z_norm"))
                nose_x_px = to_float(row.get(f"lm{NOSE_ROOT_ID}_x_px"))
                nose_y_px = to_float(row.get(f"lm{NOSE_ROOT_ID}_y_px"))

                out_row["nose_origin_landmark_id"] = NOSE_ROOT_ID
                out_row["nose_x_norm"] = "" if nose_x_norm is None else nose_x_norm
                out_row["nose_y_norm"] = "" if nose_y_norm is None else nose_y_norm
                out_row["nose_z_norm"] = "" if nose_z_norm is None else nose_z_norm
                out_row["nose_x_px"] = "" if nose_x_px is None else nose_x_px
                out_row["nose_y_px"] = "" if nose_y_px is None else nose_y_px

                # FaceMesh失敗フレームや、鼻の根が空欄のフレームは変換できない
                nose_available = (
                    nose_x_norm is not None and
                    nose_y_norm is not None and
                    nose_z_norm is not None and
                    nose_x_px is not None and
                    nose_y_px is not None
                )

                if not nose_available:
                    for lm_id in landmark_ids:
                        out_row[f"lm{lm_id}_x_rel_norm"] = ""
                        out_row[f"lm{lm_id}_y_rel_norm"] = ""
                        out_row[f"lm{lm_id}_z_rel_norm"] = ""
                        out_row[f"lm{lm_id}_x_rel_px"] = ""
                        out_row[f"lm{lm_id}_y_rel_px"] = ""

                    writer.writerow(out_row)
                    continue

                # 各ランドマークを鼻の根基準に変換
                for lm_id in landmark_ids:
                    x_norm = to_float(row.get(f"lm{lm_id}_x_norm"))
                    y_norm = to_float(row.get(f"lm{lm_id}_y_norm"))
                    z_norm = to_float(row.get(f"lm{lm_id}_z_norm"))
                    x_px = to_float(row.get(f"lm{lm_id}_x_px"))
                    y_px = to_float(row.get(f"lm{lm_id}_y_px"))

                    if (
                        x_norm is None or
                        y_norm is None or
                        z_norm is None or
                        x_px is None or
                        y_px is None
                    ):
                        out_row[f"lm{lm_id}_x_rel_norm"] = ""
                        out_row[f"lm{lm_id}_y_rel_norm"] = ""
                        out_row[f"lm{lm_id}_z_rel_norm"] = ""
                        out_row[f"lm{lm_id}_x_rel_px"] = ""
                        out_row[f"lm{lm_id}_y_rel_px"] = ""
                    else:
                        out_row[f"lm{lm_id}_x_rel_norm"] = x_norm - nose_x_norm
                        out_row[f"lm{lm_id}_y_rel_norm"] = y_norm - nose_y_norm
                        out_row[f"lm{lm_id}_z_rel_norm"] = z_norm - nose_z_norm
                        out_row[f"lm{lm_id}_x_rel_px"] = x_px - nose_x_px
                        out_row[f"lm{lm_id}_y_rel_px"] = y_px - nose_y_px

                converted_rows += 1
                writer.writerow(out_row)

    print(f"変換完了: {os.path.basename(input_path)}")
    print(f"  入力行数: {total_rows}")
    print(f"  変換できた行数: {converted_rows}")
    print(f"  保存先: {output_path}")


def main():
    if not os.path.exists(INPUT_DIR):
        print("入力フォルダが見つかりません。")
        print(f"探した場所: {INPUT_DIR}")
        return

    csv_files = [
        name for name in os.listdir(INPUT_DIR)
        if name.lower().endswith(".csv")
    ]

    if not csv_files:
        print("入力フォルダ内にCSVファイルがありません。")
        print(f"入力フォルダ: {INPUT_DIR}")
        return

    print("入力フォルダ:", INPUT_DIR)
    print("出力フォルダ:", OUTPUT_DIR)
    print(f"鼻の根として使うランドマーク番号: {NOSE_ROOT_ID}")
    print(f"CSVファイル数: {len(csv_files)}")
    print()

    for csv_name in csv_files:
        input_path = os.path.join(INPUT_DIR, csv_name)

        base_name, ext = os.path.splitext(csv_name)
        output_name = base_name + "_nose_origin.csv"
        output_path = os.path.join(OUTPUT_DIR, output_name)

        convert_one_csv(input_path, output_path)

    print()
    print("すべての変換が完了しました。")


if __name__ == "__main__":
    main()