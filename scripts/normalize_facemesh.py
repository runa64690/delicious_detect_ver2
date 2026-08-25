#!/usr/bin/env python3
"""
FaceMesh CSVをローカル座標へ正規化する。

各フレームで以下を行う。
1. 鼻根ランドマーク168を原点にする
2. 左右の目の中心を結ぶ線が水平になるように回転する
3. 目中心間距離で割り、無次元化する
4. y軸を上向き正にする
5. FaceMesh未検出フレームを除き、全フレームを保存する

出力:
    normalized_facemesh/<trial_id>.csv

未収集などで入力ファイルが存在しないlabels.csvの行はSKIPする。
存在するファイルに不備がある場合はERRORとして表示する。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


LANDMARK_COUNT = 478

NOSE_ID = 168
EYE_A_IDS = (33, 133)
EYE_B_IDS = (362, 263)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FaceMeshのピクセル座標を鼻根中心の無次元ローカル座標へ変換します。"
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("data/labels.csv"),
        help="ラベルCSVのパス（既定: data/labels.csv）",
    )
    parser.add_argument(
        "--facemesh-dir",
        type=Path,
        default=Path("data/raw_facemesh"),
        help="元FaceMesh CSVのフォルダ（既定: data/raw_facemesh）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/normalized_facemesh"),
        help="正規化後CSVの保存先（既定: data/normalized_facemesh）",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help=(
            "互換・検証用に入力の先頭Nフレームへ制限する。"
            "未指定時は全フレームを使用"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="同名の出力CSVが存在する場合に上書きする",
    )
    return parser.parse_args()


def load_labels(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"labels.csvが見つかりません: {path}")

    df = pd.read_csv(path)

    required = {"trial_id", "participant_id", "drink_label", "facemesh_csv"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            "labels.csvに必要な列がありません: " + ", ".join(missing)
        )

    if df["trial_id"].isna().any():
        raise ValueError("labels.csvのtrial_idに空欄があります")

    if df["trial_id"].duplicated().any():
        duplicated = df.loc[df["trial_id"].duplicated(False), "trial_id"].tolist()
        raise ValueError(f"trial_idが重複しています: {duplicated}")

    return df


def resolve_input_path(facemesh_dir: Path, value: object) -> Path:
    if pd.isna(value) or str(value).strip() == "":
        raise ValueError("facemesh_csvが空欄です")

    path = Path(str(value).strip())
    if not path.is_absolute():
        path = facemesh_dir / path
    return path


def required_coordinate_columns() -> list[str]:
    columns: list[str] = []
    for landmark_id in range(LANDMARK_COUNT):
        columns.extend(
            [
                f"lm{landmark_id}_x_px",
                f"lm{landmark_id}_y_px",
            ]
        )
    return columns


def normalize_dataframe(
    df: pd.DataFrame,
    *,
    trial_id: str,
    source_file: str,
    frames: int | None,
) -> pd.DataFrame:
    required_columns = ["facemesh_ok", *required_coordinate_columns()]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        preview = ", ".join(missing_columns[:8])
        suffix = " ..." if len(missing_columns) > 8 else ""
        raise ValueError(f"必要な列がありません: {preview}{suffix}")

    source = df.copy() if frames is None else df.iloc[:frames].copy()
    if source.empty:
        raise ValueError("入力CSVに処理可能な行がありません")

    facemesh_ok = pd.to_numeric(source["facemesh_ok"], errors="coerce")
    valid_detection = facemesh_ok.eq(1)
    source_frame_indices = np.flatnonzero(valid_detection.to_numpy())
    work = source.loc[valid_detection].copy()
    if work.empty:
        raise ValueError("FaceMesh検出成功フレームがありません")

    x_columns = [f"lm{i}_x_px" for i in range(LANDMARK_COUNT)]
    y_columns = [f"lm{i}_y_px" for i in range(LANDMARK_COUNT)]

    x = work[x_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    y = work[y_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)

    invalid_xy = ~np.isfinite(x) | ~np.isfinite(y)
    if invalid_xy.any():
        frame_indices = np.unique(np.where(invalid_xy)[0])
        preview = frame_indices[:10].tolist()
        raise ValueError(
            "FaceMesh検出成功フレームの座標に欠損・非数値があります: "
            f"{len(frame_indices)}フレーム、行番号例={preview}"
        )

    # 鼻根168
    nose_x = x[:, NOSE_ID]
    nose_y = y[:, NOSE_ID]

    # 左右それぞれの目頭・目尻の中点
    eye_a_x = (x[:, EYE_A_IDS[0]] + x[:, EYE_A_IDS[1]]) / 2.0
    eye_a_y = (y[:, EYE_A_IDS[0]] + y[:, EYE_A_IDS[1]]) / 2.0
    eye_b_x = (x[:, EYE_B_IDS[0]] + x[:, EYE_B_IDS[1]]) / 2.0
    eye_b_y = (y[:, EYE_B_IDS[0]] + y[:, EYE_B_IDS[1]]) / 2.0

    eye_dx = eye_b_x - eye_a_x
    eye_dy = eye_b_y - eye_a_y
    eye_distance = np.sqrt(eye_dx**2 + eye_dy**2)

    invalid_distance = ~np.isfinite(eye_distance) | (eye_distance < 1e-6)
    if invalid_distance.any():
        bad_frames = np.flatnonzero(invalid_distance).tolist()
        raise ValueError(
            "眼間距離を計算できないフレームがあります: "
            f"{bad_frames[:10]}"
        )

    # 目を結ぶ線の角度
    theta = np.arctan2(eye_dy, eye_dx)
    cos_theta = np.cos(theta)[:, None]
    sin_theta = np.sin(theta)[:, None]

    # 1. 鼻根中心化
    x_centered = x - nose_x[:, None]
    y_centered = y - nose_y[:, None]

    # 2. -theta回転で目を水平にする
    x_rotated = cos_theta * x_centered + sin_theta * y_centered
    y_rotated = -sin_theta * x_centered + cos_theta * y_centered

    # 3. 眼間距離で無次元化
    x_local = x_rotated / eye_distance[:, None]

    # 4. 画像座標は下向き正なので、上向き正へ反転
    y_local = -y_rotated / eye_distance[:, None]

    output_frames = len(work)
    metadata = pd.DataFrame(
        {
            "trial_id": [trial_id] * output_frames,
            "source_file": [source_file] * output_frames,
            # 元CSVの0始まり行番号。未検出行を除いた場合は欠番を保持する。
            "frame_index": source_frame_indices.astype(np.int64),
            "rotation_rad": theta,
            "rotation_deg": np.degrees(theta),
            "eye_distance_px": eye_distance,
        }
    )

    # 元CSVに時間情報があれば保持する
    for column in (
        "video_name",
        "rec_frame",
        "global_frame",
        "timestamp_ms",
        "elapsed_sec",
    ):
        if column in work.columns:
            metadata[column] = work[column].to_numpy()

    normalized_columns: dict[str, np.ndarray] = {}
    for landmark_id in range(LANDMARK_COUNT):
        normalized_columns[f"lm{landmark_id}_x_local"] = x_local[:, landmark_id]
        normalized_columns[f"lm{landmark_id}_y_local"] = y_local[:, landmark_id]

    normalized = pd.DataFrame(normalized_columns)
    return pd.concat(
        [metadata.reset_index(drop=True), normalized.reset_index(drop=True)],
        axis=1,
    )


def validate_normalized_output(df: pd.DataFrame, tolerance: float = 1e-5) -> None:
    nose_x = df[f"lm{NOSE_ID}_x_local"].to_numpy()
    nose_y = df[f"lm{NOSE_ID}_y_local"].to_numpy()

    if np.max(np.abs(nose_x)) > tolerance or np.max(np.abs(nose_y)) > tolerance:
        raise ValueError("正規化後も鼻根168が原点になっていません")

    eye_a_x = (
        df[f"lm{EYE_A_IDS[0]}_x_local"].to_numpy()
        + df[f"lm{EYE_A_IDS[1]}_x_local"].to_numpy()
    ) / 2.0
    eye_a_y = (
        df[f"lm{EYE_A_IDS[0]}_y_local"].to_numpy()
        + df[f"lm{EYE_A_IDS[1]}_y_local"].to_numpy()
    ) / 2.0

    eye_b_x = (
        df[f"lm{EYE_B_IDS[0]}_x_local"].to_numpy()
        + df[f"lm{EYE_B_IDS[1]}_x_local"].to_numpy()
    ) / 2.0
    eye_b_y = (
        df[f"lm{EYE_B_IDS[0]}_y_local"].to_numpy()
        + df[f"lm{EYE_B_IDS[1]}_y_local"].to_numpy()
    ) / 2.0

    horizontal_error = np.max(np.abs(eye_a_y - eye_b_y))
    normalized_eye_distance = np.sqrt(
        (eye_b_x - eye_a_x) ** 2 + (eye_b_y - eye_a_y) ** 2
    )
    scale_error = np.max(np.abs(normalized_eye_distance - 1.0))

    if horizontal_error > tolerance:
        raise ValueError(
            f"正規化後の目が水平ではありません: 最大誤差={horizontal_error:.3e}"
        )

    if scale_error > tolerance:
        raise ValueError(
            f"正規化後の眼間距離が1ではありません: 最大誤差={scale_error:.3e}"
        )


def main() -> int:
    args = parse_args()

    if args.frames is not None and args.frames <= 0:
        print("[FATAL] --framesは1以上にしてください", file=sys.stderr)
        return 2

    try:
        labels = load_labels(args.labels)
    except Exception as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)

    ok_count = 0
    skip_count = 0
    error_count = 0

    print("=" * 80)
    print("FaceMesh正規化")
    print("=" * 80)
    print(f"labels       : {args.labels}")
    print(f"input folder : {args.facemesh_dir}")
    print(f"output folder: {args.output_dir}")
    frame_setting = args.frames if args.frames is not None else "all"
    print(f"frames       : {frame_setting}")
    print(f"landmarks    : {LANDMARK_COUNT} points, XY only")
    print()

    for _, label_row in labels.iterrows():
        trial_id = str(label_row["trial_id"]).strip()

        try:
            input_path = resolve_input_path(
                args.facemesh_dir,
                label_row["facemesh_csv"],
            )
        except Exception as exc:
            print(f"[ERROR] {trial_id}: {exc}")
            error_count += 1
            continue

        if not input_path.exists():
            print(f"[SKIP]  {trial_id}: ファイルなし（未収集の可能性）")
            skip_count += 1
            continue

        output_path = args.output_dir / f"{trial_id}.csv"

        if output_path.exists() and not args.overwrite:
            print(
                f"[SKIP]  {trial_id}: 出力済み "
                f"({output_path.name})。上書きは--overwriteを指定"
            )
            skip_count += 1
            continue

        try:
            raw_df = pd.read_csv(input_path)
            normalized_df = normalize_dataframe(
                raw_df,
                trial_id=trial_id,
                source_file=input_path.name,
                frames=args.frames,
            )
            validate_normalized_output(normalized_df)

            normalized_df.to_csv(
                output_path,
                index=False,
                encoding="utf-8-sig",
                float_format="%.8f",
            )

            mean_rotation = normalized_df["rotation_deg"].mean()
            mean_eye_distance = normalized_df["eye_distance_px"].mean()
            input_frames = (
                min(len(raw_df), args.frames)
                if args.frames is not None
                else len(raw_df)
            )
            skipped_frames = input_frames - len(normalized_df)

            print(
                f"[ OK ]  {trial_id}: {output_path.name} | "
                f"{len(normalized_df)}/{input_frames} frames | "
                f"skipped={skipped_frames} | "
                f"rotation mean={mean_rotation:.2f} deg | "
                f"eye distance mean={mean_eye_distance:.1f} px"
            )
            ok_count += 1

        except Exception as exc:
            print(f"[ERROR] {trial_id}: {input_path.name} | {exc}")
            error_count += 1

    print()
    print("=" * 80)
    print("正規化結果")
    print("=" * 80)
    print(f"OK    : {ok_count}")
    print(f"SKIP  : {skip_count}")
    print(f"ERROR : {error_count}")
    print(f"合計  : {len(labels)}")

    if error_count > 0:
        print()
        print("判定: ERRORがあります。表示された試行を修正してください。")
        return 1

    print()
    print("判定: 存在するFaceMesh CSVの正規化が完了しました。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
