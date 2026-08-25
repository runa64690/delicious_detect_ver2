from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 実験条件
# ============================================================

NUM_FRAMES = 190

EXPECTED_PARTICIPANTS = 6
EXPECTED_TRIALS_PER_PARTICIPANT = 10
EXPECTED_TRIALS_PER_DRINK = 5

DRINK_LABELS = {
    "water": 0,
    "sparkling_water": 1,
}

# 方法Bで使用する80点
EYEBROW_POINTS = [
    46, 52, 53, 55, 63, 65, 66, 70, 105, 107,
    276, 282, 283, 285, 293, 295, 296, 300, 334, 336,
]

EYE_POINTS = [
    7, 33, 133, 144, 145, 153, 154, 155,
    157, 158, 159, 160, 161, 163, 173, 246,
    249, 263, 362, 373, 374, 380, 381, 382,
    384, 385, 386, 387, 388, 390, 398, 466,
]

CHEEK_POINTS = [
    50, 101, 205, 206,
    280, 330, 425, 426,
]

MOUTH_POINTS = [
    61, 185, 40, 0, 267, 409, 291,
    146, 84, 17, 314, 375,
    78, 82, 13, 312, 308, 87, 14, 317,
]

FACE_POINTS = list(dict.fromkeys(
    EYEBROW_POINTS + EYE_POINTS + CHEEK_POINTS + MOUTH_POINTS
))

# 鼻根中心化・目水平化・眼間距離正規化に必要な点
REFERENCE_POINTS = [168, 33, 133, 362, 263]
REQUIRED_POINTS = sorted(set(FACE_POINTS + REFERENCE_POINTS))

REQUIRED_LABEL_COLUMNS = [
    "trial_id",
    "participant_id",
    "trial_order",
    "drink_name",
    "drink_label",
    "facemesh_csv",
]

# FaceMesh欠損を「補間候補」とする条件
MAX_FAILURE_RATIO_FOR_INTERPOLATION = 0.05
MAX_CONSECUTIVE_FAILURES_FOR_INTERPOLATION = 3


# ============================================================
# 検査結果
# ============================================================

@dataclass
class TrialResult:
    trial_id: str
    participant_id: str
    filename: str
    status: str = "OK"
    frame_count: int | None = None
    success_rate: float | None = None
    failure_count: int | None = None
    max_consecutive_failures: int | None = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)
        if self.status == "OK":
            self.status = "WARNING"

    def add_error(self, message: str) -> None:
        self.errors.append(message)
        self.status = "ERROR"


def max_consecutive_true(mask: np.ndarray) -> int:
    """Trueが最大で何個連続しているかを返す。"""
    maximum = 0
    current = 0

    for value in mask:
        if bool(value):
            current += 1
            maximum = max(maximum, current)
        else:
            current = 0

    return maximum


def required_coordinate_columns() -> list[str]:
    columns: list[str] = []

    for landmark_id in REQUIRED_POINTS:
        columns.extend([
            f"lm{landmark_id}_x_px",
            f"lm{landmark_id}_y_px",
        ])

    return columns


def check_eye_distance(df: pd.DataFrame) -> np.ndarray:
    """
    各フレームの左右目中心間距離を返す。
    目中心:
      片側 = (33 + 133) / 2
      反対側 = (362 + 263) / 2
    """
    eye_a_x = (df["lm33_x_px"] + df["lm133_x_px"]) / 2.0
    eye_a_y = (df["lm33_y_px"] + df["lm133_y_px"]) / 2.0

    eye_b_x = (df["lm362_x_px"] + df["lm263_x_px"]) / 2.0
    eye_b_y = (df["lm362_y_px"] + df["lm263_y_px"]) / 2.0

    return np.sqrt(
        (eye_b_x - eye_a_x) ** 2
        + (eye_b_y - eye_a_y) ** 2
    ).to_numpy(dtype=float)


def validate_labels_table(labels: pd.DataFrame) -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    errors: list[str] = []

    missing_columns = [
        column for column in REQUIRED_LABEL_COLUMNS
        if column not in labels.columns
    ]
    if missing_columns:
        errors.append(
            "labels.csvに必要な列がありません: "
            + ", ".join(missing_columns)
        )
        return warnings, errors

    if labels.empty:
        errors.append("labels.csvにデータ行がありません")
        return warnings, errors

    for column in REQUIRED_LABEL_COLUMNS:
        missing_count = int(labels[column].isna().sum())
        if missing_count > 0:
            errors.append(
                f"labels.csvの{column}列に空欄が"
                f"{missing_count}件あります"
            )

    duplicated_trials = labels.loc[
        labels["trial_id"].duplicated(keep=False), "trial_id"
    ].astype(str).unique()
    if len(duplicated_trials) > 0:
        errors.append(
            "trial_idが重複しています: "
            + ", ".join(duplicated_trials)
        )

    duplicated_files = labels.loc[
        labels["facemesh_csv"].duplicated(keep=False), "facemesh_csv"
    ].astype(str).unique()
    if len(duplicated_files) > 0:
        errors.append(
            "同じFaceMesh CSVが複数試行に割り当てられています: "
            + ", ".join(duplicated_files)
        )

    # participant_idとtrial_id先頭の一致
    for row in labels.itertuples(index=False):
        trial_id = str(row.trial_id)
        participant_id = str(row.participant_id)

        expected_prefix = trial_id.split("_", maxsplit=1)[0]
        if expected_prefix != participant_id:
            errors.append(
                f"{trial_id}: participant_id={participant_id}ですが、"
                f"trial_idの先頭は{expected_prefix}です"
            )

    # 飲料名と数値ラベルの対応
    for row in labels.itertuples(index=False):
        drink_name = str(row.drink_name)
        trial_id = str(row.trial_id)

        if drink_name not in DRINK_LABELS:
            errors.append(
                f"{trial_id}: 未定義のdrink_nameです: {drink_name}"
            )
            continue

        try:
            actual_label = int(row.drink_label)
        except (TypeError, ValueError):
            errors.append(
                f"{trial_id}: drink_labelを整数に変換できません: "
                f"{row.drink_label}"
            )
            continue

        expected_label = DRINK_LABELS[drink_name]
        if actual_label != expected_label:
            errors.append(
                f"{trial_id}: {drink_name}のdrink_labelは"
                f"{expected_label}である必要があります"
                f"（現在値: {actual_label}）"
            )

    participant_count = labels["participant_id"].nunique()
    if participant_count != EXPECTED_PARTICIPANTS:
        warnings.append(
            f"参加者数が想定と異なります: "
            f"{participant_count}人 "
            f"（想定: {EXPECTED_PARTICIPANTS}人）"
        )

    expected_total = (
        EXPECTED_PARTICIPANTS * EXPECTED_TRIALS_PER_PARTICIPANT
    )
    if len(labels) != expected_total:
        warnings.append(
            f"全試行数が想定と異なります: "
            f"{len(labels)}試行（想定: {expected_total}試行）"
        )

    # 参加者ごとの試行数・クラス数・trial_order
    for participant_id, group in labels.groupby(
        "participant_id", sort=True
    ):
        if len(group) != EXPECTED_TRIALS_PER_PARTICIPANT:
            warnings.append(
                f"{participant_id}: 試行数が{len(group)}件です"
                f"（想定: {EXPECTED_TRIALS_PER_PARTICIPANT}件）"
            )

        label_counts = group["drink_label"].value_counts().to_dict()

        for drink_name, drink_label in DRINK_LABELS.items():
            count = int(label_counts.get(drink_label, 0))
            if count != EXPECTED_TRIALS_PER_DRINK:
                warnings.append(
                    f"{participant_id}: {drink_name}が{count}件です"
                    f"（想定: {EXPECTED_TRIALS_PER_DRINK}件）"
                )

        trial_order_numeric = pd.to_numeric(
            group["trial_order"], errors="coerce"
        )
        if trial_order_numeric.isna().any():
            errors.append(
                f"{participant_id}: trial_orderに数値でない値があります"
            )
        elif trial_order_numeric.duplicated().any():
            errors.append(
                f"{participant_id}: trial_orderが重複しています"
            )
        else:
            actual_orders = sorted(
                trial_order_numeric.astype(int).tolist()
            )
            expected_orders = list(range(1, len(group) + 1))
            if actual_orders != expected_orders:
                warnings.append(
                    f"{participant_id}: trial_orderが"
                    f"1〜{len(group)}の連番ではありません "
                    f"（現在: {actual_orders}）"
                )

    return warnings, errors


def validate_facemesh_csv(
    row: pd.Series,
    facemesh_dir: Path,
    num_frames: int,
) -> TrialResult:
    trial_id = str(row["trial_id"])
    participant_id = str(row["participant_id"])
    filename = str(row["facemesh_csv"])

    result = TrialResult(
        trial_id=trial_id,
        participant_id=participant_id,
        filename=filename,
    )

    csv_path = facemesh_dir / filename

    if not csv_path.is_file():
        result.add_error(f"ファイルがありません: {csv_path}")
        return result

    if Path(filename).stem != trial_id:
        result.add_warning(
            f"ファイル名とtrial_idが一致しません "
            f"({filename} / {trial_id})"
        )

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        result.add_error(
            f"CSVを読み込めません: "
            f"{type(exc).__name__}: {exc}"
        )
        return result

    result.frame_count = len(df)

    if len(df) < num_frames:
        result.add_error(
            f"フレーム不足: {len(df)} < {num_frames}"
        )
        return result

    required_columns = (
        ["facemesh_ok"]
        + required_coordinate_columns()
    )
    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        preview = ", ".join(missing_columns[:10])
        suffix = (
            f" ほか{len(missing_columns) - 10}列"
            if len(missing_columns) > 10
            else ""
        )
        result.add_error(
            f"必要な列がありません: {preview}{suffix}"
        )
        return result

    used = df.iloc[:num_frames].copy()

    # 数値へ変換できない値もNaNとして検出する
    coordinate_columns = required_coordinate_columns()
    used[coordinate_columns] = used[coordinate_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    facemesh_ok = pd.to_numeric(
        used["facemesh_ok"], errors="coerce"
    )

    coordinate_failure = used[coordinate_columns].isna().any(axis=1)
    detection_failure = facemesh_ok.ne(1) | facemesh_ok.isna()

    invalid_frame = (
        coordinate_failure.to_numpy()
        | detection_failure.to_numpy()
    )

    result.failure_count = int(invalid_frame.sum())
    result.success_rate = float(1.0 - invalid_frame.mean())
    result.max_consecutive_failures = max_consecutive_true(
        invalid_frame
    )

    if result.failure_count > 0:
        failure_ratio = result.failure_count / num_frames

        if (
            failure_ratio
            <= MAX_FAILURE_RATIO_FOR_INTERPOLATION
            and result.max_consecutive_failures
            <= MAX_CONSECUTIVE_FAILURES_FOR_INTERPOLATION
        ):
            result.add_warning(
                f"欠損フレームが{result.failure_count}件あります "
                f"（成功率{result.success_rate:.1%}、"
                f"最大連続{result.max_consecutive_failures}）"
                "。線形補間候補です"
            )
        else:
            result.add_error(
                f"欠損が多すぎます: "
                f"{result.failure_count}/{num_frames}フレーム "
                f"（成功率{result.success_rate:.1%}、"
                f"最大連続"
                f"{result.max_consecutive_failures}）"
            )

    # 正規化に使う眼間距離が計算できるか
    try:
        eye_distance = check_eye_distance(used)
    except Exception as exc:
        result.add_error(
            f"眼間距離を計算できません: "
            f"{type(exc).__name__}: {exc}"
        )
        return result

    valid_eye_distance = np.isfinite(eye_distance)
    valid_eye_distance &= eye_distance > 1e-6

    # すでに欠損扱いになっているフレームを除き、
    # 座標が存在するのに眼間距離が異常なケースを検出する
    unexpected_eye_failure = (
        ~valid_eye_distance & ~invalid_frame
    )
    if unexpected_eye_failure.any():
        result.add_error(
            "座標はありますが眼間距離が0または非数の"
            f"フレームが{int(unexpected_eye_failure.sum())}件あります"
        )

    return result


def print_section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def print_labels_summary(labels: pd.DataFrame) -> None:
    print_section("ラベル構成")

    summary = pd.crosstab(
        labels["participant_id"],
        labels["drink_name"],
        margins=True,
        margins_name="合計",
    )

    print(summary.to_string())
    print()
    print(f"全試行数: {len(labels)}")
    print(
        f"参加者数: "
        f"{labels['participant_id'].nunique()}"
    )


def print_trial_result(result: TrialResult) -> None:
    icon = {
        "OK": "[ OK ]",
        "WARNING": "[WARN]",
        "ERROR": "[ERROR]",
    }[result.status]

    details = []

    if result.frame_count is not None:
        details.append(f"{result.frame_count} frames")

    if result.success_rate is not None:
        details.append(
            f"success={result.success_rate:.1%}"
        )

    detail_text = (
        " | " + ", ".join(details)
        if details
        else ""
    )

    print(
        f"{icon} {result.trial_id:<9} "
        f"{result.filename}{detail_text}"
    )

    for message in result.warnings:
        print(f"       警告: {message}")

    for message in result.errors:
        print(f"       エラー: {message}")


def parse_args() -> argparse.Namespace:
    default_project_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        description=(
            "labels.csvとFaceMesh CSVを検査し、"
            "結果をコンソールに表示します"
        )
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=default_project_root / "data" / "labels.csv",
        help="labels.csvのパス",
    )
    parser.add_argument(
        "--facemesh-dir",
        type=Path,
        default=(
            default_project_root
            / "data"
            / "raw_facemesh"
        ),
        help="FaceMesh CSVフォルダ",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=NUM_FRAMES,
        help="各試行で使用する先頭フレーム数",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print("FaceMeshデータセット検査")
    print(f"labels.csv       : {args.labels.resolve()}")
    print(
        f"FaceMeshフォルダ : "
        f"{args.facemesh_dir.resolve()}"
    )
    print(f"使用フレーム数   : {args.frames}")
    print(f"方法Bの点数      : {len(FACE_POINTS)}")
    print(
        f"方法Bの入力形状  : "
        f"({args.frames}, {len(FACE_POINTS) * 2}) / 1試行"
    )

    if not args.labels.is_file():
        print()
        print(
            f"[ERROR] labels.csvがありません: "
            f"{args.labels.resolve()}"
        )
        return 1

    if not args.facemesh_dir.is_dir():
        print()
        print(
            f"[ERROR] FaceMeshフォルダがありません: "
            f"{args.facemesh_dir.resolve()}"
        )
        return 1

    try:
        labels = pd.read_csv(args.labels)
    except Exception as exc:
        print()
        print(
            f"[ERROR] labels.csvを読み込めません: "
            f"{type(exc).__name__}: {exc}"
        )
        return 1

    label_warnings, label_errors = validate_labels_table(
        labels
    )

    if all(
        column in labels.columns
        for column in REQUIRED_LABEL_COLUMNS
    ):
        print_labels_summary(labels)

    print_section("labels.csvの検査")

    if not label_warnings and not label_errors:
        print("[ OK ] labels.csvに問題は見つかりませんでした")

    for message in label_warnings:
        print(f"[WARN] {message}")

    for message in label_errors:
        print(f"[ERROR] {message}")

    # 必須列不足などで各試行を検査できない場合
    if label_errors and any(
        "必要な列" in message
        for message in label_errors
    ):
        print()
        print("必須列不足のためFaceMesh CSV検査を中止します")
        return 1

    print_section("FaceMesh CSVの検査")

    trial_results: list[TrialResult] = []

    for _, row in labels.iterrows():
        result = validate_facemesh_csv(
            row=row,
            facemesh_dir=args.facemesh_dir,
            num_frames=args.frames,
        )
        trial_results.append(result)
        print_trial_result(result)

    ok_count = sum(
        result.status == "OK"
        for result in trial_results
    )
    warning_count = sum(
        result.status == "WARNING"
        for result in trial_results
    )
    error_count = sum(
        result.status == "ERROR"
        for result in trial_results
    )

    print_section("検査結果")

    print(f"OK      : {ok_count}")
    print(f"WARNING : {warning_count}")
    print(f"ERROR   : {error_count}")
    print(f"合計    : {len(trial_results)}")

    if label_errors or error_count > 0:
        print()
        print(
            "判定: ERRORがあります。"
            "正規化・特徴量抽出へ進む前に修正してください。"
        )
        return 1

    if label_warnings or warning_count > 0:
        print()
        print(
            "判定: 警告があります。内容を確認してから"
            "次の処理へ進んでください。"
        )
        return 0

    print()
    print(
        "判定: 全データが検査条件を満たしています。"
        "正規化処理へ進めます。"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())