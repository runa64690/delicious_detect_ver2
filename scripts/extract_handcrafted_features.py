#!/usr/bin/env python3
"""Extract 21 handcrafted facial features from normalized FaceMesh CSV files.

One normalized CSV represents one trial.  Only ``lmXXX_x_local`` and
``lmXXX_y_local`` columns are used to calculate facial features.  Labels and
questionnaire values are attached to the output strictly as metadata; they are
never used in feature calculation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from .feature_schema import (
        FACIAL_FEATURE_COLUMNS,
        FEATURE_OUTPUT_COLUMNS,
        METADATA_COLUMNS,
        QUESTIONNAIRE_COLUMNS,
    )
except ImportError:  # Direct execution: python scripts/extract_handcrafted_features.py
    from feature_schema import (
        FACIAL_FEATURE_COLUMNS,
        FEATURE_OUTPUT_COLUMNS,
        METADATA_COLUMNS,
        QUESTIONNAIRE_COLUMNS,
    )


PROJECT_ROOT = Path(__file__).resolve().parent.parent

LABELS_PATH = Path("data/labels.csv")
EXCLUSIONS_PATH = Path("config/excluded_trials.csv")
NORMALIZED_DIR = Path("data/normalized_facemesh")
OUTPUT_PATH = Path("results/features.csv")
EXCLUDED_PATH = Path("results/excluded_trials.csv")
DEBUG_DIR = Path("results/debug_signals")

MIN_REQUIRED_FRAMES = 2
EPS = 1e-8
# Provisional value selected from the current-dataset sensitivity check.
# Revisit it with manually annotated blink frames before model evaluation.
BLINK_THRESHOLD_RATIO = 0.80
MIN_BLINK_FRAMES = 2
SAVE_DEBUG_SIGNALS = False
SAVE_DEBUG_PLOTS = False

EYE_A_POINTS = (33, 160, 158, 133, 153, 144)
EYE_B_POINTS = (362, 385, 387, 263, 373, 380)
MOUTH_POINTS = (61, 291, 13, 14)
BROW_A_POINTS = (70, 63, 105, 66, 107)
BROW_B_POINTS = (300, 293, 334, 296, 336)

REQUIRED_LANDMARKS = tuple(
    dict.fromkeys(
        EYE_A_POINTS
        + EYE_B_POINTS
        + MOUTH_POINTS
        + BROW_A_POINTS
        + BROW_B_POINTS
    )
)

# Backward-compatible aliases used throughout this script.
FEATURE_COLUMNS = FACIAL_FEATURE_COLUMNS
OUTPUT_COLUMNS = FEATURE_OUTPUT_COLUMNS
EXCLUDED_COLUMNS = [
    "trial_id",
    "participant_id",
    "reason",
    "reason_detail",
    "exclusion_source",
    "num_frames",
    "file_path",
]

REQUIRED_LABEL_COLUMNS = set(METADATA_COLUMNS + ["facemesh_csv"])


class TrialExclusion(ValueError):
    """A data problem that excludes one trial without stopping the run."""

    def __init__(
        self,
        reason: str,
        detail: str = "",
        *,
        num_frames: int | None = None,
    ) -> None:
        super().__init__(detail or reason)
        self.reason = reason
        self.detail = detail
        self.num_frames = num_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "正規化済みFaceMesh CSVから21個の手作り特徴量を抽出します。"
        )
    )
    parser.add_argument("--labels", type=Path, default=LABELS_PATH)
    parser.add_argument(
        "--exclusions", type=Path, default=EXCLUSIONS_PATH
    )
    parser.add_argument(
        "--normalized-dir", type=Path, default=NORMALIZED_DIR
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument(
        "--excluded-output", type=Path, default=EXCLUDED_PATH
    )
    parser.add_argument("--debug-dir", type=Path, default=DEBUG_DIR)
    parser.add_argument(
        "--save-debug-signals",
        action="store_true",
        default=SAVE_DEBUG_SIGNALS,
    )
    parser.add_argument(
        "--save-debug-plots",
        action="store_true",
        default=SAVE_DEBUG_PLOTS,
    )
    return parser.parse_args()


def project_path(path: Path) -> Path:
    """Resolve a CLI path relative to the project root, not the CWD."""
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_labels(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"labels.csvが見つかりません: {path}")

    labels = pd.read_csv(path)
    missing = sorted(REQUIRED_LABEL_COLUMNS - set(labels.columns))
    if missing:
        raise ValueError(
            "labels.csvに必要な列がありません: " + ", ".join(missing)
        )
    if labels["trial_id"].isna().any():
        raise ValueError("labels.csvのtrial_idに空欄があります")
    if labels["trial_id"].duplicated().any():
        duplicated = labels.loc[
            labels["trial_id"].duplicated(False), "trial_id"
        ].astype(str)
        raise ValueError(
            "labels.csvのtrial_idが重複しています: "
            + ", ".join(duplicated.unique())
        )

    missing_questionnaire = [
        column
        for column in QUESTIONNAIRE_COLUMNS
        if column not in labels.columns
    ]
    if missing_questionnaire:
        raise ValueError(
            "labels.csvにアンケート列がありません: "
            + ", ".join(missing_questionnaire)
        )
    return labels


def load_manual_exclusions(path: Path) -> pd.DataFrame:
    required = {"trial_id", "exclude", "reason", "notes"}
    if not path.is_file():
        raise FileNotFoundError(f"手動除外CSVが見つかりません: {path}")

    exclusions = pd.read_csv(path, dtype={"trial_id": str})
    missing = sorted(required - set(exclusions.columns))
    if missing:
        raise ValueError(
            "手動除外CSVに必要な列がありません: " + ", ".join(missing)
        )
    if exclusions["trial_id"].isna().any():
        raise ValueError("手動除外CSVのtrial_idに空欄があります")
    if exclusions["trial_id"].duplicated().any():
        raise ValueError("手動除外CSVのtrial_idが重複しています")

    truthy = {"1", "true", "yes", "y"}
    falsy = {"0", "false", "no", "n"}
    normalized = exclusions["exclude"].astype(str).str.strip().str.lower()
    invalid = ~normalized.isin(truthy | falsy)
    if invalid.any():
        values = exclusions.loc[invalid, "exclude"].astype(str).unique()
        raise ValueError(
            "手動除外CSVのexcludeは0/1またはtrue/falseにしてください: "
            + ", ".join(values)
        )
    exclusions = exclusions.assign(exclude=normalized.isin(truthy))
    active = exclusions.loc[exclusions["exclude"]].copy()
    if active["reason"].isna().any() or active["reason"].astype(str).str.strip().eq("").any():
        raise ValueError("有効な手動除外行のreasonに空欄があります")
    return active


def required_coordinate_columns() -> list[str]:
    columns: list[str] = []
    for landmark_id in REQUIRED_LANDMARKS:
        columns.extend(
            [f"lm{landmark_id}_x_local", f"lm{landmark_id}_y_local"]
        )
    return columns


def get_landmark_xy(df: pd.DataFrame, landmark_id: int) -> np.ndarray:
    """Return interpolated local XY coordinates for one landmark."""
    columns = [
        f"lm{landmark_id}_x_local",
        f"lm{landmark_id}_y_local",
    ]
    values = df[columns].apply(pd.to_numeric, errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan)
    values = values.interpolate(method="linear", limit_direction="both")
    array = values.to_numpy(dtype=np.float64)
    if not np.isfinite(array).all():
        raise TrialExclusion(
            "invalid_coordinates",
            f"landmark {landmark_id} has NaN/Inf after interpolation",
        )
    return array


def euclidean_distance(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Calculate frame-wise two-dimensional Euclidean distance."""
    return np.sqrt(np.sum((p1 - p2) ** 2, axis=-1))


def safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.full(np.shape(numerator), np.nan, dtype=np.float64)
    valid = np.isfinite(denominator) & (denominator > EPS)
    np.divide(numerator, denominator, out=result, where=valid)
    return result


def interpolate_signal(signal: np.ndarray, name: str) -> np.ndarray:
    series = pd.Series(signal, dtype=np.float64)
    series = series.replace([np.inf, -np.inf], np.nan)
    series = series.interpolate(method="linear", limit_direction="both")
    values = series.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise TrialExclusion(
            "invalid_signal", f"{name} has NaN/Inf after interpolation"
        )
    return values


def calculate_eye_signal(points: dict[int, np.ndarray]) -> np.ndarray:
    ear_a = safe_ratio(
        euclidean_distance(points[160], points[153])
        + euclidean_distance(points[158], points[144]),
        2.0 * euclidean_distance(points[33], points[133]),
    )
    ear_b = safe_ratio(
        euclidean_distance(points[385], points[373])
        + euclidean_distance(points[387], points[380]),
        2.0 * euclidean_distance(points[362], points[263]),
    )
    return interpolate_signal((ear_a + ear_b) / 2.0, "eye_open")


def detect_blinks(
    eye_open: np.ndarray,
) -> tuple[int, float, float, np.ndarray]:
    valid = np.isfinite(eye_open)
    if not valid.any():
        raise TrialExclusion("invalid_signal", "eye_open has no valid frame")

    threshold = float(np.median(eye_open[valid]) * BLINK_THRESHOLD_RATIO)
    closed = valid & (eye_open < threshold)

    blink_count = 0
    run_length = 0
    for is_closed in closed:
        if bool(is_closed):
            run_length += 1
        else:
            if run_length >= MIN_BLINK_FRAMES:
                blink_count += 1
            run_length = 0
    if run_length >= MIN_BLINK_FRAMES:
        blink_count += 1

    closed_ratio = float(closed.sum() / valid.sum())
    return blink_count, closed_ratio, threshold, closed


def calculate_mouth_open_signal(
    points: dict[int, np.ndarray],
) -> np.ndarray:
    height = euclidean_distance(points[13], points[14])
    width = euclidean_distance(points[61], points[291])
    return interpolate_signal(safe_ratio(height, width), "mouth_open")


def calculate_mouth_corner_signal(
    points: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    mouth_center_y = (points[13][:, 1] + points[14][:, 1]) / 2.0
    mouth_width = euclidean_distance(points[61], points[291])

    corner_a = safe_ratio(points[61][:, 1] - mouth_center_y, mouth_width)
    corner_b = safe_ratio(points[291][:, 1] - mouth_center_y, mouth_width)
    mouth_corner = interpolate_signal(
        (corner_a + corner_b) / 2.0, "mouth_corner"
    )
    asymmetry = interpolate_signal(
        np.abs(corner_a - corner_b), "mouth_corner_asymmetry"
    )
    return mouth_corner, asymmetry


def centroid(
    points: dict[int, np.ndarray], landmark_ids: Iterable[int]
) -> np.ndarray:
    return np.mean([points[landmark_id] for landmark_id in landmark_ids], axis=0)


def calculate_brow_eye_signal(
    points: dict[int, np.ndarray],
) -> np.ndarray:
    brow_a_center = centroid(points, BROW_A_POINTS)
    brow_b_center = centroid(points, BROW_B_POINTS)
    eye_a_center = centroid(points, EYE_A_POINTS)
    eye_b_center = centroid(points, EYE_B_POINTS)

    brow_eye_a = safe_ratio(
        euclidean_distance(brow_a_center, eye_a_center),
        euclidean_distance(points[33], points[133]),
    )
    brow_eye_b = safe_ratio(
        euclidean_distance(brow_b_center, eye_b_center),
        euclidean_distance(points[362], points[263]),
    )
    return interpolate_signal(
        (brow_eye_a + brow_eye_b) / 2.0, "brow_eye"
    )


def signal_std(signal: np.ndarray) -> float:
    """Population standard deviation across all frames in one trial."""
    return float(np.std(signal, ddof=0))


def signal_range(signal: np.ndarray) -> float:
    return float(np.max(signal) - np.min(signal))


def signal_motion(signal: np.ndarray) -> float:
    if len(signal) < 2:
        raise TrialExclusion("insufficient_frames", "signal has fewer than 2 frames")
    return float(np.mean(np.abs(np.diff(signal))))


def extract_trial_features(
    df: pd.DataFrame,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    missing = [
        column for column in required_coordinate_columns() if column not in df
    ]
    if missing:
        preview = ", ".join(missing[:8])
        suffix = " ..." if len(missing) > 8 else ""
        raise TrialExclusion(
            "missing_columns", f"missing columns: {preview}{suffix}"
        )

    points = {
        landmark_id: get_landmark_xy(df, landmark_id)
        for landmark_id in REQUIRED_LANDMARKS
    }

    eye_open = calculate_eye_signal(points)
    blink_count, closed_ratio, blink_threshold, is_closed = detect_blinks(
        eye_open
    )
    mouth_open = calculate_mouth_open_signal(points)
    mouth_corner, mouth_corner_asymmetry = (
        calculate_mouth_corner_signal(points)
    )
    brow_eye = calculate_brow_eye_signal(points)

    features: dict[str, float | int] = {
        "eye_open_mean": float(np.mean(eye_open)),
        "eye_open_std": signal_std(eye_open),
        "eye_open_min": float(np.min(eye_open)),
        "eye_open_range": signal_range(eye_open),
        "eye_motion_mean": signal_motion(eye_open),
        "blink_count": blink_count,
        "blink_closed_ratio": closed_ratio,
        "mouth_open_mean": float(np.mean(mouth_open)),
        "mouth_open_std": signal_std(mouth_open),
        "mouth_open_max": float(np.max(mouth_open)),
        "mouth_open_range": signal_range(mouth_open),
        "mouth_open_motion": signal_motion(mouth_open),
        "mouth_corner_mean": float(np.mean(mouth_corner)),
        "mouth_corner_std": signal_std(mouth_corner),
        "mouth_corner_range": signal_range(mouth_corner),
        "mouth_corner_motion": signal_motion(mouth_corner),
        "mouth_corner_asymmetry_mean": float(
            np.mean(mouth_corner_asymmetry)
        ),
        "brow_eye_mean": float(np.mean(brow_eye)),
        "brow_eye_std": signal_std(brow_eye),
        "brow_eye_range": signal_range(brow_eye),
        "brow_eye_motion": signal_motion(brow_eye),
    }
    validate_feature_values(features)

    debug_signals = pd.DataFrame(
        {
            "frame_index": df["frame_index"].to_numpy(),
            "eye_open": eye_open,
            "mouth_open": mouth_open,
            "mouth_corner": mouth_corner,
            "mouth_corner_asymmetry": mouth_corner_asymmetry,
            "brow_eye": brow_eye,
            "blink_threshold": blink_threshold,
            "is_closed": is_closed.astype(np.int8),
        }
    )
    return features, debug_signals


def validate_feature_values(features: dict[str, float | int]) -> None:
    if list(features) != FEATURE_COLUMNS:
        raise TrialExclusion(
            "invalid_features", "feature names/order do not match specification"
        )
    values = np.asarray([features[name] for name in FEATURE_COLUMNS], dtype=float)
    if not np.isfinite(values).all():
        raise TrialExclusion(
            "invalid_features", "21 features contain NaN/Inf"
        )


def resolve_normalized_path(
    row: pd.Series, normalized_dir: Path
) -> Path | None:
    candidates: list[Path] = []
    facemesh_csv = row.get("facemesh_csv")
    if pd.notna(facemesh_csv) and str(facemesh_csv).strip():
        candidates.append(normalized_dir / Path(str(facemesh_csv).strip()).name)
    candidates.append(normalized_dir / f"{str(row['trial_id']).strip()}.csv")

    for candidate in dict.fromkeys(candidates):
        if candidate.is_file():
            return candidate
    return None


def prepare_trial_dataframe(path: Path, trial_id: str) -> tuple[pd.DataFrame, int]:
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise TrialExclusion(
            "read_error", f"{type(exc).__name__}: {exc}"
        ) from exc

    num_frames = len(df)
    if "frame_index" not in df.columns:
        raise TrialExclusion(
            "missing_columns",
            "missing column: frame_index",
            num_frames=num_frames,
        )

    if "trial_id" in df.columns:
        csv_trial_ids = df["trial_id"].dropna().astype(str).unique()
        if len(csv_trial_ids) != 1 or csv_trial_ids[0] != trial_id:
            raise TrialExclusion(
                "trial_id_mismatch",
                f"CSV trial_id={csv_trial_ids.tolist()}, expected={trial_id}",
                num_frames=num_frames,
            )

    frame_index = pd.to_numeric(df["frame_index"], errors="coerce")
    if frame_index.isna().any():
        raise TrialExclusion(
            "invalid_frame_index",
            "frame_index contains NaN",
            num_frames=num_frames,
        )
    if frame_index.duplicated().any():
        raise TrialExclusion(
            "duplicate_frame_index",
            "frame_index contains duplicates",
            num_frames=num_frames,
        )

    df = df.assign(frame_index=frame_index).sort_values(
        "frame_index", kind="stable"
    )
    if num_frames < MIN_REQUIRED_FRAMES:
        raise TrialExclusion(
            "insufficient_frames",
            f"{num_frames} < {MIN_REQUIRED_FRAMES}",
            num_frames=num_frames,
        )
    return df.reset_index(drop=True), num_frames


def save_debug_plot(
    signals: pd.DataFrame, trial_id: str, debug_dir: Path
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "--save-debug-plotsにはmatplotlibが必要です"
        ) from exc

    figure, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    for axis, column in zip(
        axes, ["eye_open", "mouth_open", "mouth_corner", "brow_eye"]
    ):
        axis.plot(signals["frame_index"], signals[column])
        axis.set_ylabel(column)
        axis.grid(True, alpha=0.3)
    axes[-1].set_xlabel("frame_index")
    figure.suptitle(trial_id)
    figure.tight_layout()
    figure.savefig(debug_dir / f"{trial_id}_signals.png", dpi=150)
    plt.close(figure)


def validate_output(features_df: pd.DataFrame) -> None:
    errors: list[str] = []
    if list(features_df.columns) != OUTPUT_COLUMNS:
        errors.append("features.csvの列名または列順が仕様と一致しません")
    if features_df["trial_id"].duplicated().any():
        errors.append("trial_idが重複しています")
    participant = features_df["participant_id"]
    if participant.isna().any() or participant.astype(str).str.strip().eq("").any():
        errors.append("participant_idに空欄があります")

    labels = pd.to_numeric(features_df["drink_label"], errors="coerce")
    if labels.isna().any() or not set(labels.astype(int).unique()).issubset({0, 1}):
        errors.append("drink_labelに0/1以外の値があります")
    if len(FEATURE_COLUMNS) != 21:
        errors.append("顔特徴量が21列ではありません")

    numeric_features = features_df[FEATURE_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    )
    values = numeric_features.to_numpy(dtype=float)
    if np.isnan(values).any():
        errors.append("顔特徴量にNaNまたは非数値があります")
    if np.isinf(values).any():
        errors.append("顔特徴量にInfがあります")
    if errors:
        raise ValueError("出力検証エラー:\n- " + "\n- ".join(errors))


def print_sanity_checks(features_df: pd.DataFrame) -> None:
    print("\n===== Feature Sanity Check =====")
    summary = features_df[FEATURE_COLUMNS].describe().T[
        ["count", "mean", "std", "min", "max"]
    ]
    print(summary.to_string(float_format=lambda value: f"{value:.8g}"))

    constant = [
        column
        for column in FEATURE_COLUMNS
        if features_df[column].nunique(dropna=False) <= 1
    ]
    if constant:
        print("[WARNING] 全試行で同じ値の特徴量: " + ", ".join(constant))
    else:
        print("[ OK ] 全試行で一定の特徴量はありません")

    blink_distribution = (
        features_df["blink_count"].value_counts().sort_index()
    )
    print("\nblink_count distribution:")
    print(blink_distribution.to_string())
    if features_df["blink_count"].eq(0).all():
        print("[WARNING] 全試行でblink_countが0です")

    huge: list[str] = []
    relative_outliers: list[str] = []
    for column in FEATURE_COLUMNS:
        values = features_df[column].to_numpy(dtype=float)
        maximum = float(np.max(np.abs(values)))
        median = float(np.median(np.abs(values)))
        if maximum > 1e6:
            huge.append(f"{column}={maximum:.6g}")
        if median > EPS and maximum > 20.0 * median:
            row_index = int(np.argmax(np.abs(values)))
            trial_id = features_df.iloc[row_index]["trial_id"]
            relative_outliers.append(
                f"{column}: {trial_id}={values[row_index]:.6g} "
                f"(abs median={median:.6g})"
            )
    if huge:
        print("[WARNING] 1e6を超える絶対値: " + "; ".join(huge))
    else:
        print("[ OK ] 1e6を超える特徴量はありません")
    if relative_outliers:
        print("[WARNING] 絶対値中央値の20倍を超える試行:")
        for message in relative_outliers:
            print(f"  - {message}")
    else:
        print("[ OK ] 絶対値中央値の20倍を超える試行はありません")


def exclusion_row(
    row: pd.Series,
    reason: str,
    num_frames: int | None,
    file_path: Path,
    *,
    reason_detail: str = "",
    exclusion_source: str = "automatic_quality_check",
) -> dict[str, object]:
    return {
        "trial_id": str(row["trial_id"]),
        "participant_id": str(row["participant_id"]),
        "reason": reason,
        "reason_detail": reason_detail,
        "exclusion_source": exclusion_source,
        "num_frames": num_frames if num_frames is not None else "",
        "file_path": str(file_path),
    }


def main() -> int:
    args = parse_args()
    labels_path = project_path(args.labels)
    exclusions_path = project_path(args.exclusions)
    normalized_dir = project_path(args.normalized_dir)
    output_path = project_path(args.output)
    excluded_path = project_path(args.excluded_output)
    debug_dir = project_path(args.debug_dir)

    try:
        labels = load_labels(labels_path)
        manual_exclusions = load_manual_exclusions(exclusions_path)
    except Exception as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 2

    if not normalized_dir.is_dir():
        print(
            f"[FATAL] 正規化済みCSVフォルダがありません: {normalized_dir}",
            file=sys.stderr,
        )
        return 2

    output_rows: list[dict[str, object]] = []
    excluded_rows: list[dict[str, object]] = []
    label_trial_ids = set(labels["trial_id"].astype(str))
    unknown_exclusions = sorted(
        set(manual_exclusions["trial_id"].astype(str)) - label_trial_ids
    )
    if unknown_exclusions:
        print(
            "[FATAL] 手動除外CSVにlabels.csvにないtrial_idがあります: "
            + ", ".join(unknown_exclusions),
            file=sys.stderr,
        )
        return 2
    manual_by_trial = {
        str(row.trial_id): row
        for row in manual_exclusions.itertuples(index=False)
    }
    if args.save_debug_signals or args.save_debug_plots:
        debug_dir.mkdir(parents=True, exist_ok=True)

    for _, row in labels.iterrows():
        trial_id = str(row["trial_id"]).strip()
        path = resolve_normalized_path(row, normalized_dir)
        if trial_id in manual_by_trial:
            manual = manual_by_trial[trial_id]
            expected_path = path or normalized_dir / f"{trial_id}.csv"
            detail = "" if pd.isna(manual.notes) else str(manual.notes)
            manual_num_frames: int | None = None
            if path is not None:
                try:
                    manual_num_frames = len(
                        pd.read_csv(path, usecols=["frame_index"])
                    )
                except Exception:
                    # Manual exclusion remains authoritative even if its file
                    # cannot be inspected; the runtime report leaves the count blank.
                    manual_num_frames = None
            print(f"[SKIP] {trial_id}: manual exclusion ({manual.reason})")
            excluded_rows.append(
                exclusion_row(
                    row,
                    str(manual.reason),
                    manual_num_frames,
                    expected_path,
                    reason_detail=detail,
                    exclusion_source="manual_config",
                )
            )
            continue
        if path is None:
            expected_path = normalized_dir / f"{trial_id}.csv"
            print(f"WARNING: normalized FaceMesh file not found: {expected_path}")
            excluded_rows.append(
                exclusion_row(
                    row,
                    "file_not_found",
                    None,
                    expected_path,
                    exclusion_source="automatic_quality_check",
                )
            )
            continue

        num_frames: int | None = None
        try:
            trial_df, num_frames = prepare_trial_dataframe(path, trial_id)
            features, debug_signals = extract_trial_features(trial_df)

            # Questionnaire columns are metadata only.  They must not be used
            # as classification-model inputs in a later machine-learning step.
            output_row: dict[str, object] = {
                "trial_id": trial_id,
                "participant_id": row["participant_id"],
                "drink_name": row["drink_name"],
                "drink_label": row["drink_label"],
                **features,
                **{column: row[column] for column in QUESTIONNAIRE_COLUMNS},
            }
            output_rows.append(output_row)

            if args.save_debug_signals:
                debug_signals.to_csv(
                    debug_dir / f"{trial_id}_signals.csv",
                    index=False,
                    encoding="utf-8-sig",
                )
            if args.save_debug_plots:
                save_debug_plot(debug_signals, trial_id, debug_dir)
            print(f"[ OK ] {trial_id}: {len(trial_df)} frames, 21 features")
        except TrialExclusion as exc:
            if num_frames is None:
                num_frames = exc.num_frames
            detail = f" ({exc.detail})" if exc.detail else ""
            print(f"[WARN] {trial_id}: {exc.reason}{detail}")
            excluded_rows.append(
                exclusion_row(
                    row,
                    exc.reason,
                    num_frames,
                    path,
                    reason_detail=exc.detail,
                    exclusion_source="automatic_quality_check",
                )
            )
        except Exception as exc:
            print(f"[WARN] {trial_id}: processing_error ({exc})")
            excluded_rows.append(
                exclusion_row(
                    row,
                    "processing_error",
                    num_frames,
                    path,
                    reason_detail=f"{type(exc).__name__}: {exc}",
                    exclusion_source="processing_error",
                )
            )

    features_df = pd.DataFrame(output_rows, columns=OUTPUT_COLUMNS)
    excluded_df = pd.DataFrame(excluded_rows, columns=EXCLUDED_COLUMNS)

    if features_df.empty:
        print("[FATAL] 処理できた試行がありません", file=sys.stderr)
        return 1

    try:
        validate_output(features_df)
    except Exception as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    excluded_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    excluded_df.to_csv(excluded_path, index=False, encoding="utf-8-sig")

    print_sanity_checks(features_df)

    water_count = int((features_df["drink_label"] == 0).sum())
    sparkling_count = int((features_df["drink_label"] == 1).sum())
    participant_counts = features_df["participant_id"].value_counts().sort_index()

    print("\n===== Handcrafted Feature Extraction =====")
    print(f"labels trials     : {len(labels)}")
    print(f"processed trials  : {len(features_df)}")
    print(f"excluded trials   : {len(excluded_df)}")
    if not excluded_df.empty:
        source_counts = excluded_df["exclusion_source"].value_counts()
        for source, count in source_counts.items():
            print(f"  {source}: {count}")
    print()
    print(f"water trials      : {water_count}")
    print(f"sparkling trials  : {sparkling_count}")
    print()
    print(f"participants      : {features_df['participant_id'].nunique()}")
    print(f"facial features   : {len(FEATURE_COLUMNS)}")
    print("participant trials:")
    for participant_id, count in participant_counts.items():
        print(f"  {participant_id}: {count}")
    print()
    print(f"output            : {output_path}")
    print(f"excluded          : {excluded_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
