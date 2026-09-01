#!/usr/bin/env python3
"""Train and evaluate handcrafted-feature classifiers with participant LOSO.

Only the 21 columns declared in ``feature_schema.py`` are used as model
inputs. Questionnaire values and identifiers are metadata and can never enter
the estimator through this module's data-loading path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut

try:
    from .feature_schema import FACIAL_FEATURE_COLUMNS, QUESTIONNAIRE_COLUMNS
except ImportError:  # Direct execution: python scripts/train_loso_classifier.py
    from feature_schema import FACIAL_FEATURE_COLUMNS, QUESTIONNAIRE_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = Path("config/loso_experiment.json")
DEFAULT_RUN_NAME = "initial_loso_v1"
METRIC_COLUMNS = [
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "macro_f1",
    "roc_auc",
]


@dataclass(frozen=True)
class LosoFold:
    fold: int
    test_participant: str
    train_indices: np.ndarray
    test_indices: np.ndarray


@dataclass
class ModelResult:
    predictions: pd.DataFrame
    fold_metrics: pd.DataFrame
    importances: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "21個の手作り顔特徴量をDummy/ExtraTrees/Random Forestで"
            "被験者LOSO評価します。"
        )
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG_PATH
    )
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="同名runフォルダ内の既知の出力ファイルを上書きする",
    )
    return parser.parse_args()


def project_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"LOSO設定ファイルがありません: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    required = {
        "input_path",
        "excluded_trials_path",
        "output_root",
        "positive_label",
        "group_column",
        "target_column",
        "random_seed",
        "primary_metric",
        "models",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError("LOSO設定に必要な項目がありません: " + ", ".join(missing))
    if config["positive_label"] != 1:
        raise ValueError("現在の二値分類ではpositive_labelを1にしてください")
    if config["primary_metric"] not in METRIC_COLUMNS:
        raise ValueError(
            "未対応のprimary_metricです: " + str(config["primary_metric"])
        )
    if not isinstance(config["models"], dict) or not config["models"]:
        raise ValueError("models設定が空です")
    return config


def parse_exclude_flag(value: object) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"不正なexclude値です: {value}")


def load_active_exclusions(path: Path) -> pd.DataFrame:
    columns = ["trial_id", "exclude", "reason", "notes"]
    if not path.is_file():
        raise FileNotFoundError(f"手動除外CSVがありません: {path}")
    exclusions = pd.read_csv(path, dtype={"trial_id": str})
    missing = [column for column in columns if column not in exclusions]
    if missing:
        raise ValueError(
            "手動除外CSVに必要な列がありません: " + ", ".join(missing)
        )
    if exclusions["trial_id"].duplicated().any():
        raise ValueError("手動除外CSVのtrial_idが重複しています")
    active_mask = exclusions["exclude"].map(parse_exclude_flag)
    return exclusions.loc[active_mask, columns].reset_index(drop=True)


def load_and_validate_features(
    path: Path,
    *,
    group_column: str,
    target_column: str,
    positive_label: int,
    active_exclusions: pd.DataFrame,
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"features.csvがありません: {path}")
    df = pd.read_csv(path)
    required = {
        "trial_id",
        "participant_id",
        "drink_name",
        group_column,
        target_column,
        *FACIAL_FEATURE_COLUMNS,
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError("features.csvに必要な列がありません: " + ", ".join(missing))
    if df.empty:
        raise ValueError("features.csvに試行がありません")
    if df["trial_id"].isna().any() or df["trial_id"].astype(str).str.strip().eq("").any():
        raise ValueError("trial_idに空欄があります")
    if df["trial_id"].duplicated().any():
        duplicates = df.loc[df["trial_id"].duplicated(False), "trial_id"].unique()
        raise ValueError("trial_idが重複しています: " + ", ".join(map(str, duplicates)))
    groups = df[group_column]
    if groups.isna().any() or groups.astype(str).str.strip().eq("").any():
        raise ValueError(f"{group_column}に空欄があります")

    labels = pd.to_numeric(df[target_column], errors="coerce")
    if labels.isna().any() or not set(labels.astype(int).unique()).issubset({0, positive_label}):
        raise ValueError(f"{target_column}は0/{positive_label}だけにしてください")
    df[target_column] = labels.astype(int)
    if set(df[target_column].unique()) != {0, positive_label}:
        raise ValueError("データセット全体に水と炭酸水の両クラスが必要です")

    numeric = df[FACIAL_FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        bad = np.argwhere(~np.isfinite(values))[0]
        raise ValueError(
            "顔特徴量にNaN/Inf/非数値があります: "
            f"trial={df.iloc[int(bad[0])]['trial_id']}, "
            f"feature={FACIAL_FEATURE_COLUMNS[int(bad[1])]}"
        )
    df[FACIAL_FEATURE_COLUMNS] = numeric

    excluded_ids = set(active_exclusions["trial_id"].astype(str))
    overlap = sorted(excluded_ids & set(df["trial_id"].astype(str)))
    if overlap:
        raise ValueError(
            "手動除外trialがfeatures.csvへ混入しています: " + ", ".join(overlap)
        )

    if len(FACIAL_FEATURE_COLUMNS) != 21:
        raise ValueError("共通schemaの顔特徴量が21列ではありません")
    if set(FACIAL_FEATURE_COLUMNS) & set(QUESTIONNAIRE_COLUMNS):
        raise ValueError("顔特徴量schemaにアンケート項目が混入しています")
    if df[group_column].nunique() < 2:
        raise ValueError("LOSOには2人以上の被験者が必要です")
    for participant, group in df.groupby(group_column):
        if set(group[target_column].unique()) != {0, positive_label}:
            raise ValueError(
                f"test被験者{participant}に水・炭酸水の両クラスがありません"
            )
    return df.reset_index(drop=True)


def build_loso_folds(
    df: pd.DataFrame,
    *,
    group_column: str,
    target_column: str,
) -> list[LosoFold]:
    x = df[FACIAL_FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    y = df[target_column].to_numpy(dtype=int)
    groups = df[group_column].astype(str).to_numpy()
    splitter = LeaveOneGroupOut()
    folds: list[LosoFold] = []
    seen_test_indices: list[int] = []
    for fold_number, (train_indices, test_indices) in enumerate(
        splitter.split(x, y, groups), start=1
    ):
        test_groups = np.unique(groups[test_indices])
        train_groups = set(groups[train_indices])
        if len(test_groups) != 1:
            raise RuntimeError("LOSO foldのtest被験者が1人ではありません")
        if str(test_groups[0]) in train_groups:
            raise RuntimeError("同一被験者がtrainとtestに混入しています")
        folds.append(
            LosoFold(
                fold=fold_number,
                test_participant=str(test_groups[0]),
                train_indices=train_indices,
                test_indices=test_indices,
            )
        )
        seen_test_indices.extend(test_indices.tolist())
    if sorted(seen_test_indices) != list(range(len(df))):
        raise RuntimeError("全試行がちょうど1回ずつtestになっていません")
    if len(folds) != df[group_column].nunique():
        raise RuntimeError("fold数と被験者数が一致しません")
    return folds


def build_models(config: dict[str, Any]) -> dict[str, Any]:
    constructors = {
        "dummy": DummyClassifier,
        "extra_trees": ExtraTreesClassifier,
        "random_forest": RandomForestClassifier,
    }
    unknown = sorted(set(config["models"]) - set(constructors))
    if unknown:
        raise ValueError("未対応モデルです: " + ", ".join(unknown))
    return {
        name: constructors[name](**parameters)
        for name, parameters in config["models"].items()
    }


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probability_positive: np.ndarray,
    *,
    positive_label: int,
) -> dict[str, float | int]:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, positive_label])
    tn, fp, fn, tp = matrix.ravel()
    roc_auc = (
        float(roc_auc_score(y_true, probability_positive))
        if len(np.unique(y_true)) == 2
        else float("nan")
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, pos_label=positive_label, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, pos_label=positive_label, zero_division=0)
        ),
        "f1": float(f1_score(y_true, y_pred, pos_label=positive_label, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "roc_auc": roc_auc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def run_model_loso(
    df: pd.DataFrame,
    folds: list[LosoFold],
    estimator: Any,
    *,
    model_name: str,
    group_column: str,
    target_column: str,
    positive_label: int,
) -> ModelResult:
    x = df[FACIAL_FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    y = df[target_column].to_numpy(dtype=int)
    prediction_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []

    for fold in folds:
        train_groups = set(df.iloc[fold.train_indices][group_column].astype(str))
        test_groups = set(df.iloc[fold.test_indices][group_column].astype(str))
        if train_groups & test_groups:
            raise RuntimeError(
                f"fold {fold.fold}でtrain/test被験者リークを検出しました"
            )

        model = clone(estimator)
        model.fit(x[fold.train_indices], y[fold.train_indices])
        y_pred = model.predict(x[fold.test_indices]).astype(int)
        probabilities = model.predict_proba(x[fold.test_indices])
        class_to_index = {int(label): index for index, label in enumerate(model.classes_)}
        if 0 not in class_to_index or positive_label not in class_to_index:
            raise RuntimeError(f"fold {fold.fold}の学習データに両クラスがありません")
        probability_water = probabilities[:, class_to_index[0]]
        probability_positive = probabilities[:, class_to_index[positive_label]]
        if not np.allclose(probability_water + probability_positive, 1.0):
            raise RuntimeError("二値クラス確率の合計が1ではありません")

        test_df = df.iloc[fold.test_indices]
        for position, (_, row) in enumerate(test_df.iterrows()):
            prediction_rows.append(
                {
                    "model": model_name,
                    "fold": fold.fold,
                    "test_participant": fold.test_participant,
                    "trial_id": row["trial_id"],
                    "participant_id": row[group_column],
                    "drink_name": row["drink_name"],
                    "y_true": int(y[fold.test_indices][position]),
                    "y_pred": int(y_pred[position]),
                    "probability_water": float(probability_water[position]),
                    "probability_sparkling": float(probability_positive[position]),
                    "correct": int(y_pred[position] == y[fold.test_indices][position]),
                }
            )

        fold_metric = calculate_metrics(
            y[fold.test_indices],
            y_pred,
            probability_positive,
            positive_label=positive_label,
        )
        train_labels = pd.Series(y[fold.train_indices]).value_counts()
        test_labels = pd.Series(y[fold.test_indices]).value_counts()
        fold_rows.append(
            {
                "model": model_name,
                "fold": fold.fold,
                "test_participant": fold.test_participant,
                "train_trials": len(fold.train_indices),
                "test_trials": len(fold.test_indices),
                "train_water": int(train_labels.get(0, 0)),
                "train_sparkling": int(train_labels.get(positive_label, 0)),
                "test_water": int(test_labels.get(0, 0)),
                "test_sparkling": int(test_labels.get(positive_label, 0)),
                **fold_metric,
            }
        )

        if hasattr(model, "feature_importances_"):
            for feature, importance in zip(
                FACIAL_FEATURE_COLUMNS, model.feature_importances_
            ):
                importance_rows.append(
                    {
                        "model": model_name,
                        "fold": fold.fold,
                        "test_participant": fold.test_participant,
                        "feature": feature,
                        "importance": float(importance),
                    }
                )

    predictions = pd.DataFrame(prediction_rows)
    if len(predictions) != len(df):
        raise RuntimeError(f"{model_name}: 予測試行数が入力試行数と一致しません")
    if predictions["trial_id"].duplicated().any():
        raise RuntimeError(f"{model_name}: trial_idの予測が重複しています")
    if set(predictions["trial_id"]) != set(df["trial_id"]):
        raise RuntimeError(f"{model_name}: 予測されていないtrialがあります")
    return ModelResult(
        predictions=predictions.sort_values(["fold", "trial_id"]).reset_index(drop=True),
        fold_metrics=pd.DataFrame(fold_rows).sort_values("fold").reset_index(drop=True),
        importances=pd.DataFrame(importance_rows),
    )


def summarize_results(
    results: dict[str, ModelResult], *, positive_label: int, primary_metric: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    for model_name, result in results.items():
        y_true = result.predictions["y_true"].to_numpy(dtype=int)
        y_pred = result.predictions["y_pred"].to_numpy(dtype=int)
        probability = result.predictions["probability_sparkling"].to_numpy(float)
        pooled = calculate_metrics(
            y_true, y_pred, probability, positive_label=positive_label
        )
        row: dict[str, Any] = {"model": model_name, "folds": len(result.fold_metrics)}
        for metric in METRIC_COLUMNS:
            values = result.fold_metrics[metric].astype(float)
            row[f"fold_mean_{metric}"] = float(values.mean())
            row[f"fold_std_{metric}"] = float(values.std(ddof=1))
            row[f"fold_min_{metric}"] = float(values.min())
            row[f"fold_max_{metric}"] = float(values.max())
            row[f"pooled_{metric}"] = float(pooled[metric])
        row.update(
            {
                "pooled_tn": pooled["tn"],
                "pooled_fp": pooled["fp"],
                "pooled_fn": pooled["fn"],
                "pooled_tp": pooled["tp"],
            }
        )
        summary_rows.append(row)
        comparison_rows.append(
            {
                "model": model_name,
                "primary_metric": primary_metric,
                "fold_mean_primary": row[f"fold_mean_{primary_metric}"],
                "fold_std_primary": row[f"fold_std_{primary_metric}"],
                "fold_mean_accuracy": row["fold_mean_accuracy"],
                "fold_mean_f1": row["fold_mean_f1"],
                "fold_mean_roc_auc": row["fold_mean_roc_auc"],
                "pooled_balanced_accuracy": row["pooled_balanced_accuracy"],
                "pooled_accuracy": row["pooled_accuracy"],
                "pooled_f1": row["pooled_f1"],
                "pooled_roc_auc": row["pooled_roc_auc"],
            }
        )
    comparison = pd.DataFrame(comparison_rows).sort_values(
        "fold_mean_primary", ascending=False
    )
    return pd.DataFrame(summary_rows), comparison.reset_index(drop=True)


def dataframe_to_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def save_confusion_outputs(
    predictions: pd.DataFrame,
    *,
    model_name: str,
    output_dir: Path,
    positive_label: int,
) -> None:
    matrix = confusion_matrix(
        predictions["y_true"], predictions["y_pred"], labels=[0, positive_label]
    )
    matrix_df = pd.DataFrame(
        matrix,
        index=["actual_water", "actual_sparkling"],
        columns=["predicted_water", "predicted_sparkling"],
    )
    matrix_df.to_csv(
        output_dir / f"confusion_matrix_{model_name}.csv",
        encoding="utf-8-sig",
    )
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["water", "sparkling_water"],
    )
    display.plot(cmap="Blues", colorbar=False)
    plt.title(f"Pooled LOSO Confusion Matrix: {model_name}")
    plt.tight_layout()
    plt.savefig(
        output_dir / f"confusion_matrix_{model_name}.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close()


def save_feature_importances(
    importances: pd.DataFrame, *, model_name: str, output_dir: Path
) -> None:
    if importances.empty:
        return
    dataframe_to_csv(
        importances.sort_values(["fold", "importance"], ascending=[True, False]),
        output_dir / f"feature_importance_by_fold_{model_name}.csv",
    )
    summary = (
        importances.groupby("feature", as_index=False)["importance"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
        .rename(
            columns={
                "mean": "mean_importance",
                "std": "std_importance",
                "min": "min_importance",
                "max": "max_importance",
            }
        )
        .sort_values("mean_importance", ascending=False)
    )
    dataframe_to_csv(
        summary, output_dir / f"feature_importance_summary_{model_name}.csv"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_worktree_dirty() -> bool | None:
    """Return whether tracked/untracked worktree changes exist, or None outside Git."""
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return bool(completed.stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def prepare_output_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"runフォルダが既に存在します: {path}。別の--run-nameを使うか"
            "、意図的な再生成では--overwriteを指定してください"
        )
    path.mkdir(parents=True, exist_ok=True)


def save_dataset_audit(
    df: pd.DataFrame,
    folds: list[LosoFold],
    *,
    group_column: str,
    target_column: str,
    output_dir: Path,
) -> None:
    audit = (
        df.groupby(group_column)[target_column]
        .value_counts()
        .unstack(fill_value=0)
        .rename(columns={0: "water", 1: "sparkling"})
        .reset_index()
    )
    audit["total"] = audit.get("water", 0) + audit.get("sparkling", 0)
    dataframe_to_csv(audit, output_dir / "dataset_audit.csv")
    assignments: list[dict[str, Any]] = []
    for fold in folds:
        for index in fold.test_indices:
            row = df.iloc[int(index)]
            assignments.append(
                {
                    "fold": fold.fold,
                    "test_participant": fold.test_participant,
                    "trial_id": row["trial_id"],
                    "participant_id": row[group_column],
                    "drink_label": int(row[target_column]),
                    "role": "test",
                }
            )
    dataframe_to_csv(
        pd.DataFrame(assignments).sort_values(["fold", "trial_id"]),
        output_dir / "fold_assignments.csv",
    )


def run_experiment(
    config: dict[str, Any], *, run_name: str, overwrite: bool
) -> tuple[Path, pd.DataFrame, dict[str, ModelResult]]:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_name):
        raise ValueError(
            "--run-nameは英数字、ピリオド、アンダースコア、ハイフンだけにしてください"
        )
    features_path = project_path(config["input_path"])
    exclusions_path = project_path(config["excluded_trials_path"])
    output_dir = project_path(config["output_root"]) / run_name
    active_exclusions = load_active_exclusions(exclusions_path)
    df = load_and_validate_features(
        features_path,
        group_column=config["group_column"],
        target_column=config["target_column"],
        positive_label=int(config["positive_label"]),
        active_exclusions=active_exclusions,
    )
    folds = build_loso_folds(
        df,
        group_column=config["group_column"],
        target_column=config["target_column"],
    )
    models = build_models(config)
    prepare_output_dir(output_dir, overwrite=overwrite)
    save_dataset_audit(
        df,
        folds,
        group_column=config["group_column"],
        target_column=config["target_column"],
        output_dir=output_dir,
    )

    results: dict[str, ModelResult] = {}
    for model_name, estimator in models.items():
        print(f"\n===== {model_name} =====")
        result = run_model_loso(
            df,
            folds,
            estimator,
            model_name=model_name,
            group_column=config["group_column"],
            target_column=config["target_column"],
            positive_label=int(config["positive_label"]),
        )
        results[model_name] = result
        dataframe_to_csv(
            result.predictions, output_dir / f"predictions_{model_name}.csv"
        )
        dataframe_to_csv(
            result.fold_metrics, output_dir / f"fold_metrics_{model_name}.csv"
        )
        save_confusion_outputs(
            result.predictions,
            model_name=model_name,
            output_dir=output_dir,
            positive_label=int(config["positive_label"]),
        )
        save_feature_importances(
            result.importances, model_name=model_name, output_dir=output_dir
        )
        for row in result.fold_metrics.itertuples(index=False):
            print(
                f"test={row.test_participant} n={row.test_trials} "
                f"accuracy={row.accuracy:.3f} "
                f"balanced_accuracy={row.balanced_accuracy:.3f} "
                f"f1={row.f1:.3f} auc={row.roc_auc:.3f}"
            )

    summary, comparison = summarize_results(
        results,
        positive_label=int(config["positive_label"]),
        primary_metric=config["primary_metric"],
    )
    dataframe_to_csv(summary, output_dir / "summary_metrics.csv")
    dataframe_to_csv(comparison, output_dir / "model_comparison.csv")

    manifest = {
        "run_name": run_name,
        "executed_at": datetime.now().astimezone().isoformat(),
        "git_commit": git_commit(),
        "git_worktree_dirty": git_worktree_dirty(),
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "features_csv": str(features_path),
        "features_csv_sha256": sha256_file(features_path),
        "trials": len(df),
        "participants": int(df[config["group_column"]].nunique()),
        "facial_feature_count": len(FACIAL_FEATURE_COLUMNS),
        "facial_features": FACIAL_FEATURE_COLUMNS,
        "questionnaire_columns_excluded_from_model": QUESTIONNAIRE_COLUMNS,
        "active_manual_exclusions": active_exclusions.to_dict("records"),
        "folds": [
            {
                "fold": fold.fold,
                "test_participant": fold.test_participant,
                "train_trials": len(fold.train_indices),
                "test_trials": len(fold.test_indices),
            }
            for fold in folds
        ],
        "config": config,
    }
    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(json_ready(manifest), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    with (output_dir / "config_snapshot.json").open("w", encoding="utf-8") as handle:
        json.dump(json_ready(config), handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print("\n===== LOSO Model Comparison =====")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\noutput: {output_dir}")
    return output_dir, comparison, results


def main() -> int:
    args = parse_args()
    try:
        config = load_config(project_path(args.config))
        run_experiment(config, run_name=args.run_name, overwrite=args.overwrite)
    except Exception as exc:
        print(f"[FATAL] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
