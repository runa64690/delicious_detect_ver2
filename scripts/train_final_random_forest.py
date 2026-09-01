#!/usr/bin/env python3
"""Fit the deployment Random Forest on every validated trial and save it.

The final fit is for inference on future trials. Its training data must not be
used to estimate generalization performance; report the separate LOSO results.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier

try:
    from .feature_schema import FACIAL_FEATURE_COLUMNS, QUESTIONNAIRE_COLUMNS
    from .train_loso_classifier import (
        build_models,
        git_commit,
        git_worktree_dirty,
        json_ready,
        load_active_exclusions,
        load_and_validate_features,
        load_config,
        project_path,
        sha256_file,
    )
except ImportError:  # Direct execution: python scripts/train_final_random_forest.py
    from feature_schema import FACIAL_FEATURE_COLUMNS, QUESTIONNAIRE_COLUMNS
    from train_loso_classifier import (
        build_models,
        git_commit,
        git_worktree_dirty,
        json_ready,
        load_active_exclusions,
        load_and_validate_features,
        load_config,
        project_path,
        sha256_file,
    )


DEFAULT_CONFIG_PATH = Path("config/loso_experiment.json")
DEFAULT_OUTPUT_ROOT = Path("results/models")
DEFAULT_MODEL_ID = "random_forest_v1"
MODEL_FILENAME = "model.joblib"
METADATA_FILENAME = "metadata.json"
SUMMARY_FILENAME = "training_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "検証済みの全試行と21個の顔特徴量で最終Random Forestを学習し、"
            "推論用モデルと再現情報を保存します。"
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="同じmodel-idの既知の成果物を上書きする",
    )
    return parser.parse_args()


def prepare_model_dir(path: Path, *, overwrite: bool) -> None:
    known_outputs = [MODEL_FILENAME, METADATA_FILENAME, SUMMARY_FILENAME]
    existing = [name for name in known_outputs if (path / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"モデル成果物がすでにあります: {path}。別の--model-idを使うか、"
            "意図した再学習なら--overwriteを指定してください"
        )
    path.mkdir(parents=True, exist_ok=True)


def fit_final_random_forest(
    df: pd.DataFrame, config: dict[str, Any]
) -> RandomForestClassifier:
    models = build_models(config)
    if "random_forest" not in models:
        raise ValueError("設定にrandom_forestモデルがありません")
    model = models["random_forest"]
    if not isinstance(model, RandomForestClassifier):
        raise TypeError("random_forest設定からRandomForestClassifierを生成できません")

    x = df.loc[:, FACIAL_FEATURE_COLUMNS].copy()
    y = df[config["target_column"]].to_numpy(dtype=int)
    model.fit(x, y)

    if list(model.feature_names_in_) != FACIAL_FEATURE_COLUMNS:
        raise RuntimeError("学習済みモデルの特徴量名・順序が仕様と一致しません")
    if list(model.classes_) != [0, int(config["positive_label"])]:
        raise RuntimeError(f"学習済みモデルのクラスが不正です: {model.classes_}")
    return model


def save_and_verify_model(
    model: RandomForestClassifier,
    model_path: Path,
    x: pd.DataFrame,
) -> RandomForestClassifier:
    expected_prediction = model.predict(x)
    expected_probability = model.predict_proba(x)
    joblib.dump(model, model_path, compress=3)

    loaded = joblib.load(model_path)
    if not isinstance(loaded, RandomForestClassifier):
        raise RuntimeError("保存後に読み込んだモデルの型がRandom Forestではありません")
    if list(loaded.feature_names_in_) != FACIAL_FEATURE_COLUMNS:
        raise RuntimeError("再読込したモデルの特徴量名・順序が一致しません")
    np.testing.assert_array_equal(loaded.predict(x), expected_prediction)
    np.testing.assert_allclose(
        loaded.predict_proba(x), expected_probability, rtol=0, atol=1e-15
    )
    return loaded


def participant_counts(
    df: pd.DataFrame, *, group_column: str, target_column: str
) -> list[dict[str, Any]]:
    counts = (
        df.groupby(group_column)[target_column]
        .value_counts()
        .unstack(fill_value=0)
        .rename(columns={0: "water", 1: "sparkling"})
        .reset_index()
    )
    counts["total"] = counts.get("water", 0) + counts.get("sparkling", 0)
    return counts.to_dict("records")


def train_and_save_final_model(
    config: dict[str, Any],
    *,
    config_path: Path,
    output_root: Path,
    model_id: str,
    overwrite: bool,
) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", model_id):
        raise ValueError(
            "--model-idは英数字、ピリオド、アンダースコア、ハイフンだけにしてください"
        )

    features_path = project_path(config["input_path"])
    exclusions_path = project_path(config["excluded_trials_path"])
    output_dir = project_path(output_root) / model_id
    prepare_model_dir(output_dir, overwrite=overwrite)

    active_exclusions = load_active_exclusions(exclusions_path)
    df = load_and_validate_features(
        features_path,
        group_column=config["group_column"],
        target_column=config["target_column"],
        positive_label=int(config["positive_label"]),
        active_exclusions=active_exclusions,
    )
    x = df.loc[:, FACIAL_FEATURE_COLUMNS].copy()
    model = fit_final_random_forest(df, config)
    model_path = output_dir / MODEL_FILENAME
    loaded_model = save_and_verify_model(model, model_path, x)

    trained_at = datetime.now().astimezone().isoformat()
    feature_hash = sha256_file(features_path)
    class_counts = df[config["target_column"]].value_counts().sort_index()
    estimator_parameters = loaded_model.get_params(deep=False)

    metadata = {
        "artifact_schema_version": 1,
        "model_id": model_id,
        "purpose": "future_trial_inference",
        "trained_at": trained_at,
        "model_type": type(loaded_model).__name__,
        "model_file": MODEL_FILENAME,
        "model_file_sha256": sha256_file(model_path),
        "classes": [
            {"value": 0, "name": "water"},
            {"value": int(config["positive_label"]), "name": "sparkling_water"},
        ],
        "feature_count": len(FACIAL_FEATURE_COLUMNS),
        "feature_columns_in_order": FACIAL_FEATURE_COLUMNS,
        "questionnaire_columns_excluded": QUESTIONNAIRE_COLUMNS,
        "input_contract": {
            "rows_per_prediction": 1,
            "requires_finite_numeric_features": True,
            "normalization": (
                "nose landmark 168 centering, eye horizontal rotation, "
                "inter-eye-distance scale, y-axis inversion"
            ),
            "feature_window": "one complete variable-frame trial",
        },
        "estimator_parameters": estimator_parameters,
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "git_commit": git_commit(),
        "git_worktree_dirty": git_worktree_dirty(),
    }
    training_summary = {
        "model_id": model_id,
        "training_data": {
            "features_csv": str(features_path),
            "features_csv_sha256": feature_hash,
            "config_file": str(config_path),
            "config_file_sha256": sha256_file(config_path),
            "excluded_trials_file": str(exclusions_path),
            "excluded_trials_file_sha256": sha256_file(exclusions_path),
            "trials": len(df),
            "participants": int(df[config["group_column"]].nunique()),
            "water_trials": int(class_counts.get(0, 0)),
            "sparkling_trials": int(
                class_counts.get(int(config["positive_label"]), 0)
            ),
            "participant_counts": participant_counts(
                df,
                group_column=config["group_column"],
                target_column=config["target_column"],
            ),
            "active_manual_exclusions": active_exclusions.to_dict("records"),
        },
        "artifact_verification": {
            "joblib_reload_succeeded": True,
            "predictions_identical_after_reload": True,
            "probabilities_match_after_reload_atol": 1e-15,
            "feature_names_and_order_verified": True,
            "classes_verified": True,
        },
        "evaluation_note": (
            "This final model was fit on all validated trials for future inference. "
            "Do not use training-set predictions as a performance estimate; report "
            "the separately saved participant-LOSO results."
        ),
    }

    for filename, payload in (
        (METADATA_FILENAME, metadata),
        (SUMMARY_FILENAME, training_summary),
    ):
        with (output_dir / filename).open("w", encoding="utf-8") as handle:
            json.dump(json_ready(payload), handle, ensure_ascii=False, indent=2)
            handle.write("\n")

    print("===== Final Random Forest =====")
    print(f"model_id          : {model_id}")
    print(f"training trials   : {len(df)}")
    print(f"participants      : {df[config['group_column']].nunique()}")
    print(f"water/sparkling   : {int(class_counts.get(0, 0))}/{int(class_counts.get(1, 0))}")
    print(f"features          : {len(FACIAL_FEATURE_COLUMNS)}")
    print(f"trees             : {loaded_model.n_estimators}")
    print(f"reload verification: OK")
    print(f"output            : {output_dir}")
    print("evaluation        : use the separate LOSO results")
    return output_dir


def main() -> int:
    args = parse_args()
    try:
        config_path = project_path(args.config)
        config = load_config(config_path)
        train_and_save_final_model(
            config,
            config_path=config_path,
            output_root=args.output_root,
            model_id=args.model_id,
            overwrite=args.overwrite,
        )
    except Exception as exc:
        print(f"[FATAL] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
