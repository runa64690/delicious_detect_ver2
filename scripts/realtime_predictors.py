"""Model adapters used by real-time and offline trial inference.

The camera application passes one normalized FaceMesh trial to this interface.
The current adapter aggregates it into handcrafted features; a future sequence
model can consume the same normalized landmark time series directly.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

try:
    from .extract_handcrafted_features import extract_trial_features
    from .feature_schema import FACIAL_FEATURE_COLUMNS
    from .train_loso_classifier import sha256_file
except ImportError:  # Direct import from a root-level application
    from extract_handcrafted_features import extract_trial_features
    from feature_schema import FACIAL_FEATURE_COLUMNS
    from train_loso_classifier import sha256_file


@dataclass(frozen=True)
class TrialPrediction:
    model_id: str
    input_type: str
    predicted_label: int
    predicted_name: str
    probabilities: dict[str, float]
    model_inputs: dict[str, float] = field(default_factory=dict)


class TrialPredictor(ABC):
    """Interface between trial acquisition and a replaceable model backend."""

    model_id: str
    input_type: str

    @abstractmethod
    def predict_normalized(self, normalized_trial: pd.DataFrame) -> TrialPrediction:
        """Predict one complete normalized FaceMesh trial."""


class HandcraftedRandomForestPredictor(TrialPredictor):
    input_type = "handcrafted_features"

    def __init__(
        self,
        model_path: Path,
        metadata_path: Path,
        *,
        n_jobs: int | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"学習済みモデルがありません: {self.model_path}")
        if not self.metadata_path.is_file():
            raise FileNotFoundError(f"モデルmetadataがありません: {self.metadata_path}")

        with self.metadata_path.open("r", encoding="utf-8") as handle:
            self.metadata: dict[str, Any] = json.load(handle)
        self._validate_metadata()

        expected_hash = str(self.metadata["model_file_sha256"]).lower()
        actual_hash = sha256_file(self.model_path).lower()
        if actual_hash != expected_hash:
            raise ValueError(
                "モデルファイルのSHA-256がmetadataと一致しません。"
                "モデルとmetadataの組み合わせを確認してください"
            )

        model = joblib.load(self.model_path)
        if not isinstance(model, RandomForestClassifier):
            raise TypeError("handcrafted_rfにはRandomForestClassifierが必要です")
        if n_jobs is not None:
            model.set_params(n_jobs=n_jobs)
        if list(model.classes_) != [0, 1]:
            raise ValueError(f"モデルのクラス順序が[0, 1]ではありません: {model.classes_}")
        if list(model.feature_names_in_) != FACIAL_FEATURE_COLUMNS:
            raise ValueError("モデル内部の特徴量名・順序が現在の定義と一致しません")

        self.model = model
        self.model_id = str(self.metadata["model_id"])
        self.class_names = {
            int(item["value"]): str(item["name"])
            for item in self.metadata["classes"]
        }

    def _validate_metadata(self) -> None:
        required = {
            "model_id",
            "model_file_sha256",
            "classes",
            "feature_count",
            "feature_columns_in_order",
        }
        missing = sorted(required - set(self.metadata))
        if missing:
            raise ValueError("metadataに必要な項目がありません: " + ", ".join(missing))
        if int(self.metadata["feature_count"]) != len(FACIAL_FEATURE_COLUMNS):
            raise ValueError("metadataの特徴量数が現在の定義と一致しません")
        if self.metadata["feature_columns_in_order"] != FACIAL_FEATURE_COLUMNS:
            raise ValueError("metadataの特徴量名・順序が現在の定義と一致しません")
        classes = {
            int(item["value"]): str(item["name"])
            for item in self.metadata["classes"]
        }
        if classes != {0: "water", 1: "sparkling_water"}:
            raise ValueError(f"metadataのクラス対応が不正です: {classes}")

    def predict_features(self, features: dict[str, float | int]) -> TrialPrediction:
        if list(features) != FACIAL_FEATURE_COLUMNS:
            raise ValueError("推論特徴量の名前・順序がモデル仕様と一致しません")
        values = np.asarray(
            [features[name] for name in FACIAL_FEATURE_COLUMNS], dtype=np.float64
        )
        if not np.isfinite(values).all():
            raise ValueError("推論特徴量にNaN/Infがあります")

        x = pd.DataFrame([features], columns=FACIAL_FEATURE_COLUMNS)
        predicted_label = int(self.model.predict(x)[0])
        probability_values = self.model.predict_proba(x)[0]
        probability_by_label = {
            int(label): float(probability)
            for label, probability in zip(self.model.classes_, probability_values)
        }
        if not np.isclose(sum(probability_by_label.values()), 1.0):
            raise RuntimeError("モデルのクラス確率合計が1ではありません")
        return TrialPrediction(
            model_id=self.model_id,
            input_type=self.input_type,
            predicted_label=predicted_label,
            predicted_name=self.class_names[predicted_label],
            probabilities={
                self.class_names[label]: probability_by_label[label]
                for label in sorted(probability_by_label)
            },
            model_inputs={name: float(features[name]) for name in FACIAL_FEATURE_COLUMNS},
        )

    def predict_normalized(self, normalized_trial: pd.DataFrame) -> TrialPrediction:
        features, _debug_signals = extract_trial_features(normalized_trial)
        return self.predict_features(features)


def load_predictor(
    predictor_type: str,
    *,
    model_path: Path,
    metadata_path: Path,
    n_jobs: int | None = None,
) -> TrialPredictor:
    """Factory kept stable so future coordinate-sequence backends are pluggable."""
    if predictor_type == "handcrafted_rf":
        return HandcraftedRandomForestPredictor(
            model_path, metadata_path, n_jobs=n_jobs
        )
    raise ValueError(f"未対応のpredictorです: {predictor_type}")
