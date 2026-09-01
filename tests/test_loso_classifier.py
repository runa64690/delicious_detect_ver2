from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier

from scripts.feature_schema import FACIAL_FEATURE_COLUMNS, QUESTIONNAIRE_COLUMNS
from scripts.train_loso_classifier import (
    build_loso_folds,
    calculate_metrics,
    load_and_validate_features,
    run_model_loso,
)


def synthetic_features() -> pd.DataFrame:
    rows = []
    for participant_number in range(1, 7):
        participant = f"P{participant_number:02d}"
        for label in (0, 1):
            row = {
                "trial_id": f"{participant}_{label + 1:03d}",
                "participant_id": participant,
                "drink_name": "water" if label == 0 else "sparkling_water",
                "drink_label": label,
            }
            for feature_number, feature in enumerate(FACIAL_FEATURE_COLUMNS):
                row[feature] = (
                    participant_number * 0.01
                    + label * 0.1
                    + feature_number * 0.001
                )
            for questionnaire in QUESTIONNAIRE_COLUMNS:
                row[questionnaire] = 9999
            rows.append(row)
    return pd.DataFrame(rows)


class LosoClassifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = synthetic_features()
        self.empty_exclusions = pd.DataFrame(
            columns=["trial_id", "exclude", "reason", "notes"]
        )

    def validate_via_csv(self, data: pd.DataFrame | None = None) -> pd.DataFrame:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "features.csv"
            (data if data is not None else self.data).to_csv(path, index=False)
            return load_and_validate_features(
                path,
                group_column="participant_id",
                target_column="drink_label",
                positive_label=1,
                active_exclusions=self.empty_exclusions,
            )

    def test_schema_contains_exactly_21_non_questionnaire_features(self) -> None:
        self.assertEqual(len(FACIAL_FEATURE_COLUMNS), 21)
        self.assertTrue(set(FACIAL_FEATURE_COLUMNS).isdisjoint(QUESTIONNAIRE_COLUMNS))

    def test_loso_has_one_disjoint_test_fold_per_participant(self) -> None:
        data = self.validate_via_csv()
        folds = build_loso_folds(
            data,
            group_column="participant_id",
            target_column="drink_label",
        )
        self.assertEqual(len(folds), 6)
        all_test_indices: list[int] = []
        test_participants = []
        for fold in folds:
            train_participants = set(
                data.iloc[fold.train_indices]["participant_id"]
            )
            test_participants_in_fold = set(
                data.iloc[fold.test_indices]["participant_id"]
            )
            self.assertEqual(test_participants_in_fold, {fold.test_participant})
            self.assertTrue(train_participants.isdisjoint(test_participants_in_fold))
            self.assertEqual(len(fold.test_indices), 2)
            all_test_indices.extend(fold.test_indices.tolist())
            test_participants.append(fold.test_participant)
        self.assertEqual(sorted(all_test_indices), list(range(len(data))))
        self.assertEqual(len(set(test_participants)), 6)

    def test_dummy_predicts_every_trial_once_with_valid_probabilities(self) -> None:
        data = self.validate_via_csv()
        folds = build_loso_folds(
            data,
            group_column="participant_id",
            target_column="drink_label",
        )
        result = run_model_loso(
            data,
            folds,
            DummyClassifier(strategy="prior"),
            model_name="dummy",
            group_column="participant_id",
            target_column="drink_label",
            positive_label=1,
        )
        self.assertEqual(len(result.predictions), len(data))
        self.assertFalse(result.predictions["trial_id"].duplicated().any())
        probability_sum = (
            result.predictions["probability_water"]
            + result.predictions["probability_sparkling"]
        )
        np.testing.assert_allclose(probability_sum, 1.0)
        self.assertTrue(
            result.predictions["probability_sparkling"].between(0, 1).all()
        )
        self.assertEqual(int(result.fold_metrics["test_trials"].sum()), len(data))
        confusion_total = int(
            result.fold_metrics[["tn", "fp", "fn", "tp"]].to_numpy().sum()
        )
        self.assertEqual(confusion_total, len(data))

    def test_extratrees_is_reproducible_with_fixed_seed(self) -> None:
        data = self.validate_via_csv()
        folds = build_loso_folds(
            data,
            group_column="participant_id",
            target_column="drink_label",
        )

        def run_once():
            return run_model_loso(
                data,
                folds,
                ExtraTreesClassifier(
                    n_estimators=25,
                    max_features="sqrt",
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=1,
                ),
                model_name="extra_trees",
                group_column="participant_id",
                target_column="drink_label",
                positive_label=1,
            )

        first = run_once().predictions.sort_values("trial_id").reset_index(drop=True)
        second = run_once().predictions.sort_values("trial_id").reset_index(drop=True)
        np.testing.assert_array_equal(first["y_pred"], second["y_pred"])
        np.testing.assert_allclose(
            first["probability_sparkling"],
            second["probability_sparkling"],
            rtol=0,
            atol=0,
        )

    def test_manual_exclusion_must_not_be_in_features(self) -> None:
        exclusions = pd.DataFrame(
            [
                {
                    "trial_id": self.data.iloc[0]["trial_id"],
                    "exclude": 1,
                    "reason": "test",
                    "notes": "",
                }
            ]
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "features.csv"
            self.data.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "混入"):
                load_and_validate_features(
                    path,
                    group_column="participant_id",
                    target_column="drink_label",
                    positive_label=1,
                    active_exclusions=exclusions,
                )

    def test_metrics_confusion_counts_match_trial_count(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1])
        probability = np.array([0.1, 0.8, 0.4, 0.9])
        metrics = calculate_metrics(
            y_true, y_pred, probability, positive_label=1
        )
        self.assertEqual(
            metrics["tn"] + metrics["fp"] + metrics["fn"] + metrics["tp"],
            len(y_true),
        )
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["balanced_accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
