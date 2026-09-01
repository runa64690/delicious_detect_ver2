from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scripts.feature_schema import FACIAL_FEATURE_COLUMNS
from scripts.train_final_random_forest import (
    fit_final_random_forest,
    save_and_verify_model,
)


def small_training_data() -> pd.DataFrame:
    rows = []
    for participant_number in range(1, 4):
        for label in (0, 1):
            row = {
                "trial_id": f"P{participant_number:02d}_{label + 1:03d}",
                "participant_id": f"P{participant_number:02d}",
                "drink_name": "water" if label == 0 else "sparkling_water",
                "drink_label": label,
            }
            for index, feature in enumerate(FACIAL_FEATURE_COLUMNS):
                row[feature] = participant_number * 0.01 + label + index * 0.001
            rows.append(row)
    return pd.DataFrame(rows)


class FinalRandomForestTests(unittest.TestCase):
    def test_saved_model_preserves_features_classes_and_predictions(self) -> None:
        data = small_training_data()
        config = {
            "target_column": "drink_label",
            "positive_label": 1,
            "models": {
                "random_forest": {
                    "n_estimators": 25,
                    "max_features": "sqrt",
                    "min_samples_leaf": 1,
                    "class_weight": "balanced",
                    "random_state": 42,
                    "n_jobs": 1,
                }
            },
        }
        x = data.loc[:, FACIAL_FEATURE_COLUMNS]
        model = fit_final_random_forest(data, config)

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "model.joblib"
            loaded = save_and_verify_model(model, path, x)
            loaded_from_disk = joblib.load(path)

        self.assertEqual(list(loaded.feature_names_in_), FACIAL_FEATURE_COLUMNS)
        self.assertEqual(list(loaded.classes_), [0, 1])
        np.testing.assert_array_equal(
            loaded_from_disk.predict(x), model.predict(x)
        )


if __name__ == "__main__":
    unittest.main()
