from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from realtime_classify import send_control_command
from scripts.feature_schema import FACIAL_FEATURE_COLUMNS
from scripts.realtime_pipeline import (
    TrialQualityError,
    prepare_normalized_trial,
    process_trial,
)
from scripts.realtime_predictors import (
    HandcraftedRandomForestPredictor,
    TrialPrediction,
    TrialPredictor,
)
from scripts.train_loso_classifier import sha256_file


def fitted_model() -> tuple[RandomForestClassifier, pd.DataFrame]:
    rows = []
    for label in (0, 1):
        for repetition in range(3):
            rows.append(
                {
                    feature: label + repetition * 0.01 + index * 0.001
                    for index, feature in enumerate(FACIAL_FEATURE_COLUMNS)
                }
            )
    x = pd.DataFrame(rows, columns=FACIAL_FEATURE_COLUMNS)
    y = np.repeat([0, 1], 3)
    model = RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=1)
    model.fit(x, y)
    return model, x


def raw_trial_with_one_failure() -> pd.DataFrame:
    rows = []
    for frame in range(3):
        valid = frame != 1
        row: dict[str, float | int | None] = {
            "facemesh_ok": int(valid),
            "elapsed_sec": frame * 0.1,
        }
        for landmark in range(478):
            row[f"lm{landmark}_x_px"] = 100.0 + landmark * 0.01 if valid else None
            row[f"lm{landmark}_y_px"] = 100.0 + frame * 0.1 if valid else None
        if valid:
            row["lm168_x_px"] = 100.0
            row["lm168_y_px"] = 100.0
            row["lm33_x_px"] = 75.0
            row["lm33_y_px"] = 100.0
            row["lm133_x_px"] = 85.0
            row["lm133_y_px"] = 100.0
            row["lm362_x_px"] = 115.0
            row["lm362_y_px"] = 100.0
            row["lm263_x_px"] = 125.0
            row["lm263_y_px"] = 100.0
        rows.append(row)
    return pd.DataFrame(rows)


class FakePredictor(TrialPredictor):
    model_id = "fake_sequence_ready_predictor"
    input_type = "test"

    def predict_normalized(self, normalized_trial: pd.DataFrame) -> TrialPrediction:
        return TrialPrediction(
            model_id=self.model_id,
            input_type=self.input_type,
            predicted_label=0,
            predicted_name="water",
            probabilities={"water": 0.75, "sparkling_water": 0.25},
        )


class FakeControlSocket:
    def __init__(self, *, should_fail: bool = False):
        self.should_fail = should_fail
        self.sent: list[bytes] = []
        self.closed = False

    def sendall(self, payload: bytes) -> None:
        if self.should_fail:
            raise OSError("test disconnect")
        self.sent.append(payload)

    def close(self) -> None:
        self.closed = True


class RealtimeInferenceTests(unittest.TestCase):
    def test_led_control_command_is_newline_delimited(self) -> None:
        control = FakeControlSocket()
        returned = send_control_command(control, "LED_PROCESSING")
        self.assertIs(returned, control)
        self.assertEqual(control.sent, [b"LED_PROCESSING\n"])

    def test_led_control_failure_does_not_raise(self) -> None:
        control = FakeControlSocket(should_fail=True)
        returned = send_control_command(control, "LED_FACE_ON")
        self.assertIsNone(returned)
        self.assertTrue(control.closed)

    def test_predictor_validates_artifact_and_predicts_ordered_features(self) -> None:
        model, x = fitted_model()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            model_path = root / "model.joblib"
            metadata_path = root / "metadata.json"
            joblib.dump(model, model_path)
            metadata = {
                "model_id": "test_rf",
                "model_file_sha256": sha256_file(model_path),
                "classes": [
                    {"value": 0, "name": "water"},
                    {"value": 1, "name": "sparkling_water"},
                ],
                "feature_count": 21,
                "feature_columns_in_order": FACIAL_FEATURE_COLUMNS,
            }
            metadata_path.write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            predictor = HandcraftedRandomForestPredictor(
                model_path, metadata_path, n_jobs=1
            )
            result = predictor.predict_features(x.iloc[0].to_dict())

        self.assertEqual(result.model_id, "test_rf")
        self.assertIn(result.predicted_label, (0, 1))
        self.assertAlmostEqual(sum(result.probabilities.values()), 1.0)
        self.assertEqual(list(result.model_inputs), FACIAL_FEATURE_COLUMNS)

    def test_predictor_rejects_wrong_feature_order_in_metadata(self) -> None:
        model, _x = fitted_model()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            model_path = root / "model.joblib"
            metadata_path = root / "metadata.json"
            joblib.dump(model, model_path)
            metadata_path.write_text(
                json.dumps(
                    {
                        "model_id": "test_rf",
                        "model_file_sha256": sha256_file(model_path),
                        "classes": [
                            {"value": 0, "name": "water"},
                            {"value": 1, "name": "sparkling_water"},
                        ],
                        "feature_count": 21,
                        "feature_columns_in_order": list(
                            reversed(FACIAL_FEATURE_COLUMNS)
                        ),
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "順序"):
                HandcraftedRandomForestPredictor(model_path, metadata_path)

    def test_predictor_rejects_model_hash_mismatch(self) -> None:
        model, _x = fitted_model()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            model_path = root / "model.joblib"
            metadata_path = root / "metadata.json"
            joblib.dump(model, model_path)
            metadata_path.write_text(
                json.dumps(
                    {
                        "model_id": "test_rf",
                        "model_file_sha256": "0" * 64,
                        "classes": [
                            {"value": 0, "name": "water"},
                            {"value": 1, "name": "sparkling_water"},
                        ],
                        "feature_count": 21,
                        "feature_columns_in_order": FACIAL_FEATURE_COLUMNS,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                HandcraftedRandomForestPredictor(model_path, metadata_path)

    def test_quality_gate_skips_failed_frames_and_normalizes_valid_frames(self) -> None:
        normalized, quality = prepare_normalized_trial(
            raw_trial_with_one_failure(),
            trial_id="test_trial",
            minimum_valid_frames=2,
            minimum_success_rate=0.5,
        )
        self.assertEqual(quality.total_frames, 3)
        self.assertEqual(quality.valid_frames, 2)
        self.assertAlmostEqual(quality.facemesh_success_rate, 2 / 3)
        self.assertEqual(len(normalized), 2)
        self.assertEqual(normalized["frame_index"].tolist(), [0, 2])

    def test_quality_gate_rejects_low_detection_rate(self) -> None:
        with self.assertRaisesRegex(TrialQualityError, "成功率不足"):
            prepare_normalized_trial(
                raw_trial_with_one_failure(),
                trial_id="test_trial",
                minimum_valid_frames=2,
                minimum_success_rate=0.8,
            )

    def test_process_trial_logs_success_and_can_save_trial_csvs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            log_path = root / "predictions.jsonl"
            outcome = process_trial(
                raw_trial_with_one_failure(),
                trial_id="test_trial",
                predictor=FakePredictor(),
                minimum_valid_frames=2,
                minimum_success_rate=0.5,
                log_path=log_path,
                save_trials_dir=root / "trials",
            )
            record = json.loads(log_path.read_text(encoding="utf-8"))
            self.assertTrue((root / "trials/raw/test_trial.csv").is_file())
            self.assertTrue((root / "trials/normalized/test_trial.csv").is_file())

        self.assertEqual(outcome.prediction.predicted_name, "water")
        self.assertEqual(record["status"], "success")
        self.assertEqual(record["quality"]["valid_frames"], 2)

    def test_process_trial_logs_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            log_path = Path(temporary_directory) / "predictions.jsonl"
            with self.assertRaises(TrialQualityError):
                process_trial(
                    raw_trial_with_one_failure(),
                    trial_id="rejected_trial",
                    predictor=FakePredictor(),
                    minimum_valid_frames=2,
                    minimum_success_rate=0.8,
                    log_path=log_path,
                )
            record = json.loads(log_path.read_text(encoding="utf-8"))

        self.assertEqual(record["status"], "rejected")
        self.assertEqual(record["trial_id"], "rejected_trial")
        self.assertEqual(record["error_type"], "TrialQualityError")


if __name__ == "__main__":
    unittest.main()
