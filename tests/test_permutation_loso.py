from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import ExtraTreesClassifier

from scripts import permutation_test_loso as permutation
from scripts import train_loso_classifier as loso
from test_loso_classifier import synthetic_features


class PermutationLosoTests(unittest.TestCase):
    def setUp(self):
        self.data = synthetic_features()
        self.folds = loso.build_loso_folds(self.data, group_column="participant_id", target_column="drink_label")
        self.blocks = permutation.exchangeability_blocks(self.data, ["participant_id"], target_column="drink_label")
        self.model = ExtraTreesClassifier(n_estimators=5, random_state=42, n_jobs=1)

    def test_permutations_preserve_within_participant_counts_and_input(self):
        y = self.data.drink_label.to_numpy()
        original = y.copy()
        draws = []
        for index in range(1, 15):
            shuffled = permutation.permute_labels(y, self.blocks, seed=42, permutation_id=index)
            for block in self.blocks:
                np.testing.assert_array_equal(np.sort(shuffled[block]), np.sort(y[block]))
            draws.append(shuffled)
        np.testing.assert_array_equal(y, original)
        self.assertTrue(any(not np.array_equal(draw, y) for draw in draws))

    def test_extra_blocks_and_no_exchangeable_labels(self):
        data = pd.concat([self.data.assign(session=1), self.data.assign(session=2)], ignore_index=True)
        blocks = permutation.exchangeability_blocks(data, ["participant_id", "session"], target_column="drink_label")
        self.assertEqual(len(blocks), 12)
        with self.assertRaisesRegex(ValueError, "No block"):
            permutation.exchangeability_blocks(data, ["trial_id", "session"], target_column="drink_label")
        with self.assertRaisesRegex(ValueError, "target column"):
            permutation.exchangeability_blocks(data, ["drink_label"], target_column="drink_label")

    def test_score_matches_existing_loso_for_every_supported_metric(self):
        # Unequal fold sizes distinguish participant-average from pooled scores.
        data = pd.concat([self.data, self.data.iloc[[0]].assign(trial_id="extra")], ignore_index=True)
        folds = loso.build_loso_folds(data, group_column="participant_id", target_column="drink_label")
        result = loso.run_model_loso(data, folds, self.model, model_name="extra_trees",
                                     group_column="participant_id", target_column="drink_label", positive_label=1)
        for metric in loso.METRIC_COLUMNS:
            score = permutation.score_loso(data[loso.FACIAL_FEATURE_COLUMNS].to_numpy(),
                                            data.drink_label.to_numpy(), folds, self.model, metric)
            self.assertAlmostEqual(score, result.fold_metrics[metric].mean(), places=14)

    def test_pvalue_counts_ties_and_never_returns_zero(self):
        self.assertEqual(permutation.monte_carlo_pvalue(0.7, [0.6, 0.7, 0.8]), (0.75, 2))
        self.assertEqual(permutation.monte_carlo_pvalue(0.9, [0.6, 0.7, 0.8]), (0.25, 0))
        self.assertEqual(permutation.monte_carlo_pvalue(0.5, [0.5] * 5), (1, 5))
        with self.assertRaises(ValueError):
            permutation.monte_carlo_pvalue(0.5, [np.nan])

    def test_holm_preserves_original_order_and_monotonicity(self):
        np.testing.assert_allclose(permutation.holm_adjust([0.04, 0.01, 0.03]), [0.06, 0.03, 0.06])

    def test_parallel_matches_serial_and_restart_stream(self):
        kwargs = dict(x=self.data[loso.FACIAL_FEATURE_COLUMNS].to_numpy(), y=self.data.drink_label.to_numpy(),
                      blocks=self.blocks, seed=42, folds=self.folds, models={"extra_trees": self.model},
                      metric="balanced_accuracy")
        sequential = [permutation.permutation_row(i, **kwargs) for i in range(1, 4)]
        parallel = Parallel(n_jobs=2)(delayed(permutation.permutation_row)(i, **kwargs) for i in range(1, 4))
        self.assertEqual(sequential, parallel)
        self.assertEqual(sequential[2], permutation.permutation_row(3, **kwargs))

    def test_cli_outputs_resume_and_plot_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.data.to_csv(root / "features.csv", index=False)
            pd.DataFrame(columns=["trial_id", "exclude", "reason", "notes"]).to_csv(root / "excluded.csv", index=False)
            config = dict(input_path=str(root / "features.csv"), excluded_trials_path=str(root / "excluded.csv"),
                          output_root=str(root), positive_label=1, group_column="participant_id",
                          target_column="drink_label", random_seed=42, primary_metric="balanced_accuracy",
                          models={"extra_trees": dict(n_estimators=5, random_state=42, n_jobs=1)})
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            output = root / "output"
            argv = ["--config", str(config_path), "--output", str(output), "--n-permutations", "3", "--dpi", "72"]
            first = permutation.run(permutation.parse_args(argv))
            for name in ["permutation_test.pdf", "permutation_test.svg", "permutation_test.png", "figure_caption.txt"]:
                self.assertGreater((output / name).stat().st_size, 100)
            checkpoint_path = output / "permutation_scores.csv"
            complete = checkpoint_path.read_bytes()
            lines = complete.splitlines(keepends=True)
            checkpoint_path.write_bytes(b"".join(lines[:2]))
            resumed = permutation.run(permutation.parse_args([*argv, "--resume"]))
            self.assertEqual(checkpoint_path.read_bytes(), complete)
            pd.testing.assert_frame_equal(first, resumed)
            replotted = permutation.run(permutation.parse_args(["--output", str(output), "--plot-only", "--dpi", "72"]))
            pd.testing.assert_frame_equal(first, replotted)
            with self.assertRaises(FileExistsError):
                permutation.run(permutation.parse_args(argv))
            with self.assertRaisesRegex(ValueError, "differ"):
                permutation.run(permutation.parse_args([*argv, "--resume", "--seed", "43"]))

    def test_reject_corrupt_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "scores.csv"
            for contents in ("permutation_id,dummy\n2,0.5\n", "permutation_id,dummy\n1,\n"):
                path.write_text(contents, encoding="utf-8")
                with self.assertRaises(ValueError):
                    permutation.read_checkpoint(path, ["dummy"])


if __name__ == "__main__":
    unittest.main()
