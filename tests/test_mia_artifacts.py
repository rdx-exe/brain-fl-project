"""Artifact-level checks for the generated membership inference results.

The tests are skipped when `results/privacy/mia` is absent, because experiment
outputs are not tracked in git.  When artifacts are present, they verify the
claims a reader would otherwise have to trust: participant counts, absence of
the test split, calibration/evaluation disjointness, metric reproducibility, and
separation of the intermediate-checkpoint attack surface.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MIA_DIR = PROJECT_ROOT / "results" / "privacy" / "mia"
EXPECTED_SETTINGS = {"centralized", "federated", "federated_dp"}


@unittest.skipUnless(MIA_DIR.exists(), "MIA artifacts have not been generated")
class MIAArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.comparison = pd.read_csv(MIA_DIR / "mia_comparison.csv")
        cls.manifest = json.loads(
            (MIA_DIR / "attack_dataset_manifest.json").read_text()
        )

    def test_participant_counts_and_no_test_split(self) -> None:
        self.assertEqual(set(self.manifest), EXPECTED_SETTINGS)

        for setting in EXPECTED_SETTINGS:
            dataset = pd.read_csv(MIA_DIR / setting / "attack_dataset.csv")
            with self.subTest(setting=setting):
                self.assertEqual(len(dataset), 199)
                self.assertEqual(int(dataset["membership"].sum()), 164)
                self.assertEqual(int((1 - dataset["membership"]).sum()), 35)
                self.assertEqual(dataset["participant_id"].nunique(), 199)
                self.assertTrue(set(dataset["split"]) <= {"train", "validation"})
                self.assertFalse(self.manifest[setting]["test_split_used"])
                self.assertEqual(
                    self.manifest[setting]["privacy_unit"], "participant"
                )
                self.assertEqual(
                    self.manifest[setting]["slices_per_participant"], 7
                )

    def test_headline_metrics_are_reproducible(self) -> None:
        self.assertEqual(set(self.comparison["setting"]), EXPECTED_SETTINGS)
        self.assertTrue((self.comparison["evaluation_participants"] == 100).all())
        self.assertTrue((self.comparison["evaluation_members"] == 82).all())
        self.assertTrue((self.comparison["evaluation_nonmembers"] == 18).all())

        for setting in EXPECTED_SETTINGS:
            predictions = pd.read_csv(
                MIA_DIR / setting / "participant_attack_predictions.csv"
            )
            calibration = pd.read_csv(
                MIA_DIR / setting / "attack_calibration_participants.csv"
            )
            metrics = pd.read_csv(MIA_DIR / setting / "attack_metrics.csv").iloc[0]
            with self.subTest(setting=setting):
                self.assertEqual(set(predictions["attack_role"]), {"evaluation"})
                self.assertEqual(len(calibration), 99)
                self.assertFalse(
                    set(predictions["participant_id"])
                    & set(calibration["participant_id"])
                )
                self.assertTrue(
                    set(predictions["split"]) | set(calibration["split"])
                    <= {"train", "validation"}
                )

                y_true = predictions["membership"].to_numpy(dtype=int)
                scores = predictions["attack_score"].to_numpy(dtype=float)
                calls = predictions["attack_prediction"].to_numpy(dtype=int)
                self.assertAlmostEqual(
                    roc_auc_score(y_true, scores), metrics["attack_roc_auc"]
                )
                self.assertAlmostEqual(
                    balanced_accuracy_score(y_true, calls),
                    metrics["attack_balanced_accuracy"],
                )
                self.assertAlmostEqual(
                    accuracy_score(y_true, calls), metrics["attack_accuracy"]
                )
                # The primary attack must be the label-agnostic variant.
                np.testing.assert_allclose(
                    predictions["attack_score_confidence_agnostic"],
                    predictions["attack_score"],
                )

    def test_intermediate_surface_is_separate(self) -> None:
        checkpoint_csv = (
            MIA_DIR
            / "federated_checkpoints"
            / "federated_checkpoint_comparison.csv"
        )
        if not checkpoint_csv.exists():
            self.skipTest("Intermediate checkpoint artifacts are absent.")

        checkpoints = pd.read_csv(checkpoint_csv)
        self.assertEqual(list(checkpoints["round"]), list(range(6)))
        self.assertTrue((checkpoints["evaluation_participants"] == 100).all())

        federated = self.comparison.set_index("setting").loc["federated"]
        round_four = checkpoints.loc[checkpoints["round"] == 4].iloc[0]
        self.assertAlmostEqual(
            round_four["attack_roc_auc"], federated["attack_roc_auc"]
        )

        for config_path in MIA_DIR.rglob("attack_config.json"):
            config = json.loads(config_path.read_text())
            if config["setting"].startswith("federated_round_"):
                self.assertEqual(
                    config["federated_attack_surface"],
                    "intermediate_global_model",
                )

    def test_required_artifacts_exist(self) -> None:
        required = [
            "report.md",
            "threat_model.md",
            "README.md",
            "attack_dataset.csv",
            "attack_dataset_manifest.json",
            "mia_comparison.csv",
            "mia_attack_variants.csv",
            "mia_stability_summary.csv",
            "roc_comparison.png",
            "roc_comparison.svg",
            "mia_metric_comparison.png",
            "mia_metric_comparison.svg",
        ]
        for relative in required:
            path = MIA_DIR / relative
            with self.subTest(artifact=relative):
                self.assertTrue(path.exists())
                self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
