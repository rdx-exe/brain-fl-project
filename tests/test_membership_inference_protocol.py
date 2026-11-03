"""Protocol-level tests for the participant-level membership inference attack.

These tests do not load MRI data or target checkpoints.  They exercise the
invariants that make the reported MIA numbers defensible:

* one row per participant;
* disjoint, stratified attack calibration/evaluation roles;
* the test split is rejected;
* metrics behave correctly for perfect separation, chance, and the
  majority-class baseline;
* bootstrap intervals are well formed.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from privacy.attack_dataset import extract_participant_features
from privacy.membership_inference import (
    bootstrap_confidence_intervals,
    compute_metrics,
    split_attack_participants,
)


def _synthetic_participants() -> pd.DataFrame:
    """82 members and 18 non-members, each with a diagnosis label."""

    records = []
    participant_number = 0
    for membership, count in ((1, 82), (0, 18)):
        for class_label in (0, 1):
            per_class = count // 2
            for _ in range(per_class):
                participant_number += 1
                records.append(
                    {
                        "participant_id": f"P{participant_number:04d}",
                        "membership": membership,
                        "class_label": class_label,
                        "split": "train" if membership == 1 else "validation",
                    }
                )
    return pd.DataFrame(records)


class SplitProtocolTests(unittest.TestCase):
    def test_split_is_disjoint_balanced_and_complete(self) -> None:
        frame = _synthetic_participants()

        for seed in range(10):
            assigned = split_attack_participants(frame, seed=seed)
            calibration = set(
                assigned.loc[assigned["attack_role"] == "calibration", "participant_id"]
            )
            evaluation = set(
                assigned.loc[assigned["attack_role"] == "evaluation", "participant_id"]
            )

            self.assertFalse(calibration & evaluation)
            self.assertEqual(calibration | evaluation, set(frame["participant_id"]))
            self.assertEqual(len(calibration), 50)
            self.assertEqual(len(evaluation), 50)

            evaluation_members = int(
                assigned.loc[assigned["attack_role"] == "evaluation", "membership"].sum()
            )
            evaluation_nonmembers = len(evaluation) - evaluation_members
            # Stratification preserves the 82/18 membership ratio up to
            # integer rounding within the four membership-by-diagnosis strata.
            self.assertGreaterEqual(evaluation_members, 40)
            self.assertLessEqual(evaluation_members, 42)
            self.assertGreaterEqual(evaluation_nonmembers, 8)
            self.assertLessEqual(evaluation_nonmembers, 10)

    def test_duplicate_participants_are_rejected(self) -> None:
        frame = _synthetic_participants()
        duplicated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)

        with self.assertRaisesRegex(ValueError, "one row per participant"):
            split_attack_participants(duplicated, seed=42)

    def test_test_split_is_rejected(self) -> None:
        frame = _synthetic_participants()
        frame.loc[0, "split"] = "test"

        with self.assertRaisesRegex(ValueError, "Test split"):
            split_attack_participants(frame, seed=42)

    def test_non_binary_membership_is_rejected(self) -> None:
        frame = _synthetic_participants()
        frame.loc[0, "membership"] = 2

        with self.assertRaisesRegex(ValueError, "binary"):
            split_attack_participants(frame, seed=42)

    def test_extraction_refuses_test_split(self) -> None:
        with self.assertRaisesRegex(ValueError, "only read the train and validation"):
            extract_participant_features(model=None, split="test")


class MetricTests(unittest.TestCase):
    def test_perfect_separation(self) -> None:
        evaluation = pd.DataFrame(
            {
                "membership": [0, 0, 1, 1],
                "attack_score": [0.05, 0.20, 0.80, 0.95],
                "attack_prediction": [0, 0, 1, 1],
            }
        )

        metrics = compute_metrics(evaluation)

        self.assertAlmostEqual(metrics["attack_accuracy"], 1.0)
        self.assertAlmostEqual(metrics["attack_balanced_accuracy"], 1.0)
        self.assertAlmostEqual(metrics["attack_roc_auc"], 1.0)
        self.assertAlmostEqual(metrics["attack_advantage"], 1.0)
        self.assertEqual(metrics["evaluation_members"], 2)
        self.assertEqual(metrics["evaluation_nonmembers"], 2)

    def test_all_member_prediction_matches_majority_baseline(self) -> None:
        y_true = np.array([1] * 82 + [0] * 18)
        evaluation = pd.DataFrame(
            {
                "membership": y_true,
                "attack_score": np.full(len(y_true), 0.9),
                "attack_prediction": np.ones(len(y_true), dtype=int),
            }
        )

        metrics = compute_metrics(evaluation)

        self.assertAlmostEqual(metrics["attack_accuracy"], 0.82)
        self.assertAlmostEqual(metrics["majority_class_accuracy"], 0.82)
        self.assertAlmostEqual(metrics["attack_balanced_accuracy"], 0.5)
        self.assertAlmostEqual(metrics["attack_advantage"], 0.0)
        self.assertAlmostEqual(metrics["attack_roc_auc"], 0.5)

    def test_bootstrap_intervals_are_ordered_and_bounded(self) -> None:
        rng = np.random.default_rng(0)
        n = 60
        membership = np.array([1] * 40 + [0] * 20)
        scores = np.clip(
            rng.normal(loc=np.where(membership == 1, 0.65, 0.35), scale=0.15),
            0.0,
            1.0,
        )
        evaluation = pd.DataFrame(
            {
                "membership": membership,
                "attack_score": scores,
                "attack_prediction": (scores >= 0.5).astype(int),
            }
        )

        intervals = bootstrap_confidence_intervals(
            evaluation, n_bootstrap=200, seed=7
        )

        for metric in ("attack_roc_auc", "attack_balanced_accuracy", "attack_accuracy"):
            lower = intervals[f"{metric}_ci_lower"]
            upper = intervals[f"{metric}_ci_upper"]
            self.assertTrue(np.isfinite(lower))
            self.assertTrue(np.isfinite(upper))
            self.assertLessEqual(lower, upper)
            self.assertGreaterEqual(lower, 0.0)
            self.assertLessEqual(upper, 1.0)


if __name__ == "__main__":
    unittest.main()
