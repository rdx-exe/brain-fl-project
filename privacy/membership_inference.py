"""Participant-level membership inference against OASIS-1 MRI classifiers.

This module implements a confidence-based membership inference attack (MIA)
against three target models:

1. a centralized ResNet-18;
2. a federated ResNet-18 trained with FedAvg;
3. a federated ResNet-18 trained with participant-level local DP-SGD.

The attack is deliberately scoped to the *final* model output.  A separate
supplementary analysis can evaluate intermediate global FedAvg checkpoints.
Individual client-update attacks are a different threat model and are not
combined with the final-model results.

Protocol summary
----------------
* Privacy unit: participant (all seven slices stay together).
* Members: the 164 training participants.
* Non-members: the 35 validation participants.
* The 36-participant test split is never loaded by the attack.
* Attack calibration and evaluation participants are disjoint.
* The primary attack is a label-agnostic logistic regression on participant-
  level confidence features.  A label-aware loss variant is a sensitivity
  analysis.
* Metrics include accuracy, precision, recall, F1, balanced accuracy,
  ROC-AUC, TPR-FPR advantage, and TPR at low false-positive rates.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from privacy.attack_dataset import (
    CONFIDENCE_FEATURES,
    LABEL_AWARE_FEATURES,
    TARGETS,
    TargetSpec,
    build_attack_dataset,
    checkpoint_paths_for_intermediate_rounds,
)

DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "privacy" / "mia"
DEFAULT_SEED = 42
DEFAULT_REPEATS = 50
DEFAULT_BOOTSTRAP = 2000
FPR_TARGETS = (0.01, 0.05, 0.10)
PRIMARY_VARIANT = "confidence_agnostic"


# ---------------------------------------------------------------------------
# Splitting and attack fitting
# ---------------------------------------------------------------------------


def _strata(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["membership"].astype(str)
        + "_class"
        + frame["class_label"].astype(str)
    )


def split_attack_participants(
    frame: pd.DataFrame,
    seed: int,
    evaluation_fraction: float = 0.5,
) -> pd.DataFrame:
    """Assign each participant to attack calibration or evaluation.

    Splitting is stratified by membership status and diagnosis class.  Because
    each participant occurs exactly once, the split is automatically grouped at
    participant level.  The function asserts that the two roles are disjoint.
    """

    if not 0.0 < evaluation_fraction < 1.0:
        raise ValueError("evaluation_fraction must be strictly between 0 and 1.")

    if frame["participant_id"].duplicated().any():
        raise ValueError(
            "Attack dataset must contain exactly one row per participant; "
            "found duplicate participant_id values."
        )
    if not set(frame["membership"].unique()).issubset({0, 1}):
        raise ValueError("Membership labels must be binary (0/1).")
    if "test" in set(frame["split"]):
        raise ValueError("Test split must not enter the attack dataset.")

    indices = np.arange(len(frame))
    calibration_idx, evaluation_idx = train_test_split(
        indices,
        test_size=evaluation_fraction,
        random_state=seed,
        stratify=_strata(frame),
    )

    roles = np.empty(len(frame), dtype=object)
    roles[calibration_idx] = "calibration"
    roles[evaluation_idx] = "evaluation"
    result = frame.copy()
    result["attack_role"] = roles

    calibration_ids = set(result.loc[result["attack_role"] == "calibration", "participant_id"])
    evaluation_ids = set(result.loc[result["attack_role"] == "evaluation", "participant_id"])
    if calibration_ids & evaluation_ids:
        raise ValueError("Attack calibration/evaluation participant leakage detected.")
    if calibration_ids | evaluation_ids != set(result["participant_id"]):
        raise ValueError("Attack role assignment did not cover every participant exactly once.")
    if "test" in set(result["split"]):
        raise ValueError("Test split must not enter the attack dataset.")
    return result


def make_attack_model(seed: int) -> Pipeline:
    """Standardized, class-balanced logistic-regression attack model."""

    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    solver="liblinear",
                    C=1.0,
                    max_iter=1000,
                    random_state=seed,
                ),
            ),
        ]
    )


def fit_and_score(
    calibration: pd.DataFrame,
    evaluation: pd.DataFrame,
    features: list[str],
    seed: int,
    allow_overlap: bool = False,
) -> tuple[np.ndarray, Pipeline]:
    """Fit the attack on calibration participants and score evaluation rows.

    ``allow_overlap`` is used only when re-scoring the calibration participants
    for transparency outputs.  It must remain false for every evaluation metric.
    """

    if not allow_overlap and set(calibration["participant_id"]) & set(
        evaluation["participant_id"]
    ):
        raise ValueError("Participant leakage between attack calibration and evaluation.")
    if not set(calibration.columns).issuperset(features):
        raise ValueError("Calibration frame is missing requested attack features.")
    if not set(evaluation.columns).issuperset(features):
        raise ValueError("Evaluation frame is missing requested attack features.")

    model = make_attack_model(seed)
    model.fit(calibration[features], calibration["membership"].astype(int))
    scores = model.predict_proba(evaluation[features])[:, 1]
    return scores, model


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _operating_point_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = np.asarray(y_pred, dtype=int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) else float("nan")
    tpr = float(tp / (tp + fn)) if (tp + fn) else float("nan")

    metrics: dict[str, float] = {
        "attack_accuracy": float(accuracy_score(y_true, y_pred)),
        "attack_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "attack_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "attack_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "attack_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "attack_specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "attack_fpr_at_threshold": fpr,
        "attack_tpr_at_threshold": tpr,
        "attack_advantage": float(tpr - fpr),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }

    if len(np.unique(y_true)) == 2:
        curve_fpr, curve_tpr, _ = roc_curve(y_true, y_score)
        auc = float(roc_auc_score(y_true, y_score))
        metrics["attack_roc_auc"] = auc
        metrics["attack_auc_advantage"] = float(2.0 * auc - 1.0)
        for target in FPR_TARGETS:
            label = f"{target:.2f}"
            tpr_at_target = float(np.interp(target, curve_fpr, curve_tpr))
            metrics[f"attack_tpr_at_fpr_{label}"] = tpr_at_target
            metrics[f"attack_advantage_at_fpr_{label}"] = float(tpr_at_target - target)
    else:
        metrics["attack_roc_auc"] = float("nan")
        metrics["attack_auc_advantage"] = float("nan")
        for target in FPR_TARGETS:
            label = f"{target:.2f}"
            metrics[f"attack_tpr_at_fpr_{label}"] = float("nan")
            metrics[f"attack_advantage_at_fpr_{label}"] = float("nan")

    return metrics


def compute_metrics(
    evaluation: pd.DataFrame,
    score_column: str = "attack_score",
    prediction_column: str = "attack_prediction",
) -> dict[str, Any]:
    """Compute participant-level attack metrics for one evaluation set."""

    y_true = evaluation["membership"].to_numpy(dtype=int)
    y_score = evaluation[score_column].to_numpy(dtype=float)
    y_pred = evaluation[prediction_column].to_numpy(dtype=int)

    metrics: dict[str, Any] = {
        "evaluation_participants": int(len(evaluation)),
        "evaluation_members": int(y_true.sum()),
        "evaluation_nonmembers": int((1 - y_true).sum()),
        "member_prevalence": float(y_true.mean()),
        "majority_class_accuracy": float(max(y_true.mean(), 1.0 - y_true.mean())),
    }
    metrics.update(_operating_point_metrics(y_true, y_score, y_pred))
    return metrics


def bootstrap_confidence_intervals(
    evaluation: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
    score_column: str = "attack_score",
    prediction_column: str = "attack_prediction",
) -> dict[str, float]:
    """Stratified participant-level bootstrap 95% intervals.

    The bootstrap preserves the observed member/non-member counts, so it does
    not accidentally create a one-class resample.
    """

    y_true = evaluation["membership"].to_numpy(dtype=int)
    y_score = evaluation[score_column].to_numpy(dtype=float)
    y_pred = evaluation[prediction_column].to_numpy(dtype=int)
    member_idx = np.flatnonzero(y_true == 1)
    nonmember_idx = np.flatnonzero(y_true == 0)

    metric_names = [
        "attack_roc_auc",
        "attack_advantage",
        "attack_balanced_accuracy",
        "attack_accuracy",
        "attack_tpr_at_fpr_0.10",
    ]
    samples: dict[str, list[float]] = {name: [] for name in metric_names}
    rng = np.random.default_rng(seed)

    for _ in range(n_bootstrap):
        chosen = np.concatenate(
            [
                rng.choice(member_idx, size=len(member_idx), replace=True),
                rng.choice(nonmember_idx, size=len(nonmember_idx), replace=True),
            ]
        )
        resampled = evaluation.iloc[chosen]
        values = _operating_point_metrics(
            resampled["membership"].to_numpy(dtype=int),
            resampled[score_column].to_numpy(dtype=float),
            resampled[prediction_column].to_numpy(dtype=int),
        )
        for name in metric_names:
            value = values.get(name, float("nan"))
            if np.isfinite(value):
                samples[name].append(float(value))

    intervals: dict[str, float] = {}
    for name, values in samples.items():
        if not values:
            intervals[f"{name}_ci_lower"] = float("nan")
            intervals[f"{name}_ci_upper"] = float("nan")
            continue
        lower, upper = np.percentile(values, [2.5, 97.5])
        intervals[f"{name}_ci_lower"] = float(lower)
        intervals[f"{name}_ci_upper"] = float(upper)
    return intervals


# ---------------------------------------------------------------------------
# Stability analysis
# ---------------------------------------------------------------------------


def repeated_split_stability(
    dataset: pd.DataFrame,
    features: list[str],
    seed: int,
    repeats: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Repeat the calibration/evaluation split to quantify split sensitivity."""

    rows: list[dict[str, Any]] = []
    for repeat in range(repeats):
        split_seed = seed + 1000 + repeat
        assigned = split_attack_participants(dataset, split_seed)
        calibration = assigned[assigned["attack_role"] == "calibration"]
        evaluation = assigned[assigned["attack_role"] == "evaluation"]
        scores, _ = fit_and_score(calibration, evaluation, features, split_seed)
        evaluation = evaluation.copy()
        evaluation["attack_score"] = scores
        evaluation["attack_prediction"] = (scores >= 0.5).astype(int)
        metrics = compute_metrics(evaluation)
        rows.append(
            {
                "repeat": repeat,
                "split_seed": split_seed,
                "calibration_participants": int(len(calibration)),
                "evaluation_participants": int(len(evaluation)),
                **metrics,
            }
        )

    repeats_frame = pd.DataFrame(rows)
    summary_rows = []
    for column in [
        "attack_accuracy",
        "attack_precision",
        "attack_recall",
        "attack_f1",
        "attack_balanced_accuracy",
        "attack_roc_auc",
        "attack_advantage",
        "attack_tpr_at_fpr_0.10",
        "attack_advantage_at_fpr_0.10",
    ]:
        values = repeats_frame[column].dropna().to_numpy(dtype=float)
        summary_rows.append(
            {
                "metric": column,
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)),
                "min": float(values.min()),
                "p2.5": float(np.percentile(values, 2.5)),
                "median": float(np.percentile(values, 50)),
                "p97.5": float(np.percentile(values, 97.5)),
                "max": float(values.max()),
                "repeats": int(len(values)),
            }
        )
    return repeats_frame, pd.DataFrame(summary_rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _save_figure(fig, path_stem: Path) -> None:
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(path_stem.with_suffix(".png"), dpi=200, bbox_inches="tight")


def plot_roc_curve(
    evaluation: pd.DataFrame,
    output_stem: Path,
    title: str,
    score_column: str = "attack_score",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y_true = evaluation["membership"].to_numpy(dtype=int)
    y_score = evaluation[score_column].to_numpy(dtype=float)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    tpr_at_10 = float(np.interp(0.10, fpr, tpr))

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.plot(fpr, tpr, color="#b2182b", linewidth=2.5, label=f"Attack ROC (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=1.2, label="Random guess")
    ax.scatter([0.10], [tpr_at_10], color="#2166ac", zorder=5)
    ax.annotate(
        f"TPR@FPR=10%: {tpr_at_10:.3f}",
        xy=(0.10, tpr_at_10),
        xytext=(0.18, max(0.05, tpr_at_10 - 0.18)),
        arrowprops={"arrowstyle": "->", "color": "#2166ac"},
        fontsize=9,
        color="#2166ac",
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("False positive rate (non-member falsely called member)")
    ax.set_ylabel("True positive rate (member correctly identified)")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    _save_figure(fig, output_stem)
    plt.close(fig)


def plot_roc_comparison(
    results: dict[str, tuple[pd.DataFrame, dict[str, Any]]],
    output_stem: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "centralized": "#b2182b",
        "federated": "#2166ac",
        "federated_dp": "#1b7837",
    }
    labels = {
        "centralized": "Centralized ResNet-18",
        "federated": "Federated FedAvg ResNet-18",
        "federated_dp": "Federated ResNet-18 + DP",
    }

    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    for setting, (evaluation, metrics) in results.items():
        fpr, tpr, _ = roc_curve(
            evaluation["membership"].to_numpy(dtype=int),
            evaluation["attack_score"].to_numpy(dtype=float),
        )
        ax.plot(
            fpr,
            tpr,
            linewidth=2.5,
            color=colors.get(setting, None),
            label=f"{labels.get(setting, setting)} (AUC={metrics['attack_roc_auc']:.3f})",
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=1.2, label="Random guess")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Participant-level membership inference ROC")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    _save_figure(fig, output_stem)
    plt.close(fig)


def plot_metric_comparison(
    comparison: pd.DataFrame,
    output_stem: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {
        "centralized": "Centralized",
        "federated": "FedAvg",
        "federated_dp": "FedAvg + DP",
    }
    x = np.arange(len(comparison))
    width = 0.36

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.bar(
        x - width / 2,
        comparison["attack_roc_auc"],
        width,
        label="ROC-AUC",
        color="#2166ac",
    )
    advantages = comparison["attack_advantage_at_fpr_0.10"]
    ax.bar(
        x + width / 2,
        advantages,
        width,
        label="Advantage at FPR=10%",
        color="#b2182b",
    )
    ax.axhline(0.5, color="#777777", linestyle=":", linewidth=1, label="AUC chance")
    ax.axhline(0.0, color="#777777", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([labels.get(value, value) for value in comparison["setting"]])
    lower = min(0.0, float(advantages.min())) - 0.05
    ax.set_ylim(lower, 1.02)
    ax.set_ylabel("Attack performance")
    ax.set_title("MIA performance across target models")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    _save_figure(fig, output_stem)
    plt.close(fig)


def plot_score_distributions(
    results: dict[str, tuple[pd.DataFrame, dict[str, Any]]],
    output_stem: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {
        "centralized": "Centralized",
        "federated": "FedAvg",
        "federated_dp": "FedAvg + DP",
    }
    if len(results) == 1:
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        axes = [ax]
    else:
        fig, axes = plt.subplots(1, len(results), figsize=(4.0 * len(results), 4.0), sharey=True)
    bins = np.linspace(0, 1, 16)
    for ax, (setting, (evaluation, _)) in zip(axes, results.items()):
        members = evaluation.loc[evaluation["membership"] == 1, "attack_score"]
        nonmembers = evaluation.loc[evaluation["membership"] == 0, "attack_score"]
        ax.hist(members, bins=bins, alpha=0.65, label="Members", color="#b2182b")
        ax.hist(nonmembers, bins=bins, alpha=0.65, label="Non-members", color="#2166ac")
        ax.set_title(labels.get(setting, setting))
        ax.set_xlabel("Attack score P(member)")
        ax.grid(alpha=0.2)
    if len(results) == 1:
        axes[0].set_ylabel("Participants")
        axes[0].legend(fontsize=8)
    else:
        axes[0].set_ylabel("Participants")
        axes[0].legend(fontsize=8)
        fig.suptitle("Participant-level attack-score distributions", y=1.02)
    fig.tight_layout()
    _save_figure(fig, output_stem)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Documentation helpers
# ---------------------------------------------------------------------------


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def write_threat_model(
    output_path: Path,
    dataset_metadata: dict[str, dict[str, Any]],
    comparison: pd.DataFrame,
) -> None:
    base_metadata = dataset_metadata.get("centralized") or next(iter(dataset_metadata.values()))
    counts = base_metadata
    text = f"""# Membership Inference Threat Model

## 1. Research question

Can an adversary determine whether a participant's OASIS-1 MRI examination was
used to train a centralized or federated Alzheimer's Disease classifier?  The
attack is evaluated against three target models:

| Target | Attack surface |
| --- | --- |
| Centralized ResNet-18 | final centralized model checkpoint |
| Federated ResNet-18 (FedAvg) | final global model checkpoint |
| Federated ResNet-18 + local DP | final global model checkpoint |

## 2. Attacker access

The primary attacker is a **black-box, participant-level adversary** with:

* query access to the final model's class probabilities for arbitrary slices;
* knowledge of the model architecture family (ResNet-18), input preprocessing
  (seven fixed axial slices, per-slice percentile normalization, 224x224,
  three-channel replication), and participant-level aggregation rule;
* knowledge of the public OASIS-1 participant manifest and the general
  train/validation split construction, but **not** which specific evaluation
  participants were used for training;
* a set of calibration participants for whom the adversary knows membership
  status.  In this experiment these are disjoint subsets of the training and
  validation pools.

The primary attack does **not** require the participant's true diagnosis: it
uses only the prediction probability, confidence, entropy, and slice-level
confidence statistics.  A label-aware loss variant is reported as a sensitivity
analysis for an adversary who also knows the diagnosis.

The attacker does **not** have:

* access to raw training MRI volumes beyond the query interface;
* model gradients, optimizer state, or raw client updates;
* the 36-participant test split;
* the ability to modify the training process.

## 3. Membership definition

* **Member** (`membership = 1`): a participant in the target model's `train`
  split ({counts['members']} participants).
* **Non-member** (`membership = 0`): a participant in the target model's
  `validation` split ({counts['nonmembers']} participants).

The two sets are disjoint by construction and the attack code asserts this.
The test split is never loaded, trained on, tuned on, or used to evaluate the
attack.  Participants are the privacy unit: all seven slices from one
participant stay in the same calibration or evaluation role.

## 4. Calibration/evaluation separation

The attack dataset is split by participant into:

* **attack calibration** participants: used to fit the attack model;
* **attack evaluation** participants: used only to compute the reported
  metrics and participant-level predictions.

The split is stratified by membership status and diagnosis class.  No
participant appears in both roles, and the code raises an error if it detects
overlap.  The test split is excluded from both roles.

## 5. Attack methodology

The primary attack is a **confidence-based logistic-regression attack**:

1. Query the target model on each of a participant's seven slices.
2. Aggregate the seven class-1 probabilities into one participant score
   (mean probability) and derive participant-level confidence, entropy,
   mean slice confidence, slice-probability standard deviation, minimum, and
   maximum.
3. Standardize the features and fit a class-balanced logistic regression on the
   calibration participants.
4. Apply the fitted attack to the disjoint evaluation participants.
5. Predict membership at score >= 0.5 and compute participant-level metrics.

The label-aware variant adds the participant's binary cross-entropy loss to the
feature set.  It models a more informed attacker who also knows the diagnosis;
it is a sensitivity analysis, not the primary threat model.  Having more
information does not guarantee a higher measured attack score on a finite
evaluation sample.

## 6. Federated attack surfaces (kept separate)

| Surface | What the attacker sees | Status in this study |
| --- | --- | --- |
| Final global model | Final aggregated parameters / predictions | **Evaluated** (main comparison) |
| Intermediate global checkpoints | Global parameters after selected FedAvg rounds | **Evaluated as a supplementary trajectory analysis**; each round uses the same final-model-style query attack, but results are not pooled with the final-model row |
| Individual client updates | Per-client parameter deltas or gradients received by the server | **Not evaluated**. The training pipeline does not retain client updates, and an update-level attack has a different threat model (norm/direction/gradient-inversion analysis). It must not be conflated with final-model MIA. |

The DP mechanism in this project is a **local client-update DP-SGD mechanism**.
Its privacy accounting is scoped to the client update, not to the final global
model.  The final-model MIA results therefore do not validate or invalidate the
DP guarantee.  A server that observes individual client updates is outside the
final-model threat model and is not covered by this attack.

The DP configuration specifies participant-level mean-logit aggregation, while
the attack queries the project's slice-evaluation wrapper and aggregates slice
probabilities.  The saved DP checkpoint is a round-0 artifact with no completed
training round, and its outputs are nearly constant, so both views are near
chance.  The DP row is descriptive and not a precise estimate of a particular
deployment interface.

## 7. What the results do and do not mean

* A high attack AUC means the target model's confidence distribution leaks
  membership information to this attacker.
* An attack accuracy close to the member prevalence is **not** evidence of
  strong privacy.  The report therefore emphasizes balanced accuracy,
  TPR-FPR advantage, and TPR at low FPR rather than raw accuracy.
* Federated learning does not automatically prevent MIA.  The FedAvg result is
  an empirical measurement under this protocol, not a privacy proof.
* DP may reduce membership signal, but a DP result on a model with near-chance
  utility is confounded by utility collapse.  The DP checkpoint in this
  repository is a round-0 artifact with no completed DP training round; its
  attack result cannot be attributed to DP alone.
* The non-member set is the validation split, which was also used for
  checkpoint selection in the centralized and federated runs.  Those
  participants are therefore *selection-exposed*, not perfectly held out.  This
  is a limitation and makes the non-member group a conservative proxy for truly
  unseen participants.

## 8. Ethical scope

This is a research evaluation on de-identified OASIS-1 data.  The attack is
implemented to measure privacy risk, not to target individuals.  No attempt is
made to re-identify participants or recover raw images.
"""
    output_path.write_text(text)


def _metric_or_na(
    comparison: pd.DataFrame,
    setting: str,
    column: str,
    fmt: str = ".3f",
) -> str:
    row = comparison.loc[comparison["setting"] == setting]
    if row.empty:
        return "not run"
    value = row.iloc[0][column]
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return format(float(value), fmt)
    return str(value)


def write_report(
    output_path: Path,
    comparison: pd.DataFrame,
    variant_metrics: pd.DataFrame,
    stability_summary: pd.DataFrame,
    dataset_metadata: dict[str, dict[str, Any]],
    intermediate_comparison: pd.DataFrame | None,
    repeats: int,
    bootstrap: int,
    seed: int,
) -> None:
    base_metadata = dataset_metadata.get("centralized") or next(iter(dataset_metadata.values()))
    rows = comparison.copy()
    table = _markdown_table(
        rows,
        [
            "setting",
            "evaluation_participants",
            "evaluation_members",
            "evaluation_nonmembers",
            "majority_class_accuracy",
            "attack_accuracy",
            "attack_balanced_accuracy",
            "attack_roc_auc",
            "attack_advantage",
            "attack_tpr_at_fpr_0.10",
            "attack_advantage_at_fpr_0.10",
        ],
    )

    variant_table = _markdown_table(
        variant_metrics,
        [
            "setting",
            "attack_variant",
            "attack_roc_auc",
            "attack_advantage",
            "attack_balanced_accuracy",
            "attack_accuracy",
            "attack_precision",
            "attack_recall",
            "attack_f1",
        ],
    )

    stability_table = _markdown_table(
        stability_summary,
        ["setting", "metric", "mean", "std", "p2.5", "p97.5"],
    )

    utility_rows = []
    for setting, metadata in dataset_metadata.items():
        validation = metadata.get("validation_metrics") or {}
        utility_rows.append(
            {
                "setting": setting,
                "checkpoint": metadata.get("checkpoint"),
                "round_or_epoch": (
                    metadata.get("checkpoint_round")
                    if metadata.get("checkpoint_round") is not None
                    else metadata.get("checkpoint_epoch")
                ),
                "validation_accuracy": validation.get("val_accuracy", float("nan")),
                "validation_auc": validation.get("val_auc", float("nan")),
                "validation_loss": validation.get("val_loss", float("nan")),
            }
        )
    utility_table = _markdown_table(
        pd.DataFrame(utility_rows),
        [
            "setting",
            "round_or_epoch",
            "validation_accuracy",
            "validation_auc",
            "validation_loss",
        ],
    )

    ci_table = _markdown_table(
        comparison,
        [
            "setting",
            "attack_roc_auc",
            "attack_roc_auc_ci_lower",
            "attack_roc_auc_ci_upper",
            "attack_advantage",
            "attack_advantage_ci_lower",
            "attack_advantage_ci_upper",
        ],
    )

    intermediate_text = ""
    if intermediate_comparison is not None and not intermediate_comparison.empty:
        intermediate_table = _markdown_table(
            intermediate_comparison,
            [
                "round",
                "evaluation_participants",
                "majority_class_accuracy",
                "attack_balanced_accuracy",
                "attack_roc_auc",
                "attack_advantage",
                "attack_tpr_at_fpr_0.10",
            ],
        )
        intermediate_text = f"""
## 9. Intermediate global checkpoints (separate attack surface)

The table below repeats the final-model query attack against each saved global
FedAvg checkpoint.  This is a **separate threat model**: an adversary who can
obtain intermediate global parameters or query intermediate global models.  The
rounds are not pooled with the final-model comparison in Section 7.  Every row
uses the same participant-level calibration/evaluation split as the final-model
analysis, so round 4 (the validation-selected final checkpoint) matches the
FedAvg row in Section 7.

{intermediate_table}

The best final federated checkpoint is selected by validation loss, so its
round may not be the last round.  The trajectory is descriptive; the small
number of rounds and the validation-selection effect limit causal
interpretation.
"""

    text = f"""# Membership Inference Against OASIS-1 Alzheimer's Classifiers

**Date:** {date.today().isoformat()}  
**Scope:** centralized ResNet-18, federated FedAvg ResNet-18, and federated
ResNet-18 with participant-level local DP.  
**Privacy unit:** participant (seven slices grouped).  
**Test split:** excluded from attack construction, calibration, tuning, and
evaluation.

## 1. Executive summary

This report replaces the earlier MIA artifacts, which were not methodologically
valid: the implementation did not run, the attack was not class-balanced, the
reported accuracy was dominated by the 82% member prevalence, and the saved
comparison file contained serialized optimizer state rather than clean metrics.
The rebuilt analysis uses participant-level grouping, disjoint
calibration/evaluation sets, class-balanced attacks, prevalence-aware metrics,
bootstrap intervals, repeated-split stability checks, and explicit threat-model
documentation.

The main findings are:

* The centralized model leaks substantial participant-level membership signal
  (ROC-AUC approximately {_metric_or_na(comparison, 'centralized', 'attack_roc_auc')}).
* The FedAvg global model leaks less but is still above chance
  (ROC-AUC approximately {_metric_or_na(comparison, 'federated', 'attack_roc_auc')}).
* The DP checkpoint is near chance
  (ROC-AUC approximately {_metric_or_na(comparison, 'federated_dp', 'attack_roc_auc')});
  however, this checkpoint is a round-0 artifact with no completed DP training
  round and near-chance utility.  The result is confounded by utility collapse
  and is **not** evidence that DP prevents all membership inference.
* Raw accuracy is misleading under the approximately 82% member prevalence.  The
  class-balanced attack and TPR-FPR advantage are the primary comparisons.

## 2. Data and participant splits

OASIS-1 contains 235 usable participants after quality control:

| Split | Participants | Role in this study |
| --- | ---: | --- |
| train | {base_metadata['members']} | members for all target models |
| validation | {base_metadata['nonmembers']} | non-members; also used for target-model checkpoint selection |
| test | 36 | excluded from the attack entirely |

Each participant contributes the same seven fixed axial slices
({', '.join(map(str, [20, 40, 60, 78, 100, 120, 140]))}).  The seven slices are
aggregated into one participant representation; they are never counted as seven
independent membership observations.

The validation split is not a perfect non-member set because it influenced
checkpoint selection.  This makes the non-member group selection-exposed and
the attack estimate conservative relative to truly unseen participants.  Using
the test split as a cleaner non-member set would consume the final model test
set, so it is deliberately avoided.

## 3. Attack dataset construction

For each target model and each participant:

1. Load the participant's seven preprocessed slices.
2. Query the target model and obtain the class-1 probability for each slice.
3. Aggregate to participant level:
   * mean class-1 probability;
   * confidence = max(p, 1-p);
   * binary entropy of p;
   * mean slice confidence;
   * slice-probability standard deviation, minimum, and maximum;
   * label-aware sensitivity feature: binary cross-entropy loss using the true
     diagnosis.
4. Attach membership status (`train` = 1, `validation` = 0).

The resulting participant-level dataset has {base_metadata['total_participants']}
rows per target model.  The full dataset is saved as `attack_dataset.csv` and
per target under `<setting>/attack_dataset.csv`.

All three targets are queried through the same slice-level interface and
aggregated by the mean class-1 probability over the seven slices.  The DP model
was trained with mean-logit aggregation, so querying slice probabilities is not
identical to its training-time participant aggregation.  The attack uses the
same wrapper as the project's DP utility evaluation, and the DP model's outputs
are nearly constant, so the qualitative near-chance interpretation is unchanged;
the difference is documented as a limitation rather than treated as a precise
DP effect estimate.

## 4. Attack protocol

* **Primary attack:** class-balanced logistic regression on label-agnostic
  confidence features.  Features are standardized before fitting.
* **Calibration/evaluation split:** 50/50 by participant, stratified by
  membership status and diagnosis class, seed {seed}.  The split is
  disjoint at participant level.
* **Prediction threshold:** 0.5 on the attack score P(member).
* **Sensitivity attack:** label-aware logistic regression that adds the
  participant loss.
* **Stability:** the 50/50 split is repeated {repeats} times with
  different seeds; metrics are summarized by mean, standard deviation, and
  2.5/97.5 percentiles.
* **Uncertainty:** stratified participant-level bootstrap with
  {bootstrap} resamples for the primary evaluation metrics.
* **Test set:** never loaded by any attack code path.

## 5. Metrics

* **Accuracy:** (TP+TN)/N.  Reported with the majority-class baseline because
  members are 82% of the evaluation pool.
* **Precision, recall, F1:** threshold-dependent attack performance.
* **Balanced accuracy:** mean of TPR and TNR; robust to the 82/18 imbalance.
* **ROC-AUC:** probability that a randomly chosen member receives a higher
  attack score than a randomly chosen non-member.
* **Attack advantage:** TPR - FPR at the 0.5 threshold.
* **AUC advantage:** 2*AUC - 1.
* **TPR at FPR = 1%, 5%, 10%:** privacy-relevant low-false-positive operating
  points.  The advantage at FPR=10% is TPR@10% - 0.10.

## 6. Target-model utility (context for privacy results)

Privacy risk must be interpreted alongside model utility.  The table reports
the target checkpoint's own validation metrics (35 participants).  These are
provided for context only; the attack does not use them for fitting.

{utility_table}

The centralized and FedAvg models have modest diagnostic utility.  The DP
artifact is a round-0 checkpoint: only the initial global model was saved, with
no post-training round-1 checkpoint and no client DP metric files.  It has
near-chance validation utility, so its near-chance MIA result is confounded by
utility collapse.  Its participant probabilities are nearly constant (mean
approximately 0.77 for both members and non-members), leaving almost no
confidence variation for the attack to exploit.  The available DP artifacts do
not record epsilon; the DP configuration is participant-level local DP-SGD with
noise multiplier 0.5, max gradient norm 1.0, delta 1e-5, one configured round,
and one local epoch.  No formal end-to-end DP claim is made here.

## 7. Main results

{table}

**Interpretation.** The centralized model's confidence distribution is highly
separable: members are usually predicted with high confidence in either class,
while non-members are closer to the decision boundary.  The FedAvg model
preserves some of this signal, but substantially less.  The DP row is close to
chance, but the DP artifact itself has near-chance validation utility and no
completed training round, so the row primarily demonstrates that an incomplete
round-0 artifact provides little membership signal; it is not a clean
DP-vs-non-DP causal comparison.

Raw accuracy must be read against the majority baseline.  For example, an
accuracy of 0.82 with 82% members is exactly the trivial all-member baseline.
Balanced accuracy and TPR-FPR advantage are therefore the primary evidence.

### 7.1 Bootstrap uncertainty

{ci_table}

Intervals are stratified participant-level bootstrap 95% intervals.  They
quantify uncertainty from the finite number of validation non-members; they do
not address dataset shift or the selection-exposed non-member limitation.

### 7.2 Attack-variant sensitivity

{variant_table}

The label-aware variant models an attacker who also knows the participant's
diagnosis.  Although it uses strictly more information, it does not dominate the
label-agnostic attack on this finite evaluation set: for FedAvg and the DP
checkpoint its measured AUC and advantage are lower, while for the centralized
model the two variants are effectively tied.  The added loss feature is highly
correlated with the confidence features and adds noise on this small sample.
The primary conclusions therefore rely on the label-agnostic attacker, and the
label-aware results are reported as a sensitivity check rather than as the
headline risk estimate.

### 7.3 Repeated-split stability

{stability_table}

The repeated splits show that the ordering of the three target models is stable
and is not an artifact of the single primary split.  The repeated-split means are
somewhat lower than the single primary split for FedAvg and the DP checkpoint,
so the primary split is not the most conservative estimate; the ordering of the
targets is unchanged.

## 8. ROC curves

* `roc_comparison.svg` / `roc_comparison.png`: overlay of the three main ROC
  curves.
* `<setting>/attack_roc.svg` / `.png`: per-target ROC curve with the
  TPR@FPR=10% operating point.
* `<setting>/attack_score_distribution.svg` / `.png`: attack-score distributions
  for members and non-members.

![Overlaid ROC curves for the three target models](roc_comparison.png)

![ROC-AUC and low-FPR advantage by target model](mia_metric_comparison.png)

{intermediate_text}

## 10. Federated attack surfaces

The main comparison targets the **final global model**.  Intermediate global
checkpoints are analyzed separately in the trajectory table above.  Individual
client updates are **not** analyzed here: the training pipeline does not retain
per-client parameter deltas, and update-level attacks require a different
threat model (access to client updates, update norms/directions, and potentially
gradient inversion).  Combining the final-model and client-update threat models
without separation would be misleading.

In the FedAvg run, all four clients participate in every round
(`fraction_fit = 1.0`), and the four partitions together cover all 164 training
participants with no overlap.  Every training participant is therefore a member
of the final global model under this protocol; client sampling does not create a
subset of non-contributing "members".

The DP mechanism is a local participant-level DP-SGD mechanism.  Its guarantee
is scoped to the client update under the accountant's assumptions.  FedAvg does
not provide secure aggregation, so a server that observes individual updates is
outside the final-model threat model.  The final-model MIA results should not be
used to claim that the DP mechanism is effective or ineffective at the update
level.

## 11. Limitations and assumptions

1. **Selection-exposed non-members.** The validation split was used for
   checkpoint selection.  The attack's non-member group is therefore not a
   perfectly held-out population.
2. **Small non-member sample.** Only 35 validation participants are available;
   the 50/50 split leaves about 17-18 non-members for evaluation.  Bootstrap and
   repeated splits quantify but do not eliminate this uncertainty.
3. **Single dataset and architecture.** Results are specific to OASIS-1,
   seven-slice preprocessing, ResNet-18, and these training configurations.
4. **Attack-model choice.** Logistic regression is a defensible confidence
   attack but not an exhaustive search over attack families.  A shadow-model or
   stronger adaptive attack could find more leakage.
5. **Label assumptions.** The primary attack is label-agnostic; the label-aware
   variant models an attacker who also knows the diagnosis.  It uses more
   information but does not improve every measured metric on this finite sample,
   so it is reported only as a sensitivity check.
6. **DP confound.** The DP checkpoint is a round-0 artifact with no completed
   DP training round, near-chance validation utility, and nearly constant
   participant probabilities.  Its near-chance MIA result cannot be attributed
   to DP alone, and no epsilon value was recorded in the available artifacts.
   The DP configuration specifies mean-logit aggregation, while the attack
   queries slice probabilities through the project's DP evaluation wrapper;
   this interface difference is a further reason to treat the DP row as
   descriptive only.
7. **No client-update evaluation.** The server-side update attack is documented
   but not measured because updates were not retained.
8. **No causal claim about FedAvg.** Federated learning changes optimization,
   aggregation, and checkpoint selection simultaneously; the FedAvg-vs-
   centralized difference is an empirical association, not a controlled causal
   effect of federation alone.

## 12. Reproduction

From the project root:

```bash
python -m privacy.membership_inference \\
  --output-dir results/privacy/mia \\
  --seed 42 \\
  --repeats 50 \\
  --bootstrap 2000 \\
  --include-intermediate-federated
```

The command reads only the train and validation splits and writes all artifacts
under `results/privacy/mia/`.

Protocol-level checks can be run with:

```bash
python -m unittest discover -s tests -v
```

These tests verify one row per participant, disjoint calibration/evaluation
roles, rejection of the test split, class-balanced metric behaviour, and
well-formed bootstrap intervals.  They do not retrain the target models.

## 13. Conclusion

Participant-level membership inference succeeds strongly against the
centralized model, moderately against the FedAvg global model, and is near
chance against the available DP checkpoint.  The DP result is confounded by
utility collapse and must not be read as proof that DP prevents membership
inference.  Federated learning reduces but does not eliminate the observed
membership signal.  The appropriate conclusion is a measured, threat-model-
specific risk statement, not a claim of universal privacy or universal
vulnerability.
"""
    output_path.write_text(text)


# ---------------------------------------------------------------------------
# Main experiment driver
# ---------------------------------------------------------------------------


def run_setting(
    spec: TargetSpec,
    output_dir: Path,
    seed: int = DEFAULT_SEED,
    repeats: int = DEFAULT_REPEATS,
    bootstrap: int = DEFAULT_BOOTSTRAP,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Run the full MIA pipeline for one target model."""

    setting_dir = output_dir / spec.setting
    setting_dir.mkdir(parents=True, exist_ok=True)

    dataset, metadata = build_attack_dataset(spec)
    assigned = split_attack_participants(dataset, seed)
    calibration = assigned[assigned["attack_role"] == "calibration"].copy()
    evaluation = assigned[assigned["attack_role"] == "evaluation"].copy()

    variant_rows: list[dict[str, Any]] = []
    predictions = evaluation.copy()
    calibration_predictions = calibration.copy()

    for variant, features in (
        ("confidence_agnostic", CONFIDENCE_FEATURES),
        ("label_aware", LABEL_AWARE_FEATURES),
    ):
        scores, _ = fit_and_score(calibration, evaluation, features, seed)
        predictions[f"attack_score_{variant}"] = scores
        predictions[f"attack_prediction_{variant}"] = (scores >= 0.5).astype(int)

        calibration_scores, _ = fit_and_score(
            calibration, calibration, features, seed, allow_overlap=True
        )
        calibration_predictions[f"attack_score_{variant}"] = calibration_scores
        calibration_predictions[f"attack_prediction_{variant}"] = (
            calibration_scores >= 0.5
        ).astype(int)

        metrics = compute_metrics(
            predictions,
            score_column=f"attack_score_{variant}",
            prediction_column=f"attack_prediction_{variant}",
        )
        variant_rows.append(
            {
                "setting": spec.setting,
                "attack_variant": variant,
                "features": ",".join(features),
                **metrics,
            }
        )

    primary_score = f"attack_score_{PRIMARY_VARIANT}"
    primary_prediction = f"attack_prediction_{PRIMARY_VARIANT}"
    predictions["attack_score"] = predictions[primary_score]
    predictions["attack_prediction"] = predictions[primary_prediction]
    calibration_predictions["attack_score"] = calibration_predictions[primary_score]
    calibration_predictions["attack_prediction"] = calibration_predictions[
        primary_prediction
    ]

    primary_metrics = compute_metrics(
        predictions,
        score_column="attack_score",
        prediction_column="attack_prediction",
    )
    primary_metrics.update(
        bootstrap_confidence_intervals(
            predictions,
            n_bootstrap=bootstrap,
            seed=seed,
            score_column="attack_score",
            prediction_column="attack_prediction",
        )
    )
    primary_metrics.update(
        {
            "setting": spec.setting,
            "checkpoint": str(spec.checkpoint),
            "attack_target": spec.attack_target,
            "attack_method": "participant-level class-balanced logistic confidence attack",
            "attack_variant": PRIMARY_VARIANT,
            "attack_features": ",".join(CONFIDENCE_FEATURES),
            "calibration_participants": int(len(calibration)),
            "calibration_members": int(calibration["membership"].sum()),
            "calibration_nonmembers": int((1 - calibration["membership"]).sum()),
            "test_split_used": False,
            "privacy_unit": "participant",
            "slices_per_participant": int(metadata["slices_per_participant"]),
            "attack_protocol": "stratified 50/50 participant split; disjoint calibration/evaluation",
            "split_seed": seed,
            "primary_variant": PRIMARY_VARIANT,
        }
    )

    # Persist the participant-level dataset and predictions.
    assigned.to_csv(setting_dir / "attack_dataset.csv", index=False)
    predictions.to_csv(setting_dir / "participant_attack_predictions.csv", index=False)
    calibration_predictions.to_csv(
        setting_dir / "attack_calibration_participants.csv", index=False
    )
    pd.DataFrame([primary_metrics]).to_csv(
        setting_dir / "attack_metrics.csv", index=False
    )
    pd.DataFrame(variant_rows).to_csv(
        setting_dir / "attack_metrics_by_variant.csv", index=False
    )

    # Stability of the split.
    stability_repeats, stability_summary = repeated_split_stability(
        dataset,
        CONFIDENCE_FEATURES,
        seed=seed,
        repeats=repeats,
    )
    stability_repeats.insert(0, "setting", spec.setting)
    stability_summary.insert(0, "setting", spec.setting)
    stability_repeats.to_csv(setting_dir / "attack_stability.csv", index=False)
    stability_summary.to_csv(
        setting_dir / "attack_stability_summary.csv", index=False
    )

    # ROC and score distribution.
    plot_roc_curve(
        predictions,
        setting_dir / "attack_roc",
        title=f"Membership inference ROC: {spec.setting}",
    )
    plot_score_distributions(
        {spec.setting: (predictions, primary_metrics)},
        setting_dir / "attack_score_distribution",
    )

    config = {
        "setting": spec.setting,
        "checkpoint": str(spec.checkpoint),
        "attack_target": spec.attack_target,
        "target_description": spec.description,
        "target_metadata": metadata,
        "seed": seed,
        "privacy_unit": "participant",
        "slices_per_participant": int(metadata["slices_per_participant"]),
        "members": "train split",
        "nonmembers": "validation split",
        "test_split_used": False,
        "attack_protocol": "stratified 50/50 participant split; disjoint calibration/evaluation",
        "attack_model": "StandardScaler + class-balanced LogisticRegression(liblinear, C=1.0)",
        "primary_attack_variant": PRIMARY_VARIANT,
        "primary_features": CONFIDENCE_FEATURES,
        "sensitivity_features": LABEL_AWARE_FEATURES,
        "prediction_threshold": 0.5,
        "repeats": repeats,
        "bootstrap_resamples": bootstrap,
        "metrics": "accuracy, precision, recall, F1, balanced accuracy, ROC-AUC, TPR-FPR advantage, TPR at FPR 1/5/10%",
        "federated_attack_surface": (
            "final_global_model"
            if spec.setting == "federated"
            else "final_global_model_with_local_dp"
            if spec.setting == "federated_dp"
            else "intermediate_global_model"
            if spec.setting.startswith("federated_round_")
            else "not_applicable"
        ),
        "intermediate_checkpoints_note": (
            "Intermediate global checkpoints are evaluated separately; "
            "individual client updates are not evaluated."
        ),
    }
    (setting_dir / "attack_config.json").write_text(
        json.dumps(config, indent=2, default=str) + "\n"
    )

    return (
        pd.DataFrame([primary_metrics]),
        metadata,
        {"evaluation": predictions, "metrics": primary_metrics},
        pd.DataFrame(variant_rows),
        stability_summary,
    )


def run_intermediate_federated_checkpoints(
    output_dir: Path,
    seed: int = DEFAULT_SEED,
    repeats: int = DEFAULT_REPEATS,
    bootstrap: int = DEFAULT_BOOTSTRAP,
) -> pd.DataFrame:
    """Evaluate each saved FedAvg global checkpoint as a separate surface."""

    checkpoint_dir = (
        PROJECT_ROOT / "results" / "federated" / "iid" / "checkpoints"
    )
    paths = checkpoint_paths_for_intermediate_rounds(checkpoint_dir)
    rows: list[dict[str, Any]] = []
    intermediate_root = output_dir / "federated_checkpoints"
    for path in paths:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        server_round = checkpoint.get("round", int(path.stem.split("_")[-1]))
        spec = TargetSpec(
            setting=f"federated_round_{int(server_round):03d}",
            checkpoint=path,
            attack_target="intermediate_global_model",
            model_kind="resnet18",
            description=f"FedAvg global checkpoint after round {server_round}.",
        )
        metrics, _, _, _, _ = run_setting(
            spec,
            intermediate_root,
            seed=seed,
            repeats=min(repeats, 20),
            bootstrap=bootstrap,
        )
        row = metrics.iloc[0].to_dict()
        row["round"] = int(server_round)
        row["validation_participants"] = int(
            checkpoint.get("validation_metrics", {}).get("participants", 0)
        )
        row["validation_accuracy"] = float(
            checkpoint.get("validation_metrics", {}).get("val_accuracy", float("nan"))
        )
        row["validation_auc"] = float(
            checkpoint.get("validation_metrics", {}).get("val_auc", float("nan"))
        )
        rows.append(row)
    comparison = pd.DataFrame(rows).sort_values("round").reset_index(drop=True)
    comparison.to_csv(
        intermediate_root / "federated_checkpoint_comparison.csv", index=False
    )
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    parser.add_argument(
        "--settings",
        nargs="+",
        choices=sorted(TARGETS),
        default=sorted(TARGETS),
        help="Target models to evaluate.",
    )
    parser.add_argument(
        "--include-intermediate-federated",
        action="store_true",
        help="Also evaluate each saved Federated FedAvg round checkpoint.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)

    comparison_rows: list[pd.DataFrame] = []
    variant_frames: list[pd.DataFrame] = []
    stability_frames: list[pd.DataFrame] = []
    dataset_metadata: dict[str, dict[str, Any]] = {}
    plot_results: dict[str, tuple[pd.DataFrame, dict[str, Any]]] = {}

    for setting in args.settings:
        spec = TARGETS[setting]
        print(f"\n=== Running MIA against {setting} ===")
        metrics, metadata, plot_payload, variants, stability = run_setting(
            spec,
            args.output_dir,
            seed=args.seed,
            repeats=args.repeats,
            bootstrap=args.bootstrap,
        )
        comparison_rows.append(metrics)
        variant_frames.append(variants)
        stability_frames.append(stability)
        dataset_metadata[setting] = metadata
        plot_results[setting] = (
            plot_payload["evaluation"],
            plot_payload["metrics"],
        )

    comparison = pd.concat(comparison_rows, ignore_index=True)
    comparison.to_csv(args.output_dir / "mia_comparison.csv", index=False)

    variant_metrics = pd.concat(variant_frames, ignore_index=True)
    variant_metrics.to_csv(args.output_dir / "mia_attack_variants.csv", index=False)

    stability_summary = pd.concat(stability_frames, ignore_index=True)
    stability_summary.to_csv(
        args.output_dir / "mia_stability_summary.csv", index=False
    )

    # Combined participant-level attack dataset for all main target models.
    combined = []
    for setting in args.settings:
        path = args.output_dir / setting / "attack_dataset.csv"
        frame = pd.read_csv(path)
        frame["setting"] = setting
        combined.append(frame)
    pd.concat(combined, ignore_index=True).to_csv(
        args.output_dir / "attack_dataset.csv", index=False
    )
    (args.output_dir / "attack_dataset_manifest.json").write_text(
        json.dumps(dataset_metadata, indent=2, default=str) + "\n"
    )

    plot_roc_comparison(plot_results, args.output_dir / "roc_comparison")
    plot_metric_comparison(comparison, args.output_dir / "mia_metric_comparison")

    intermediate_comparison = None
    if args.include_intermediate_federated:
        print("\n=== Running MIA against intermediate federated checkpoints ===")
        intermediate_comparison = run_intermediate_federated_checkpoints(
            args.output_dir,
            seed=args.seed,
            repeats=args.repeats,
            bootstrap=args.bootstrap,
        )

    write_threat_model(
        args.output_dir / "threat_model.md",
        dataset_metadata,
        comparison,
    )
    write_report(
        args.output_dir / "report.md",
        comparison,
        variant_metrics,
        stability_summary,
        dataset_metadata,
        intermediate_comparison,
        repeats=args.repeats,
        bootstrap=args.bootstrap,
        seed=args.seed,
    )

    readme = """# MIA artifacts

| File | Meaning |
| --- | --- |
| `attack_dataset.csv` | Participant-level attack dataset for the three main targets. |
| `attack_dataset_manifest.json` | Construction metadata for each target. |
| `mia_comparison.csv` | Headline metrics for the three targets. |
| `mia_attack_variants.csv` | Label-agnostic vs label-aware attack metrics. |
| `mia_stability_summary.csv` | Repeated-split stability summary. |
| `roc_comparison.svg` / `.png` | Overlaid ROC curves for the three targets. |
| `mia_metric_comparison.svg` / `.png` | Bar chart of AUC and low-FPR advantage. |
| `threat_model.md` | Threat model, attacker access, membership definition, federated surfaces. |
| `report.md` | Full research report, results, limitations, and reproduction command. |
| `<setting>/` | Per-target dataset, predictions, metrics, ROC, stability, and config. |
| `federated_checkpoints/` | Supplementary intermediate-global-checkpoint analysis, when requested. |

## Validation

```bash
python -m unittest discover -s tests -v
```

The protocol tests check one row per participant, disjoint attack
calibration/evaluation roles, rejection of the test split, metric sanity, and
bootstrap interval shape. They do not retrain the target models.
"""
    (args.output_dir / "README.md").write_text(readme)

    print("\n=== Main comparison ===")
    print(comparison.to_string(index=False))
    print(f"\nArtifacts written to {args.output_dir}")


if __name__ == "__main__":
    main()
