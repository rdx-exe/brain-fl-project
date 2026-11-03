"""Participant-level attack-dataset construction for membership inference.

The privacy unit is the OASIS-1 participant.  A participant contributes seven
correlated axial slices to the classifier; those slices are never treated as
independent membership observations.  This module queries a trained target
model once per slice, aggregates the seven slice probabilities into one
participant representation, and attaches the participant's membership status.

Membership definition
---------------------
* member (``membership = 1``): participant is in the target model's training
  split;
* non-member (``membership = 0``): participant is in the target model's
  validation split.

The test split is deliberately never loaded.  It is reserved for final model
utility evaluation and must not be used to train, tune, or evaluate the attack.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from federated.dp_server import DPSliceEvaluationModel
from federated.federated_config import get_device
from models.resnet18_model import build_resnet18
from preprocessing.pytorch_dataset import OASISSliceDataset, SELECTED_SLICES

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_EXTRACTION_REPORT = (
    PROJECT_ROOT / "results" / "quality_control" / "extraction_report.csv"
)

EXPECTED_SLICES = len(SELECTED_SLICES)
IMAGE_SIZE = 224
BATCH_SIZE = 16
EPS = 1e-6

# The primary attack is deliberately label-agnostic: an adversary who only sees
# model outputs does not need to know the participant's true diagnosis.
CONFIDENCE_FEATURES = [
    "probability",
    "confidence",
    "entropy",
    "mean_slice_confidence",
    "slice_probability_std",
    "slice_probability_min",
    "slice_probability_max",
]

# Sensitivity analysis: this variant additionally assumes the adversary knows
# the participant's true label and can compute the per-participant loss.
LABEL_AWARE_FEATURES = CONFIDENCE_FEATURES + ["loss"]


@dataclass(frozen=True)
class TargetSpec:
    """A target model and the attack surface it represents."""

    setting: str
    checkpoint: Path
    attack_target: str
    model_kind: str  # "resnet18" or "dp_groupnorm"
    description: str


TARGETS: dict[str, TargetSpec] = {
    "centralized": TargetSpec(
        setting="centralized",
        checkpoint=PROJECT_ROOT / "results" / "centralized" / "checkpoints" / "best_model.pth",
        attack_target="final_centralized_model",
        model_kind="resnet18",
        description="Centralized ResNet-18 trained on the 164-participant training split.",
    ),
    "federated": TargetSpec(
        setting="federated",
        checkpoint=PROJECT_ROOT / "results" / "federated" / "iid" / "best_global_model.pth",
        attack_target="final_global_model",
        model_kind="resnet18",
        description="Final global FedAvg ResNet-18 (best validation-loss round).",
    ),
    "federated_dp": TargetSpec(
        setting="federated_dp",
        checkpoint=PROJECT_ROOT
        / "results"
        / "privacy"
        / "dp_experiments"
        / "dp_nm0.5_c1.0"
        / "best_global_model.pth",
        attack_target="final_global_model_with_local_dp",
        model_kind="dp_groupnorm",
        description=(
            "Final global FedAvg model trained with participant-level local "
            "DP-SGD (Opacus, GroupNorm backbone)."
        ),
    ),
}


def _binary_entropy(probability: np.ndarray | float) -> np.ndarray | float:
    clipped = np.clip(probability, EPS, 1.0 - EPS)
    return -(clipped * np.log(clipped) + (1.0 - clipped) * np.log(1.0 - clipped))


def load_target_model(
    spec: TargetSpec,
    device: torch.device | None = None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load a target checkpoint without reading any participant data."""

    if not spec.checkpoint.exists():
        raise FileNotFoundError(f"Missing target checkpoint: {spec.checkpoint}")

    device = device or get_device()
    checkpoint = torch.load(spec.checkpoint, map_location=device, weights_only=False)

    if spec.model_kind == "dp_groupnorm":
        model: torch.nn.Module = DPSliceEvaluationModel(num_classes=2)
    elif spec.model_kind == "resnet18":
        model = build_resnet18(num_classes=2, pretrained=False)
    else:
        raise ValueError(f"Unsupported target model kind: {spec.model_kind}")

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, checkpoint


def extract_participant_features(
    model: torch.nn.Module,
    split: str,
    extraction_report: Path = DEFAULT_EXTRACTION_REPORT,
    device: torch.device | None = None,
    batch_size: int = BATCH_SIZE,
) -> pd.DataFrame:
    """Query a model on one split and aggregate seven slices per participant.

    The function refuses to load the test split.  It raises if a participant
    does not contribute exactly seven distinct slice predictions or has an
    inconsistent label.
    """

    if split not in {"train", "validation"}:
        raise ValueError(
            "The membership attack may only read the train and validation "
            f"splits; received {split!r}. The test split is reserved."
        )

    device = device or get_device()
    dataset = OASISSliceDataset(
        extraction_report,
        split,
        SELECTED_SLICES,
        IMAGE_SIZE,
        cache_size=2,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    records: dict[str, dict[str, Any]] = {}
    with torch.no_grad():
        for images, labels, participant_ids in loader:
            probabilities = torch.softmax(model(images.to(device)), dim=1)[:, 1]
            probabilities = probabilities.cpu().numpy()
            for participant_id, label, probability in zip(
                participant_ids, labels.numpy(), probabilities
            ):
                participant_id = str(participant_id)
                record = records.setdefault(
                    participant_id,
                    {"label": int(label), "probabilities": []},
                )
                if record["label"] != int(label):
                    raise ValueError(
                        f"Inconsistent labels for participant {participant_id}."
                    )
                record["probabilities"].append(float(probability))

    rows: list[dict[str, Any]] = []
    for participant_id, record in records.items():
        probabilities = np.asarray(record["probabilities"], dtype=np.float64)
        if len(probabilities) != EXPECTED_SLICES:
            raise ValueError(
                f"Participant {participant_id} has {len(probabilities)} slice "
                f"predictions; expected {EXPECTED_SLICES}."
            )

        label = int(record["label"])
        probability = float(probabilities.mean())
        clipped = float(np.clip(probability, EPS, 1.0 - EPS))
        confidence = float(max(clipped, 1.0 - clipped))
        rows.append(
            {
                "participant_id": participant_id,
                "class_label": label,
                "probability": probability,
                "confidence": confidence,
                "entropy": float(_binary_entropy(probability)),
                "loss": float(
                    -(label * np.log(clipped) + (1 - label) * np.log(1 - clipped))
                ),
                "mean_slice_confidence": float(
                    np.mean(np.maximum(probabilities, 1.0 - probabilities))
                ),
                "mean_slice_entropy": float(
                    np.mean(_binary_entropy(probabilities))
                ),
                "slice_probability_std": float(probabilities.std(ddof=0)),
                "slice_probability_min": float(probabilities.min()),
                "slice_probability_max": float(probabilities.max()),
                "slice_probability_range": float(
                    probabilities.max() - probabilities.min()
                ),
                "slice_count": int(len(probabilities)),
                "split": split,
            }
        )

    frame = pd.DataFrame(rows).sort_values("participant_id").reset_index(drop=True)
    if frame.empty:
        raise RuntimeError(f"No participant features were extracted for {split}.")
    return frame


def build_attack_dataset(
    spec: TargetSpec,
    extraction_report: Path = DEFAULT_EXTRACTION_REPORT,
    device: torch.device | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the full participant-level attack dataset for one target model."""

    device = device or get_device()
    model, checkpoint = load_target_model(spec, device=device)

    train = extract_participant_features(
        model, "train", extraction_report=extraction_report, device=device
    )
    validation = extract_participant_features(
        model, "validation", extraction_report=extraction_report, device=device
    )

    train["membership"] = 1
    validation["membership"] = 0

    train_ids = set(train["participant_id"])
    validation_ids = set(validation["participant_id"])
    if train_ids & validation_ids:
        raise ValueError(
            "Member/non-member participant overlap detected: "
            f"{sorted(train_ids & validation_ids)}"
        )

    dataset = pd.concat([train, validation], ignore_index=True)
    dataset["setting"] = spec.setting
    dataset["checkpoint"] = str(spec.checkpoint)
    dataset["attack_target"] = spec.attack_target

    metadata = {
        "setting": spec.setting,
        "checkpoint": str(spec.checkpoint),
        "checkpoint_round": checkpoint.get("round"),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "validation_metrics": checkpoint.get("validation_metrics"),
        "attack_target": spec.attack_target,
        "target_description": spec.description,
        "members": int(train["membership"].sum()),
        "nonmembers": int(len(validation)),
        "total_participants": int(len(dataset)),
        "member_split": "train",
        "nonmember_split": "validation",
        "test_split_used": False,
        "privacy_unit": "participant",
        "slices_per_participant": EXPECTED_SLICES,
        "participant_aggregation": "mean class-1 probability over seven slices",
        "feature_sets": {
            "confidence_agnostic": CONFIDENCE_FEATURES,
            "label_aware": LABEL_AWARE_FEATURES,
        },
    }
    return dataset, metadata


def checkpoint_paths_for_intermediate_rounds(
    checkpoint_dir: Path,
) -> list[Path]:
    """Return sorted round checkpoints from a federated output directory."""

    paths = sorted(checkpoint_dir.glob("round_*.pth"))
    if not paths:
        raise FileNotFoundError(f"No round checkpoints found in {checkpoint_dir}")
    return paths
