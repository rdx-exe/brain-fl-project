import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    log_loss
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Make project root importable when this script is run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.pytorch_dataset import create_dataloaders
from models.resnet18_model import build_resnet18


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42

NUM_CLASSES = 2
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

MAX_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 7

PRETRAINED = False

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "centralized"
)

CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"

BEST_CHECKPOINT = CHECKPOINT_DIR / "best_model.pth"
HISTORY_PATH = LOG_DIR / "training_history.csv"


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Prefer reproducible operations where supported.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# DEVICE CONFIGURATION
# ============================================================

def get_device():

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    return device


# ============================================================
# TRAINING ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):

    model.train()

    running_loss = 0.0
    total_samples = 0

    for images, labels, participant_ids in loader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)

        loss = criterion(
            logits,
            labels
        )

        loss.backward()

        optimizer.step()

        batch_size = images.size(0)

        running_loss += loss.item() * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise RuntimeError("Training loader is empty.")

    epoch_loss = running_loss / total_samples

    return epoch_loss


# ============================================================
# PARTICIPANT-LEVEL PREDICTION AGGREGATION
# ============================================================

def aggregate_participant_predictions(
    model,
    loader,
    device
):
    """
    Collect slice probabilities and aggregate them by participant.

    Each participant must have exactly seven slices.

    Aggregation:
        Mean probability of class 1 across selected slices.

    Returns:
        participant_ids
        participant_labels
        participant_probabilities
    """

    model.eval()

    participant_records = {}

    with torch.no_grad():

        for images, labels, participant_ids in loader:

            images = images.to(
                device,
                non_blocking=True
            )

            logits = model(images)

            probabilities = torch.softmax(
                logits,
                dim=1
            )[:, 1]

            probabilities = probabilities.cpu().numpy()

            labels = labels.cpu().numpy()

            for participant_id, label, probability in zip(
                participant_ids,
                labels,
                probabilities
            ):

                if participant_id not in participant_records:

                    participant_records[participant_id] = {
                        "label": int(label),
                        "probabilities": []
                    }

                record = participant_records[participant_id]

                if record["label"] != int(label):

                    raise ValueError(
                        f"Inconsistent labels for participant "
                        f"{participant_id}"
                    )

                record["probabilities"].append(
                    float(probability)
                )

    participant_ids = []
    participant_labels = []
    participant_probabilities = []

    expected_slices = 7

    for participant_id, record in participant_records.items():

        probabilities = record["probabilities"]

        if len(probabilities) != expected_slices:

            raise ValueError(
                f"Participant {participant_id} has "
                f"{len(probabilities)} predictions; "
                f"expected {expected_slices}."
            )

        participant_ids.append(participant_id)

        participant_labels.append(
            record["label"]
        )

        participant_probabilities.append(
            float(np.mean(probabilities))
        )

    if not participant_ids:
        raise RuntimeError(
            "No participant predictions were generated."
        )

    return (
        participant_ids,
        np.asarray(participant_labels, dtype=np.int64),
        np.asarray(participant_probabilities, dtype=np.float64)
    )


# ============================================================
# PARTICIPANT-LEVEL VALIDATION
# ============================================================

def validate_model(
    model,
    loader,
    device
):

    (
        participant_ids,
        labels,
        probabilities
    ) = aggregate_participant_predictions(
        model,
        loader,
        device
    )

    predictions = (
        probabilities >= 0.5
    ).astype(np.int64)

    participant_loss = log_loss(
        labels,
        probabilities,
        labels=[0, 1]
    )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    if len(np.unique(labels)) == 2:

        auc = roc_auc_score(
            labels,
            probabilities
        )

    else:
        auc = float("nan")

    metrics = {
        "val_loss": float(participant_loss),
        "val_accuracy": float(accuracy),
        "val_auc": float(auc),
        "participants": len(participant_ids)
    }

    return metrics


# ============================================================
# CHECKPOINT SAVING
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    epoch,
    validation_metrics,
    checkpoint_path
):

    checkpoint = {

        "epoch": epoch,

        "model_state_dict": model.state_dict(),

        "optimizer_state_dict": optimizer.state_dict(),

        "validation_metrics": validation_metrics,

        "model_name": "ResNet18",

        "num_classes": NUM_CLASSES,

        "pretrained": PRETRAINED,

        "random_seed": RANDOM_SEED

    }

    torch.save(
        checkpoint,
        checkpoint_path
    )


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def run_training(max_epochs=MAX_EPOCHS):

    if max_epochs <= 0:
        raise ValueError(
            "Number of epochs must be greater than zero."
        )

    set_seed(RANDOM_SEED)

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    device = get_device()

    print("=" * 70)
    print("CENTRALIZED OASIS-1 RESNET-18 TRAINING")
    print("=" * 70)

    print("\nDevice:", device)
    print("Random seed:", RANDOM_SEED)
    print("Maximum epochs:", max_epochs)
    print("Learning rate:", LEARNING_RATE)
    print("Weight decay:", WEIGHT_DECAY)
    print("Pretrained:", PRETRAINED)

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    (
        train_dataset,
        validation_dataset,
        test_dataset,
        train_loader,
        validation_loader,
        test_loader
    ) = create_dataloaders()

    print("\nTraining participants:", len(train_dataset.report))
    print("Validation participants:", len(validation_dataset.report))
    print("Test participants:", len(test_dataset.report))

    # We deliberately do not iterate over test_loader here.
    # The test split remains untouched until final evaluation.

    del test_dataset
    del test_loader

    # --------------------------------------------------------
    # Initialize model
    # --------------------------------------------------------

    model = build_resnet18(
        num_classes=NUM_CLASSES,
        pretrained=PRETRAINED
    )

    model = model.to(device)

    # --------------------------------------------------------
    # Loss and optimizer
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2
    )

    # --------------------------------------------------------
    # Training state
    # --------------------------------------------------------

    best_validation_loss = float("inf")

    epochs_without_improvement = 0

    history = []

    # --------------------------------------------------------
    # Epoch loop
    # --------------------------------------------------------

    for epoch in range(1, max_epochs + 1):

        print("\n" + "-" * 70)

        print(f"EPOCH {epoch}/{max_epochs}")

        print("-" * 70)

        # Training uses slice-level batches.
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        # Validation metrics are participant-level.
        validation_metrics = validate_model(
            model,
            validation_loader,
            device
        )

        scheduler.step(
            validation_metrics["val_loss"]
        )

        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Training loss: {train_loss:.5f}")

        print(
            "Validation participant loss:",
            f"{validation_metrics['val_loss']:.5f}"
        )

        print(
            "Validation participant accuracy:",
            f"{validation_metrics['val_accuracy']:.4f}"
        )

        print(
            "Validation participant ROC-AUC:",
            f"{validation_metrics['val_auc']:.4f}"
        )

        print(
            "Current learning rate:",
            f"{current_lr:.8f}"
        )

        # ----------------------------------------------------
        # Record history
        # ----------------------------------------------------

        epoch_record = {

            "epoch": epoch,

            "train_loss": train_loss,

            "val_loss": validation_metrics["val_loss"],

            "val_accuracy": validation_metrics["val_accuracy"],

            "val_auc": validation_metrics["val_auc"],

            "learning_rate": current_lr

        }

        history.append(epoch_record)

        pd.DataFrame(history).to_csv(
            HISTORY_PATH,
            index=False
        )

        # ----------------------------------------------------
        # Best checkpoint
        # ----------------------------------------------------

        if (
            validation_metrics["val_loss"]
            < best_validation_loss
        ):

            best_validation_loss = (
                validation_metrics["val_loss"]
            )

            epochs_without_improvement = 0

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                validation_metrics=validation_metrics,
                checkpoint_path=BEST_CHECKPOINT
            )

            print("\nBest checkpoint saved.")

        else:

            epochs_without_improvement += 1

            print(
                "\nNo validation-loss improvement:",
                epochs_without_improvement
            )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):

            print(
                "\nEarly stopping triggered."
            )

            break

    # --------------------------------------------------------
    # Training completion
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print("CENTRALIZED TRAINING COMPLETED")

    print("=" * 70)

    print("\nBest validation loss:", best_validation_loss)

    print("Best checkpoint:", BEST_CHECKPOINT)

    print("Training history:", HISTORY_PATH)

    print("Epochs completed:", len(history))

    return history


# ============================================================
# COMMAND LINE INTERFACE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Train centralized OASIS-1 ResNet-18."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=MAX_EPOCHS,
        help="Maximum number of training epochs."
    )

    args = parser.parse_args()

    run_training(
        max_epochs=args.epochs
    )


if __name__ == "__main__":
    main()