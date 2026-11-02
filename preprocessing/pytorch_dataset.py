from pathlib import Path
from collections import OrderedDict, Counter

import numpy as np
import pandas as pd
import nibabel as nib
import torch

from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPORT_PATH = (
    PROJECT_ROOT
    / "results"
    / "quality_control"
    / "extraction_report.csv"
)

# Seven representative axial slices.
SELECTED_SLICES = [20, 40, 60, 78, 100, 120, 140]

IMAGE_SIZE = 224
BATCH_SIZE = 8
NUM_WORKERS = 0
CACHE_SIZE = 2

# Labels:
# 0 = CDR 0 (cognitively normal)
# 1 = CDR > 0 (CDR-positive)


# ============================================================
# MRI VOLUME LOADING
# ============================================================

def load_mri_volume(image_path):
    """
    Load an Analyze/NIfTI-compatible MRI volume.

    Supports:
        (X, Y, Z)
        (X, Y, Z, 1)

    Returns:
        float32 NumPy array with shape (X, Y, Z).
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"MRI image not found: {image_path}"
        )

    image = nib.load(str(image_path))

    volume = np.asarray(
        image.dataobj,
        dtype=np.float32
    )

    if volume.ndim == 4 and volume.shape[-1] == 1:
        volume = volume[..., 0]

    if volume.ndim != 3:
        raise ValueError(
            "Expected a 3D MRI volume or singleton-channel "
            f"4D volume. Received shape: {volume.shape}"
        )

    if not np.isfinite(volume).all():
        raise ValueError(
            f"MRI contains NaN or infinite values: {image_path}"
        )

    return volume


# ============================================================
# MRI INTENSITY NORMALIZATION
# ============================================================

def normalize_slice(slice_data):
    """
    Normalize one MRI slice.

    Processing:
        1. Identify non-zero voxels.
        2. Calculate the 1st and 99.5th percentiles.
        3. Clip extreme intensities.
        4. Scale non-zero intensities to [0, 1].
        5. Preserve zero-valued background.
    """

    slice_data = np.asarray(
        slice_data,
        dtype=np.float32
    )

    normalized = np.zeros_like(
        slice_data,
        dtype=np.float32
    )

    brain_mask = slice_data > 0

    if not brain_mask.any():
        return normalized

    brain_values = slice_data[brain_mask]

    lower = np.percentile(brain_values, 1)
    upper = np.percentile(brain_values, 99.5)

    if upper <= lower:
        return normalized

    clipped = np.clip(
        slice_data,
        lower,
        upper
    )

    normalized[brain_mask] = (
        (clipped[brain_mask] - lower)
        / (upper - lower)
    )

    return normalized


# ============================================================
# PYTORCH DATASET
# ============================================================

class OASISSliceDataset(Dataset):

    def __init__(
        self,
        report_path,
        split,
        selected_slices=None,
        image_size=224,
        cache_size=2
    ):

        self.report_path = Path(report_path)
        self.split = split
        self.image_size = image_size
        self.cache_size = cache_size

        self.volume_cache = OrderedDict()

        self.selected_slices = (
            list(selected_slices)
            if selected_slices is not None
            else list(SELECTED_SLICES)
        )

        # ----------------------------------------------------
        # Validate configuration
        # ----------------------------------------------------

        if not self.selected_slices:
            raise ValueError(
                "At least one slice must be selected."
            )

        if len(set(self.selected_slices)) != len(
            self.selected_slices
        ):
            raise ValueError(
                "Duplicate slice indices detected."
            )

        if any(
            not isinstance(index, (int, np.integer))
            or index < 0
            for index in self.selected_slices
        ):
            raise ValueError(
                "Slice indices must be non-negative integers."
            )

        if self.image_size <= 0:
            raise ValueError(
                "Image size must be greater than zero."
            )

        if self.cache_size < 0:
            raise ValueError(
                "Cache size cannot be negative."
            )

        # ----------------------------------------------------
        # Load extraction report
        # ----------------------------------------------------

        if not self.report_path.exists():
            raise FileNotFoundError(
                f"Extraction report not found: {self.report_path}"
            )

        self.report = pd.read_csv(self.report_path)

        required_columns = [
            "participant_id",
            "session_id",
            "split",
            "label",
            "image_path",
            "status"
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in self.report.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing columns in extraction report: "
                f"{missing_columns}"
            )

        # ----------------------------------------------------
        # Filter requested split
        # ----------------------------------------------------

        self.report = self.report[
            (self.report["split"] == split)
            & (self.report["status"] == "PASSED")
        ].copy()

        if self.report.empty:
            raise ValueError(
                f"No valid records found for split: {split}"
            )

        # ----------------------------------------------------
        # Validate metadata
        # ----------------------------------------------------

        if self.report[
            ["participant_id", "session_id", "image_path"]
        ].isnull().any().any():

            raise ValueError(
                f"Missing participant, session, or image path "
                f"in {split} records."
            )

        self.report["participant_id"] = (
            self.report["participant_id"].astype(str)
        )

        self.report["session_id"] = (
            self.report["session_id"].astype(str)
        )

        self.report["label"] = pd.to_numeric(
            self.report["label"],
            errors="raise"
        ).astype(int)

        if not self.report["label"].isin([0, 1]).all():
            raise ValueError(
                "Labels must contain only 0 and 1."
            )

        if self.report["participant_id"].duplicated().any():
            raise ValueError(
                f"Multiple records found for a participant "
                f"in the {split} split."
            )

        # ----------------------------------------------------
        # Validate label consistency
        # ----------------------------------------------------

        label_counts = (
            self.report.groupby("participant_id")["label"]
            .nunique()
        )

        if (label_counts > 1).any():
            raise ValueError(
                "Inconsistent labels found for participants."
            )

        # ----------------------------------------------------
        # Expand participants into slice records
        # ----------------------------------------------------

        self.samples = []

        for _, record in self.report.iterrows():

            for slice_index in self.selected_slices:

                self.samples.append({
                    "participant_id": record["participant_id"],
                    "session_id": record["session_id"],
                    "label": int(record["label"]),
                    "image_path": record["image_path"],
                    "slice_index": int(slice_index)
                })

        # ----------------------------------------------------
        # Verify slice count per participant
        # ----------------------------------------------------

        slice_counts = Counter(
            sample["participant_id"]
            for sample in self.samples
        )

        expected_slices = len(self.selected_slices)

        invalid_counts = {
            participant: count
            for participant, count in slice_counts.items()
            if count != expected_slices
        }

        if invalid_counts:
            raise ValueError(
                "Unexpected slice count for participants: "
                f"{invalid_counts}"
            )

        print(
            f"{split.capitalize()} dataset initialized: "
            f"{len(self.samples)} slices from "
            f"{len(self.report)} participants"
        )

    # ========================================================
    # VOLUME CACHE
    # ========================================================

    def get_volume(self, image_path):

        image_path = str(image_path)

        if image_path in self.volume_cache:

            self.volume_cache.move_to_end(image_path)

            return self.volume_cache[image_path]

        absolute_path = PROJECT_ROOT / image_path

        volume = load_mri_volume(absolute_path)

        # Validate selected slice indices against volume depth.
        if max(self.selected_slices) >= volume.shape[2]:

            raise IndexError(
                f"Selected slice index {max(self.selected_slices)} "
                f"exceeds volume depth {volume.shape[2]} "
                f"for {image_path}"
            )

        if self.cache_size > 0:

            self.volume_cache[image_path] = volume

            if len(self.volume_cache) > self.cache_size:

                self.volume_cache.popitem(last=False)

        return volume

    # ========================================================
    # IMAGE PREPARATION
    # ========================================================

    def prepare_slice(self, slice_data):

        normalized = normalize_slice(slice_data)

        # Convert to tensor: (1, H, W).
        image = torch.from_numpy(
            normalized.copy()
        ).unsqueeze(0)

        # Resize to 224 x 224.
        image = F.interpolate(
            image.unsqueeze(0),
            size=(
                self.image_size,
                self.image_size
            ),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        # Convert grayscale to three channels for ResNet-18.
        image = image.repeat(3, 1, 1)

        return image.float()

    # ========================================================
    # DATASET LENGTH
    # ========================================================

    def __len__(self):

        return len(self.samples)

    # ========================================================
    # GET ITEM
    # ========================================================

    def __getitem__(self, index):

        sample = self.samples[index]

        volume = self.get_volume(
            sample["image_path"]
        )

        slice_index = sample["slice_index"]

        # Axis 2 is the axial slice axis in this project.
        slice_data = volume[:, :, slice_index]

        image = self.prepare_slice(slice_data)

        label = torch.tensor(
            sample["label"],
            dtype=torch.long
        )

        participant_id = sample["participant_id"]

        # Return participant identity for participant-level evaluation.
        return image, label, participant_id


# ============================================================
# DATALOADER CREATION
# ============================================================

def create_dataloaders():

    train_dataset = OASISSliceDataset(
        report_path=REPORT_PATH,
        split="train",
        selected_slices=SELECTED_SLICES,
        image_size=IMAGE_SIZE,
        cache_size=CACHE_SIZE
    )

    validation_dataset = OASISSliceDataset(
        report_path=REPORT_PATH,
        split="validation",
        selected_slices=SELECTED_SLICES,
        image_size=IMAGE_SIZE,
        cache_size=CACHE_SIZE
    )

    test_dataset = OASISSliceDataset(
        report_path=REPORT_PATH,
        split="test",
        selected_slices=SELECTED_SLICES,
        image_size=IMAGE_SIZE,
        cache_size=CACHE_SIZE
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    return (
        train_dataset,
        validation_dataset,
        test_dataset,
        train_loader,
        validation_loader,
        test_loader
    )


# ============================================================
# DATASET INTEGRITY VALIDATION
# ============================================================

def validate_dataset_integrity(dataset, split_name):

    print(
        f"\n========== {split_name.upper()} INTEGRITY CHECK =========="
    )

    participant_ids = dataset.report["participant_id"].tolist()

    assert len(participant_ids) == len(set(participant_ids)), (
        "Duplicate participant records detected."
    )

    expected_slices = len(dataset.selected_slices)

    participant_slice_counts = Counter(
        sample["participant_id"]
        for sample in dataset.samples
    )

    assert all(
        count == expected_slices
        for count in participant_slice_counts.values()
    ), "Unexpected number of slices per participant."

    participant_labels = {}

    for sample in dataset.samples:

        participant_id = sample["participant_id"]
        label = sample["label"]

        if participant_id in participant_labels:

            assert participant_labels[participant_id] == label, (
                f"Inconsistent label for {participant_id}"
            )

        participant_labels[participant_id] = label

    print("Unique participants:", len(participant_ids))
    print("Slices per participant:", expected_slices)
    print("Total samples:", len(dataset))
    print("Participant labels: consistent")
    print("Dataset integrity: PASSED")


# ============================================================
# BATCH VALIDATION
# ============================================================

def validate_batch(loader, split_name):

    print(
        f"\n========== {split_name.upper()} BATCH VALIDATION =========="
    )

    images, labels, participant_ids = next(iter(loader))

    print("Image tensor shape:", tuple(images.shape))
    print("Label tensor shape:", tuple(labels.shape))
    print("Participant IDs in batch:", list(participant_ids))

    print("Image data type:", images.dtype)
    print("Label data type:", labels.dtype)

    print("Minimum image intensity:", images.min().item())
    print("Maximum image intensity:", images.max().item())

    print("Labels in batch:", torch.unique(labels).tolist())

    assert images.ndim == 4
    assert images.shape[1] == 3
    assert images.shape[2] == IMAGE_SIZE
    assert images.shape[3] == IMAGE_SIZE

    assert labels.ndim == 1
    assert len(participant_ids) == images.shape[0]

    assert torch.isfinite(images).all()

    assert images.min() >= 0
    assert images.max() <= 1

    assert set(
        torch.unique(labels).tolist()
    ).issubset({0, 1})

    print(f"{split_name.capitalize()} batch validation PASSED.")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("OASIS-1 PYTORCH DATASET VALIDATION")
    print("=" * 65)

    print("\nPyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    (
        train_dataset,
        validation_dataset,
        test_dataset,
        train_loader,
        validation_loader,
        test_loader
    ) = create_dataloaders()

    # --------------------------------------------------------
    # Dataset sizes
    # --------------------------------------------------------

    print("\n========== DATASET SIZES ==========")

    for name, dataset in [
        ("Training", train_dataset),
        ("Validation", validation_dataset),
        ("Test", test_dataset)
    ]:

        print(
            f"{name} participants:",
            dataset.report["participant_id"].nunique()
        )

        print(
            f"{name} slices:",
            len(dataset)
        )

    print("\n========== DATALOADER SIZES ==========")

    print("Training batches:", len(train_loader))
    print("Validation batches:", len(validation_loader))
    print("Test batches:", len(test_loader))

    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    print("\n========== CLASS DISTRIBUTION ==========")

    for name, dataset in [
        ("Train", train_dataset),
        ("Validation", validation_dataset),
        ("Test", test_dataset)
    ]:

        distribution = (
            dataset.report["label"]
            .value_counts()
            .sort_index()
        )

        print(f"\n{name}:")

        print("Class 0:", int(distribution.get(0, 0)))
        print("Class 1:", int(distribution.get(1, 0)))

    # --------------------------------------------------------
    # Integrity checks
    # --------------------------------------------------------

    validate_dataset_integrity(
        train_dataset,
        "train"
    )

    validate_dataset_integrity(
        validation_dataset,
        "validation"
    )

    validate_dataset_integrity(
        test_dataset,
        "test"
    )

    # --------------------------------------------------------
    # Participant isolation check
    # --------------------------------------------------------

    train_ids = set(
        train_dataset.report["participant_id"]
    )

    validation_ids = set(
        validation_dataset.report["participant_id"]
    )

    test_ids = set(
        test_dataset.report["participant_id"]
    )

    assert train_ids.isdisjoint(validation_ids)
    assert train_ids.isdisjoint(test_ids)
    assert validation_ids.isdisjoint(test_ids)

    print("\nParticipant isolation: PASSED")

    # --------------------------------------------------------
    # Batch checks
    # --------------------------------------------------------

    validate_batch(train_loader, "train")
    validate_batch(validation_loader, "validation")
    validate_batch(test_loader, "test")

    print("\n" + "=" * 65)
    print("PYTORCH DATASET VALIDATION COMPLETED SUCCESSFULLY")
    print("=" * 65)


if __name__ == "__main__":
    main()