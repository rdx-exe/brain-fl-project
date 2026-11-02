
from pathlib import Path
from collections import OrderedDict

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
# These are selected from the previously inspected MRI volumes.
SELECTED_SLICES = [20, 40, 60, 78, 100, 120, 140]

IMAGE_SIZE = 224

BATCH_SIZE = 8

NUM_WORKERS = 0

# Small LRU cache to avoid repeatedly loading the same MRI volume.
CACHE_SIZE = 2

# Labels:
# 0 = CDR 0 (non-demented)
# 1 = CDR > 0 (CDR-positive dementia)


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

    # Remove only a singleton fourth dimension.
    if volume.ndim == 4 and volume.shape[-1] == 1:
        volume = volume[..., 0]

    if volume.ndim != 3:
        raise ValueError(
            f"Expected a 3D MRI volume or singleton-channel 4D volume. "
            f"Received shape: {volume.shape}"
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
    Normalize a single MRI slice.

    Steps:
        1. Convert to float32.
        2. Identify non-zero voxels.
        3. Calculate 1st and 99.5th percentiles.
        4. Clip extreme intensity values.
        5. Scale intensities to [0, 1].
        6. Preserve the zero-valued background.

    Returns:
        Normalized 2D NumPy array.
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

    lower = np.percentile(
        brain_values,
        1
    )

    upper = np.percentile(
        brain_values,
        99.5
    )

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

        self.selected_slices = (
            selected_slices
            if selected_slices is not None
            else SELECTED_SLICES
        )

        self.image_size = image_size

        self.cache_size = cache_size

        self.volume_cache = OrderedDict()

        if not self.report_path.exists():
            raise FileNotFoundError(
                f"Extraction report not found: {self.report_path}"
            )

        self.report = pd.read_csv(
            self.report_path
        )

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
                f"Missing columns in extraction report: {missing_columns}"
            )

        # Keep only the requested split and successfully extracted volumes.
        self.report = self.report[
            (self.report["split"] == split)
            & (self.report["status"] == "PASSED")
        ].copy()

        if self.report.empty:
            raise ValueError(
                f"No valid records found for split: {split}"
            )

        self.report["label"] = pd.to_numeric(
            self.report["label"],
            errors="raise"
        ).astype(int)

        if not self.report["label"].isin([0, 1]).all():
            raise ValueError(
                "Labels must contain only 0 and 1."
            )

        # Expand each participant into individual slice records.
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

        print(
            f"{split.capitalize()} dataset initialized: "
            f"{len(self.samples)} slices"
        )

    # --------------------------------------------------------
    # VOLUME CACHE
    # --------------------------------------------------------

    def get_volume(self, image_path):

        image_path = str(image_path)

        if image_path in self.volume_cache:

            self.volume_cache.move_to_end(
                image_path
            )

            return self.volume_cache[image_path]

        absolute_path = (
            PROJECT_ROOT / image_path
        )

        volume = load_mri_volume(
            absolute_path
        )

        if self.cache_size > 0:

            self.volume_cache[image_path] = volume

            if len(self.volume_cache) > self.cache_size:

                self.volume_cache.popitem(
                    last=False
                )

        return volume

    # --------------------------------------------------------
    # IMAGE PREPARATION
    # --------------------------------------------------------

    def prepare_slice(self, slice_data):

        normalized = normalize_slice(
            slice_data
        )

        # Convert to PyTorch tensor: (1, H, W).
        image = torch.from_numpy(
            normalized.copy()
        ).unsqueeze(0)

        # Resize to the ResNet-compatible dimensions.
        image = F.interpolate(
            image.unsqueeze(0),
            size=(
                self.image_size,
                self.image_size
            ),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        # ResNet-18 expects three input channels.
        image = image.repeat(
            3,
            1,
            1
        )

        return image.float()

    # --------------------------------------------------------
    # DATASET LENGTH
    # --------------------------------------------------------

    def __len__(self):

        return len(self.samples)

    # --------------------------------------------------------
    # GET ITEM
    # --------------------------------------------------------

    def __getitem__(self, index):

        sample = self.samples[index]

        volume = self.get_volume(
            sample["image_path"]
        )

        slice_index = sample["slice_index"]

        if slice_index >= volume.shape[2]:

            raise IndexError(
                f"Slice {slice_index} exceeds volume depth "
                f"{volume.shape[2]} for "
                f"{sample['participant_id']}"
            )

        # Axis 2 is the axial slice axis in this project.
        slice_data = volume[:, :, slice_index]

        image = self.prepare_slice(
            slice_data
        )

        label = torch.tensor(
            sample["label"],
            dtype=torch.long
        )

        return image, label


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
# DATASET VALIDATION
# ============================================================

def validate_batch(loader, split_name):

    print(
        f"\n========== {split_name.upper()} BATCH VALIDATION =========="
    )

    images, labels = next(iter(loader))

    print(
        "Image tensor shape:",
        tuple(images.shape)
    )

    print(
        "Label tensor shape:",
        tuple(labels.shape)
    )

    print(
        "Image data type:",
        images.dtype
    )

    print(
        "Label data type:",
        labels.dtype
    )

    print(
        "Minimum image intensity:",
        images.min().item()
    )

    print(
        "Maximum image intensity:",
        images.max().item()
    )

    print(
        "Labels in batch:",
        torch.unique(labels).tolist()
    )

    assert images.ndim == 4
    assert images.shape[1] == 3
    assert images.shape[2] == IMAGE_SIZE
    assert images.shape[3] == IMAGE_SIZE

    assert labels.ndim == 1

    assert torch.isfinite(images).all()

    assert images.min() >= 0
    assert images.max() <= 1

    assert set(torch.unique(labels).tolist()).issubset({0, 1})

    print(
        f"{split_name.capitalize()} batch validation PASSED."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)

    print("OASIS-1 PYTORCH DATASET VALIDATION")

    print("=" * 65)

    print(
        "\nPyTorch version:",
        torch.__version__
    )

    print(
        "CUDA available:",
        torch.cuda.is_available()
    )

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

    print("\n========== DATASET SIZES ==========")

    print(
        "Training participants:",
        train_dataset.report["participant_id"].nunique()
    )

    print(
        "Validation participants:",
        validation_dataset.report["participant_id"].nunique()
    )

    print(
        "Test participants:",
        test_dataset.report["participant_id"].nunique()
    )

    print(
        "Training slices:",
        len(train_dataset)
    )

    print(
        "Validation slices:",
        len(validation_dataset)
    )

    print(
        "Test slices:",
        len(test_dataset)
    )

    print(
        "\nTraining batches:",
        len(train_loader)
    )

    print(
        "Validation batches:",
        len(validation_loader)
    )

    print(
        "Test batches:",
        len(test_loader)
    )

    print("\n========== CLASS DISTRIBUTION ==========")

    for name, dataset in [
        ("Train", train_dataset),
        ("Validation", validation_dataset),
        ("Test", test_dataset)
    ]:

        distribution = dataset.report["label"].value_counts().sort_index()

        print(f"\n{name}:")

        print(
            "Class 0:",
            int(distribution.get(0, 0))
        )

        print(
            "Class 1:",
            int(distribution.get(1, 0))
        )

    validate_batch(
        train_loader,
        "train"
    )

    validate_batch(
        validation_loader,
        "validation"
    )

    validate_batch(
        test_loader,
        "test"
    )

    print("\n" + "=" * 65)

    print("PYTORCH DATASET VALIDATION COMPLETED SUCCESSFULLY")

    print("=" * 65)


if __name__ == "__main__":
    main()
