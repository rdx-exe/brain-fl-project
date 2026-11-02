
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "preprocessing_analysis"
    / "visual_quality_control"
)

# Representative axial slices
SLICE_INDICES = [20, 40, 60, 78, 100, 120, 140]


# ============================================================
# MRI VOLUME LOADING
# ============================================================

def load_volume(image_path):

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"MRI volume not found: {image_path}"
        )

    image = nib.load(str(image_path))

    volume = np.asarray(
        image.dataobj,
        dtype=np.float32
    )

    # Handle singleton dimensions.
    # Example: (176, 208, 176, 1) -> (176, 208, 176)
    volume = np.squeeze(volume)

    # Ensure the final volume is genuinely 3D.
    if volume.ndim != 3:
        raise ValueError(
            f"Expected a 3D MRI volume after removing "
            f"singleton dimensions, got shape: {volume.shape}"
        )

    # Check for invalid voxel values.
    if not np.isfinite(volume).all():
        raise ValueError(
            f"MRI volume contains NaN or infinite values: {image_path}"
        )

    return image, volume


# ============================================================
# INTENSITY NORMALIZATION
# ============================================================

def normalize_for_display(slice_data):

    nonzero = slice_data[
        np.isfinite(slice_data) & (slice_data > 0)
    ]

    if nonzero.size == 0:
        return np.zeros_like(
            slice_data,
            dtype=np.float32
        )

    low, high = np.percentile(
        nonzero,
        [1, 99]
    )

    if high <= low:
        return np.zeros_like(
            slice_data,
            dtype=np.float32
        )

    normalized = np.clip(
        (slice_data - low) / (high - low),
        0,
        1
    )

    return normalized


# ============================================================
# PARTICIPANT VISUALIZATION
# ============================================================

def visualize_participant(record):

    image_path = PROJECT_ROOT / record["image_path"]

    image, volume = load_volume(image_path)

    print(
        f"\nParticipant: {record['participant_id']}"
    )

    print(
        f"Original/processed volume shape: {volume.shape}"
    )

    # Validate requested slice indices.
    depth = volume.shape[2]

    valid_slices = [
        index for index in SLICE_INDICES
        if 0 <= index < depth
    ]

    if not valid_slices:
        raise ValueError(
            f"No valid slice indices for volume depth: {depth}"
        )

    fig, axes = plt.subplots(
        1,
        len(valid_slices),
        figsize=(21, 4),
        squeeze=False
    )

    axes = axes.flatten()

    for axis, slice_index in zip(
        axes,
        valid_slices
    ):

        slice_data = volume[:, :, slice_index]

        displayed = normalize_for_display(
            slice_data
        )

        axis.imshow(
            displayed.T,
            cmap="gray",
            origin="lower"
        )

        axis.set_title(
            f"Slice {slice_index}"
        )

        axis.axis("off")

    fig.suptitle(
        f'{record["participant_id"]} | '
        f'Label: {record["label"]} | '
        f'Shape: {volume.shape}',
        fontsize=12
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / f'{record["participant_id"]}_montage.png'
    )

    plt.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Saved:", output_path)


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():

    print("=" * 65)
    print("OASIS-1 TRAINING VOLUME VISUAL QUALITY CONTROL")
    print("=" * 65)

    # Validate extraction report.
    if not REPORT_PATH.exists():
        raise FileNotFoundError(
            f"Extraction report not found: {REPORT_PATH}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df = pd.read_csv(REPORT_PATH)

    required_columns = {
        "image_path",
        "participant_id",
        "label",
        "split",
        "status"
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise KeyError(
            f"Missing required columns in extraction report: "
            f"{missing_columns}"
        )

    # Select valid training volumes.
    train_df = df[
        (df["split"] == "train")
        & (df["status"] == "PASSED")
    ].copy()

    if train_df.empty:
        raise ValueError(
            "No valid training volumes found."
        )

    # Select representative examples from each class.
    selected = []

    for label in sorted(train_df["label"].dropna().unique()):

        class_records = train_df[
            train_df["label"] == label
        ]

        selected.append(
            class_records.iloc[0]
        )

        if len(class_records) > 1:
            selected.append(
                class_records.iloc[len(class_records) // 2]
            )

    print(
        f"\nParticipants selected: {len(selected)}"
    )

    # Visualize selected participants.
    successful = 0
    failed = 0

    for record in selected:

        try:
            visualize_participant(record)
            successful += 1

        except Exception as error:

            failed += 1

            print(
                f"\nERROR processing "
                f"{record['participant_id']}: {error}"
            )

    # Final report.
    print("\n" + "=" * 65)
    print("VISUAL QUALITY CONTROL SUMMARY")
    print("=" * 65)

    print(f"Selected participants: {len(selected)}")
    print(f"Successfully visualized: {successful}")
    print(f"Failed: {failed}")

    print("\nOutput directory:", OUTPUT_DIR)

    if failed:
        print("\nSome participants require manual inspection.")
    else:
        print("\nAll selected MRI volumes visualized successfully.")


if __name__ == "__main__":
    main()
