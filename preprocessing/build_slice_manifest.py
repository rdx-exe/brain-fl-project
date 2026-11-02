
from pathlib import Path

import pandas as pd
import nibabel as nib
import numpy as np


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
    / "data"
    / "partitions"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "slice_manifest.csv"
)


# Fixed axial slices selected for every participant.
SLICE_INDICES = [60, 70, 80, 90, 100, 110, 120]

EXPECTED_SPLITS = {
    "train",
    "validation",
    "test"
}


# ============================================================
# MRI VALIDATION
# ============================================================

def validate_mri_volume(image_path, participant_id):

    image = nib.load(str(image_path))

    original_shape = image.shape

    # Remove singleton dimensions.
    # Example: (176, 208, 176, 1) -> (176, 208, 176)
    volume = np.squeeze(
        np.asarray(image.dataobj)
    )

    # Ensure the resulting volume is 3D.
    if volume.ndim != 3:
        raise ValueError(
            f"Expected 3D MRI after removing singleton "
            f"dimensions: {participant_id}. "
            f"Original shape: {original_shape}, "
            f"Processed shape: {volume.shape}"
        )

    if not np.isfinite(volume).all():
        raise ValueError(
            f"Non-finite voxel values detected: {participant_id}"
        )

    return image, volume, original_shape


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():

    print("=" * 65)
    print("BUILDING OASIS-1 2D SLICE MANIFEST")
    print("=" * 65)

    # --------------------------------------------------------
    # 1. Validate extraction report
    # --------------------------------------------------------

    if not REPORT_PATH.exists():
        raise FileNotFoundError(
            f"Extraction report not found: {REPORT_PATH}"
        )

    df = pd.read_csv(REPORT_PATH)

    required_columns = {
        "participant_id",
        "session_id",
        "split",
        "label",
        "image_path",
        "status"
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise KeyError(
            f"Missing extraction report columns: {missing_columns}"
        )

    df = df[
        df["status"] == "PASSED"
    ].copy()

    if df.empty:
        raise ValueError(
            "No successfully extracted MRI volumes."
        )

    # --------------------------------------------------------
    # 2. Validate participant records
    # --------------------------------------------------------

    if df["participant_id"].isna().any():
        raise ValueError(
            "Missing participant IDs in extraction report."
        )

    if df["participant_id"].duplicated().any():
        raise ValueError(
            "Duplicate participants detected in extraction report."
        )

    if df["split"].isna().any():
        raise ValueError("Missing dataset split assignments.")

    actual_splits = set(df["split"].unique())

    if actual_splits != EXPECTED_SPLITS:
        raise ValueError(
            f"Unexpected dataset splits. "
            f"Expected {EXPECTED_SPLITS}, got {actual_splits}"
        )

    if df["label"].isna().any():
        raise ValueError(
            "Missing labels detected in extraction report."
        )

    # --------------------------------------------------------
    # 3. Build slice manifest
    # --------------------------------------------------------

    manifest_records = []

    print(f"\nValid participants: {len(df)}")

    for _, record in df.iterrows():

        participant_id = record["participant_id"]

        image_path = (
            PROJECT_ROOT / record["image_path"]
        )

        if not image_path.exists():
            raise FileNotFoundError(
                f"MRI file not found: {image_path}"
            )

        # Load and validate MRI.
        image, volume, original_shape = validate_mri_volume(
            image_path,
            participant_id
        )

        shape = volume.shape

        print(
            f"\nParticipant: {participant_id}"
        )

        print(
            f"Original shape: {original_shape}"
        )

        print(
            f"Processed shape: {shape}"
        )

        # Validate selected slice indices.
        for slice_index in SLICE_INDICES:

            if not (0 <= slice_index < shape[2]):
                raise ValueError(
                    f"Invalid slice index {slice_index} "
                    f"for {participant_id}. "
                    f"Volume depth: {shape[2]}"
                )

            manifest_records.append({

                "sample_id": (
                    f"{participant_id}"
                    f"_slice_{slice_index:03d}"
                ),

                "participant_id": participant_id,

                "session_id": record["session_id"],

                "split": record["split"],

                "label": int(record["label"]),

                "image_path": record["image_path"],

                "slice_index": slice_index,

                "volume_shape": str(shape),

                "original_volume_shape": str(original_shape),

            })

    # --------------------------------------------------------
    # 4. Create manifest DataFrame
    # --------------------------------------------------------

    manifest = pd.DataFrame(manifest_records)

    if manifest.empty:
        raise RuntimeError(
            "Slice manifest generation produced no samples."
        )

    # --------------------------------------------------------
    # 5. Validate manifest integrity
    # --------------------------------------------------------

    expected_count = (
        len(df) * len(SLICE_INDICES)
    )

    if len(manifest) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} samples, "
            f"generated {len(manifest)}."
        )

    if manifest["sample_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate sample IDs detected."
        )

    if manifest["participant_id"].isna().any():
        raise RuntimeError(
            "Missing participant IDs."
        )

    if manifest["label"].isna().any():
        raise RuntimeError(
            "Missing labels."
        )

    # --------------------------------------------------------
    # 6. Verify slices per participant
    # --------------------------------------------------------

    samples_per_participant = (
        manifest.groupby("participant_id").size()
    )

    if not (
        samples_per_participant == len(SLICE_INDICES)
    ).all():

        raise RuntimeError(
            "Inconsistent number of slices per participant."
        )

    # --------------------------------------------------------
    # 7. Verify participant-level split separation
    # --------------------------------------------------------

    split_participants = {

        split: set(
            manifest.loc[
                manifest["split"] == split,
                "participant_id"
            ]
        )

        for split in EXPECTED_SPLITS
    }

    for first, second in [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ]:

        overlap = (
            split_participants[first]
            & split_participants[second]
        )

        if overlap:
            raise RuntimeError(
                f"Participant leakage detected: "
                f"{first} vs {second}: {overlap}"
            )

    # --------------------------------------------------------
    # 8. Save manifest
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    manifest.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # --------------------------------------------------------
    # 9. Display summary
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("MANIFEST SUMMARY")
    print("=" * 65)

    print("\nParticipants:", len(df))

    print(
        "Slices per participant:",
        len(SLICE_INDICES)
    )

    print(
        "Total slice samples:",
        len(manifest)
    )

    print("\nSplit distribution:")

    print(
        manifest.groupby("split").size()
    )

    print("\nClass distribution:")

    print(
        pd.crosstab(
            manifest["split"],
            manifest["label"]
        )
    )

    print("\nSelected slices:", SLICE_INDICES)

    print("\nParticipant leakage: NONE")

    print("\nSaved manifest:")
    print(OUTPUT_PATH)

    print("\nSLICE MANIFEST BUILD COMPLETED SUCCESSFULLY.")


if __name__ == "__main__":
    main()
