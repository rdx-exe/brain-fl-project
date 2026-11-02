
import shutil
import tarfile

from pathlib import Path, PurePosixPath

import nibabel as nib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PARTITION_DIR = PROJECT_ROOT / "data" / "partitions"

ARCHIVE_DIR = PROJECT_ROOT / "data" / "raw" / "archives"

EXTRACTED_DIR = PROJECT_ROOT / "data" / "extracted"

QC_REPORT = (
    PROJECT_ROOT
    / "results"
    / "quality_control"
    / "mri_volume_validation.csv"
)

EXPECTED_SHAPE = (176, 208, 176)

SPLIT_FILES = [
    "train.csv",
    "validation.csv",
    "test.csv",
]


def load_and_verify_manifests():

    frames = []

    for filename in SPLIT_FILES:

        path = PARTITION_DIR / filename

        if not path.exists():
            raise FileNotFoundError(path)

        df = pd.read_csv(path)

        frames.append(df)

    df = pd.concat(
        frames,
        ignore_index=True
    )

    required = [
        "participant_id",
        "session_id",
        "archive",
        "image_member",
        "split",
        "label",
    ]

    missing = [
        column for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing manifest columns: {missing}"
        )

    if df["participant_id"].duplicated().any():
        raise ValueError(
            "Duplicate participant assignments detected."
        )

    if df["session_id"].duplicated().any():
        raise ValueError(
            "Duplicate MRI sessions detected."
        )

    # Verify the previous full-volume QC report.
    qc = pd.read_csv(QC_REPORT)

    if len(qc) != len(df):
        raise ValueError(
            "QC report and partition manifests have "
            "different record counts."
        )

    if not (qc["status"] == "PASSED").all():
        raise ValueError(
            "Not all MRI volumes passed validation."
        )

    return df


def copy_member(archive, member, destination):

    source = archive.extractfile(member)

    if source is None:
        raise RuntimeError(
            f"Could not read archive member: {member.name}"
        )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with source, destination.open("wb") as output:

        shutil.copyfileobj(
            source,
            output
        )


def validate_extracted_pair(image_path):

    image = nib.load(str(image_path))

    data = np.asanyarray(image.dataobj)

    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]

    if data.shape != EXPECTED_SHAPE:
        raise ValueError(
            f"Unexpected extracted dimensions: {data.shape}"
        )

    if not np.isfinite(data).all():
        raise ValueError(
            "Extracted image contains invalid values."
        )

    if data.max() <= 0:
        raise ValueError(
            "Extracted image is empty."
        )

    return {
        "shape": str(data.shape),
        "minimum": float(data.min()),
        "maximum": float(data.max()),
        "mean": float(data.mean()),
    }


def main():

    print("Loading and verifying manifests...")

    df = load_and_verify_manifests()

    print("MRI volumes to extract:", len(df))

    extraction_records = []

    grouped = df.groupby("archive")

    for archive_name, group in grouped:

        archive_path = ARCHIVE_DIR / archive_name

        if not archive_path.exists():
            raise FileNotFoundError(archive_path)

        print("\nOpening:", archive_name)

        with tarfile.open(
            archive_path,
            "r:*"
        ) as archive:

            members = {
                member.name: member
                for member in archive.getmembers()
                if member.isfile()
            }

            for _, record in group.iterrows():

                session_id = record["session_id"]

                participant_id = record["participant_id"]

                split = record["split"]

                image_member_name = record["image_member"]

                header_member_name = str(
                    PurePosixPath(
                        image_member_name
                    ).with_suffix(".hdr")
                )

                if image_member_name not in members:
                    raise FileNotFoundError(
                        image_member_name
                    )

                if header_member_name not in members:
                    raise FileNotFoundError(
                        header_member_name
                    )

                # Ensure that we only extract masked GFC images.
                if not image_member_name.lower().endswith(
                    "_t88_masked_gfc.img"
                ):
                    raise ValueError(
                        f"Unexpected MRI representation: "
                        f"{image_member_name}"
                    )

                output_directory = (
                    EXTRACTED_DIR
                    / split
                    / participant_id
                )

                output_image = (
                    output_directory
                    / f"{session_id}_t88_masked_gfc.img"
                )

                output_header = (
                    output_directory
                    / f"{session_id}_t88_masked_gfc.hdr"
                )

                print(
                    f"Extracting {session_id} [{split}]..."
                )

                copy_member(
                    archive,
                    members[image_member_name],
                    output_image
                )

                copy_member(
                    archive,
                    members[header_member_name],
                    output_header
                )

                stats = validate_extracted_pair(
                    output_image
                )

                extraction_records.append({

                    "participant_id": participant_id,

                    "session_id": session_id,

                    "split": split,

                    "label": record["label"],

                    "image_path": str(
                        output_image.relative_to(PROJECT_ROOT)
                    ),

                    "header_path": str(
                        output_header.relative_to(PROJECT_ROOT)
                    ),

                    "status": "PASSED",

                    **stats,

                })

    report = pd.DataFrame(
        extraction_records
    )

    report_path = (
        PROJECT_ROOT
        / "results"
        / "quality_control"
        / "extraction_report.csv"
    )

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    report.to_csv(
        report_path,
        index=False
    )

    print("\n========== EXTRACTION SUMMARY ==========")

    print("\nExtraction status:")
    print(report["status"].value_counts())

    print("\nVolumes by split:")
    print(report["split"].value_counts())

    print("\nExtracted volume count:", len(report))

    print("\nExtraction report:")
    print(report_path)

    print("\nExtracted dataset:")
    print(EXTRACTED_DIR)

    print("\nExtraction completed successfully.")


if __name__ == "__main__":
    main()
