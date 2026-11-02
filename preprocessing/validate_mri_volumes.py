
import tarfile
import tempfile

from pathlib import Path, PurePosixPath
from collections import defaultdict

import nibabel as nib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PARTITION_DIR = PROJECT_ROOT / "data" / "partitions"

ARCHIVE_DIR = PROJECT_ROOT / "data" / "raw" / "archives"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "quality_control"
)

EXPECTED_SHAPE = (176, 208, 176)
EXPECTED_SPACING = (1.0, 1.0, 1.0)

SPLIT_FILES = [
    "train.csv",
    "validation.csv",
    "test.csv",
]


def load_splits():

    frames = []

    for filename in SPLIT_FILES:

        path = PARTITION_DIR / filename

        if not path.exists():
            raise FileNotFoundError(path)

        df = pd.read_csv(path)

        frames.append(df)

    combined = pd.concat(
        frames,
        ignore_index=True
    )

    required_columns = [
        "participant_id",
        "session_id",
        "archive",
        "image_member",
        "split",
        "label",
    ]

    missing = [
        column
        for column in required_columns
        if column not in combined.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns in split files: {missing}"
        )

    if combined["session_id"].duplicated().any():
        raise ValueError(
            "Duplicate session IDs found in split manifests."
        )

    if combined["participant_id"].duplicated().any():
        raise ValueError(
            "Participant leakage or duplicate participant "
            "assignment detected."
        )

    return combined


def write_archive_member(archive, member, destination):

    source = archive.extractfile(member)

    if source is None:
        raise RuntimeError(
            f"Cannot read archive member: {member.name}"
        )

    with source, destination.open("wb") as output:
        output.write(source.read())


def validate_volume(archive, members, image_member_name):

    image_path_in_archive = PurePosixPath(
        image_member_name
    )

    header_member_name = str(
        image_path_in_archive.with_suffix(".hdr")
    )

    if image_member_name not in members:
        raise FileNotFoundError(
            f"Image missing: {image_member_name}"
        )

    if header_member_name not in members:
        raise FileNotFoundError(
            f"Header missing: {header_member_name}"
        )

    with tempfile.TemporaryDirectory(
        prefix="oasis_mri_qc_"
    ) as temporary_directory:

        temporary_directory = Path(
            temporary_directory
        )

        image_path = (
            temporary_directory
            / image_path_in_archive.name
        )

        header_path = (
            temporary_directory
            / image_path_in_archive.with_suffix(".hdr").name
        )

        write_archive_member(
            archive,
            members[image_member_name],
            image_path
        )

        write_archive_member(
            archive,
            members[header_member_name],
            header_path
        )

        image = nib.load(str(image_path))

        raw_data = np.asanyarray(image.dataobj)

        original_shape = raw_data.shape

        # OASIS images may have a singleton fourth dimension.
        if raw_data.ndim == 4 and raw_data.shape[-1] == 1:
            data = raw_data[..., 0]

        elif raw_data.ndim == 3:
            data = raw_data

        else:
            raise ValueError(
                f"Unexpected image dimensions: {original_shape}"
            )

        if data.shape != EXPECTED_SHAPE:
            raise ValueError(
                f"Unexpected 3D shape: {data.shape}"
            )

        spacing = tuple(
            float(value)
            for value in image.header.get_zooms()[:3]
        )

        if not np.allclose(
            spacing,
            EXPECTED_SPACING,
            atol=1e-4
        ):
            raise ValueError(
                f"Unexpected voxel spacing: {spacing}"
            )

        finite = np.isfinite(data)

        if not finite.all():
            raise ValueError(
                "Image contains NaN or infinite values."
            )

        minimum = float(data.min())
        maximum = float(data.max())
        mean = float(data.mean())
        std = float(data.std())

        nonzero_fraction = float(
            np.count_nonzero(data) / data.size
        )

        if maximum <= 0 or std <= 0:
            raise ValueError(
                "Image is empty or has no intensity variation."
            )

        if nonzero_fraction <= 0:
            raise ValueError(
                "Image contains no non-zero voxels."
            )

        return {

            "original_shape": str(original_shape),

            "shape": str(data.shape),

            "voxel_spacing": str(spacing),

            "data_type": str(data.dtype),

            "minimum": minimum,

            "maximum": maximum,

            "mean": mean,

            "standard_deviation": std,

            "nonzero_fraction": nonzero_fraction,

            "finite_values": bool(finite.all()),

        }


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Loading participant splits...")

    df = load_splits()

    print("Participants to validate:", len(df))

    print("\nSplit distribution:")
    print(df["split"].value_counts())

    results = []

    # Process one archive at a time.
    grouped = df.groupby("archive")

    for archive_name, group in grouped:

        archive_path = ARCHIVE_DIR / archive_name

        print(
            f"\nOpening archive: {archive_name}"
        )

        if not archive_path.exists():
            raise FileNotFoundError(archive_path)

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

                image_member = record["image_member"]

                print(
                    f"Validating {session_id}...",
                    end=" "
                )

                try:

                    header_name = str(
                        PurePosixPath(
                            image_member
                        ).with_suffix(".hdr")
                    )

                    if image_member not in members:
                        raise FileNotFoundError(
                            "Image member not found."
                        )

                    if header_name not in members:
                        raise FileNotFoundError(
                            "Header member not found."
                        )

                    stats = validate_volume(
                        archive,
                        members,
                        image_member
                    )

                    result = {

                        "participant_id": record["participant_id"],

                        "session_id": session_id,

                        "split": record["split"],

                        "label": record["label"],

                        "archive": archive_name,

                        "image_member": image_member,

                        "status": "PASSED",

                        "error": None,

                        **stats,

                    }

                    print("PASSED")

                except Exception as error:

                    result = {

                        "participant_id": record["participant_id"],

                        "session_id": session_id,

                        "split": record["split"],

                        "label": record["label"],

                        "archive": archive_name,

                        "image_member": image_member,

                        "status": "FAILED",

                        "error": str(error),

                    }

                    print("FAILED:", error)

                results.append(result)

    report = pd.DataFrame(results)

    report_path = (
        OUTPUT_DIR
        / "mri_volume_validation.csv"
    )

    report.to_csv(
        report_path,
        index=False
    )

    print("\n========== VALIDATION SUMMARY ==========")

    print("\nValidation status:")
    print(report["status"].value_counts())

    print("\nValidation by split:")

    print(
        pd.crosstab(
            report["split"],
            report["status"]
        )
    )

    print("\nValidation by label:")

    print(
        pd.crosstab(
            report["label"],
            report["status"]
        )
    )

    print("\nReport saved to:")
    print(report_path)

    if not (report["status"] == "PASSED").all():

        print(
            "\nWARNING: Some MRI volumes failed validation."
        )

    else:

        print(
            "\nAll selected MRI volumes passed validation."
        )


if __name__ == "__main__":
    main()
