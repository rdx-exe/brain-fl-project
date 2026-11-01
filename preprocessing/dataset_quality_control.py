
import re
import tarfile
import tempfile

from pathlib import Path, PurePosixPath

import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ARCHIVE_DIR = PROJECT_ROOT / "data" / "raw" / "archives"

METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "oasis_cross-sectional-5708aa0a98d82080.xlsx"
)

OUTPUT_DIR = PROJECT_ROOT / "results" / "quality_control"

# Number of participants to inspect per CDR category.
SAMPLES_PER_CDR = 3

MASKED_SUFFIX = "_t88_masked_gfc.img"


def load_metadata():

    df = pd.read_excel(METADATA_PATH)

    required = ["ID", "CDR"]

    for column in required:
        if column not in df.columns:
            raise ValueError(f"Missing metadata column: {column}")

    df["session_id"] = df["ID"].astype(str)

    df["participant_id"] = (
        df["session_id"]
        .str.replace(r"_MR\d+$", "", regex=True)
    )

    # Only use records with an available CDR value.
    df = df.dropna(subset=["CDR"]).copy()

    df["CDR"] = pd.to_numeric(df["CDR"], errors="coerce")
    df = df.dropna(subset=["CDR"])

    # Keep one session per participant for this QC sample.
    df["session_number"] = (
        df["session_id"]
        .str.extract(r"_MR(\d+)$")[0]
        .astype(float)
    )

    df = df.sort_values(
        ["participant_id", "session_number"]
    )

    df = df.drop_duplicates(
        subset=["participant_id"],
        keep="first"
    )

    return df


def build_archive_index():

    """
    Maps OASIS session IDs to masked GFC image/header pairs.

    Only archive metadata is indexed here.
    No MRI volumes are extracted at this stage.
    """

    archives = sorted(
        path
        for path in ARCHIVE_DIR.iterdir()
        if path.is_file()
        and (
            path.name.endswith(".tar.gz")
            or path.name.endswith(".tgz")
            or path.name.endswith(".tar")
        )
    )

    if not archives:
        raise FileNotFoundError(
            f"No TAR archives found in {ARCHIVE_DIR}"
        )

    print("Archives found:", len(archives))

    image_index = {}

    for archive_path in archives:

        print("Indexing:", archive_path.name)

        with tarfile.open(archive_path, "r:*") as archive:

            members = archive.getmembers()

            member_names = {
                member.name
                for member in members
                if member.isfile()
            }

            for member in members:

                if not member.isfile():
                    continue

                member_path = PurePosixPath(member.name)
                filename = member_path.name

                if not filename.lower().endswith(
                    MASKED_SUFFIX
                ):
                    continue

                match = re.match(
                    r"^(OAS1_\d{4}_MR\d+)",
                    filename,
                    re.IGNORECASE
                )

                if not match:
                    continue

                session_id = match.group(1)

                hdr_path = str(
                    member_path.with_suffix(".hdr")
                )

                if hdr_path not in member_names:
                    continue

                image_index.setdefault(
                    session_id,
                    {
                        "archive": archive_path,
                        "img_member": member.name,
                        "hdr_member": hdr_path,
                    }
                )

    print(
        "\nSessions with complete masked GFC pairs:",
        len(image_index)
    )

    return image_index


def select_qc_participants(metadata, image_index):

    """
    Select participants from every available CDR category.
    Prefer the earliest MRI session for each participant.
    """

    selected = []

    categories = sorted(metadata["CDR"].unique())

    for cdr in categories:

        group = metadata[metadata["CDR"] == cdr]

        available = group[
            group["session_id"].isin(image_index)
        ]

        chosen = available.head(SAMPLES_PER_CDR)

        print(
            f"CDR {cdr}: "
            f"{len(chosen)} selected from "
            f"{len(available)} available participants"
        )

        selected.extend(chosen.to_dict("records"))

    return selected


def extract_image_pair(record, image_index, output_dir):

    """
    Extract only one .img/.hdr pair.

    The files are written using their basenames rather than
    archive paths, preventing archive path traversal.
    """

    session_id = record["session_id"]
    info = image_index[session_id]

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    with tarfile.open(info["archive"], "r:*") as archive:

        for member_name in [
            info["img_member"],
            info["hdr_member"]
        ]:

            member = archive.getmember(member_name)

            source = archive.extractfile(member)

            if source is None:
                raise RuntimeError(
                    f"Cannot read archive member: {member_name}"
                )

            destination = (
                output_dir
                / PurePosixPath(member_name).name
            )

            with source, destination.open("wb") as output:
                output.write(source.read())

    return (
        output_dir
        / PurePosixPath(info["img_member"]).name
    )


def inspect_volume(image_path):

    image = nib.load(str(image_path))

    data = image.get_fdata(
        dtype=np.float32
    )

    # Convert a singleton-channel 4D volume into 3D.
    if data.ndim == 4 and data.shape[-1] == 1:
        data = np.squeeze(data, axis=-1)

    if data.ndim != 3:
        raise ValueError(
            f"Unexpected volume dimensions: {data.shape}"
        )

    if not np.isfinite(data).all():
        raise ValueError("Image contains NaN or infinite values")

    nonzero = data[data > 0]

    if nonzero.size == 0:
        raise ValueError("Image contains no positive voxels")

    stats = {
        "shape": str(data.shape),
        "voxel_spacing": str(image.header.get_zooms()[:3]),
        "data_type": str(image.get_data_dtype()),
        "minimum": float(data.min()),
        "maximum": float(data.max()),
        "mean": float(data.mean()),
        "std": float(data.std()),
        "nonzero_fraction": float(np.count_nonzero(data) / data.size),
        "positive_p01": float(np.percentile(nonzero, 1)),
        "positive_p50": float(np.percentile(nonzero, 50)),
        "positive_p99": float(np.percentile(nonzero, 99)),
    }

    return data, stats


def save_visualization(data, participant_id, cdr):

    x, y, z = [
        dimension // 2
        for dimension in data.shape
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5)
    )

    views = [
        (data[:, :, z].T, "Axial"),
        (data[:, y, :].T, "Coronal"),
        (data[x, :, :].T, "Sagittal"),
    ]

    for axis, (view, title) in zip(axes, views):

        axis.imshow(
            view,
            cmap="gray",
            origin="lower"
        )

        axis.set_title(title)
        axis.axis("off")

    fig.suptitle(
        f"{participant_id} | CDR={cdr} | Masked GFC"
    )

    plt.tight_layout()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        OUTPUT_DIR
        / f"{participant_id}_cdr_{cdr}.png"
    )

    plt.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Saved:", output_path)


def main():

    print("\nLoading metadata...")

    metadata = load_metadata()

    print(
        "Unique participants with valid CDR:",
        len(metadata)
    )

    print("\nBuilding archive index...")

    image_index = build_archive_index()

    print("\nSelecting QC participants...")

    selected = select_qc_participants(
        metadata,
        image_index
    )

    report = []

    print("\nInspecting selected MRI volumes...")

    for record in selected:

        participant_id = record["participant_id"]
        session_id = record["session_id"]
        cdr = record["CDR"]

        print(
            f"\nProcessing {participant_id} "
            f"| {session_id} | CDR={cdr}"
        )

        row = {
            "participant_id": participant_id,
            "session_id": session_id,
            "CDR": cdr,
            "status": "FAILED",
        }

        try:

            # Use a temporary directory for each image pair.
            with tempfile.TemporaryDirectory() as temp_dir:

                image_path = extract_image_pair(
                    record,
                    image_index,
                    Path(temp_dir)
                )

                data, stats = inspect_volume(image_path)

                save_visualization(
                    data,
                    participant_id,
                    cdr
                )

                row.update(stats)
                row["status"] = "PASS"

        except Exception as error:

            row["error"] = str(error)

            print("QC ERROR:", error)

        report.append(row)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    report_df = pd.DataFrame(report)

    report_path = (
        OUTPUT_DIR
        / "mri_quality_control_report.csv"
    )

    report_df.to_csv(
        report_path,
        index=False
    )

    print("\n========== QC SUMMARY ==========")

    print(
        report_df["status"].value_counts(
            dropna=False
        )
    )

    print("\nFull report saved to:")
    print(report_path)

    print("\nQC completed.")


if __name__ == "__main__":
    main()
