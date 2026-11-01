
import re
import tarfile

from pathlib import Path, PurePosixPath
from collections import defaultdict

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ARCHIVE_DIR = PROJECT_ROOT / "data" / "raw" / "archives"

METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "oasis_cross-sectional-5708aa0a98d82080.xlsx"
)

OUTPUT_DIR = PROJECT_ROOT / "results" / "quality_control"

MASKED_SUFFIX = "_t88_masked_gfc.img"


def load_metadata():

    df = pd.read_excel(METADATA_PATH)

    df["session_id"] = df["ID"].astype(str).str.strip()

    df["participant_id"] = (
        df["session_id"]
        .str.replace(r"_MR\d+$", "", regex=True)
    )

    return df


def get_archives():

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

    return archives


def index_masked_images():

    """
    Index all masked GFC images and their corresponding
    headers across every archive.

    No MRI volumes are extracted.
    """

    index = defaultdict(list)

    archives = get_archives()

    print("Archives found:", len(archives))

    for archive_path in archives:

        print("\nInspecting:", archive_path.name)

        with tarfile.open(archive_path, "r:*") as archive:

            members = archive.getmembers()

            file_members = {
                member.name: member
                for member in members
                if member.isfile()
            }

            for member_name, member in file_members.items():

                path = PurePosixPath(member_name)

                filename = path.name

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

                header_path = str(
                    path.with_suffix(".hdr")
                )

                header_exists = (
                    header_path in file_members
                )

                index[session_id].append({

                    "session_id": session_id,

                    "archive": archive_path.name,

                    "image_member": member_name,

                    "header_member": header_path,

                    "header_exists": header_exists,

                    "image_size_bytes": member.size,

                    "header_size_bytes": (
                        file_members[header_path].size
                        if header_exists
                        else None
                    ),

                })

    return index


def create_audit(metadata, image_index):

    metadata_sessions = set(
        metadata["session_id"]
    )

    image_sessions = set(
        image_index.keys()
    )

    rows = []

    # Check every metadata session.
    for _, record in metadata.iterrows():

        session_id = record["session_id"]

        matches = image_index.get(
            session_id,
            []
        )

        if len(matches) == 0:

            status = "MISSING_IMAGE"

            archive = None
            image_member = None
            header_exists = False
            duplicate_count = 0

        elif len(matches) > 1:

            status = "DUPLICATE_IMAGE_SESSION"

            archive = ";".join(
                item["archive"]
                for item in matches
            )

            image_member = ";".join(
                item["image_member"]
                for item in matches
            )

            header_exists = all(
                item["header_exists"]
                for item in matches
            )

            duplicate_count = len(matches)

        else:

            match = matches[0]

            archive = match["archive"]
            image_member = match["image_member"]
            header_exists = match["header_exists"]

            duplicate_count = 1

            status = (
                "MATCHED"
                if header_exists
                else "MISSING_HEADER"
            )

        rows.append({

            "participant_id": record["participant_id"],

            "session_id": session_id,

            "CDR": record["CDR"],

            "MMSE": record["MMSE"]
            if "MMSE" in metadata.columns
            else None,

            "archive": archive,

            "image_member": image_member,

            "header_exists": header_exists,

            "matching_image_count": duplicate_count,

            "status": status,

        })

    report = pd.DataFrame(rows)

    # Find archive image sessions that have no metadata record.
    unmatched_image_sessions = (
        image_sessions - metadata_sessions
    )

    unmatched_rows = []

    for session_id in sorted(unmatched_image_sessions):

        for match in image_index[session_id]:

            unmatched_rows.append({

                "participant_id": re.sub(
                    r"_MR\d+$",
                    "",
                    session_id
                ),

                "session_id": session_id,

                "CDR": None,

                "MMSE": None,

                "archive": match["archive"],

                "image_member": match["image_member"],

                "header_exists": match["header_exists"],

                "matching_image_count": len(
                    image_index[session_id]
                ),

                "status": "NO_METADATA_RECORD",

            })

    if unmatched_rows:

        report = pd.concat(
            [
                report,
                pd.DataFrame(unmatched_rows)
            ],
            ignore_index=True
        )

    return report


def print_summary(report, image_index):

    print("\n========== DATASET AUDIT ==========")

    print(
        "\nMetadata records:",
        report[
            report["status"] != "NO_METADATA_RECORD"
        ].shape[0]
    )

    print(
        "Unique MRI sessions indexed:",
        len(image_index)
    )

    print(
        "Total masked image entries:",
        sum(len(items) for items in image_index.values())
    )

    print("\nMatching status:")

    print(
        report["status"].value_counts(
            dropna=False
        )
    )

    print("\nCDR distribution in metadata:")

    print(
        report[
            report["status"] != "NO_METADATA_RECORD"
        ]["CDR"].value_counts(
            dropna=False
        ).sort_index()
    )

    print("\nParticipants with repeated metadata sessions:")

    metadata_rows = report[
        report["status"] != "NO_METADATA_RECORD"
    ]

    repeated = (
        metadata_rows.groupby("participant_id")
        ["session_id"]
        .nunique()
    )

    repeated = repeated[repeated > 1]

    print(repeated)

    print("\n===================================")


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Loading metadata...")

    metadata = load_metadata()

    print(
        "Metadata records:",
        len(metadata)
    )

    print("\nIndexing MRI archives...")

    image_index = index_masked_images()

    print("\nCreating matching report...")

    report = create_audit(
        metadata,
        image_index
    )

    report_path = (
        OUTPUT_DIR
        / "full_dataset_audit.csv"
    )

    report.to_csv(
        report_path,
        index=False
    )

    print_summary(
        report,
        image_index
    )

    print("\nAudit report saved to:")
    print(report_path)


if __name__ == "__main__":
    main()
