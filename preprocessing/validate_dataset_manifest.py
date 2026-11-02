
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPORT_PATH = (
    PROJECT_ROOT
    / "results"
    / "quality_control"
    / "extraction_report.csv"
)

EXPECTED_SPLITS = {"train", "validation", "test"}


def main():

    print("Loading extraction report...")

    if not REPORT_PATH.exists():
        raise FileNotFoundError(REPORT_PATH)

    df = pd.read_csv(REPORT_PATH)

    required_columns = [
        "participant_id",
        "session_id",
        "split",
        "label",
        "image_path",
        "header_path",
        "status",
        "shape",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    print("\n========== DATASET OVERVIEW ==========")

    print("Total records:", len(df))
    print("Unique participants:", df["participant_id"].nunique())
    print("Unique sessions:", df["session_id"].nunique())

    print("\n========== SPLIT DISTRIBUTION ==========")

    print(
        df["split"]
        .value_counts()
        .reindex(sorted(EXPECTED_SPLITS), fill_value=0)
    )

    print("\n========== CLASS DISTRIBUTION ==========")

    class_counts = pd.crosstab(
        df["split"],
        df["label"]
    ).reindex(
        index=["train", "validation", "test"],
        fill_value=0
    )

    print(class_counts)

    print("\n========== CLASS PROPORTIONS ==========")

    class_proportions = class_counts.div(
        class_counts.sum(axis=1),
        axis=0
    )

    print(
        class_proportions.round(3)
    )

    print("\n========== PARTICIPANT LEAKAGE CHECK ==========")

    split_sets = {
        split: set(
            df.loc[
                df["split"] == split,
                "participant_id"
            ].dropna()
        )
        for split in EXPECTED_SPLITS
    }

    leakage_found = False

    split_names = sorted(EXPECTED_SPLITS)

    for i, first in enumerate(split_names):

        for second in split_names[i + 1:]:

            overlap = split_sets[first] & split_sets[second]

            print(
                f"{first} vs {second}: "
                f"{len(overlap)} overlapping participants"
            )

            if overlap:
                leakage_found = True
                print("Overlapping IDs:", sorted(overlap))

    if leakage_found:
        print("WARNING: Participant leakage detected!")
    else:
        print("PASS: No participant overlap between splits.")

    print("\n========== DUPLICATE CHECKS ==========")

    duplicate_sessions = df[
        df.duplicated("session_id", keep=False)
    ]

    duplicate_participants = df[
        df.duplicated("participant_id", keep=False)
    ]

    print(
        "Duplicate session records:",
        len(duplicate_sessions)
    )

    print(
        "Participants with multiple records:",
        duplicate_participants["participant_id"].nunique()
    )

    print("\n========== LABEL CONSISTENCY ==========")

    label_counts = df["label"].value_counts(dropna=False)

    print(label_counts)

    inconsistent_labels = (
        df.groupby("participant_id")["label"]
        .nunique(dropna=False)
    )

    inconsistent_labels = inconsistent_labels[
        inconsistent_labels > 1
    ]

    print(
        "Participants with inconsistent labels:",
        len(inconsistent_labels)
    )

    if len(inconsistent_labels):
        print(inconsistent_labels)

    print("\n========== FILE VALIDATION ==========")

    image_exists = df["image_path"].apply(
        lambda path: (PROJECT_ROOT / str(path)).is_file()
    )

    header_exists = df["header_path"].apply(
        lambda path: (PROJECT_ROOT / str(path)).is_file()
    )

    print("Missing image files:", int((~image_exists).sum()))
    print("Missing header files:", int((~header_exists).sum()))

    print("\n========== EXTRACTION STATUS ==========")

    print(df["status"].value_counts(dropna=False))

    print("\n========== SHAPE CONSISTENCY ==========")

    print(df["shape"].value_counts(dropna=False))

    # Final validation summary.

    checks = {
        "All records passed extraction": (
            df["status"].eq("PASSED").all()
        ),
        "All splits are valid": (
            set(df["split"].dropna()).issubset(EXPECTED_SPLITS)
        ),
        "No missing labels": (
            df["label"].notna().all()
        ),
        "No missing participant IDs": (
            df["participant_id"].notna().all()
        ),
        "No participant leakage": (
            not leakage_found
        ),
        "No missing image files": (
            image_exists.all()
        ),
        "No missing header files": (
            header_exists.all()
        ),
        "No inconsistent participant labels": (
            len(inconsistent_labels) == 0
        ),
    }

    print("\n========== FINAL VALIDATION ==========")

    for description, passed in checks.items():

        status = "PASS" if passed else "FAIL"

        print(f"[{status}] {description}")

    if all(checks.values()):

        print("\nDATASET MANIFEST VALIDATION PASSED.")

    else:

        print("\nDATASET MANIFEST VALIDATION FAILED.")
        raise RuntimeError(
            "Resolve the failed checks before model development."
        )


if __name__ == "__main__":
    main()
