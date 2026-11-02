
from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]

AUDIT_PATH = (
    PROJECT_ROOT
    / "results"
    / "quality_control"
    / "full_dataset_audit.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "partitions"

RANDOM_STATE = 42


def load_labeled_cohort():

    if not AUDIT_PATH.exists():
        raise FileNotFoundError(AUDIT_PATH)

    df = pd.read_csv(AUDIT_PATH)

    # Only use sessions with confirmed image/header matches.
    df = df[df["status"] == "MATCHED"].copy()

    # Exclude records with missing CDR.
    df = df.dropna(subset=["CDR"]).copy()

    # Binary classification:
    # 0 = CDR 0.0
    # 1 = CDR greater than 0
    df["label"] = (df["CDR"] > 0).astype(int)

    df["label_name"] = df["label"].map({
        0: "CN",
        1: "CDR_positive"
    })

    # Verify that every participant appears only once
    # in the labelled cohort.
    duplicate_ids = df[
        df.duplicated(
            subset=["participant_id"],
            keep=False
        )
    ]

    if not duplicate_ids.empty:

        raise ValueError(
            "Repeated participants found in the labelled "
            "cohort. Resolve them before splitting:\n"
            f"{duplicate_ids[['participant_id', 'session_id', 'CDR']]}"
        )

    return df


def create_splits(df):

    # First split:
    # 70% training, 30% temporary.
    train_df, temporary_df = train_test_split(

        df,

        test_size=0.30,

        random_state=RANDOM_STATE,

        stratify=df["label"]
    )

    # Second split:
    # Divide temporary data equally into validation and test.
    val_df, test_df = train_test_split(

        temporary_df,

        test_size=0.50,

        random_state=RANDOM_STATE,

        stratify=temporary_df["label"]
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "validation"
    test_df["split"] = "test"

    return train_df, val_df, test_df


def verify_splits(train_df, val_df, test_df):

    splits = {
        "train": train_df,
        "validation": val_df,
        "test": test_df
    }

    # Check participant-level isolation.
    participant_sets = {
        name: set(df["participant_id"])
        for name, df in splits.items()
    }

    names = list(participant_sets.keys())

    for i in range(len(names)):

        for j in range(i + 1, len(names)):

            first = participant_sets[names[i]]
            second = participant_sets[names[j]]

            overlap = first.intersection(second)

            if overlap:

                raise ValueError(
                    f"Participant leakage between "
                    f"{names[i]} and {names[j]}: {overlap}"
                )

    # Verify that no participant was lost or duplicated.
    all_ids = (
        participant_sets["train"]
        | participant_sets["validation"]
        | participant_sets["test"]
    )

    total_rows = (
        len(train_df)
        + len(val_df)
        + len(test_df)
    )

    if len(all_ids) != total_rows:
        raise ValueError(
            "Duplicate participant assignment detected."
        )

    print("\nParticipant isolation: PASSED")

    print("Unique participants:", len(all_ids))


def create_summary(train_df, val_df, test_df):

    rows = []

    for name, df in [
        ("train", train_df),
        ("validation", val_df),
        ("test", test_df)
    ]:

        counts = df["label"].value_counts()

        rows.append({

            "split": name,

            "total_participants": len(df),

            "CN_count": int(counts.get(0, 0)),

            "CDR_positive_count": int(counts.get(1, 0)),

            "CN_percentage": round(
                counts.get(0, 0) / len(df) * 100,
                2
            ),

            "CDR_positive_percentage": round(
                counts.get(1, 0) / len(df) * 100,
                2
            )

        })

    return pd.DataFrame(rows)


def main():

    print("Loading labelled cohort...")

    cohort = load_labeled_cohort()

    print("\nLabel distribution:")

    print(
        cohort["label_name"].value_counts()
    )

    print("\nCreating participant-level splits...")

    train_df, val_df, test_df = create_splits(cohort)

    verify_splits(
        train_df,
        val_df,
        test_df
    )

    # Combine all splits.
    combined = pd.concat(
        [train_df, val_df, test_df],
        ignore_index=True
    )

    # Save the full labelled cohort with split assignments.
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    combined.to_csv(
        OUTPUT_DIR / "participant_splits.csv",
        index=False
    )

    # Save individual split manifests.
    train_df.to_csv(
        OUTPUT_DIR / "train.csv",
        index=False
    )

    val_df.to_csv(
        OUTPUT_DIR / "validation.csv",
        index=False
    )

    test_df.to_csv(
        OUTPUT_DIR / "test.csv",
        index=False
    )

    # Generate summary.
    summary = create_summary(
        train_df,
        val_df,
        test_df
    )

    summary.to_csv(
        OUTPUT_DIR / "split_summary.csv",
        index=False
    )

    print("\n========== SPLIT SUMMARY ==========")

    print(summary.to_string(index=False))

    print("\nFiles saved in:")
    print(OUTPUT_DIR)

    print("\nDataset splitting completed.")


if __name__ == "__main__":
    main()
