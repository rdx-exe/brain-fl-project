
import sys
from pathlib import Path

# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Add project root to Python module search path
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.metadata_loader import OASISMetadataLoader


# ============================================================
# FILE PATHS
# ============================================================

METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "oasis_cross-sectional-5708aa0a98d82080.xlsx"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "oasis_metadata_audit.csv"
)


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():

    print("=" * 65)
    print("OASIS-1 METADATA OUTPUT GENERATION")
    print("=" * 65)

    # Validate metadata file
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {METADATA_PATH}"
        )

    # Initialize metadata loader
    loader = OASISMetadataLoader(METADATA_PATH)

    # Load metadata
    loader.load()

    # Inspect diagnosis information
    loader.inspect_diagnosis()

    # Generate audit table
    audit = loader.create_audit_table()

    if audit is None or audit.empty:
        raise ValueError("Generated metadata audit table is empty.")

    # Validate required columns
    required_columns = {"label", "participant_id"}
    missing_columns = required_columns - set(audit.columns)

    if missing_columns:
        raise KeyError(
            f"Missing required audit columns: {missing_columns}"
        )

    # Display label distribution
    print("\n" + "=" * 65)
    print("LABEL DISTRIBUTION")
    print("=" * 65)

    print(audit["label"].value_counts(dropna=False))

    # Display participant statistics
    print("\nUnique participants:")
    print(audit["participant_id"].nunique())

    # Identify repeated participant records
    print("\nRepeated participant records:")

    repeated = (
        audit["participant_id"]
        .value_counts()
        .loc[lambda x: x > 1]
    )

    if repeated.empty:
        print("No repeated participant records found.")
    else:
        print(repeated)

    # Create output directory if necessary
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Export audit table
    audit.to_csv(OUTPUT_PATH, index=False)

    print("\n" + "=" * 65)
    print("METADATA OUTPUT GENERATED SUCCESSFULLY")
    print("=" * 65)

    print(f"\nTotal records: {len(audit)}")
    print(f"Unique participants: {audit['participant_id'].nunique()}")
    print(f"Output file: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
