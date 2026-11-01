
import sys
from pathlib import Path

# ============================================================
# PROJECT CONFIGURATION
# ============================================================

# Resolve project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Register project root for absolute imports
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.metadata_loader import OASISMetadataLoader


# ============================================================
# METADATA CONFIGURATION
# ============================================================

METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "oasis_cross-sectional-5708aa0a98d82080.xlsx"
)


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():

    print("=" * 65)
    print("OASIS-1 METADATA INSPECTION")
    print("=" * 65)

    # Verify metadata file exists
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {METADATA_PATH}"
        )

    # Initialize metadata loader
    loader = OASISMetadataLoader(METADATA_PATH)

    # Load metadata
    loader.load()

    # Inspect dataset structure
    loader.inspect()

    # Inspect clinical variables
    loader.inspect_clinical_columns()

    print("\nMetadata inspection completed successfully.")


if __name__ == "__main__":
    main()
