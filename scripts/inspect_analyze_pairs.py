
import sys
from pathlib import Path

# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Register project root for absolute imports
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.analyze_inspector import inspect_analyze_archive


# ============================================================
# ARCHIVE CONFIGURATION
# ============================================================

ARCHIVE_DIR = PROJECT_ROOT / "data" / "raw" / "archives"


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():

    print("=" * 65)
    print("OASIS-1 ANALYZE PAIR INSPECTION")
    print("=" * 65)

    # Verify archive directory
    if not ARCHIVE_DIR.exists():
        raise FileNotFoundError(
            f"Archive directory not found: {ARCHIVE_DIR}"
        )

    # Discover archives
    archives = sorted(ARCHIVE_DIR.glob("*.tar.gz"))

    if not archives:
        raise FileNotFoundError(
            f"No .tar.gz archives found in: {ARCHIVE_DIR}"
        )

    print(f"\nArchive directory: {ARCHIVE_DIR}")
    print(f"Archives found: {len(archives)}")

    # Inspect each archive
    for index, archive in enumerate(archives, start=1):

        print(f"\n[{index}/{len(archives)}] Inspecting: {archive.name}")

        inspect_analyze_archive(archive)

    print("\n" + "=" * 65)
    print("ARCHIVE INSPECTION COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    main()
