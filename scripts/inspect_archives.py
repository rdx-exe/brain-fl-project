import sys
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Add project root to Python module search path
sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.archive_inspector import inspect_archive


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ARCHIVE_DIR = PROJECT_ROOT / "data" / "raw" / "archives"


def main():

    archives = list(ARCHIVE_DIR.glob("*.tar.gz"))

    archives += list(ARCHIVE_DIR.glob("*.tgz"))

    if not archives:
        print("[ERROR] No TAR archives found.")
        print("Expected location:", ARCHIVE_DIR)
        return

    print(f"Found {len(archives)} archives.")

    for archive in sorted(set(archives)):
        inspect_archive(archive)


if __name__ == "__main__":
    main()