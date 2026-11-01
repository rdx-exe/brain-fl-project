import tarfile
from pathlib import Path
from collections import Counter


def inspect_archive(archive_path):
    """
    Inspect a TAR/TAR.GZ archive without extracting it.
    """

    archive_path = Path(archive_path)

    if not archive_path.exists():
        print(f"[ERROR] File not found: {archive_path}")
        return

    print("\n" + "=" * 65)
    print(f"ARCHIVE: {archive_path.name}")
    print("=" * 65)

    try:
        with tarfile.open(archive_path, mode="r:*") as tar:

            members = tar.getmembers()

            print(f"Total entries: {len(members)}")

            file_count = sum(1 for m in members if m.isfile())
            directory_count = sum(1 for m in members if m.isdir())

            print(f"Files: {file_count}")
            print(f"Directories: {directory_count}")

            extensions = Counter()

            for member in members:
                if member.isfile():
                    extension = Path(member.name).suffix.lower()
                    extensions[extension or "[no extension]"] += 1

            print("\nFile extensions:")

            for extension, count in extensions.most_common():
                print(f"{extension:20} {count}")

            print("\nFirst 30 archive entries:")

            for member in members[:30]:
                print(member.name)

            print("\nPotential MRI files:")

            mri_files = [
                member.name
                for member in members
                if member.isfile()
                and (
                    member.name.lower().endswith(".nii")
                    or member.name.lower().endswith(".nii.gz")
                )
            ]

            for filename in mri_files[:20]:
                print(filename)

            print(f"\nTotal NIfTI files: {len(mri_files)}")

    except (tarfile.TarError, OSError) as error:
        print(f"[ERROR] Could not inspect archive: {error}")