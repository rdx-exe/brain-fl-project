import tarfile
from pathlib import PurePosixPath


def inspect_analyze_archive(archive_path):

    print(f"\nInspecting: {archive_path.name}")

    with tarfile.open(archive_path, "r:gz") as tar:

        members = tar.getmembers()

        files = {
            member.name: member
            for member in members
            if member.isfile()
        }

        img_files = [
            name for name in files
            if name.lower().endswith(".img")
        ]

        paired = []
        unpaired = []

        for img_path in img_files:

            hdr_path = str(PurePosixPath(img_path).with_suffix(".hdr"))

            if hdr_path in files:
                paired.append(img_path)
            else:
                unpaired.append(img_path)

        print("Analyze .img files:", len(img_files))
        print("Complete .img/.hdr pairs:", len(paired))
        print("Unpaired .img files:", len(unpaired))

        categories = {
            "RAW": [],
            "T88_111": [],
            "SUBJ_111": [],
        }

        for path in paired:

            if "/RAW/" in path:
                categories["RAW"].append(path)

            if "/T88_111/" in path:
                categories["T88_111"].append(path)

            if "/SUBJ_111/" in path:
                categories["SUBJ_111"].append(path)

        for category, paths in categories.items():

            print(f"\n{category}: {len(paths)} paired images")

            for path in paths[:5]:
                print(" ", path)

        print("\nPotential processed GFC images:")

        candidates = [
            path for path in paired
            if "_gfc.img" in path.lower()
        ]

        for path in candidates[:10]:
            print(" ", path)

        print("\nPotential masked GFC images:")

        masked = [
            path for path in paired
            if "masked_gfc.img" in path.lower()
        ]

        for path in masked[:10]:
            print(" ", path)