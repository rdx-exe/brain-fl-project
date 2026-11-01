
from pathlib import Path

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_DIR = PROJECT_ROOT / "data" / "raw" / "archives"

# We will inspect one representative participant from disc1.
# These are paths inside the TAR archive, not local filesystem paths.

UNMASKED_NAME = (
    "disc1/OAS1_0001_MR1/PROCESSED/MPRAGE/T88_111/"
    "OAS1_0001_MR1_mpr_n4_anon_111_t88_gfc.img"
)

MASKED_NAME = (
    "disc1/OAS1_0001_MR1/PROCESSED/MPRAGE/T88_111/"
    "OAS1_0001_MR1_mpr_n4_anon_111_t88_masked_gfc.img"
)


def extract_sample_pair(archive_path, internal_path, output_dir):

    import tarfile

    output_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive_path, "r:gz") as archive:

        for extension in [".img", ".hdr"]:

            member_name = str(Path(internal_path))

            if extension == ".hdr":
                member_name = str(Path(internal_path).with_suffix(".hdr"))

            member = archive.getmember(member_name)

            destination = output_dir / Path(member_name).name

            source = archive.extractfile(member)

            if source is None:
                raise RuntimeError(f"Cannot read {member_name}")

            with source, destination.open("wb") as output:
                output.write(source.read())

    return output_dir / Path(internal_path).name


def inspect_image(image_path):

    image = nib.load(str(image_path))
    data = image.get_fdata()

    print("\nFile:", image_path.name)
    print("Shape:", data.shape)
    print("Voxel spacing:", image.header.get_zooms())
    print("Data type:", image.get_data_dtype())
    print("Affine:\n", image.affine)
    print("Intensity minimum:", np.min(data))
    print("Intensity maximum:", np.max(data))
    print("Intensity mean:", np.mean(data))
    print("Intensity standard deviation:", np.std(data))

    return data



def visualize_mri(data, title):

    # Remove the final singleton dimension, if present.
    if data.ndim == 4 and data.shape[-1] == 1:
        data = np.squeeze(data, axis=-1)

    if data.ndim != 3:
        raise ValueError(
            f"Expected 3D MRI, received shape {data.shape}"
        )

    print(f"\nVisualizing: {title}")
    print("3D volume shape:", data.shape)

    x, y, z = [dimension // 2 for dimension in data.shape]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(
        data[:, :, z].T,
        cmap="gray",
        origin="lower"
    )
    axes[0].set_title("Axial")

    axes[1].imshow(
        data[:, y, :].T,
        cmap="gray",
        origin="lower"
    )
    axes[1].set_title("Coronal")

    axes[2].imshow(
        data[x, :, :].T,
        cmap="gray",
        origin="lower"
    )
    axes[2].set_title("Sagittal")

    fig.suptitle(title)

    for axis in axes:
        axis.axis("off")

    plt.tight_layout()

    output_path = (
        PROJECT_ROOT
        / "data"
        / "sample_inspection"
        / f"{title.lower().replace(' ', '_')}.png"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=200, bbox_inches="tight")

    print("Saved visualization:", output_path)

    plt.show()


def main():

    archive_path = (
        ARCHIVE_DIR / "oasis_cross-sectional_disc1.tar.gz"
    )

    sample_dir = PROJECT_ROOT / "data" / "sample_inspection"

    unmasked_path = extract_sample_pair(
        archive_path,
        UNMASKED_NAME,
        sample_dir / "unmasked"
    )

    masked_path = extract_sample_pair(
        archive_path,
        MASKED_NAME,
        sample_dir / "masked"
    )

    unmasked_data = inspect_image(unmasked_path)
    masked_data = inspect_image(masked_path)

    visualize_mri(unmasked_data, "Unmasked GFC MRI")
    visualize_mri(masked_data, "Masked GFC MRI")


if __name__ == "__main__":
    main()
