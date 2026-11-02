
from pathlib import Path
from collections import Counter

import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPORT_PATH = (
    PROJECT_ROOT
    / "results"
    / "quality_control"
    / "extraction_report.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "preprocessing_analysis"
)

TRAIN_SPLIT = "train"


# ============================================================
# MRI LOADING
# ============================================================

def load_volume(image_path):
    """
    Load an MRI volume and return:
        - 3D MRI volume
        - Anatomical orientation
        - Voxel spacing
        - Original shape
    """

    image = nib.load(str(image_path))

    volume = np.asarray(
        image.dataobj,
        dtype=np.float32
    )

    # Remove singleton fourth dimension if present.
    if volume.ndim == 4 and volume.shape[-1] == 1:
        volume = volume[..., 0]

    if volume.ndim != 3:
        raise ValueError(
            f"Expected 3D volume, got {volume.shape}"
        )

    # Determine anatomical orientation from affine.
    orientation = nib.aff2axcodes(image.affine)

    # Retrieve voxel spacing.
    voxel_spacing = image.header.get_zooms()[:3]

    return {
        "volume": volume,
        "orientation": orientation,
        "voxel_spacing": voxel_spacing,
        "shape": volume.shape
    }


# ============================================================
# SLICE ANALYSIS
# ============================================================

def analyze_volume(volume):
    """
    Calculate slice-wise:
        - Non-zero voxel occupancy
        - Mean non-zero intensity

    Axis 2 is used as the slice axis.
    Its anatomical interpretation is verified separately
    using the image affine.
    """

    total_slices = volume.shape[2]

    occupancy = []
    mean_intensity = []

    for index in range(total_slices):

        slice_data = volume[:, :, index]

        nonzero = slice_data > 0

        occupancy.append(
            float(nonzero.mean())
        )

        if nonzero.any():

            mean_intensity.append(
                float(slice_data[nonzero].mean())
            )

        else:

            mean_intensity.append(0.0)

    return (
        np.asarray(occupancy, dtype=np.float32),
        np.asarray(mean_intensity, dtype=np.float32)
    )


# ============================================================
# POPULATION-LEVEL STATISTICS
# ============================================================

def calculate_statistics(
    occupancy_array,
    intensity_array
):
    """
    Calculate cohort-level slice statistics.
    """

    mean_occupancy = occupancy_array.mean(axis=0)

    median_occupancy = np.median(
        occupancy_array,
        axis=0
    )

    p25_occupancy = np.percentile(
        occupancy_array,
        25,
        axis=0
    )

    p75_occupancy = np.percentile(
        occupancy_array,
        75,
        axis=0
    )

    occupancy_std = occupancy_array.std(
        axis=0
    )

    mean_intensity = intensity_array.mean(
        axis=0
    )

    median_intensity = np.median(
        intensity_array,
        axis=0
    )

    intensity_std = intensity_array.std(
        axis=0
    )

    slice_indices = np.arange(
        occupancy_array.shape[1]
    )

    results = pd.DataFrame({

        "slice_index": slice_indices,

        "mean_occupancy": mean_occupancy,

        "median_occupancy": median_occupancy,

        "p25_occupancy": p25_occupancy,

        "p75_occupancy": p75_occupancy,

        "occupancy_std": occupancy_std,

        "mean_nonzero_intensity": mean_intensity,

        "median_nonzero_intensity": median_intensity,

        "intensity_std": intensity_std,

    })

    return results


# ============================================================
# VARIABILITY DIAGNOSTICS
# ============================================================

def analyze_variability(occupancy_array):
    """
    Investigate whether participants have identical
    or highly similar occupancy profiles.
    """

    print("\n========== OCCUPANCY VARIABILITY ==========")

    unique_profiles = np.unique(
        occupancy_array,
        axis=0
    )

    occupancy_std = occupancy_array.std(
        axis=0
    )

    print(
        "Total participants:",
        occupancy_array.shape[0]
    )

    print(
        "Unique occupancy profiles:",
        unique_profiles.shape[0]
    )

    print(
        "Mean standard deviation across slices:",
        float(occupancy_std.mean())
    )

    print(
        "Maximum standard deviation across slices:",
        float(occupancy_std.max())
    )

    print(
        "Minimum standard deviation across slices:",
        float(occupancy_std.min())
    )

    identical = np.allclose(
        occupancy_array,
        occupancy_array[0]
    )

    print(
        "All occupancy profiles identical:",
        identical
    )

    if identical:

        print(
            "\nWARNING: All participants have identical "
            "occupancy profiles."
        )

        print(
            "Investigate image masks, preprocessing, "
            "and extraction consistency."
        )

    elif unique_profiles.shape[0] < occupancy_array.shape[0]:

        print(
            "\nNOTE: Some participants share identical "
            "occupancy profiles."
        )

    else:

        print(
            "\nOccupancy profiles vary across participants."
        )


# ============================================================
# ORIENTATION DIAGNOSTICS
# ============================================================

def analyze_orientations(orientation_list):

    print("\n========== MRI ORIENTATION ==========")

    orientation_counts = Counter(
        orientation_list
    )

    for orientation, count in orientation_counts.items():

        print(
            f"{orientation}: {count} volumes"
        )

    if len(orientation_counts) == 1:

        print(
            "\nAll MRI volumes share the same "
            "anatomical orientation."
        )

    else:

        print(
            "\nWARNING: Multiple anatomical orientations "
            "were detected."
        )

        print(
            "Volumes should be reoriented consistently "
            "before model training."
        )


# ============================================================
# VISUALIZATION
# ============================================================

def plot_occupancy(results):

    slice_indices = results["slice_index"]

    plt.figure(figsize=(13, 5))

    plt.plot(
        slice_indices,
        results["mean_occupancy"],
        label="Mean occupancy"
    )

    plt.plot(
        slice_indices,
        results["median_occupancy"],
        label="Median occupancy"
    )

    plt.fill_between(
        slice_indices,
        results["p25_occupancy"],
        results["p75_occupancy"],
        alpha=0.2,
        label="25th–75th percentile"
    )

    plt.xlabel("Slice index")

    plt.ylabel("Non-zero voxel fraction")

    plt.title(
        "Training Cohort: Slice Occupancy"
    )

    plt.legend()

    plt.grid(alpha=0.25)

    plt.tight_layout()

    output_path = OUTPUT_DIR / "slice_occupancy.png"

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved:", output_path)


def plot_intensity(results):

    slice_indices = results["slice_index"]

    plt.figure(figsize=(13, 5))

    plt.plot(
        slice_indices,
        results["mean_nonzero_intensity"],
        label="Mean intensity"
    )

    plt.plot(
        slice_indices,
        results["median_nonzero_intensity"],
        label="Median intensity"
    )

    plt.xlabel("Slice index")

    plt.ylabel("Mean non-zero intensity")

    plt.title(
        "Training Cohort: Slice Intensity"
    )

    plt.legend()

    plt.grid(alpha=0.25)

    plt.tight_layout()

    output_path = OUTPUT_DIR / "slice_intensity.png"

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved:", output_path)


def plot_occupancy_variability(results):

    plt.figure(figsize=(13, 5))

    plt.plot(
        results["slice_index"],
        results["occupancy_std"],
        label="Occupancy standard deviation"
    )

    plt.xlabel("Slice index")

    plt.ylabel("Occupancy standard deviation")

    plt.title(
        "Training Cohort: Slice-wise Occupancy Variability"
    )

    plt.legend()

    plt.grid(alpha=0.25)

    plt.tight_layout()

    output_path = OUTPUT_DIR / "slice_occupancy_variability.png"

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved:", output_path)


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if not REPORT_PATH.exists():

        raise FileNotFoundError(
            f"Extraction report not found: {REPORT_PATH}"
        )

    print("Loading extraction report...")

    df = pd.read_csv(REPORT_PATH)

    required_columns = {
        "split",
        "image_path"
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:

        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    train_df = df[
        df["split"] == TRAIN_SPLIT
    ].copy()

    if train_df.empty:

        raise ValueError(
            "No training records found."
        )

    print(
        "\nTraining volumes:",
        len(train_df)
    )

    occupancy_curves = []
    intensity_curves = []

    orientation_list = []
    voxel_spacing_list = []
    volume_shapes = []

    # --------------------------------------------------------
    # Load and analyze each training volume.
    # --------------------------------------------------------

    for _, record in tqdm(
        train_df.iterrows(),
        total=len(train_df),
        desc="Analyzing MRI volumes"
    ):

        image_path = (
            PROJECT_ROOT / record["image_path"]
        )

        if not image_path.exists():

            raise FileNotFoundError(
                f"MRI volume not found: {image_path}"
            )

        loaded = load_volume(image_path)

        volume = loaded["volume"]

        orientation_list.append(
            loaded["orientation"]
        )

        voxel_spacing_list.append(
            loaded["voxel_spacing"]
        )

        volume_shapes.append(
            loaded["shape"]
        )

        occupancy, intensity = analyze_volume(
            volume
        )

        occupancy_curves.append(occupancy)

        intensity_curves.append(intensity)

    # --------------------------------------------------------
    # Validate dimensional consistency.
    # --------------------------------------------------------

    print("\n========== VOLUME CONSISTENCY ==========")

    unique_shapes = Counter(volume_shapes)

    for shape, count in unique_shapes.items():

        print(
            f"Shape {shape}: {count} volumes"
        )

    if len(unique_shapes) != 1:

        raise ValueError(
            "MRI volumes have different dimensions. "
            "Standardize volume dimensions before "
            "cohort-level slice analysis."
        )

    unique_spacing = Counter(
        tuple(round(float(x), 5) for x in spacing)
        for spacing in voxel_spacing_list
    )

    print("\nVoxel spacing distribution:")

    for spacing, count in unique_spacing.items():

        print(
            f"{spacing}: {count} volumes"
        )

    # --------------------------------------------------------
    # Stack participant-level statistics.
    # --------------------------------------------------------

    occupancy_array = np.stack(
        occupancy_curves
    )

    intensity_array = np.stack(
        intensity_curves
    )

    # --------------------------------------------------------
    # Orientation and variability diagnostics.
    # --------------------------------------------------------

    analyze_orientations(
        orientation_list
    )

    analyze_variability(
        occupancy_array
    )

    # --------------------------------------------------------
    # Calculate cohort-level statistics.
    # --------------------------------------------------------

    results = calculate_statistics(
        occupancy_array,
        intensity_array
    )

    csv_path = (
        OUTPUT_DIR
        / "axial_slice_coverage.csv"
    )

    results.to_csv(
        csv_path,
        index=False
    )

    # --------------------------------------------------------
    # Generate visualizations.
    # --------------------------------------------------------

    plot_occupancy(results)

    plot_intensity(results)

    plot_occupancy_variability(results)

    # --------------------------------------------------------
    # Identify active slices.
    # --------------------------------------------------------

    active = np.where(
        results["median_occupancy"].to_numpy() > 0
    )[0]

    print("\n========== SLICE ANALYSIS ==========")

    print(
        "Total slices:",
        len(results)
    )

    print(
        "Slices with non-zero median occupancy:",
        len(active)
    )

    if len(active):

        print(
            "First active slice:",
            int(active.min())
        )

        print(
            "Last active slice:",
            int(active.max())
        )

    # --------------------------------------------------------
    # Highest occupancy slices.
    # --------------------------------------------------------

    print("\nHighest mean occupancy slices:")

    print(
        results.nlargest(
            10,
            "mean_occupancy"
        ).to_string(index=False)
    )

    # --------------------------------------------------------
    # Highest occupancy variability.
    # --------------------------------------------------------

    print("\nHighest occupancy variability slices:")

    print(
        results.nlargest(
            10,
            "occupancy_std"
        )[
            [
                "slice_index",
                "mean_occupancy",
                "occupancy_std"
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # Save summary.
    # --------------------------------------------------------

    summary_path = (
        OUTPUT_DIR
        / "slice_analysis_summary.txt"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "MRI SLICE ANALYSIS SUMMARY\n"
        )

        file.write("==========================\n\n")

        file.write(
            f"Training volumes: {len(train_df)}\n"
        )

        file.write(
            f"Volume shape: {volume_shapes[0]}\n"
        )

        file.write(
            f"Unique orientations: {len(set(orientation_list))}\n"
        )

        file.write(
            f"Unique occupancy profiles: "
            f"{np.unique(occupancy_array, axis=0).shape[0]}\n"
        )

        file.write(
            f"Mean occupancy standard deviation: "
            f"{occupancy_array.std(axis=0).mean():.8f}\n"
        )

        file.write(
            f"First active slice: "
            f"{int(active.min()) if len(active) else 'N/A'}\n"
        )

        file.write(
            f"Last active slice: "
            f"{int(active.max()) if len(active) else 'N/A'}\n"
        )

    # --------------------------------------------------------
    # Final output.
    # --------------------------------------------------------

    print("\n========== ANALYSIS COMPLETED ==========")

    print("\nSaved files:")

    print(csv_path)

    print(
        OUTPUT_DIR / "slice_occupancy.png"
    )

    print(
        OUTPUT_DIR / "slice_intensity.png"
    )

    print(
        OUTPUT_DIR / "slice_occupancy_variability.png"
    )

    print(summary_path)


if __name__ == "__main__":
    main()
