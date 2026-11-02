
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm


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


def load_volume(image_path):

    image = nib.load(str(image_path))

    volume = np.asarray(
        image.dataobj,
        dtype=np.float32
    )

    if volume.ndim == 4 and volume.shape[-1] == 1:
        volume = volume[..., 0]

    if volume.ndim != 3:
        raise ValueError(
            f"Expected 3D MRI, got {volume.shape}"
        )

    return volume


def calculate_intensity_statistics(volume):

    # Analyze positive voxels only.
    # These correspond to the non-zero region
    # in the selected masked MRI representation.

    values = volume[volume > 0]

    if values.size == 0:
        raise ValueError("Volume contains no positive voxels.")

    percentiles = np.percentile(
        values,
        [1, 5, 25, 50, 75, 95, 99]
    )

    return {
        "voxel_count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "p01": float(percentiles[0]),
        "p05": float(percentiles[1]),
        "p25": float(percentiles[2]),
        "p50": float(percentiles[3]),
        "p75": float(percentiles[4]),
        "p95": float(percentiles[5]),
        "p99": float(percentiles[6]),
    }


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df = pd.read_csv(REPORT_PATH)

    train_df = df[
        df["split"] == "train"
    ].copy()

    print(
        "Training volumes:",
        len(train_df)
    )

    records = []

    sample_distributions = []

    for _, record in tqdm(
        train_df.iterrows(),
        total=len(train_df)
    ):

        image_path = (
            PROJECT_ROOT / record["image_path"]
        )

        volume = load_volume(image_path)

        statistics = calculate_intensity_statistics(
            volume
        )

        statistics["session_id"] = record.get(
            "session_id",
            ""
        )

        records.append(statistics)

        # Subsample positive voxels for visualization.
        values = volume[volume > 0]

        if values.size > 50000:

            rng = np.random.default_rng(42)

            values = rng.choice(
                values,
                size=50000,
                replace=False
            )

        sample_distributions.append(values)

    result_df = pd.DataFrame(records)

    output_csv = (
        OUTPUT_DIR
        / "intensity_distribution.csv"
    )

    result_df.to_csv(
        output_csv,
        index=False
    )

    # --------------------------------------------------------
    # Summary statistics across participants.
    # --------------------------------------------------------

    columns = [
        "mean",
        "std",
        "p01",
        "p05",
        "p25",
        "p50",
        "p75",
        "p95",
        "p99",
    ]

    print("\n========== INTENSITY SUMMARY ==========")

    print(
        result_df[columns].describe().round(4)
    )

    # --------------------------------------------------------
    # Plot participant-level intensity distributions.
    # --------------------------------------------------------

    plt.figure(figsize=(13, 6))

    for values in sample_distributions:

        plt.hist(
            values,
            bins=100,
            density=True,
            histtype="step",
            alpha=0.12
        )

    plt.xlabel("MRI intensity")

    plt.ylabel("Density")

    plt.title(
        "Training Cohort: MRI Intensity Distributions"
    )

    plt.grid(alpha=0.2)

    plt.tight_layout()

    histogram_path = (
        OUTPUT_DIR
        / "intensity_distributions.png"
    )

    plt.savefig(
        histogram_path,
        dpi=200
    )

    plt.close()

    # --------------------------------------------------------
    # Plot participant-level intensity percentiles.
    # --------------------------------------------------------

    plt.figure(figsize=(13, 6))

    for column in [
        "p01",
        "p05",
        "p25",
        "p50",
        "p75",
        "p95",
        "p99",
    ]:

        plt.plot(
            result_df[column].values,
            label=column
        )

    plt.xlabel("Training participant index")

    plt.ylabel("MRI intensity")

    plt.title(
        "Training Cohort: Participant Intensity Percentiles"
    )

    plt.legend()

    plt.grid(alpha=0.2)

    plt.tight_layout()

    percentile_path = (
        OUTPUT_DIR
        / "intensity_percentiles.png"
    )

    plt.savefig(
        percentile_path,
        dpi=200
    )

    plt.close()

    print("\nSaved files:")

    print(output_csv)

    print(histogram_path)

    print(percentile_path)


if __name__ == "__main__":
    main()
