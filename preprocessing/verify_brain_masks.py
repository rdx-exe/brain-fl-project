
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

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


def load_mask(image_path):

    image = nib.load(str(image_path))

    volume = np.asarray(
        image.dataobj,
        dtype=np.float32
    )

    return volume > 0


def main():

    df = pd.read_csv(REPORT_PATH)

    train_df = df[
        df["split"] == "train"
    ].copy()

    print("Training volumes:", len(train_df))

    reference_mask = None

    identical_count = 0
    different_count = 0

    results = []

    for _, record in tqdm(
        train_df.iterrows(),
        total=len(train_df)
    ):

        image_path = (
            PROJECT_ROOT / record["image_path"]
        )

        mask = load_mask(image_path)

        if reference_mask is None:

            reference_mask = mask.copy()

            identical = True

        else:

            identical = np.array_equal(
                reference_mask,
                mask
            )

        if identical:

            identical_count += 1

        else:

            different_count += 1

        intersection = np.logical_and(
            reference_mask,
            mask
        ).sum()

        union = np.logical_or(
            reference_mask,
            mask
        ).sum()

        dice = (
            2 * intersection
            / (
                reference_mask.sum()
                + mask.sum()
            )
        )

        results.append({

            "session_id": record.get(
                "session_id",
                ""
            ),

            "identical_to_reference": identical,

            "mask_voxels": int(mask.sum()),

            "dice_with_reference": float(dice),

        })

    result_df = pd.DataFrame(results)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        OUTPUT_DIR
        / "brain_mask_comparison.csv"
    )

    result_df.to_csv(
        output_path,
        index=False
    )

    print("\n========== MASK COMPARISON ==========")

    print(
        "Identical masks:",
        identical_count
    )

    print(
        "Different masks:",
        different_count
    )

    print(
        "Mean Dice similarity:",
        result_df["dice_with_reference"].mean()
    )

    print(
        "Minimum Dice similarity:",
        result_df["dice_with_reference"].min()
    )

    print(
        "Maximum Dice similarity:",
        result_df["dice_with_reference"].max()
    )

    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()
