import pandas as pd
from pathlib import Path


class OASISMetadataLoader:

    def __init__(self, file_path):

        self.file_path = Path(file_path)
        self.data = None

    def load(self):

        if not self.file_path.exists():
            raise FileNotFoundError(self.file_path)

        extension = self.file_path.suffix.lower()

        if extension in [".xlsx", ".xls"]:
            self.data = pd.read_excel(self.file_path)

        elif extension == ".csv":
            self.data = pd.read_csv(self.file_path)

        else:
            raise ValueError(
                f"Unsupported metadata format: {extension}"
            )

        print("Metadata loaded:", self.file_path.name)
        return self.data

    def validate_columns(self):

        required = ["ID", "Age", "CDR", "MMSE"]

        missing = [
            column for column in required
            if column not in self.data.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required columns: {missing}"
            )

    def inspect_diagnosis(self):

        self.validate_columns()

        print("\nCDR distribution:")
        print(self.data["CDR"].value_counts(dropna=False).sort_index())

        print("\nMMSE distribution:")
        print(self.data["MMSE"].value_counts(dropna=False).sort_index())

        print("\nRecords with missing CDR:")
        print(self.data["CDR"].isna().sum())

        print("\nRecords by MRI session:")
        print(
            self.data["ID"]
            .astype(str)
            .str.extract(r"(_MR\d+)$")[0]
            .value_counts(dropna=False)
        )

    def create_audit_table(self):

        self.validate_columns()

        df = self.data.copy()

        df["session_id"] = df["ID"].astype(str)

        # Remove the MRI session suffix to obtain
        # the participant-level identifier.
        df["participant_id"] = (
            df["session_id"]
            .str.replace(r"_MR\d+$", "", regex=True)
        )

        df["label"] = pd.NA

        df.loc[df["CDR"] == 0.0, "label"] = "CN"

        df.loc[df["CDR"] > 0.0, "label"] = "Dementia"

        # Keep unknown ratings separate.
        df["label"] = df["label"].astype("string")

        return df