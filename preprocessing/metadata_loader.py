
import pandas as pd
from pathlib import Path


class OASISMetadataLoader:

    def __init__(self, metadata_path):
        self.metadata_path = Path(metadata_path)
        self.data = None

    def load(self):

        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {self.metadata_path}"
            )

        suffix = self.metadata_path.suffix.lower()

        # Load according to file format
        if suffix == ".csv":
            self.data = pd.read_csv(self.metadata_path)

        elif suffix in [".xlsx", ".xls"]:
            self.data = pd.read_excel(self.metadata_path)

        else:
            raise ValueError(
                f"Unsupported metadata format: {suffix}"
            )

        print("\nMetadata loaded successfully.")
        print(f"File: {self.metadata_path.name}")
        print(f"Records: {len(self.data)}")
        print(f"Columns: {len(self.data.columns)}")

        return self.data

    def inspect(self):

        if self.data is None:
            raise RuntimeError("Load metadata before inspection.")

        print("\n" + "=" * 65)
        print("METADATA STRUCTURE")
        print("=" * 65)

        print("\nDataset shape:")
        print(self.data.shape)

        print("\nColumn names:")
        for column in self.data.columns:
            print(f" - {column}")

        print("\nFirst five records:")
        print(self.data.head().to_string())

        print("\nData types:")
        print(self.data.dtypes)

        print("\nMissing values:")
        print(self.data.isnull().sum())

    def inspect_clinical_columns(self):

        if self.data is None:
            raise RuntimeError("Load metadata before inspection.")

        print("\n" + "=" * 65)
        print("CLINICAL VARIABLE INSPECTION")
        print("=" * 65)

        # Identify potentially relevant clinical variables
        keywords = [
            "cdr",
            "age",
            "gender",
            "educ",
            "ses",
            "mmse",
            "hand",
            "group",
            "diagnosis"
        ]

        clinical_columns = [
            column for column in self.data.columns
            if any(
                keyword in str(column).lower()
                for keyword in keywords
            )
        ]

        if not clinical_columns:
            print("\nNo matching clinical columns found.")
            return

        for column in clinical_columns:

            print(f"\nColumn: {column}")
            print(f"Unique values: {self.data[column].nunique(dropna=False)}")
            print(self.data[column].value_counts(dropna=False).head(20))
