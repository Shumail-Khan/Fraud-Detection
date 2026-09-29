"""
Data loading and ingestion utilities.
Supports loading full datasets, stratified samples, and chunked processing.
"""

from pathlib import Path
from typing import Iterator, Optional, Union
import pandas as pd

from fraud_detection.utils.logger import setup_logger

logger = setup_logger(__name__)

EXPECTED_COLUMNS = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
]


class DataLoader:
    """Loads and validates transaction datasets."""

    def __init__(self, data_path: Union[str, Path]):
        self.data_path = Path(data_path)

    def validate_schema(self, df: pd.DataFrame) -> bool:
        """Validates that all essential columns are present."""
        missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"Dataset missing required columns: {missing}")
        return True

    def load(self, nrows: Optional[int] = None) -> pd.DataFrame:
        """Loads dataset into a Pandas DataFrame."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found at: {self.data_path}")

        logger.info(f"Loading data from: {self.data_path} (nrows={nrows})")
        df = pd.read_csv(self.data_path, nrows=nrows)
        self.validate_schema(df)
        logger.info(
            f"Loaded {len(df):,} transactions. Fraud cases: {df['isFraud'].sum():,} "
            f"({(df['isFraud'].mean() * 100):.3f}%)"
        )
        return df

    def stream_chunks(self, chunksize: int = 50_000) -> Iterator[pd.DataFrame]:
        """Yields chunks of the dataset for memory-efficient streaming."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found at: {self.data_path}")

        logger.info(f"Streaming data in chunks of {chunksize:,} rows from: {self.data_path}")
        for chunk in pd.read_csv(self.data_path, chunksize=chunksize):
            yield chunk
