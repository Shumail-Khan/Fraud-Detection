"""
Unit tests for data loading and preprocessing.
"""

import pytest
import pandas as pd
from pathlib import Path

from fraud_detection.data.loader import DataLoader
from fraud_detection.data.preprocessor import DataPreprocessor
from fraud_detection.config import AppConfig


@pytest.fixture
def sample_data_path():
    config = AppConfig()
    return config.data.sample_path


def test_data_loader_sample_exists(sample_data_path):
    assert Path(sample_data_path).exists(), "Sample dataset should exist in data/sample/"


def test_data_loader_load(sample_data_path):
    loader = DataLoader(sample_data_path)
    df = loader.load(nrows=50)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 50
    assert "isFraud" in df.columns
    assert "amount" in df.columns


def test_data_preprocessor_encoding():
    df = pd.DataFrame({
        "type": ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"],
        "amount": [10.0, 20.0, 30.0, 40.0, 50.0],
    })
    preprocessor = DataPreprocessor()
    encoded_df = preprocessor.encode_transaction_types(df)

    assert "type_TRANSFER" in encoded_df.columns
    assert "type_CASH_OUT" in encoded_df.columns
    assert encoded_df.loc[1, "type_TRANSFER"] == 1
    assert encoded_df.loc[1, "type_PAYMENT"] == 0


def test_data_preprocessor_split():
    df = pd.DataFrame({
        "isFraud": [0] * 80 + [1] * 20,
        "amount": list(range(100)),
    })
    preprocessor = DataPreprocessor(test_size=0.25, random_state=42)
    train_df, test_df = preprocessor.split_data(df, target_col="isFraud")

    assert len(train_df) == 75
    assert len(test_df) == 25
    # Check stratification
    assert test_df["isFraud"].sum() == 5
