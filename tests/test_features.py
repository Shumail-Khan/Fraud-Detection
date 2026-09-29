"""
Unit tests for feature engineering transformations.
"""

import numpy as np
import pandas as pd
from fraud_detection.features.engineering import FeatureEngineer


def test_feature_engineer_log_transform():
    df = pd.DataFrame({
        "nameOrig": ["C1", "C2"],
        "amount": [0.0, 99.0],
        "oldbalanceOrg": [100.0, 200.0],
        "newbalanceOrig": [100.0, 101.0],
        "oldbalanceDest": [0.0, 0.0],
        "newbalanceDest": [0.0, 99.0],
        "isFraud": [0, 1],
    })
    fe = FeatureEngineer()
    out = fe.fit_transform(df)

    assert "log_amount" in out.columns
    assert np.isclose(out.loc[0, "log_amount"], 0.0)
    assert np.isclose(out.loc[1, "log_amount"], np.log1p(99.0))


def test_feature_engineer_balance_discrepancy():
    # Valid transaction: old (100) - amount (50) == new (50) -> error == 0
    # Fraud transaction: old (100) - amount (50) != new (0) -> error != 0
    df = pd.DataFrame({
        "nameOrig": ["C1", "C2"],
        "amount": [50.0, 50.0],
        "oldbalanceOrg": [100.0, 100.0],
        "newbalanceOrig": [50.0, 0.0],
        "oldbalanceDest": [0.0, 0.0],
        "newbalanceDest": [50.0, 0.0],
        "isFraud": [0, 1],
    })
    fe = FeatureEngineer()
    out = fe.fit_transform(df)

    assert "error_balance_orig" in out.columns
    assert out.loc[0, "error_balance_orig"] == 0.0
    assert out.loc[1, "error_balance_orig"] == -50.0
