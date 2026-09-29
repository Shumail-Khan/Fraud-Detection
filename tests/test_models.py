"""
Unit tests for models and ensemble functionality.
"""

import numpy as np
import pandas as pd
import pytest

from fraud_detection.models.isolation_forest import IsolationForestDetector
from fraud_detection.models.random_forest import RandomForestDetector
from fraud_detection.models.ensemble import HybridFraudEnsemble
from fraud_detection.models.evaluate import evaluate_model


@pytest.fixture
def synthetic_features():
    np.random.seed(42)
    n = 100
    X = pd.DataFrame({
        "f1": np.random.randn(n),
        "f2": np.random.rand(n) * 100,
        "f3": np.random.exponential(1.0, n),
    })
    # Imbalanced labels: 90 normal, 10 fraud
    y = pd.Series([0] * 90 + [1] * 10)
    return X, y


def test_isolation_forest(synthetic_features):
    X, _ = synthetic_features
    iso = IsolationForestDetector(n_estimators=20, contamination=0.1, random_state=42)
    iso.fit(X)

    preds = iso.predict(X)
    proba = iso.predict_proba(X)

    assert len(preds) == len(X)
    assert set(np.unique(preds)).issubset({0, 1})
    assert len(proba) == len(X)
    assert np.all((proba >= 0.0) & (proba <= 1.0))


def test_random_forest(synthetic_features):
    X, y = synthetic_features
    rf = RandomForestDetector(n_estimators=10, max_depth=4, random_state=42)
    rf.fit(X, y)

    preds = rf.predict(X)
    proba = rf.predict_proba(X)

    assert len(preds) == len(X)
    assert len(proba) == len(X)
    assert np.all((proba >= 0.0) & (proba <= 1.0))


def test_hybrid_ensemble(synthetic_features):
    X, y = synthetic_features
    iso = IsolationForestDetector(n_estimators=10, random_state=42)
    rf = RandomForestDetector(n_estimators=10, max_depth=3, random_state=42)

    ensemble = HybridFraudEnsemble(strategy="union", rf_detector=rf, iso_detector=iso)
    ensemble.fit(X, y)

    preds = ensemble.predict(X)
    proba = ensemble.predict_proba(X)

    assert len(preds) == len(X)
    assert len(proba) == len(X)
    assert np.all((proba >= 0.0) & (proba <= 1.0))


def test_evaluate_metrics():
    y_true = pd.Series([0, 0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1, 0])
    amounts = pd.Series([100.0, 200.0, 50.0, 500.0, 300.0])

    metrics = evaluate_model(y_true, y_pred, amounts=amounts, fp_cost_usd=10.0)
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "financial_kpis" in metrics
    # Total fraud = 500 + 300 = 800
    assert metrics["financial_kpis"]["total_fraud_exposure_usd"] == 800.0
    # Prevented fraud = 500
    assert metrics["financial_kpis"]["prevented_fraud_loss_usd"] == 500.0
    # Missed fraud = 300
    assert metrics["financial_kpis"]["missed_fraud_loss_usd"] == 300.0
    # False positive cost = 1 * $10 = $10
    assert metrics["financial_kpis"]["operational_fp_cost_usd"] == 10.0
    # Net savings = 500 - 10 = 490
    assert metrics["financial_kpis"]["net_financial_savings_usd"] == 490.0
