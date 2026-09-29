"""
Unit tests for the real-time stream processing simulator.
"""

import pandas as pd
from fraud_detection.streaming.simulator import StreamSimulator
from fraud_detection.models.isolation_forest import IsolationForestDetector


def test_stream_simulator_execution():
    df = pd.DataFrame({
        "f1": [1.0, 2.0, 50.0],
        "amount": [10.0, 20.0, 50000.0],
        "type": ["PAYMENT", "TRANSFER", "CASH_OUT"],
        "nameOrig": ["C1", "C2", "C3"],
        "nameDest": ["M1", "C4", "C5"],
    })
    feature_cols = ["f1", "amount"]

    model = IsolationForestDetector(n_estimators=10, random_state=42)
    model.fit(df[feature_cols])

    alerts = []
    simulator = StreamSimulator(
        model=model,
        feature_columns=feature_cols,
        alert_callback=lambda event: alerts.append(event),
    )

    summary = simulator.simulate(df, delay_seconds=0.0, max_events=3)

    assert summary["total_transactions_processed"] == 3
    assert "throughput_tx_per_sec" in summary
    assert "latency_avg_ms" in summary
    assert summary["latency_avg_ms"] > 0
