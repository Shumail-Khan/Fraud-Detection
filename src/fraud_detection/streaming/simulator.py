"""
Near-Real-Time Stream Processing Simulator.
Simulates live transaction ingestion, continuous scoring, latency tracking, and fraud alerting.
"""

import time
from typing import Callable, Dict, Any, List, Optional
import numpy as np
import pandas as pd

from fraud_detection.models.base import BaseFraudDetector
from fraud_detection.utils.logger import setup_logger

logger = setup_logger(__name__)


class StreamSimulator:
    """Simulates near-real-time streaming transaction processing."""

    def __init__(
        self,
        model: BaseFraudDetector,
        feature_columns: List[str],
        alert_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self.model = model
        self.feature_columns = feature_columns
        self.alert_callback = alert_callback or self._default_alert

    @staticmethod
    def _default_alert(event: Dict[str, Any]) -> None:
        """Default alerting action when fraud is detected."""
        print(
            f"[ALERT] [FRAUD DETECTED] Tx #{event['index']} | Amount: ${event['amount']:,.2f} | "
            f"Type: {event.get('type', 'N/A')} | Sender: {event.get('nameOrig', 'N/A')} -> "
            f"Dest: {event.get('nameDest', 'N/A')} | Risk Score: {event['risk_score']:.4f} | "
            f"Latency: {event['latency_ms']:.2f}ms"
        )

    def process_single(self, row: pd.Series, event_index: int = 1) -> Dict[str, Any]:
        """Scores a single incoming transaction record in near-real-time."""
        start_time = time.perf_counter()

        # Build feature vector
        features_dict = {col: [row.get(col, 0.0)] for col in self.feature_columns}
        X_vec = pd.DataFrame(features_dict)

        # Score with model
        is_fraud = int(self.model.predict(X_vec)[0])
        risk_score = float(self.model.predict_proba(X_vec)[0])

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        event_result = {
            "index": event_index,
            "is_fraud": is_fraud,
            "risk_score": risk_score,
            "latency_ms": latency_ms,
            "amount": float(row.get("amount", 0.0)),
            "type": row.get("type", "UNKNOWN"),
            "nameOrig": row.get("nameOrig", "N/A"),
            "nameDest": row.get("nameDest", "N/A"),
        }

        if is_fraud:
            self.alert_callback(event_result)

        return event_result

    def simulate(
        self,
        stream_df: pd.DataFrame,
        delay_seconds: float = 0.01,
        max_events: Optional[int] = 100,
    ) -> Dict[str, Any]:
        """
        Executes a real-time streaming simulation across candidate transactions.
        
        Args:
            stream_df: DataFrame of incoming transactions.
            delay_seconds: Artificial sleep between events to simulate real network interval.
            max_events: Maximum number of transactions to stream.
        """
        records = stream_df.head(max_events) if max_events else stream_df
        logger.info(f"Starting real-time simulation for {len(records)} transactions...")

        latencies: List[float] = []
        frauds_detected = 0
        total_volume_processed = 0.0
        fraud_volume_intercepted = 0.0

        sim_start = time.perf_counter()

        for idx, (_, row) in enumerate(records.iterrows(), start=1):
            res = self.process_single(row, event_index=idx)
            latencies.append(res["latency_ms"])
            total_volume_processed += res["amount"]

            if res["is_fraud"]:
                frauds_detected += 1
                fraud_volume_intercepted += res["amount"]

            if delay_seconds > 0:
                time.sleep(delay_seconds)

        total_sim_time = time.perf_counter() - sim_start

        latencies_arr = np.array(latencies)
        summary = {
            "total_transactions_processed": len(records),
            "total_frauds_intercepted": frauds_detected,
            "fraud_rate_pct": (frauds_detected / len(records) * 100.0) if len(records) > 0 else 0.0,
            "total_volume_usd": total_volume_processed,
            "fraud_volume_intercepted_usd": fraud_volume_intercepted,
            "total_runtime_seconds": total_sim_time,
            "throughput_tx_per_sec": len(records) / total_sim_time if total_sim_time > 0 else 0.0,
            "latency_p50_ms": float(np.percentile(latencies_arr, 50)),
            "latency_p95_ms": float(np.percentile(latencies_arr, 95)),
            "latency_p99_ms": float(np.percentile(latencies_arr, 99)),
            "latency_avg_ms": float(np.mean(latencies_arr)),
        }

        print("\n" + "=" * 65)
        print("  STREAMING SIMULATION BENCHMARK RESULTS")
        print("=" * 65)
        print(f"  Processed Transactions      : {summary['total_transactions_processed']:,}")
        print(f"  Frauds Intercepted          : {summary['total_frauds_intercepted']:,} ({summary['fraud_rate_pct']:.2f}%)")
        print(f"  Total Processed Volume      : ${summary['total_volume_usd']:,.2f}")
        print(f"  Intercepted Fraud Volume    : ${summary['fraud_volume_intercepted_usd']:,.2f}")
        print(f"  Throughput                  : {summary['throughput_tx_per_sec']:.2f} tx/sec")
        print(f"  Inference Latency (Avg)     : {summary['latency_avg_ms']:.2f} ms")
        print(f"  Inference Latency (P50)     : {summary['latency_p50_ms']:.2f} ms")
        print(f"  Inference Latency (P95)     : {summary['latency_p95_ms']:.2f} ms")
        print(f"  Inference Latency (P99)     : {summary['latency_p99_ms']:.2f} ms")
        print("=" * 65 + "\n")

        return summary
