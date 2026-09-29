"""
Command-line interface to simulate near-real-time streaming transaction processing.
Loads serialized models and streams transactions measuring latency, alerts, and throughput.
"""

import argparse
import sys
from pathlib import Path
import joblib
import pandas as pd

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fraud_detection.config import AppConfig
from fraud_detection.data.loader import DataLoader
from fraud_detection.data.preprocessor import DataPreprocessor
from fraud_detection.features.engineering import FeatureEngineer
from fraud_detection.streaming.simulator import StreamSimulator
from fraud_detection.utils.logger import setup_logger

logger = setup_logger("stream_cli")


def run_streaming(
    data_path: str = None,
    model_path: str = "models/ensemble_pipeline.joblib",
    fe_path: str = "models/feature_engineer.joblib",
    events: int = 100,
    delay: float = 0.02,
) -> None:
    config = AppConfig()
    selected_data_path = Path(data_path) if data_path else config.data.active_data_path

    # Verify model artifact exists
    m_path = Path(model_path)
    if not m_path.exists():
        logger.warning(f"Trained model not found at '{m_path}'. Training quick model first...")
        from scripts.train import run_pipeline
        run_pipeline(data_path=str(selected_data_path), nrows=5000)

    logger.info(f"Loading trained model from: {m_path}")
    model = joblib.load(m_path)

    fe_file = Path(fe_path)
    if fe_file.exists():
        logger.info(f"Loading fitted FeatureEngineer from: {fe_file}")
        feature_engineer = joblib.load(fe_file)
    else:
        feature_engineer = FeatureEngineer()

    # Load stream data
    loader = DataLoader(selected_data_path)
    df = loader.load(nrows=events * 5)

    preprocessor = DataPreprocessor()
    df_encoded = preprocessor.encode_transaction_types(df)
    df_features = feature_engineer.transform(df_encoded)

    # Initialize Simulator
    feature_cols = config.features.model_features
    simulator = StreamSimulator(
        model=model,
        feature_columns=feature_cols,
    )

    # Run stream
    simulator.simulate(
        stream_df=df_features,
        delay_seconds=delay,
        max_events=events,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate real-time streaming transaction scoring")
    parser.add_argument("--data", type=str, default=None, help="Path to input transactions CSV")
    parser.add_argument("--model", type=str, default="models/ensemble_pipeline.joblib", help="Model path")
    parser.add_argument("--events", type=int, default=100, help="Number of transactions to stream")
    parser.add_argument("--delay", type=float, default=0.01, help="Simulated network delay in seconds")
    args = parser.parse_args()

    run_streaming(
        data_path=args.data,
        model_path=args.model,
        events=args.events,
        delay=args.delay,
    )
