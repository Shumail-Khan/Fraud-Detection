"""
Command-line interface for training, evaluating, and serializing
the Fraud Detection model pipeline.
"""

import argparse
import json
import os
import sys
from pathlib import Path
import joblib

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fraud_detection.config import AppConfig
from fraud_detection.data.loader import DataLoader
from fraud_detection.data.preprocessor import DataPreprocessor
from fraud_detection.features.engineering import FeatureEngineer
from fraud_detection.models.isolation_forest import IsolationForestDetector
from fraud_detection.models.random_forest import RandomForestDetector
from fraud_detection.models.ensemble import HybridFraudEnsemble
from fraud_detection.models.evaluate import evaluate_model, print_evaluation_summary
from fraud_detection.utils.logger import setup_logger

logger = setup_logger("train_pipeline")


def run_pipeline(
    data_path: str = None,
    strategy: str = "union",
    output_dir: str = "models",
    nrows: int = None,
) -> None:
    config = AppConfig()
    selected_data_path = Path(data_path) if data_path else config.data.active_data_path

    logger.info(f"Target dataset: {selected_data_path}")
    loader = DataLoader(selected_data_path)
    df = loader.load(nrows=nrows)

    # 1. Preprocessing & Stratified Train/Test Split
    preprocessor = DataPreprocessor(test_size=config.data.test_size, random_state=config.data.random_state)
    df_encoded = preprocessor.encode_transaction_types(df)
    train_df, test_df = preprocessor.split_data(df_encoded, target_col=config.features.target_col)

    # 2. Feature Engineering (Fitted strictly on Train to avoid leakage)
    feature_engineer = FeatureEngineer()
    train_feat = feature_engineer.fit_transform(train_df)
    test_feat = feature_engineer.transform(test_df)

    feature_cols = config.features.model_features
    X_train, y_train = feature_engineer.extract_feature_matrix(train_feat, feature_cols)
    X_test, y_test = feature_engineer.extract_feature_matrix(test_feat, feature_cols)

    logger.info(f"Feature matrix shape: Train={X_train.shape}, Test={X_test.shape}")
    logger.info(f"Features: {feature_cols}")

    # 3. Model Training
    # Unsupervised detector
    iso = IsolationForestDetector(
        n_estimators=config.models.iso_n_estimators,
        contamination=config.models.iso_contamination,
        random_state=config.models.iso_random_state,
        n_jobs=config.models.iso_n_jobs,
    )
    # Supervised detector
    rf = RandomForestDetector(
        n_estimators=config.models.rf_n_estimators,
        max_depth=config.models.rf_max_depth,
        class_weight=config.models.rf_class_weight,
        random_state=config.models.rf_random_state,
        n_jobs=config.models.rf_n_jobs,
    )
    # Hybrid ensemble
    ensemble = HybridFraudEnsemble(
        strategy=strategy,
        rf_detector=rf,
        iso_detector=iso,
    )
    ensemble.fit(X_train, y_train)

    # 4. Evaluation on Hold-Out Test Set
    logger.info("Evaluating models on unseen test split...")
    test_amounts = test_df["amount"]

    # Evaluate Isolation Forest
    iso_preds = iso.predict(X_test)
    iso_metrics = evaluate_model(y_test, iso_preds, amounts=test_amounts)
    print_evaluation_summary(iso_metrics, title="Isolation Forest (Unsupervised Baseline)")

    # Evaluate Random Forest
    rf_preds = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)
    rf_metrics = evaluate_model(y_test, rf_preds, y_proba=rf_proba, amounts=test_amounts)
    print_evaluation_summary(rf_metrics, title="Random Forest (Supervised Classifier)")

    # Evaluate Hybrid Ensemble
    ens_preds = ensemble.predict(X_test)
    ens_proba = ensemble.predict_proba(X_test)
    ens_metrics = evaluate_model(y_test, ens_preds, y_proba=ens_proba, amounts=test_amounts)
    print_evaluation_summary(ens_metrics, title=f"Hybrid Ensemble (Strategy: {strategy})")

    # 5. Serialization
    out_dir_path = Path(output_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    joblib.dump(ensemble, out_dir_path / "ensemble_pipeline.joblib")
    joblib.dump(iso, out_dir_path / "isolation_forest.joblib")
    joblib.dump(rf, out_dir_path / "random_forest.joblib")
    joblib.dump(feature_engineer, out_dir_path / "feature_engineer.joblib")

    all_metrics = {
        "dataset": str(selected_data_path),
        "test_samples": len(test_df),
        "isolation_forest": iso_metrics,
        "random_forest": rf_metrics,
        "ensemble": ens_metrics,
    }
    with open(out_dir_path / "metrics_report.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    logger.info(f"Artifacts successfully saved to: {out_dir_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Fraud Detection Pipeline")
    parser.add_argument("--data", type=str, default=None, help="Path to CSV dataset")
    parser.add_argument(
        "--strategy",
        type=str,
        default="union",
        choices=["union", "intersection", "soft_voting"],
        help="Ensemble strategy",
    )
    parser.add_argument("--output", type=str, default="models", help="Directory to save models")
    parser.add_argument("--nrows", type=int, default=None, help="Limit number of rows to load")
    args = parser.parse_args()

    run_pipeline(
        data_path=args.data,
        strategy=args.strategy,
        output_dir=args.output,
        nrows=args.nrows,
    )
