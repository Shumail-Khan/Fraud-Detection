"""
Feature engineering module for financial and e-commerce fraud detection.
Transforms raw transaction metrics into predictive behavioral signals.
"""

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd

from fraud_detection.utils.logger import setup_logger

logger = setup_logger(__name__)


class FeatureEngineer:
    """Engineers statistical and behavioral features from transaction records."""

    def __init__(self):
        # Stored customer transaction statistics to prevent data leakage during test/streaming inference
        self.orig_stats_: Optional[pd.DataFrame] = None

    def fit(self, df: pd.DataFrame) -> "FeatureEngineer":
        """Computes customer behavioral baselines from historical/training data."""
        logger.info("Computing customer transaction behavioral baselines...")
        stats = df.groupby("nameOrig")["amount"].agg(
            tx_count_orig="count",
            tx_sum_orig="sum"
        ).reset_index()
        self.orig_stats_ = stats
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies feature transformations to the dataset."""
        df = df.copy()

        # 1. Non-linear scaling of monetary values
        df["log_amount"] = np.log1p(df["amount"])

        # 2. Historical User Behavioral Proxies
        if self.orig_stats_ is not None and "nameOrig" in df.columns:
            df = df.merge(self.orig_stats_, on="nameOrig", how="left")
            df["tx_count_orig"] = df["tx_count_orig"].fillna(1.0)
            df["tx_sum_orig"] = df["tx_sum_orig"].fillna(df["amount"])
        else:
            # Fallback for online / single transactions
            df["tx_count_orig"] = 1.0
            df["tx_sum_orig"] = df["amount"]

        # Ratio of transaction amount to user's average
        avg_tx = df["tx_sum_orig"] / df["tx_count_orig"].clip(lower=1.0)
        df["amount_to_avg_ratio"] = df["amount"] / avg_tx.replace(0, np.nan)
        df["amount_to_avg_ratio"] = df["amount_to_avg_ratio"].fillna(1.0)

        # 3. Balance Discrepancy (Account Inconsistency Detection)
        # In a consistent transaction: oldbalanceOrg - amount == newbalanceOrig
        df["error_balance_orig"] = (df["newbalanceOrig"] + df["amount"]) - df["oldbalanceOrg"]
        df["error_balance_dest"] = (df["oldbalanceDest"] + df["amount"]) - df["newbalanceDest"]

        # Clean numerical anomalies
        num_cols = ["amount_to_avg_ratio", "error_balance_orig", "error_balance_dest"]
        df[num_cols] = df[num_cols].replace([np.inf, -np.inf], 0.0).fillna(0.0)

        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fits behavioral baselines and transforms the dataset."""
        return self.fit(df).transform(df)

    def extract_feature_matrix(
        self, df: pd.DataFrame, feature_columns: List[str]
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Extracts X (feature matrix) and y (target) aligned with required feature columns."""
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0.0

        X = df[feature_columns].copy()
        X = X.replace([np.inf, -np.inf], 0.0).fillna(0.0)

        y = df["isFraud"].copy() if "isFraud" in df.columns else None
        return X, y
