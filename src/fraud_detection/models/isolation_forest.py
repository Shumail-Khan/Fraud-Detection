"""
Unsupervised Anomaly Detection using Isolation Forest.
Detects zero-day, novel, and out-of-distribution fraudulent behaviors without relying on labels.
"""

from typing import Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from fraud_detection.models.base import BaseFraudDetector
from fraud_detection.utils.logger import setup_logger

logger = setup_logger(__name__)


class IsolationForestDetector(BaseFraudDetector):
    """Unsupervised Isolation Forest fraud detector."""

    def __init__(
        self,
        n_estimators: int = 150,
        contamination: float = 0.01,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "IsolationForestDetector":
        """Fits Isolation Forest on transaction feature matrix (labels not used)."""
        logger.info(f"Fitting IsolationForest (n_estimators={self.n_estimators}, contamination={self.contamination})")
        self.model.fit(X)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Returns binary prediction: 1 for Anomaly (Fraud), 0 for Normal."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict.")
        raw_preds = self.model.predict(X)
        # scikit-learn: -1 = anomaly, 1 = inlier
        return np.where(raw_preds == -1, 1, 0)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Calculates normalized anomaly risk score in [0.0, 1.0].
        Lower decision_function values indicate higher anomaly severity.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict_proba.")
        decision_scores = self.model.decision_function(X)
        # Normalize decision scores: smaller means higher anomaly score
        # Using min-max inverted or logistic sigmoid on inverted decision scores
        scores = -decision_scores
        # Logistic sigmoid scaling for stable probability-like output
        proba = 1.0 / (1.0 + np.exp(-scores * 10))
        return proba
