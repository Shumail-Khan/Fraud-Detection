"""
Hybrid Fraud Detection Ensemble.
Combines Unsupervised Anomaly Detection (Isolation Forest) with
Supervised Machine Learning (Random Forest) for robust, defense-in-depth fraud prevention.
"""

from typing import Optional, Literal
import numpy as np
import pandas as pd

from fraud_detection.models.base import BaseFraudDetector
from fraud_detection.models.isolation_forest import IsolationForestDetector
from fraud_detection.models.random_forest import RandomForestDetector
from fraud_detection.utils.logger import setup_logger

logger = setup_logger(__name__)


class HybridFraudEnsemble(BaseFraudDetector):
    """
    Ensemble combining Isolation Forest and Random Forest.
    
    Strategies:
    - 'union': Flags fraud if EITHER detector flags fraud (maximizes Recall & fraud catch rate).
    - 'intersection': Flags fraud only if BOTH agree (maximizes Precision & minimizes customer friction).
    - 'soft_voting': Blends risk probabilities: weight_rf * p_rf + weight_iso * p_iso >= threshold.
    """

    def __init__(
        self,
        strategy: Literal["union", "intersection", "soft_voting"] = "union",
        rf_detector: Optional[RandomForestDetector] = None,
        iso_detector: Optional[IsolationForestDetector] = None,
        rf_weight: float = 0.7,
        iso_weight: float = 0.3,
        threshold: float = 0.5,
    ):
        self.strategy = strategy
        self.rf_detector = rf_detector or RandomForestDetector()
        self.iso_detector = iso_detector or IsolationForestDetector()
        self.rf_weight = rf_weight
        self.iso_weight = iso_weight
        self.threshold = threshold
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "HybridFraudEnsemble":
        """Fits both underlying detectors on feature matrix X."""
        logger.info(f"Training HybridFraudEnsemble using strategy='{self.strategy}'")
        self.iso_detector.fit(X)
        self.rf_detector.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generates ensemble binary fraud predictions."""
        if not self.is_fitted:
            raise RuntimeError("HybridFraudEnsemble must be fitted before predict.")

        iso_preds = self.iso_detector.predict(X)
        rf_preds = self.rf_detector.predict(X)

        if self.strategy == "union":
            return ((iso_preds == 1) | (rf_preds == 1)).astype(int)
        elif self.strategy == "intersection":
            return ((iso_preds == 1) & (rf_preds == 1)).astype(int)
        elif self.strategy == "soft_voting":
            probs = self.predict_proba(X)
            return (probs >= self.threshold).astype(int)
        else:
            raise ValueError(f"Unknown ensemble strategy: {self.strategy}")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns weighted ensemble continuous risk score."""
        if not self.is_fitted:
            raise RuntimeError("HybridFraudEnsemble must be fitted before predict_proba.")

        rf_prob = self.rf_detector.predict_proba(X)
        iso_prob = self.iso_detector.predict_proba(X)

        blended = (self.rf_weight * rf_prob) + (self.iso_weight * iso_prob)
        return np.clip(blended, 0.0, 1.0)
