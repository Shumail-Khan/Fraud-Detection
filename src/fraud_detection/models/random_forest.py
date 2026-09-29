"""
Supervised Random Forest Classifier with Cost-Sensitive Class Balancing.
"""

from typing import Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from fraud_detection.models.base import BaseFraudDetector
from fraud_detection.utils.logger import setup_logger

logger = setup_logger(__name__)


class RandomForestDetector(BaseFraudDetector):
    """Supervised Random Forest classifier designed for imbalanced fraud datasets."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = 12,
        class_weight: str = "balanced",
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.class_weight = class_weight
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "RandomForestDetector":
        """Fits Random Forest classifier on labeled training data."""
        if y is None:
            raise ValueError("Supervised RandomForest requires target labels 'y'.")
        logger.info(
            f"Fitting RandomForest (n_estimators={self.n_estimators}, max_depth={self.max_depth}, "
            f"class_weight={self.class_weight})"
        )
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Returns binary fraud predictions."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict.")
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns continuous predicted probability of fraud (class 1)."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict_proba.")
        return self.model.predict_proba(X)[:, 1]

    @property
    def feature_importances_(self) -> np.ndarray:
        """Returns feature importances from fitted trees."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted to retrieve feature importances.")
        return self.model.feature_importances_
