"""
Base interfaces and contracts for fraud detection models.
"""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
import pandas as pd


class BaseFraudDetector(ABC):
    """Abstract base class for all fraud detection algorithms."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "BaseFraudDetector":
        """Fits the detector to the training data."""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts binary fraud flags (1 = Fraud, 0 = Legitimate)."""
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts continuous fraud risk scores / probabilities in [0.0, 1.0]."""
        pass
