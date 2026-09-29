"""
Fraud Detection Package
An end-to-end framework for detecting fraudulent e-commerce and mobile financial transactions.
"""

__version__ = "0.1.0"

from fraud_detection.config import AppConfig
from fraud_detection.models.ensemble import HybridFraudEnsemble

__all__ = ["AppConfig", "HybridFraudEnsemble", "__version__"]
