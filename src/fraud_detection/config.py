"""
Configuration management for the Fraud Detection system.
Provides centralized path management, hyperparameters, and feature definitions.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


# Base directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
SAMPLE_DATA_DIR = DATA_DIR / "sample"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


@dataclass
class DataConfig:
    """Dataset file locations and settings."""
    sample_path: Path = SAMPLE_DATA_DIR / "sample_transactions.csv"
    raw_path: Path = RAW_DATA_DIR / "PS_20174392719_1491204439457_log.csv"
    root_fallback_raw: Path = PROJECT_ROOT / "PS_20174392719_1491204439457_log.csv"
    test_size: float = 0.2
    random_state: int = 42

    @property
    def active_data_path(self) -> Path:
        """Returns the full raw path if available, otherwise defaults to the sample dataset."""
        if self.raw_path.exists():
            return self.raw_path
        if self.root_fallback_raw.exists():
            return self.root_fallback_raw
        return self.sample_path


@dataclass
class FeatureConfig:
    """Feature columns and transformations."""
    target_col: str = "isFraud"
    id_cols: List[str] = field(default_factory=lambda: ["nameOrig", "nameDest", "isFlaggedFraud"])
    base_numeric_features: List[str] = field(
        default_factory=lambda: [
            "step",
            "amount",
            "oldbalanceOrg",
            "newbalanceOrig",
            "oldbalanceDest",
            "newbalanceDest",
        ]
    )
    categorical_features: List[str] = field(default_factory=lambda: ["type"])
    engineered_features: List[str] = field(
        default_factory=lambda: [
            "log_amount",
            "tx_count_orig",
            "tx_sum_orig",
            "amount_to_avg_ratio",
            "error_balance_orig",
            "error_balance_dest",
        ]
    )

    @property
    def model_features(self) -> List[str]:
        """All features utilized in the modeling pipeline."""
        return [
            "amount",
            "log_amount",
            "tx_count_orig",
            "tx_sum_orig",
            "amount_to_avg_ratio",
            "error_balance_orig",
            "error_balance_dest",
            "oldbalanceOrg",
            "newbalanceOrig",
            "oldbalanceDest",
            "newbalanceDest",
            "type_CASH_OUT",
            "type_TRANSFER",
            "type_PAYMENT",
            "type_DEBIT",
            "type_CASH_IN",
        ]


@dataclass
class ModelConfig:
    """Hyperparameters for models and ensemble."""
    # Isolation Forest (Unsupervised)
    iso_n_estimators: int = 150
    iso_contamination: float = 0.01
    iso_random_state: int = 42
    iso_n_jobs: int = -1

    # Random Forest (Supervised)
    rf_n_estimators: int = 100
    rf_max_depth: int = 12
    rf_class_weight: str = "balanced"
    rf_random_state: int = 42
    rf_n_jobs: int = -1

    # Ensemble logic: 'union' (OR), 'intersection' (AND), or 'soft_voting'
    ensemble_strategy: str = "union"
    soft_voting_threshold: float = 0.5


@dataclass
class AppConfig:
    """Master application configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    models_dir: Path = MODELS_DIR
