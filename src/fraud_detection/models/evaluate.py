"""
Model evaluation and business ROI impact quantification.
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

from fraud_detection.utils.logger import setup_logger

logger = setup_logger(__name__)


def evaluate_model(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    amounts: Optional[pd.Series] = None,
    fp_cost_usd: float = 15.0,
) -> Dict[str, Any]:
    """
    Evaluates both standard statistical metrics and financial business KPIs.
    
    Args:
        y_true: Ground truth binary target (1 = Fraud, 0 = Legitimate).
        y_pred: Predicted binary labels.
        y_proba: Predicted continuous risk probabilities (optional).
        amounts: Monetary transaction values for financial ROI analysis (optional).
        fp_cost_usd: Estimated manual operational review cost per False Positive alert.
    
    Returns:
        Dictionary containing statistical metrics and financial KPIs.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
    }

    if y_proba is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
            metrics["pr_auc"] = float(average_precision_score(y_true, y_proba))
        except Exception:
            metrics["roc_auc"] = None
            metrics["pr_auc"] = None

    # Financial & Business Impact Analysis
    if amounts is not None:
        amounts = np.array(amounts)
        actual_fraud_mask = (np.array(y_true) == 1)
        pred_fraud_mask = (np.array(y_pred) == 1)

        total_fraud_loss = float(np.sum(amounts[actual_fraud_mask]))
        prevented_loss = float(np.sum(amounts[actual_fraud_mask & pred_fraud_mask]))
        missed_loss = float(np.sum(amounts[actual_fraud_mask & (~pred_fraud_mask)]))
        operational_investigation_cost = float(fp * fp_cost_usd)
        net_savings = float(prevented_loss - operational_investigation_cost)
        catch_rate_pct = (prevented_loss / total_fraud_loss * 100.0) if total_fraud_loss > 0 else 0.0

        metrics["financial_kpis"] = {
            "total_fraud_exposure_usd": total_fraud_loss,
            "prevented_fraud_loss_usd": prevented_loss,
            "missed_fraud_loss_usd": missed_loss,
            "operational_fp_cost_usd": operational_investigation_cost,
            "net_financial_savings_usd": net_savings,
            "fraud_volume_catch_rate_pct": catch_rate_pct,
        }

    return metrics


def print_evaluation_summary(metrics: Dict[str, Any], title: str = "Evaluation Summary") -> None:
    """Pretty prints evaluation metrics and financial impact to console."""
    print("\n" + "=" * 65)
    print(f"  {title.upper()}")
    print("=" * 65)
    print(f"  Accuracy           : {metrics['accuracy']:.4f}")
    print(f"  Precision          : {metrics['precision']:.4f}")
    print(f"  Recall             : {metrics['recall']:.4f}")
    print(f"  F1-Score           : {metrics['f1_score']:.4f}")
    if metrics.get("roc_auc") is not None:
        print(f"  ROC-AUC            : {metrics['roc_auc']:.4f}")
    if metrics.get("pr_auc") is not None:
        print(f"  PR-AUC             : {metrics['pr_auc']:.4f}")

    cm = metrics["confusion_matrix"]
    print("-" * 65)
    print("  CONFUSION MATRIX")
    print(f"  True Positives (Caught Frauds)  : {cm['true_positives']:,}")
    print(f"  False Positives (False Alarms)  : {cm['false_positives']:,}")
    print(f"  False Negatives (Missed Frauds) : {cm['false_negatives']:,}")
    print(f"  True Negatives (Correct Normal) : {cm['true_negatives']:,}")

    if "financial_kpis" in metrics:
        fin = metrics["financial_kpis"]
        print("-" * 65)
        print("  FINANCIAL BUSINESS IMPACT")
        print(f"  Total Fraud Exposure            : ${fin['total_fraud_exposure_usd']:,.2f}")
        print(f"  Prevented Fraud Loss            : ${fin['prevented_fraud_loss_usd']:,.2f}")
        print(f"  Missed Fraud Loss               : ${fin['missed_fraud_loss_usd']:,.2f}")
        print(f"  Operational FP Investigation Cost: ${fin['operational_fp_cost_usd']:,.2f}")
        print(f"  NET FINANCIAL BENEFIT           : ${fin['net_financial_savings_usd']:,.2f}")
        print(f"  Fraud Dollar Catch Rate         : {fin['fraud_volume_catch_rate_pct']:.2f}%")
    print("=" * 65 + "\n")
