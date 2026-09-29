# Model Registry

This directory stores serialized model artifacts generated during training.

Artifact files (`*.joblib`, `*.pkl`) are git-ignored to prevent bloating repository history.

## Saved Artifacts
When executing `python scripts/train.py`:
- `isolation_forest.joblib`: Trained unsupervised anomaly detector
- `random_forest.joblib`: Trained supervised ensemble classifier
- `ensemble_pipeline.joblib`: Unified hybrid detection pipeline
- `metrics_report.json`: Precision, Recall, F1, PR-AUC, and financial savings KPIs
