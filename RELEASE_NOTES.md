# Release v0.1.0 — Real-Time E-Commerce & Financial Fraud Detection System

**Release Date:** September 2026  
**Tag:** `v0.1.0`  
**License:** MIT  

---

## 🌟 Highlights & Summary

The initial release of the **Fraud Detection System** delivers an end-to-end, production-ready framework for detecting fraudulent transactions in e-commerce and mobile financial services (PaySim simulation). 

This release introduces a **dual-layer defense-in-depth architecture** uniting unsupervised anomaly detection with supervised machine learning, alongside a near-real-time streaming transaction simulator and automated financial ROI quantification.

---

## 🚀 Key Features Included in this Release

### 1. Hybrid Defense-in-Depth Ensemble
- **Unsupervised Anomaly Detection (`IsolationForestDetector`):** Detects emerging zero-day attack vectors and anomalous distribution shifts without label dependency.
- **Cost-Sensitive Supervised Classification (`RandomForestDetector`):** Resolves extreme class imbalance (<0.13% fraud prevalence) with balanced tree weighting.
- **Multi-Strategy Hybrid Combiner (`HybridFraudEnsemble`):** Supports `union` (maximize recall/prevent loss leakage), `intersection` (minimize false alarms), and `soft_voting` (blended probability thresholds).

### 2. Behavioral Domain Feature Engineering
- **User Velocity & History:** `amount_to_avg_ratio`, `tx_count_orig`, `tx_sum_orig`.
- **Account Discrepancy Signals:** Mathematical balance tracking (`error_balance_orig`, `error_balance_dest`) identifying account drains and phantom destination transfers.
- **Non-Linear Monetary Compression:** `log_amount = log1p(amount)`.

### 3. Near-Real-Time Stream Simulator
- Simulates continuous transaction ingestion with live fraud alerting.
- Benchmarks latency SLAs: P50, P95, and P99 inference latencies in milliseconds.
- Computes real-time streaming throughput (`tx/sec`).

### 4. Financial ROI & Business Impact Engine
- Computes Net Financial Benefit: `Prevented Losses - Operational Review Overhead`.
- Tracks dollar-volume catch rates and false positive operational review overhead.

### 5. Large File Management & Developer Experience
- Solves GitHub's 100MB file limit by git-ignoring raw ~500MB CSV files.
- Includes a lightweight 20,000-row stratified sample (`data/sample/sample_transactions.csv`, ~1.48MB) for instant setup and zero-dependency CI.
- Automated Kaggle download script (`scripts/download_data.py`).
- 11 unit & integration tests passing with 100% success rate.
- Automated CI pipeline via GitHub Actions across Python 3.10, 3.11, and 3.12.

---

## 📦 Getting Started

```bash
# Clone the repository
git clone https://github.com/your-username/fraud-detection-system.git
cd fraud-detection-system

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Run test suite
pytest tests/ -v

# Train and evaluate models
python scripts/train.py

# Simulate live transaction streaming
python scripts/simulate_stream.py --events 50
```

---

## 👥 Contributors
- Shumail Khan (@shumailkhan7878)
