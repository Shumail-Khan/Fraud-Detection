# PaySim Fraud Detection Data Directory

This directory manages datasets for the Fraud Detection system.

## ⚠️ Large File Policy & GitHub Restrictions
The complete PaySim raw dataset (`PS_20174392719_1491204439457_log.csv`) is **~493.5 MB** and contains **6,362,620 transactions**.
- GitHub enforces a strict **100 MB hard limit** on file uploads.
- The raw dataset is permanently excluded from version control via `.gitignore`.
- **Never attempt to commit raw CSV files to GitHub.**

---

## 🚀 Instant Quickstart: Sample Dataset
For rapid testing, continuous integration (CI), and local prototyping, a lightweight stratified sample is committed directly in this repository:
- **Location:** `data/sample/sample_transactions.csv`
- **Size:** ~1.48 MB
- **Total Records:** 20,000 transactions
- **Class Distribution:** 5,000 Fraud (25%) / 15,000 Non-Fraud (75%)
- **Purpose:** Enables end-to-end execution of training, unit tests, and streaming simulation out-of-the-box without downloading 500MB.

---

## 📥 How to Obtain the Full Dataset

### Option 1: Automated Download via Kagglehub (Recommended)
Run the automated downloader script included in the repository:
```bash
python scripts/download_data.py
```
This script downloads the latest version of the PaySim dataset from Kaggle and places it in `data/raw/`.

### Option 2: Kaggle CLI
If you have the Kaggle CLI configured:
```bash
kaggle datasets download -d ealaxi/paysim1 -p data/raw/ --unzip
```

### Option 3: Manual Download from Kaggle Web UI
1. Visit [Kaggle: PaySim - Mobile Money Fraud Simulation](https://www.kaggle.com/datasets/ealaxi/paysim1).
2. Click **Download** (archive `archive.zip`).
3. Extract `PS_20174392719_1491204439457_log.csv` into `data/raw/` (or repository root).

---

## 📊 Data Dictionary

| Column Name | Type | Description |
| :--- | :--- | :--- |
| `step` | `int64` | Time step unit in real hours (1 step = 1 hour). Total simulation covers 744 steps (30 days). |
| `type` | `object` | Transaction category: `PAYMENT`, `TRANSFER`, `CASH_OUT`, `DEBIT`, `CASH_IN`. |
| `amount` | `float64` | Transaction monetary value in local currency. |
| `nameOrig` | `object` | Customer ID initiating the transaction. |
| `oldbalanceOrg` | `float64` | Sender balance before the transaction. |
| `newbalanceOrig` | `float64` | Sender balance after the transaction. |
| `nameDest` | `object` | Recipient ID receiving the transaction (Merchant `M*` or Customer `C*`). |
| `oldbalanceDest` | `float64` | Recipient initial balance before transaction. |
| `newbalanceDest` | `float64` | Recipient new balance after transaction. |
| `isFraud` | `int64` | Ground-truth binary fraud indicator (`1` = Fraudulent, `0` = Legitimate). |
| `isFlaggedFraud` | `int64` | Legacy rule-based business flag (marked if single transaction > 200,000). |

---

## 🛠️ Regenerating the Sample Dataset
If you wish to re-sample with different proportions or sizes:
```bash
python scripts/create_sample.py [path_to_raw_csv] [path_to_sample_output]
```
