"""
Automated downloader for the PaySim Fraud Detection dataset.
Supports downloading via kagglehub or providing guidance for Kaggle CLI.
"""

import os
import shutil
import sys


def download_dataset(target_dir: str = "data/raw") -> str:
    print("=" * 60)
    print("PaySim Fraud Detection - Automated Dataset Fetcher")
    print("=" * 60)

    os.makedirs(target_dir, exist_ok=True)
    target_csv = os.path.join(target_dir, "PS_20174392719_1491204439457_log.csv")

    # Check if file already exists in root or data/raw
    if os.path.exists("PS_20174392719_1491204439457_log.csv"):
        print("[i] Found raw dataset in project root.")
        print(f"    Moving to standardized directory: {target_csv}")
        shutil.copy("PS_20174392719_1491204439457_log.csv", target_csv)
        return target_csv

    if os.path.exists(target_csv):
        print(f"[+] Raw dataset already exists at: {target_csv}")
        return target_csv

    try:
        import kagglehub
        print("[i] Attempting download via kagglehub from 'ealaxi/paysim1'...")
        download_path = kagglehub.dataset_download("ealaxi/paysim1")
        print(f"[+] Download completed to cache: {download_path}")

        # Locate csv in downloaded files
        for fname in os.listdir(download_path):
            if fname.endswith(".csv"):
                src_file = os.path.join(download_path, fname)
                shutil.copy(src_file, target_csv)
                print(f"[+] Successfully copied dataset to: {target_csv}")
                return target_csv

    except ImportError:
        print("[!] 'kagglehub' package not installed.")
        print("    Install it with: pip install kagglehub")
    except Exception as e:
        print(f"[!] kagglehub download failed: {e}")

    print("\nAlternative Options:")
    print("1. Use Kaggle CLI:")
    print("   kaggle datasets download -d ealaxi/paysim1 -p data/raw/ --unzip")
    print("2. Manual download from Kaggle:")
    print("   https://www.kaggle.com/datasets/ealaxi/paysim1")
    print(f"   Extract 'PS_20174392719_1491204439457_log.csv' into: {target_dir}")
    return ""


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    download_dataset(out_dir)
