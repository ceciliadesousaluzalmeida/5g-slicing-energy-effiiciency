# All comments in English
import os
import shutil
import pandas as pd


RESULTS_DIR = "results"
REQUIRED_SEEDS = {1, 2, 3, 4, 5}


def folder_is_empty(path):
    """Return True if directory has no files."""
    return len(os.listdir(path)) == 0


def find_scalability_csv(path):
    """
    Find a CSV file containing 'scalability' in the name.
    Return full path or None.
    """
    for f in os.listdir(path):
        if f.endswith(".csv") and "scalability" in f.lower():
            return os.path.join(path, f)
    return None


def seeds_are_valid(csv_path):
    """
    Check if CSV contains seeds 1..5.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return False

    if "seed" not in df.columns:
        return False

    seeds_present = set(df["seed"].dropna().astype(int).unique())
    return REQUIRED_SEEDS.issubset(seeds_present)


def clean_results():
    for root, dirs, files in os.walk(RESULTS_DIR, topdown=False):
        for d in dirs:
            folder = os.path.join(root, d)

            # 1️⃣ remove empty folders
            if folder_is_empty(folder):
                print(f"[DELETE] empty folder: {folder}")
                shutil.rmtree(folder)
                continue

            # 2️⃣ check scalability csv
            csv_path = find_scalability_csv(folder)

            if csv_path is None:
                print(f"[DELETE] no scalability csv: {folder}")
                shutil.rmtree(folder)
                continue

            if not seeds_are_valid(csv_path):
                print(f"[DELETE] seeds missing (1..5): {folder}")
                shutil.rmtree(folder)


if __name__ == "__main__":
    clean_results()