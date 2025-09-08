# eda_feature_vs_label.py
# ------------------------------------------------------------
# Creates quick EDA plots showing how each feature relates to
# LABEL_UNDELIVERED_CB:
#  - Numeric: boxplot (by label) + histogram overlay
#  - Categorical: bar chart of positive rate per category
# Saves PNGs into an output folder.
# ------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from main import TARGET

# ----------- CONFIGURE THESE -----------
# ---------- Paths ----------
TRAIN_CSV = r"C:\Users\cghiuta\Desktop\Training_Data.csv"
TEST_CSV  = r"C:\Users\cghiuta\Desktop\Testing_Data.csv"  # optional
OUTPUT_DIR = Path("feature_plots")  # output folder (will be created)
ID_LIKE = {"MERCHANT_ID", "RN"}     # columns to ignore
MAX_CATS = 15           # max categories to display for categorical plots
MIN_NON_NULL = 100      # skip columns with fewer non-null values
BINS = 30               # histogram bins
# --------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(TRAIN_CSV)
    if TARGET not in df.columns:
        raise ValueError(f"Target '{TARGET}' not found. Columns: {df.columns.tolist()}")

    usable_cols = [c for c in df.columns if c not in ID_LIKE and c != TARGET]

    # Identify numeric vs categorical
    numeric_cols = [c for c in usable_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in usable_cols if not pd.api.types.is_numeric_dtype(df[c])]

    created = []

    # ------- Numeric features -------
    for col in numeric_cols:
        series = df[col]
        if series.notna().sum() < MIN_NON_NULL:
            continue

        x0 = df.loc[df[TARGET] == 0, col].dropna()
        x1 = df.loc[df[TARGET] == 1, col].dropna()
        if len(x0) < 5 or len(x1) < 5:
            continue

        # Boxplot by label
        try:
            plt.figure(figsize=(7, 5))
            plt.boxplot([x0.values, x1.values],
                        labels=["label=0 (delivered)", "label=1 (undelivered)"],
                        showfliers=False)
            plt.title(f"{col} vs {TARGET} (boxplot)")
            plt.ylabel(col)
            plt.xlabel("Class")
            plt.tight_layout()
            out_path = OUTPUT_DIR / f"num_{col}_box.png"
            plt.savefig(out_path, dpi=130)
            plt.close()
            created.append(out_path)
        except Exception:
            plt.close()

        # Histogram overlay
        try:
            plt.figure(figsize=(7, 5))
            plt.hist(x0.values, bins=BINS, alpha=0.5, label="label=0", density=True)
            plt.hist(x1.values, bins=BINS, alpha=0.5, label="label=1", density=True)
            plt.title(f"{col} distribution by class (hist)")
            plt.xlabel(col)
            plt.ylabel("Density")
            plt.legend()
            plt.tight_layout()
            out_path = OUTPUT_DIR / f"num_{col}_hist.png"
            plt.savefig(out_path, dpi=130)
            plt.close()
            created.append(out_path)
        except Exception:
            plt.close()

    # ------- Categorical features -------
    for col in categorical_cols:
        series = df[col].astype("object")
        if series.notna().sum() < MIN_NON_NULL:
            continue

        tmp = (
            df[[col, TARGET]]
            .dropna()
            .groupby(col)[TARGET]
            .agg(["mean", "count"])
            .rename(columns={"mean": "pos_rate", "count": "n"})
            .sort_values("n", ascending=False)
        )
        if tmp.empty:
            continue

        top = tmp.head(MAX_CATS)

        try:
            plt.figure(figsize=(9, 6))
            plt.bar(top.index.astype(str), top["pos_rate"].values)
            plt.title(f"{col}: positive rate by category (top {len(top)})")
            plt.ylabel("Positive rate (P[label=1])")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            out_path = OUTPUT_DIR / f"cat_{col}_posrate.png"
            plt.savefig(out_path, dpi=130)
            plt.close()
            created.append(out_path)
        except Exception:
            plt.close()

    # Write a small index file
    with open(OUTPUT_DIR / "PLOTS_README.txt", "w", encoding="utf-8") as f:
        f.write("Generated plots:\n")
        for p in created:
            f.write(str(p.resolve()) + "\n")

    print(f"Done. Saved {len(created)} plots to: {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
