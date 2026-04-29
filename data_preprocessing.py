"""
data_preprocessing.py
----------------------
Cleaning, feature engineering & encoding for the Google Play Store dataset.
"""

import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# ─── Helpers ──────────────────────────────────────────────────────────────────

def clean_installs(val) -> float:
    """'1,000,000+' → 1000000"""
    if pd.isna(val):
        return np.nan
    return float(re.sub(r"[^0-9]", "", str(val)) or np.nan)


def clean_price(val) -> float:
    """'$4.99' or 0 → 4.99"""
    if pd.isna(val):
        return 0.0
    return float(re.sub(r"[^0-9.]", "", str(val)) or 0.0)


def clean_size(val) -> float:
    """'19M' → 19.0, '512k' → 0.5, 'Varies…' → NaN"""
    if pd.isna(val) or "varies" in str(val).lower():
        return np.nan
    val = str(val).strip()
    if val.endswith("M") or val.endswith("m"):
        return float(val[:-1])
    if val.endswith("k") or val.endswith("K"):
        return float(val[:-1]) / 1024
    try:
        return float(val)
    except ValueError:
        return np.nan


def assign_tier(installs: float) -> str:
    """Low: <10K | Medium: 10K–1M | High: >1M"""
    if installs < 10_000:
        return "Low"
    elif installs <= 1_000_000:
        return "Medium"
    else:
        return "High"


# ─── Main pipeline ─────────────────────────────────────────────────────────────

def preprocess(path: str = "Play_Store_Data.csv"):
    """
    Returns
    -------
    df_encoded   : fully encoded DataFrame ready for ML
    df_clean     : human-readable cleaned DataFrame
    label_encoders : dict of fitted LabelEncoders
    """

    # 1. Load
    df = pd.read_csv(path)

    # 2. Drop exact duplicates
    df = df.drop_duplicates(subset=["App"], keep="first").reset_index(drop=True)

    # 3. Clean columns
    df["Installs"]  = df["Installs"].apply(clean_installs)
    df["Price"]     = df["Price"].apply(clean_price)
    df["Size_MB"]   = df["Size"].apply(clean_size)
    df["Reviews"]   = pd.to_numeric(df["Reviews"], errors="coerce")
    df["Rating"]    = pd.to_numeric(df["Rating"],  errors="coerce")

    # 4. Drop rows where Installs is missing/zero (can't build target)
    df = df[df["Installs"].notna() & (df["Installs"] > 0)].copy()

    # 5. Handle Type column
    df["Type"] = df["Type"].replace("0", "Free")
    df["Type"] = df["Type"].fillna("Free")
    df = df[df["Type"].isin(["Free", "Paid"])].copy()

    # 6. Handle Content Rating nulls
    df["Content Rating"] = df["Content Rating"].fillna("Everyone")

    # 7. Fill Rating with category mean (fallback to overall mean)
    overall_mean = df["Rating"].mean()
    df["Rating"] = df.groupby("Category")["Rating"].transform(
        lambda x: x.fillna(x.mean())
    )
    df["Rating"] = df["Rating"].fillna(overall_mean)
    df["Rating"] = df["Rating"].clip(1.0, 5.0)

    # 8. Fill Size_MB with median
    df["Size_MB"] = df["Size_MB"].fillna(df["Size_MB"].median())

    # 9. Fill Reviews nulls
    df["Reviews"] = df["Reviews"].fillna(df["Reviews"].median()).astype(int)

    # 10. Feature Engineering
    df["Paid_App"] = (df["Type"] == "Paid").astype(int)

    df["Engagement"] = (df["Reviews"] / df["Installs"]).replace(
        [np.inf, -np.inf], 0
    ).fillna(0)

    cat_avg = df.groupby("Category")["Installs"].transform("mean")
    df["Category_Avg_Installs"] = cat_avg

    # 11. Target label (needed before category high-rate)
    df["Install_Tier"] = df["Installs"].apply(assign_tier)

    # ── Supervised category features ───────────────────────────────────────────

    # Category_App_Count: total apps per category (market saturation / competition)
    # More apps in a category = tougher market
    cat_counts = df.groupby("Category")["App"].transform("count")
    df["Category_App_Count"] = cat_counts.astype(float)

    # Category_High_Rate: fraction of High-tier apps per category (learned from data)
    # This is the KEY feature that tells the model each category's success ceiling
    # e.g. MEDICAL ≈ 0.5%, ENTERTAINMENT ≈ 49%, FAMILY ≈ 14.8%
    cat_high_rate = df.groupby("Category")["Install_Tier"].transform(
        lambda x: (x == "High").mean()
    )
    df["Category_High_Rate"] = cat_high_rate

    # Category_Saturation: normalised competition score 0-1
    # High = crowded market, lower odds of standing out
    max_count = df["Category_App_Count"].max()
    df["Category_Saturation"] = df["Category_App_Count"] / max_count

    # ──────────────────────────────────────────────────────────────────────────

    # 12. Keep clean copy
    df_clean = df.copy()

    # 13. Encode categoricals
    label_encoders = {}
    for col in ["Category", "Content Rating", "Install_Tier"]:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    df_encoded = df.copy()

    return df_encoded, df_clean, label_encoders


if __name__ == "__main__":
    enc, clean, les = preprocess()
    print("Encoded shape:", enc.shape)
    print("Tier counts:\n", clean["Install_Tier"].value_counts())
    print("\nNew features sample:")
    print(clean[["Category", "Category_App_Count", "Category_High_Rate",
                  "Category_Saturation"]].drop_duplicates("Category").sort_values(
        "Category_High_Rate", ascending=False).to_string())
