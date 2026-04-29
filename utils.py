"""
utils.py
--------
Descriptive statistics, t-test, correlation summary, and business
recommendations for the Google Play Store dataset.
"""

import numpy as np
import pandas as pd
from scipy import stats


# ─── Descriptive stats ────────────────────────────────────────────────────────
def compute_statistics(df: pd.DataFrame) -> dict:
    cols = ["Rating", "Reviews", "Installs", "Price", "Size_MB",
            "Engagement", "Category_Avg_Installs"]
    cols = [c for c in cols if c in df.columns]

    out = {}
    for col in cols:
        s = df[col].dropna()
        out[col] = {
            "Mean":    round(s.mean(),   2),
            "Std Dev": round(s.std(),    2),
            "Min":     round(s.min(),    2),
            "Max":     round(s.max(),    2),
            "Median":  round(s.median(), 2),
        }
    return out


# ─── T-test: Paid vs Free installs ───────────────────────────────────────────
def compute_ttest(df: pd.DataFrame) -> dict:
    free = df[df["Paid_App"] == 0]["Installs"].dropna()
    paid = df[df["Paid_App"] == 1]["Installs"].dropna()

    t, p = stats.ttest_ind(free, paid, equal_var=False)  # Welch's t-test

    return {
        "group1":      "Free",
        "group2":      "Paid",
        "mean_free":   round(free.mean(), 0),
        "mean_paid":   round(paid.mean(), 0),
        "n_free":      len(free),
        "n_paid":      len(paid),
        "t_stat":      round(t, 4),
        "p_value":     round(p, 6),
        "significant": p < 0.05,
    }


# ─── Correlation with Installs ────────────────────────────────────────────────
def compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include=[np.number])
    corr = (
        num.corr()["Installs"]
        .drop("Installs", errors="ignore")
        .reset_index()
        .rename(columns={"index": "Feature", "Installs": "Correlation with Installs"})
    )
    corr["Abs"] = corr["Correlation with Installs"].abs()
    corr = corr.sort_values("Abs", ascending=False).drop(columns="Abs")
    corr["Correlation with Installs"] = corr["Correlation with Installs"].round(4)
    return corr.reset_index(drop=True)


# ─── Business Recommendations ────────────────────────────────────────────────
def generate_recommendations(df: pd.DataFrame) -> list:
    recs = []

    # Best category
    best_cat = df.groupby("Category")["Installs"].mean().idxmax()
    best_val = df.groupby("Category")["Installs"].mean().max()
    recs.append(
        f"🏆 **{best_cat}** has the highest average installs "
        f"({best_val/1e6:.1f}M). Launching in this category maximises reach."
    )

    # Free vs paid
    free_avg = df[df["Paid_App"] == 0]["Installs"].mean()
    paid_avg = df[df["Paid_App"] == 1]["Installs"].mean()
    recs.append(
        f"💸 Free apps average **{free_avg/1e6:.1f}M installs** vs "
        f"**{paid_avg/1e6:.2f}M** for paid. Go free — monetise via ads/IAP."
    )

    # Rating sweet spot
    high_avg_rating = df[df["Install_Tier"] == "High"]["Rating"].mean()
    recs.append(
        f"⭐ High-install apps average a rating of **{high_avg_rating:.2f}**. "
        "Aim for ≥ 4.0 stars by responding to reviews and shipping updates."
    )

    # Engagement
    eng_corr = df["Engagement"].corr(df["Installs"])
    recs.append(
        f"💬 Review-to-install engagement has a **{eng_corr:.2f}** correlation with installs. "
        "Encourage ratings — a short in-app prompt significantly boosts review count."
    )

    # Size
    size_corr = df["Size_MB"].corr(df["Installs"])
    recs.append(
        f"📦 App size correlation with installs: **{size_corr:.2f}**. "
        "Keep your APK under 30 MB to reduce install friction on low-storage devices."
    )

    # Content rating
    top_cr = df.groupby("Content Rating")["Installs"].mean().idxmax()
    recs.append(
        f"👨‍👩‍👧 **'{top_cr}'** content rating yields the highest average installs. "
        "Targeting everyone widens the potential user base."
    )

    # Most reviewed category
    top_review_cat = df.groupby("Category")["Reviews"].mean().idxmax()
    recs.append(
        f"📝 **{top_review_cat}** generates the most reviews per app. "
        "High review volume signals strong community engagement — a key growth lever."
    )

    return recs


# ─── Format stats as DataFrame ────────────────────────────────────────────────
def format_stats_table(stats_dict: dict) -> pd.DataFrame:
    rows = [{"Feature": k, **v} for k, v in stats_dict.items()]
    return pd.DataFrame(rows)
