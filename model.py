"""
model.py
--------
Train Logistic Regression & Random Forest to predict Install Tier.
Includes confusion matrix plot, feature importance plot,
and a smart rule-adjusted predict_success with probability breakdown.

Key improvements over v1:
  • Random Forest replaces Decision Tree — better generalisation, less overfit
  • Category_High_Rate + Category_Saturation features carry the real category signal
  • Rating drives the rule engine much harder (dominant signal as expected)
  • Category saturation penalises overcrowded markets in the rule engine
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics         import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing   import StandardScaler


# ─── Feature columns ─────────────────────────────────────────────────────────
FEATURE_COLS = [
    "Rating",
    "Reviews",
    "Size_MB",
    "Price",
    "Paid_App",
    "Engagement",
    "Category_Avg_Installs",
    "Category_enc",
    "Content Rating_enc",
    # Supervised category features (NEW)
    "Category_High_Rate",    # how often does this category produce High-tier apps?
    "Category_Saturation",   # how crowded is this category? (0=niche, 1=FAMILY-level)
    "Category_App_Count",    # raw app count — lets RF learn non-linear saturation effects
]

TARGET     = "Install_Tier_enc"
TIER_NAMES = None   # filled after LabelEncoder fit (alphabetical: High, Low, Medium)


def train_models(df_encoded: pd.DataFrame, label_encoders: dict):
    """
    Train both models, return results, fitted objects, and test data.
    """
    global TIER_NAMES
    le_tier    = label_encoders["Install_Tier"]
    TIER_NAMES = list(le_tier.classes_)   # ['High', 'Low', 'Medium']

    available = [c for c in FEATURE_COLS if c in df_encoded.columns]
    X = df_encoded[available].fillna(0).values
    y = df_encoded[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        # Random Forest: 300 trees, class-weight balanced so minority tiers get fair weight
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,         # let trees grow fully — forest averaging prevents overfit
            min_samples_leaf=5,     # avoid noisy leaf splits
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = {}
    for name, mdl in models.items():
        if name == "Logistic Regression":
            mdl.fit(X_train_sc, y_train)
            y_pred = mdl.predict(X_test_sc)
        else:
            mdl.fit(X_train, y_train)
            y_pred = mdl.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        cm  = confusion_matrix(y_test, y_pred)
        cr  = classification_report(
            y_test, y_pred, target_names=TIER_NAMES, output_dict=True
        )
        results[name] = {"accuracy": acc, "confusion_matrix": cm, "report": cr}

    return results, models, scaler, X_test, y_test, available


# ─── Confusion matrix plot ────────────────────────────────────────────────────
def plot_confusion_matrix(cm: np.ndarray, model_name: str) -> plt.Figure:
    names = TIER_NAMES or ["High", "Low", "Medium"]
    fig, ax = plt.subplots(figsize=(5, 4), facecolor="#F8F9FA")
    ax.set_facecolor("#F8F9FA")

    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=9)
    ax.set_yticklabels(names, fontsize=9)

    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=12,
                    color="white" if cm[i, j] > cm.max() * 0.5 else "black")

    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("Actual", fontsize=10)
    ax.set_title(f"Confusion Matrix – {model_name}", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


# ─── Feature importance plot ──────────────────────────────────────────────────
def plot_feature_importance(models: dict, feature_names: list) -> plt.Figure:
    rf = models.get("Random Forest")
    if rf is None or not hasattr(rf, "feature_importances_"):
        return None

    importances = rf.feature_importances_
    idx = np.argsort(importances)
    feats  = [feature_names[i] for i in idx]
    values = importances[idx]

    PAL = ["#3A86FF", "#FF6B6B", "#06D6A0", "#FFB703", "#8338EC",
           "#FB5607", "#219EBC", "#023047", "#F72585", "#2EC4B6", "#E71D36", "#FF9F1C"]

    fig, ax = plt.subplots(figsize=(9, 5), facecolor="#F8F9FA")
    ax.set_facecolor("#F8F9FA")

    colors = [PAL[i % len(PAL)] for i in range(len(feats))]
    ax.barh(feats, values, color=colors, edgecolor="white", height=0.7)

    for i, (f, v) in enumerate(zip(feats, values)):
        ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=8)

    ax.set_xlabel("Importance Score", fontsize=10)
    ax.set_title("Feature Importance – Random Forest", fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.grid(True, color="#DEE2E6", linewidth=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


# ─── Rule-based score adjustment ─────────────────────────────────────────────
def _compute_rule_score(input_dict: dict) -> float:
    """
    Returns a score in [-1, +1] based on user-controllable quality signals.
    Positive  → push toward High
    Negative  → push toward Low

    Rating is a strong signal, but Category_High_Rate is the dominant ceiling.
    A great rating in a dead-end category (MEDICAL ~0.5% High rate) should NOT
    override the category reality. Category in a thriving category (ENTERTAINMENT
    ~49% High rate) should amplify the rating bonus.
    """
    score = 0.0

    # ── Rating: important but calibrated ─────────────────────────────────────
    rating = input_dict.get("Rating", 3.0)
    if rating >= 4.5:
        score += 0.45
    elif rating >= 4.2:
        score += 0.30
    elif rating >= 4.0:
        score += 0.18
    elif rating >= 3.7:
        score += 0.06
    elif rating >= 3.5:
        score += 0.00
    elif rating >= 3.0:
        score -= 0.15
    elif rating >= 2.0:
        score -= 0.35
    else:
        score -= 0.55

    # ── Price ────────────────────────────────────────────────────────────────
    price = input_dict.get("Price", 0.0)
    if price == 0:
        score += 0.10
    elif price <= 1.0:
        score -= 0.05
    elif price <= 5.0:
        score -= 0.15
    else:
        score -= 0.25

    # ── Size ─────────────────────────────────────────────────────────────────
    size = input_dict.get("Size_MB", 15.0)
    if size <= 10:
        score += 0.05
    elif size <= 30:
        score += 0.02
    elif size <= 60:
        score -= 0.03
    else:
        score -= 0.08

    # ── Content rating ───────────────────────────────────────────────────────
    content_name = input_dict.get("_content_rating_name", "")
    if "Everyone" in content_name:
        score += 0.08
    elif "Teen" in content_name:
        score += 0.02
    elif "Mature" in content_name or "Adults" in content_name:
        score -= 0.05

    # ── Category High Rate — the dominant category signal ─────────────────────
    # Learned from actual data: what fraction of apps in this category are High-tier?
    # e.g. MEDICAL ≈ 0.005 (only 0.5% make High), ENTERTAINMENT ≈ 0.49 (49% make High)
    # Use a multiplier of 2.5 so extreme categories (MEDICAL, ENTERTAINMENT) dominate.
    cat_high_rate = input_dict.get("Category_High_Rate", 0.20)
    overall_avg   = 0.205   # dataset average ~20.5% High
    rate_delta    = cat_high_rate - overall_avg
    score += rate_delta * 2.5   # ±0.29 delta → ±0.73 score contribution

    # ── Category Saturation penalty ──────────────────────────────────────────
    # Crowded categories (FAMILY, GAME, TOOLS) → harder to break through.
    saturation = input_dict.get("Category_Saturation", 0.20)
    if saturation > 0.7:
        score -= 0.18
    elif saturation > 0.4:
        score -= 0.10
    elif saturation > 0.2:
        score -= 0.04
    elif saturation < 0.05:
        score += 0.03   # niche category: less competition bonus

    return max(-1.0, min(1.0, score))


def _raw_probs(
    input_dict: dict,
    models: dict,
    scaler: StandardScaler,
    feature_names: list,
    chosen_model: str,
    tier_names: list,          # alphabetical: ['High', 'Low', 'Medium']
) -> dict:
    """
    Return raw ML probability dict  {'High': p, 'Low': p, 'Medium': p}
    Falls back to one-hot if model has no predict_proba.
    """
    row = np.array([input_dict.get(f, 0) for f in feature_names]).reshape(1, -1)
    mdl = models[chosen_model]

    if chosen_model == "Logistic Regression":
        row_in = scaler.transform(row)
    else:
        row_in = row

    if hasattr(mdl, "predict_proba"):
        proba = mdl.predict_proba(row_in)[0]
        return {tier_names[i]: float(proba[i]) for i in range(len(tier_names))}
    else:
        pred_enc = mdl.predict(row_in)[0]
        probs = {t: 0.0 for t in tier_names}
        probs[tier_names[pred_enc]] = 1.0
        return probs


def _adjust_probs(raw: dict, rule_score: float) -> dict:
    """
    Shift probability mass between High and Low based on rule_score.
    Positive score → move mass toward High.
    Negative score → move mass toward Low.
    Medium is gently adjusted.
    Rating-driven shift is stronger (max 60 pp) to make rating differences visible.
    """
    high = raw.get("High", 0.0)
    med  = raw.get("Medium", 0.0)
    low  = raw.get("Low",  0.0)

    shift = rule_score * 0.50   # max 50 pp shift — strong enough to be visible, won't override RF completely

    if shift > 0:
        # Take from Low, add to High
        take = min(low, shift)
        low  -= take
        high += take
        # Also pull from Medium toward High (25% of shift) — makes high ratings more visible
        take2 = min(med, shift * 0.25)
        med  -= take2
        high += take2
    else:
        # Take from High, add to Low
        take = min(high, abs(shift))
        high -= take
        low  += take
        take2 = min(med, abs(shift) * 0.20)
        med  -= take2
        low  += take2

    total = high + med + low
    if total > 0:
        high /= total
        med  /= total
        low  /= total

    return {"High": round(high, 4), "Medium": round(med, 4), "Low": round(low, 4)}


def _probs_to_tier(probs: dict) -> str:
    return max(probs, key=probs.get)


# ─── Public predict function ──────────────────────────────────────────────────
def predict_success(
    input_dict: dict,
    models: dict,
    scaler: StandardScaler,
    feature_names: list,
    label_encoder,
    chosen_model: str = "Random Forest",
) -> tuple:
    """
    Returns (tier_str, adjusted_probs_dict)
    adjusted_probs_dict = {'High': float, 'Medium': float, 'Low': float}  (sum to 1)
    """
    tier_names = list(label_encoder.classes_)   # alphabetical: High, Low, Medium

    raw   = _raw_probs(input_dict, models, scaler, feature_names, chosen_model, tier_names)
    score = _compute_rule_score(input_dict)
    adj   = _adjust_probs(raw, score)
    tier  = _probs_to_tier(adj)

    return tier, adj
