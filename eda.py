"""
eda.py
------
Reusable Matplotlib plotting functions. Every function returns a Figure
so Streamlit can render it with st.pyplot().
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ─── Design tokens (minimal SaaS dark) ───────────────────────────────────────
PAL  = ["#4F46E5", "#6366F1", "#818CF8", "#A5B4FC"]
BG   = "#0B0F19"
AX_BG = "#111827"
GRID = "#2A3648"
TXT = "#E5E7EB"
TXT_MUTED = "#9CA3AF"
SPINE = "#1F2937"


def _base(ax, title, xlabel="", ylabel=""):
    ax.set_title(title, fontsize=13, fontweight="bold", color=TXT, pad=10)
    ax.set_xlabel(xlabel, fontsize=10, color=TXT_MUTED)
    ax.set_ylabel(ylabel, fontsize=10, color=TXT_MUTED)
    ax.tick_params(labelsize=8, colors=TXT_MUTED)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SPINE)
    ax.spines["bottom"].set_color(SPINE)
    ax.yaxis.grid(True, color=GRID, linewidth=0.7, linestyle="--")
    ax.set_axisbelow(True)


# ─── 1. Rating Distribution ───────────────────────────────────────────────────
def plot_rating_distribution(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 4), facecolor=BG)
    ax.set_facecolor(AX_BG)

    counts, bins, patches = ax.hist(
        df["Rating"].dropna(), bins=20,
        color=PAL[1], edgecolor="#E5E7EB", linewidth=0.4, alpha=0.92
    )
    # colour bars by rating band
    for patch, left in zip(patches, bins[:-1]):
        if left < 3.0:
            patch.set_facecolor("#6B7280")
        elif left < 4.0:
            patch.set_facecolor(PAL[2])
        else:
            patch.set_facecolor(PAL[0])

    mean_r = df["Rating"].mean()
    ax.axvline(mean_r, color="#DDE8F8", linestyle="--", linewidth=1.5,
               label=f"Mean = {mean_r:.2f}")
    ax.legend(fontsize=9, facecolor=AX_BG, edgecolor=SPINE, labelcolor=TXT)

    low_p  = mpatches.Patch(color="#6B7280", label="< 3.0")
    mid_p  = mpatches.Patch(color=PAL[2], label="3.0 – 4.0")
    high_p = mpatches.Patch(color=PAL[0], label="> 4.0")
    ax.legend(handles=[low_p, mid_p, high_p], fontsize=9, loc="upper left",
              facecolor=AX_BG, edgecolor=SPINE, labelcolor=TXT)

    _base(ax, "App Rating Distribution", "Rating", "Number of Apps")
    fig.tight_layout()
    return fig


# ─── 2. Top Categories by Installs ───────────────────────────────────────────
def plot_category_installs(df: pd.DataFrame) -> plt.Figure:
    cat_avg = (
        df.groupby("Category")["Installs"]
        .mean()
        .sort_values(ascending=False)
        .head(12)
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG)
    ax.set_facecolor(AX_BG)

    colors = [PAL[i % 4] for i in range(len(cat_avg))]
    bars = ax.barh(cat_avg["Category"][::-1], cat_avg["Installs"][::-1],
                   color=colors[::-1], edgecolor="#E5E7EB", linewidth=0.3, height=0.7)

    for bar, val in zip(bars, cat_avg["Installs"][::-1]):
        ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                f"{val/1e6:.1f}M", va="center", fontsize=8, color=TXT)

    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M")
    )
    _base(ax, "Top 12 Categories by Average Installs", "Avg Installs", "Category")
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.7, linestyle="--")
    fig.tight_layout()
    return fig


# ─── 3. Price vs Installs ─────────────────────────────────────────────────────
def plot_price_vs_installs(df: pd.DataFrame) -> plt.Figure:
    paid = df[df["Paid_App"] == 1].copy()
    paid = paid[paid["Price"] < 30]     # clip extreme outliers for readability

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), facecolor=BG)

    # Left: scatter paid apps
    ax = axes[0]
    ax.set_facecolor(AX_BG)
    sc = ax.scatter(
        paid["Price"], np.log1p(paid["Installs"]),
        c=paid["Rating"], cmap="Blues", alpha=0.6, s=20, vmin=1, vmax=5
    )
    cbar = plt.colorbar(sc, ax=ax, label="Rating")
    cbar.ax.yaxis.set_tick_params(color=TXT_MUTED)
    plt.setp(cbar.ax.get_yticklabels(), color=TXT_MUTED)
    cbar.set_label("Rating", color=TXT_MUTED)
    _base(ax, "Paid Apps: Price vs Log(Installs)", "Price ($)", "Log(Installs)")

    # Right: free vs paid avg installs
    ax2 = axes[1]
    ax2.set_facecolor(AX_BG)
    avg = df.groupby("Paid_App")["Installs"].mean()
    labels = ["Free", "Paid"]
    bar_colors = [PAL[0], "#6B7280"]
    bars = ax2.bar(labels, [avg.get(0, 0), avg.get(1, 0)],
                   color=bar_colors, edgecolor="#E5E7EB", linewidth=0.4, width=0.5)
    for bar, val in zip(bars, [avg.get(0, 0), avg.get(1, 0)]):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                 f"{val/1e6:.2f}M", ha="center", fontsize=10, fontweight="bold", color=TXT)
    _base(ax2, "Avg Installs: Free vs Paid", "App Type", "Avg Installs")

    fig.tight_layout()
    return fig


# ─── 4. Correlation Heatmap ───────────────────────────────────────────────────
def plot_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    num_cols = ["Rating", "Reviews", "Installs", "Price",
                "Size_MB", "Paid_App", "Engagement", "Category_Avg_Installs"]
    num_cols = [c for c in num_cols if c in df.columns]
    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(9, 7), facecolor=BG)
    ax.set_facecolor(AX_BG)

    im = ax.imshow(corr, cmap="cividis", vmin=-1, vmax=1, aspect="auto")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color=TXT_MUTED)
    plt.setp(cbar.ax.get_yticklabels(), color=TXT_MUTED)

    labels = corr.columns.tolist()
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8, color=TXT_MUTED)
    ax.set_yticklabels(labels, fontsize=8, color=TXT_MUTED)

    for i in range(len(labels)):
        for j in range(len(labels)):
            v = corr.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=7.5,
                    color="#0B1220" if abs(v) < 0.6 else "#F8FAFC")

    ax.set_title("Feature Correlation Heatmap", fontsize=13,
                 fontweight="bold", color=TXT, pad=10)
    fig.tight_layout()
    return fig


# ─── 5. Install Tier Distribution ────────────────────────────────────────────
def plot_tier_distribution(df: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor=BG)

    # Pie
    counts = df["Install_Tier"].value_counts()
    tier_colors = {"Low": "#6B7280", "Medium": PAL[2], "High": PAL[0]}
    colors = [tier_colors.get(t, PAL[0]) for t in counts.index]

    axes[0].set_facecolor(AX_BG)
    axes[0].pie(counts.values, labels=counts.index, colors=colors,
                autopct="%1.1f%%", startangle=140,
                textprops=dict(color=TXT_MUTED),
                wedgeprops=dict(edgecolor="#E5E7EB", linewidth=0.8))
    axes[0].set_title("Install Tier Distribution", fontsize=13, fontweight="bold", color=TXT)

    # Bar by type
    axes[1].set_facecolor(AX_BG)
    tier_type = df.groupby(["Install_Tier", "Type"]).size().unstack(fill_value=0)
    tier_order = ["Low", "Medium", "High"]
    tier_type = tier_type.reindex([t for t in tier_order if t in tier_type.index])

    x = range(len(tier_type))
    width = 0.35
    for i, col in enumerate(tier_type.columns):
        offset = (i - 0.5) * width
        bars = axes[1].bar(
            [xi + offset for xi in x], tier_type[col],
            width=width, label=col,
            color=PAL[0] if col == "Free" else "#6B7280",
            edgecolor="#E5E7EB",
            linewidth=0.3
        )
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(tier_type.index)
    axes[1].legend(fontsize=9, facecolor=AX_BG, edgecolor=SPINE, labelcolor=TXT)
    _base(axes[1], "Install Tiers: Free vs Paid", "Tier", "Count")

    fig.tight_layout()
    return fig


# ─── 6. Rating by Category (top 10) ──────────────────────────────────────────
def plot_rating_by_category(df: pd.DataFrame) -> plt.Figure:
    top_cats = df["Category"].value_counts().head(10).index
    df_top = df[df["Category"].isin(top_cats)]

    groups = [df_top[df_top["Category"] == c]["Rating"].dropna().values
              for c in top_cats]

    fig, ax = plt.subplots(figsize=(11, 5), facecolor=BG)
    ax.set_facecolor(AX_BG)

    bp = ax.boxplot(groups, labels=top_cats, patch_artist=True,
                    medianprops=dict(color="#E2ECFA", linewidth=2))
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(PAL[i % 4])
        patch.set_alpha(0.8)

    ax.tick_params(axis="x", rotation=30)
    _base(ax, "Rating Distribution – Top 10 Categories", "Category", "Rating")
    fig.tight_layout()
    return fig


# ─── 7. Top Apps by Reviews ───────────────────────────────────────────────────
def plot_top_apps_reviews(df: pd.DataFrame) -> plt.Figure:
    top = df.nlargest(10, "Reviews")[["App", "Reviews", "Install_Tier"]].copy()
    top["App_short"] = top["App"].str[:25]

    tier_colors = {"Low": "#6B7280", "Medium": PAL[2], "High": PAL[0]}
    colors = [tier_colors.get(t, PAL[0]) for t in top["Install_Tier"]]

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG)
    ax.set_facecolor(AX_BG)

    bars = ax.barh(top["App_short"][::-1], top["Reviews"][::-1],
                   color=colors[::-1], edgecolor="#E5E7EB", linewidth=0.3, height=0.7)
    for bar, val in zip(bars, top["Reviews"][::-1]):
        ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                f"{val/1e6:.1f}M", va="center", fontsize=8, color=TXT)

    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M")
    )
    _base(ax, "Top 10 Apps by Number of Reviews", "Reviews", "")
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.7, linestyle="--")
    fig.tight_layout()
    return fig


import matplotlib.ticker
