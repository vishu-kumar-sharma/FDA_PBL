"""
app.py  –  Google Play Store App Success Predictor
----------------------------------------------------
Run:  streamlit run app.py
"""

import os
import streamlit as st
import pandas as pd
import numpy as np

# ── Page config (must be very first Streamlit call) ───────────────────────────
st.set_page_config(
    page_title="Play Store Success Predictor",
    page_icon="PS",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #090E1A;
    --surface: #111827;
    --surface-2: #0D1424;
    --border: #1E2A40;
    --text-primary: #F0F4FF;
    --text-secondary: #7A8BA8;
    --accent: #7C5CFC;
    --accent-light: #1E1640;
    --accent2: #FF6B35;
    --success: #00C48C;
    --warning: #FFB020;
    --danger: #FF4757;
    --grad-start: #7C5CFC;
    --grad-end: #FF6B35;
}

html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
.stApp { background: #090E1A !important; color: #F0F4FF !important; }
.main, .block-container, [data-testid="stAppViewContainer"] { background: #090E1A !important; }
header[data-testid="stHeader"] { background: #090E1A !important; border-bottom: 1px solid #1E2A40 !important; }
.block-container { max-width: 1240px; padding-top: 24px; padding-bottom: 24px; }

/* Top bar */
.topbar {
    display: flex; justify-content: space-between; align-items: center;
    background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
    padding: 14px 20px; margin-bottom: 20px;
    box-shadow: 0 2px 8px rgba(108,63,255,0.06);
}
.topbar-title { font-size: 20px; font-weight: 700; color: var(--text-primary); margin: 0; }
.topbar-sub { font-size: 13px; color: var(--text-secondary); margin: 2px 0 0 0; }
.deploy-btn {
    background: linear-gradient(135deg, var(--grad-start), var(--grad-end));
    color: #FFFFFF; border: none;
    border-radius: 8px; padding: 9px 18px; font-size: 14px; font-weight: 700;
    box-shadow: 0 4px 12px rgba(108,63,255,0.30);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0D1424;
    border-right: 1px solid var(--border);
    min-width: 225px !important; max-width: 225px !important;
}
[data-testid="stSidebar"] * { color: var(--text-secondary) !important; }
[data-testid="stSidebar"] .stRadio label {
    border: 1px solid transparent; border-radius: 8px; padding: 9px 12px; font-weight: 500;
    transition: all 0.15s ease;
}
[data-testid="stSidebar"] .stRadio label:hover { background: #1E1640; border-color: #3D2A80; color: #C4B3FF !important; }
[data-testid="stSidebar"] .stRadio label:has(input:checked) {
    background: #1E1640; border-color: #7C5CFC; color: #C4B3FF !important;
    font-weight: 600 !important;
}

/* Hero banner */
.hero-card {
    background: linear-gradient(135deg, #6C3FFF 0%, #A855F7 50%, #FF6B35 100%);
    border-radius: 16px; padding: 32px 28px; margin-bottom: 24px;
    box-shadow: 0 8px 32px rgba(108,63,255,0.25);
    position: relative; overflow: hidden;
}
.hero-card::before {
    content: ''; position: absolute; top: -60px; right: -60px;
    width: 200px; height: 200px; border-radius: 50%;
    background: rgba(255,255,255,0.08);
}
.hero-card::after {
    content: ''; position: absolute; bottom: -40px; left: 30%;
    width: 150px; height: 150px; border-radius: 50%;
    background: rgba(255,255,255,0.06);
}
.hero-title { font-size: 28px; line-height: 1.3; font-weight: 800; color: #FFFFFF !important; margin: 0; text-shadow: 0 2px 8px rgba(0,0,0,0.35); }
.hero-sub { color: rgba(255,255,255,0.92) !important; margin: 10px 0 0 0; font-size: 15px; text-shadow: 0 1px 4px rgba(0,0,0,0.3); }
.hero-badge {
    display: inline-block; background: rgba(255,255,255,0.18); border: 1px solid rgba(255,255,255,0.30);
    border-radius: 999px; padding: 4px 14px; font-size: 12px; font-weight: 600;
    color: #FFFFFF; margin-bottom: 14px; backdrop-filter: blur(4px);
}

/* Cards and hierarchy */
.panel, .quick-card, .result-card {
    background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
    box-shadow: 0 2px 8px rgba(15,22,48,0.06);
}
.panel { padding: 20px; }
.quick-grid { margin: 0 0 20px 0; }
.quick-card { padding: 18px; transition: transform 0.15s ease, box-shadow 0.15s ease; }
.quick-card:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(108,63,255,0.10); }
.quick-icon { font-size: 22px; margin-bottom: 10px; }
.quick-label { color: var(--text-secondary); font-size: 12px; font-weight: 600; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.05em; }
.quick-value { color: var(--text-primary); font-size: 30px; font-weight: 800; line-height: 1.1; }
.quick-value-accent { background: linear-gradient(135deg, var(--grad-start), var(--grad-end)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
.sub-card-title { color: var(--text-primary); font-size: 18px; font-weight: 700; margin-bottom: 10px; }
.sec-hdr {
    font-size: 20px; font-weight: 700; color: var(--text-primary);
    margin-bottom: 8px; padding: 0;
}
.muted { color: var(--text-secondary); font-size: 14px; margin-bottom: 16px; }

/* About section */
.about-panel {
    background: linear-gradient(135deg, #13102A 0%, #1A1020 100%);
    border: 1px solid #2A1F50; border-radius: 12px; padding: 20px;
}
.about-row { display: flex; align-items: flex-start; gap: 12px; margin-bottom: 14px; }
.about-icon {
    width: 36px; height: 36px; border-radius: 10px; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center; font-size: 16px;
}
.about-icon-purple { background: rgba(124,92,252,0.18); }
.about-icon-orange { background: rgba(255,107,53,0.18); }
.about-icon-green { background: rgba(0,196,140,0.18); }
.about-text-title { font-size: 14px; font-weight: 700; color: var(--text-primary); margin: 0; }
.about-text-body { font-size: 13px; color: var(--text-secondary); margin: 2px 0 0 0; }

/* Stack chips */
.stack-chip {
    display: inline-block; background: rgba(124,92,252,0.15); border: 1px solid rgba(124,92,252,0.35);
    color: #C4B3FF; border-radius: 999px; padding: 3px 12px;
    font-size: 12px; font-weight: 600; margin: 3px 4px 3px 0;
}

/* Metrics + typography */
div[data-testid="metric-container"] {
    background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
    padding: 16px; box-shadow: 0 2px 8px rgba(15,22,48,0.06);
}
div[data-testid="metric-container"] label { color: var(--text-secondary) !important; font-size: 12px !important; font-weight: 600 !important; text-transform: uppercase; letter-spacing: 0.04em; }
div[data-testid="metric-container"] div[data-testid="stMetricValue"] { color: var(--text-primary); font-size: 28px; font-weight: 800; }

h1 { font-size: 24px; font-weight: 700; color: var(--text-primary); }
h2 { font-size: 18px; font-weight: 700; color: var(--text-primary); }
h3 { font-size: 16px; font-weight: 600; color: var(--text-primary); }
p, li { color: var(--text-secondary) !important; font-size: 14px; }
label, .stCaption { color: var(--text-secondary) !important; font-size: 14px; }
strong { color: var(--text-primary); }

/* Tabs, controls, buttons */
button[data-baseweb="tab"] {
    border-radius: 8px; border: 1px solid var(--border); background: var(--surface-2);
    color: var(--text-secondary); font-size: 14px; font-weight: 500;
}
button[data-baseweb="tab"][aria-selected="true"] {
    background: var(--accent-light); color: var(--accent); border-color: #C4B3FF; font-weight: 700;
}
button[kind="primary"] {
    background: linear-gradient(135deg, var(--grad-start), var(--grad-end));
    border: none; border-radius: 8px;
    color: #FFFFFF; font-size: 14px; font-weight: 700;
    box-shadow: 0 4px 12px rgba(108,63,255,0.25);
}
button[kind="primary"]:hover { filter: brightness(1.05); transform: translateY(-1px); }

.stTextInput > div > div > input, .stNumberInput input, .stSelectbox > div > div, .stTextArea textarea {
    border-radius: 8px !important; border: 1px solid var(--border) !important;
    background: var(--surface-2) !important; color: var(--text-primary) !important;
}
.stSlider [data-baseweb="slider"] > div > div { background: #E0D9FF !important; }
.stSlider [role="slider"] { border: 2px solid var(--accent) !important; background: var(--accent) !important; box-shadow: 0 0 0 4px rgba(108,63,255,0.15) !important; }

.stDataFrame, div[data-testid="stTable"], details[data-testid="stExpander"], [data-testid="stAlert"] {
    border: 1px solid var(--border); border-radius: 12px; background: var(--surface);
}
hr { border-color: var(--border); }

/* Status blocks */
.badge-high, .badge-medium, .badge-low { border-radius: 10px; padding: 16px; font-size: 18px; font-weight: 700; text-align: center; border: 1px solid; }
.badge-high { background: linear-gradient(135deg, rgba(0,196,140,0.10), rgba(0,196,140,0.04)); color: #007A5A; border-color: rgba(0,196,140,0.35); }
.badge-medium { background: linear-gradient(135deg, rgba(255,176,32,0.10), rgba(255,176,32,0.04)); color: #8A5C00; border-color: rgba(255,176,32,0.35); }
.badge-low { background: linear-gradient(135deg, rgba(255,71,87,0.10), rgba(255,71,87,0.04)); color: #A0001A; border-color: rgba(255,71,87,0.35); }
.result-card { padding: 16px; margin-bottom: 8px; }
.factor-chip { border-radius: 999px; padding: 3px 10px; font-size: 12px; font-weight: 700; display: inline-block; margin-right: 8px; }
.factor-strong { background: rgba(0,196,140,0.12); color: #007A5A; border: 1px solid rgba(0,196,140,0.30); }
.factor-neutral { background: rgba(108,63,255,0.10); color: var(--accent); border: 1px solid rgba(108,63,255,0.25); }
.factor-risk { background: rgba(255,71,87,0.10); color: #A0001A; border: 1px solid rgba(255,71,87,0.25); }

/* Nav info footer */
.nav-info { font-size: 12px; color: var(--text-secondary); line-height: 1.6; }
</style>
""", unsafe_allow_html=True)

# ── Module imports ────────────────────────────────────────────────────────────
from data_preprocessing import preprocess
from eda import (
    plot_rating_distribution, plot_category_installs,
    plot_price_vs_installs,   plot_correlation_heatmap,
    plot_tier_distribution,   plot_rating_by_category,
    plot_top_apps_reviews,
)
from model import (
    train_models, plot_confusion_matrix,
    plot_feature_importance, predict_success,
)
from utils import (
    compute_statistics, compute_ttest, compute_correlations,
    generate_recommendations, format_stats_table,
)

# ── Data & model loading (cached) ─────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "Play_Store_Data.csv")

@st.cache_data(show_spinner="Loading and cleaning data...")
def load_data():
    return preprocess(DATA_PATH)

@st.cache_resource(show_spinner="Training ML models...")
def load_models(enc_hash):
    return train_models(df_encoded, label_encoders)

df_encoded, df_clean, label_encoders = load_data()
ml_results, models, scaler, _, _, feat_names = load_models(id(df_encoded))

def render_topbar(title: str, subtitle: str, show_deploy: bool = False):
    action_html = '<button class="deploy-btn">Deploy</button>' if show_deploy else ""
    st.markdown(
        f"""
        <div class="topbar">
            <div>
                <p class="topbar-title">{title}</p>
                <p class="topbar-sub">{subtitle}</p>
            </div>
            {action_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Sidebar Navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Navigation")
    st.markdown("---")
    page = st.radio("", [
        "⌂ Home",
        "◫ Data Overview",
        "◪ EDA",
        "◎ Statistics",
        "◩ ML Model",
        "◧ Prediction",
    ], label_visibility="collapsed", key="nav_page")
    st.markdown("---")
    st.markdown(
        f"<small>Google Play Store Dataset<br>"
        f"Apps: {len(df_clean):,} | Features: {df_clean.shape[1]}</small>",
        unsafe_allow_html=True,
    )

page = page.split(" ", 1)[1]

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 – HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "Home":
    render_topbar("Play Store Analytics", "Production-style dashboard for app growth prediction", show_deploy=False)

    # ── Hero Banner ──────────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-badge">🚀 ML-Powered Analytics</div>
            <p class="hero-title" style="color:#FFFFFF !important; text-shadow: 0 2px 8px rgba(0,0,0,0.4);">Google Play Store<br>App Success Predictor</p>
            <p class="hero-sub" style="color:rgba(255,255,255,0.92) !important; text-shadow: 0 1px 4px rgba(0,0,0,0.3);">Predict install tier using rating, pricing, category &amp; engagement signals — powered by Random Forest &amp; Logistic Regression.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── KPI Cards ────────────────────────────────────────────────────────────
    total  = len(df_clean)
    high_n = (df_clean["Install_Tier"] == "High").sum()
    mid_n  = (df_clean["Install_Tier"] == "Medium").sum()
    low_n  = (df_clean["Install_Tier"] == "Low").sum()
    avg_r  = df_clean["Rating"].mean()
    free_n = (df_clean["Paid_App"] == 0).sum()

    q1, q2, q3 = st.columns(3)
    with q1:
        st.markdown(
            f"""<div class='quick-card'>
                <div class='quick-icon'>📦</div>
                <div class='quick-label'>Total Apps</div>
                <div class='quick-value quick-value-accent'>{total:,}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with q2:
        st.markdown(
            f"""<div class='quick-card'>
                <div class='quick-icon'>⭐</div>
                <div class='quick-label'>Average Rating</div>
                <div class='quick-value quick-value-accent'>{avg_r:.2f}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with q3:
        st.markdown(
            f"""<div class='quick-card'>
                <div class='quick-icon'>🆓</div>
                <div class='quick-label'>Free Apps</div>
                <div class='quick-value quick-value-accent'>{free_n:,}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    q4, q5, q6 = st.columns(3)
    with q4:
        pct = high_n / total * 100
        st.markdown(
            f"""<div class='quick-card'>
                <div class='quick-icon'>🟢</div>
                <div class='quick-label'>High Tier (&gt;1M installs)</div>
                <div class='quick-value' style='color:#00C48C;'>{high_n:,}</div>
                <div style='margin-top:8px;height:6px;background:#E0F9F1;border-radius:999px;'>
                  <div style='width:{pct:.1f}%;height:100%;background:#00C48C;border-radius:999px;'></div>
                </div>
                <div style='font-size:12px;color:#5A6480;margin-top:4px;'>{pct:.0f}% of total</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with q5:
        pct = mid_n / total * 100
        st.markdown(
            f"""<div class='quick-card'>
                <div class='quick-icon'>🟡</div>
                <div class='quick-label'>Medium Tier (10K–1M)</div>
                <div class='quick-value' style='color:#FFB020;'>{mid_n:,}</div>
                <div style='margin-top:8px;height:6px;background:#FFF8E6;border-radius:999px;'>
                  <div style='width:{pct:.1f}%;height:100%;background:#FFB020;border-radius:999px;'></div>
                </div>
                <div style='font-size:12px;color:#5A6480;margin-top:4px;'>{pct:.0f}% of total</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with q6:
        pct = low_n / total * 100
        st.markdown(
            f"""<div class='quick-card'>
                <div class='quick-icon'>🔴</div>
                <div class='quick-label'>Low Tier (&lt;10K installs)</div>
                <div class='quick-value' style='color:#FF4757;'>{low_n:,}</div>
                <div style='margin-top:8px;height:6px;background:#FFF0F1;border-radius:999px;'>
                  <div style='width:{pct:.1f}%;height:100%;background:#FF4757;border-radius:999px;'></div>
                </div>
                <div style='font-size:12px;color:#5A6480;margin-top:4px;'>{pct:.0f}% of total</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── About + Navigation Guide ─────────────────────────────────────────────
    col_about, col_nav = st.columns([3, 2])

    with col_about:
        st.markdown("<div class='about-panel'>", unsafe_allow_html=True)
        st.markdown("<div class='sub-card-title'>About This Project</div>", unsafe_allow_html=True)
        st.markdown("""
<div class='about-row'>
  <div class='about-icon about-icon-purple'>🎯</div>
  <div>
    <p class='about-text-title'>What it does</p>
    <p class='about-text-body'>Predicts whether a new app is likely to be Low, Medium, or High install tier using ML models trained on real Play Store data.</p>
  </div>
</div>
<div class='about-row'>
  <div class='about-icon about-icon-orange'>📊</div>
  <div>
    <p class='about-text-title'>Install Tier Rules</p>
    <p class='about-text-body'><strong>High:</strong> &gt;1M installs &nbsp;|&nbsp; <strong>Medium:</strong> 10K–1M &nbsp;|&nbsp; <strong>Low:</strong> &lt;10K installs</p>
  </div>
</div>
<div class='about-row' style='margin-bottom:0;'>
  <div class='about-icon about-icon-green'>🛠️</div>
  <div>
    <p class='about-text-title'>Tech Stack</p>
    <p class='about-text-body'>
      <span class='stack-chip'>Python</span>
      <span class='stack-chip'>Pandas</span>
      <span class='stack-chip'>NumPy</span>
      <span class='stack-chip'>Scikit-learn</span>
      <span class='stack-chip'>Streamlit</span>
      <span class='stack-chip'>Matplotlib</span>
      <span class='stack-chip'>SciPy</span>
    </p>
  </div>
</div>
""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_nav:
        st.markdown("""
<div class='panel'>
  <div class='sub-card-title'>🗺️ Navigation Guide</div>
  <div style='display:flex;flex-direction:column;gap:10px;'>
    <div style='display:flex;align-items:center;gap:10px;padding:10px;background:rgba(124,92,252,0.10);border-radius:8px;border-left:3px solid #7C5CFC;'>
      <span style='font-size:18px;'>◫</span>
      <div><strong style='font-size:13px;color:#F0F4FF;'>Data Overview</strong><br><span style='font-size:12px;color:#7A8BA8;'>Schema, sample rows, stats</span></div>
    </div>
    <div style='display:flex;align-items:center;gap:10px;padding:10px;background:rgba(255,107,53,0.10);border-radius:8px;border-left:3px solid #FF6B35;'>
      <span style='font-size:18px;'>◪</span>
      <div><strong style='font-size:13px;color:#F0F4FF;'>EDA</strong><br><span style='font-size:12px;color:#7A8BA8;'>Charts &amp; visual analysis</span></div>
    </div>
    <div style='display:flex;align-items:center;gap:10px;padding:10px;background:rgba(0,196,140,0.10);border-radius:8px;border-left:3px solid #00C48C;'>
      <span style='font-size:18px;'>◎</span>
      <div><strong style='font-size:13px;color:#F0F4FF;'>Statistics</strong><br><span style='font-size:12px;color:#7A8BA8;'>T-tests, correlations</span></div>
    </div>
    <div style='display:flex;align-items:center;gap:10px;padding:10px;background:rgba(168,85,247,0.10);border-radius:8px;border-left:3px solid #A855F7;'>
      <span style='font-size:18px;'>◩</span>
      <div><strong style='font-size:13px;color:#F0F4FF;'>ML Model</strong><br><span style='font-size:12px;color:#7A8BA8;'>Accuracy, confusion matrix</span></div>
    </div>
    <div style='display:flex;align-items:center;gap:10px;padding:10px;background:rgba(255,71,87,0.10);border-radius:8px;border-left:3px solid #FF4757;'>
      <span style='font-size:18px;'>◧</span>
      <div><strong style='font-size:13px;color:#F0F4FF;'>Prediction</strong><br><span style='font-size:12px;color:#7A8BA8;'>Run a custom scenario</span></div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 – DATA OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Data Overview":
    render_topbar("Data Overview", "Dataset shape, schema, and engineered features")
    st.markdown('<div class="sec-hdr">Data Overview</div>', unsafe_allow_html=True)
    st.markdown("<div class='muted'>Inspect dataset structure, engineered features, and data quality indicators.</div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows",   f"{df_clean.shape[0]:,}")
    c2.metric("Columns", df_clean.shape[1])
    c3.metric("Missing Values", int(df_clean.isnull().sum().sum()))
    c4.metric("Categories", df_clean["Category"].nunique())

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["Dataset Sample", "Column Info", "Summary Stats"])

    with tab1:
        display_cols = ["App", "Category", "Rating", "Reviews", "Installs",
                        "Type", "Price", "Size_MB", "Content Rating",
                        "Paid_App", "Engagement", "Install_Tier"]
        display_cols = [c for c in display_cols if c in df_clean.columns]
        st.dataframe(df_clean[display_cols].head(25), use_container_width=True)

    with tab2:
        info = pd.DataFrame({
            "Column":    df_clean.columns,
            "Dtype":     df_clean.dtypes.astype(str).values,
            "Non-Null":  df_clean.notnull().sum().values,
            "Unique":    df_clean.nunique().values,
        })
        st.dataframe(info, use_container_width=True)
        st.markdown("#### Engineered Features")
        st.markdown("""
| Feature | Description |
|---------|-------------|
| `Paid_App` | 1 if Paid, 0 if Free |
| `Engagement` | Reviews ÷ Installs — proxy for user loyalty |
| `Category_Avg_Installs` | Mean installs in the same category — market size signal |
| `Size_MB` | App size normalised to Megabytes |
| `Install_Tier` | Target: Low / Medium / High |
""")

    with tab3:
        numeric_display = ["Rating", "Reviews", "Installs", "Price",
                           "Size_MB", "Engagement", "Category_Avg_Installs"]
        numeric_display = [c for c in numeric_display if c in df_clean.columns]
        st.dataframe(
            df_clean[numeric_display].describe().T.round(2)
            .reset_index().rename(columns={"index": "Feature"}),
            use_container_width=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 – EDA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "EDA":
    render_topbar("EDA", "Visual exploration of signals and patterns")
    st.markdown('<div class="sec-hdr">Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.markdown("<div class='muted'>Visual analysis of category behavior, install trends, and feature relationships.</div>", unsafe_allow_html=True)

    chart = st.selectbox("Select a chart:", [
        "Rating Distribution",
        "Top Categories by Installs",
        "Price vs Installs",
        "Correlation Heatmap",
        "Install Tier Distribution",
        "Rating by Category (Box)",
        "Top Apps by Reviews",
    ])

    st.markdown("---")

    if chart == "Rating Distribution":
        st.markdown("#### App Rating Distribution")
        st.markdown("Histogram coloured by rating band. Mean marked with dashed line.")
        st.pyplot(plot_rating_distribution(df_clean))

    elif chart == "Top Categories by Installs":
        st.markdown("#### Top 12 Categories by Average Installs")
        st.markdown("Horizontal bar chart sorted by average install count.")
        st.pyplot(plot_category_installs(df_clean))

    elif chart == "Price vs Installs":
        st.markdown("#### Price vs Installs  |  Free vs Paid")
        st.markdown("Left: scatter of paid apps coloured by rating. Right: free vs paid average installs.")
        st.pyplot(plot_price_vs_installs(df_clean))

    elif chart == "Correlation Heatmap":
        st.markdown("#### Feature Correlation Heatmap")
        st.markdown("Pearson correlation between numeric features (green = positive, red = negative).")
        st.pyplot(plot_correlation_heatmap(df_clean))

    elif chart == "Install Tier Distribution":
        st.markdown("#### Install Tier Distribution")
        st.markdown("Pie chart of tier share and grouped bar by app type.")
        st.pyplot(plot_tier_distribution(df_clean))

    elif chart == "Rating by Category (Box)":
        st.markdown("#### Rating Distribution – Top 10 Categories")
        st.markdown("Box-plots showing rating spread for the 10 most common categories.")
        st.pyplot(plot_rating_by_category(df_clean))

    else:
        st.markdown("#### Top 10 Apps by Number of Reviews")
        st.markdown("Bars coloured by Install Tier.")
        st.pyplot(plot_top_apps_reviews(df_clean))

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 – STATISTICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Statistics":
    render_topbar("Statistics", "Statistical tests and numerical insights")
    st.markdown('<div class="sec-hdr">Statistical Analysis</div>', unsafe_allow_html=True)
    st.markdown("<div class='muted'>Statistical validation for product, monetization, and growth decisions.</div>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Descriptive Stats", "Correlations", "T-Test", "Recommendations"]
    )

    with tab1:
        st.markdown("#### Descriptive Statistics for Key Features")
        st.dataframe(
            format_stats_table(compute_statistics(df_clean)),
            use_container_width=True,
        )

    with tab2:
        st.markdown("#### Pearson Correlation with Installs")
        corr_df = compute_correlations(df_clean)
        st.dataframe(
            corr_df.style.background_gradient(
                subset=["Correlation with Installs"], cmap="RdYlGn", vmin=-1, vmax=1
            ),
            use_container_width=True,
        )

    with tab3:
        t = compute_ttest(df_clean)
        st.markdown("#### Welch's T-Test: Free vs Paid App Installs")
        st.markdown(f"""
| Metric | Value |
|--------|-------|
| Free Apps (n) | {t['n_free']:,} |
| Paid Apps (n) | {t['n_paid']:,} |
| Mean Installs – Free | {t['mean_free']:,.0f} |
| Mean Installs – Paid | {t['mean_paid']:,.0f} |
| T-statistic | {t['t_stat']} |
| P-value | {t['p_value']} |
| Significant (α = 0.05)? | {'Yes' if t['significant'] else 'No'} |
""")
        if t["significant"]:
            st.success(
                f"**Interpretation:** Free apps receive significantly more installs than paid apps "
                f"(p = {t['p_value']}). The difference is not due to chance."
            )
        else:
            st.info(f"No statistically significant difference found (p = {t['p_value']}).")

    with tab4:
        st.markdown("#### Business Recommendations")
        for rec in generate_recommendations(df_clean):
            st.markdown(f"> {rec}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 – ML MODEL
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ML Model":
    render_topbar("ML Model", "Model performance and error analysis")
    st.markdown('<div class="sec-hdr">Machine Learning Models</div>', unsafe_allow_html=True)
    st.markdown("<div class='muted'>Compare model quality, class-level performance, and feature influence.</div>", unsafe_allow_html=True)

    model_names = list(ml_results.keys())

    st.markdown("### Model Accuracy")
    acc_cols = st.columns(len(model_names))
    for col, name in zip(acc_cols, model_names):
        acc = ml_results[name]["accuracy"]
        col.metric(name, f"{acc*100:.2f}%", help="Test-set accuracy (25% hold-out)")

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["Confusion Matrix", "Classification Report", "Feature Importance"])

    with tab1:
        sel = st.selectbox("Select model:", model_names, key="cm")
        cm  = ml_results[sel]["confusion_matrix"]
        c1, c2 = st.columns([1, 1])
        with c1:
            st.pyplot(plot_confusion_matrix(cm, sel))

    with tab2:
        sel2  = st.selectbox("Select model:", model_names, key="cr")
        report = ml_results[sel2]["report"]
        tnames = list(label_encoders["Install_Tier"].classes_)
        rows = []
        for cls in tnames:
            if cls in report:
                rows.append({
                    "Class":     cls,
                    "Precision": round(report[cls]["precision"], 3),
                    "Recall":    round(report[cls]["recall"],    3),
                    "F1-Score":  round(report[cls]["f1-score"],  3),
                    "Support":   int(report[cls]["support"]),
                })
        rows.append({
            "Class":     "Overall Accuracy",
            "Precision": "—", "Recall": "—",
            "F1-Score":  round(ml_results[sel2]["accuracy"], 3),
            "Support":   sum(r["Support"] for r in rows),
        })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    with tab3:
        fig_fi = plot_feature_importance(models, feat_names)
        if fig_fi:
            st.pyplot(fig_fi)
        else:
            st.info("Feature importance not available.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 – PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Prediction":
    render_topbar("Prediction", "Run a scenario and estimate install tier")
    st.markdown('<div class="sec-hdr">Predict App Success</div>', unsafe_allow_html=True)
    st.markdown("<div class='muted'>Enter your app profile to estimate success potential across Low, Medium, and High install tiers.</div>", unsafe_allow_html=True)

    categories      = sorted(df_clean["Category"].unique())
    content_ratings = sorted(df_clean["Content Rating"].dropna().unique())

    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    with st.form("pred_form"):
        st.markdown("### App Details")
        st.markdown("<div class='muted'>Use realistic launch assumptions for more reliable predictions.</div>", unsafe_allow_html=True)
        r1c1, r1c2, r1c3 = st.columns(3)

        with r1c1:
            rating   = st.slider("Rating (1.0 – 5.0)", 1.0, 5.0, 4.0, 0.1)
            reviews  = st.number_input("Number of Reviews", 0, 100_000_000, 10_000, 1000)
            price    = st.number_input("Price ($) — 0 for Free", 0.0, 400.0, 0.0, 0.5)

        with r1c2:
            size_mb     = st.slider("App Size (MB)", 0.5, 100.0, 15.0, 0.5)
            category    = st.selectbox("Category", categories)
            content_r   = st.selectbox("Content Rating", content_ratings)

        with r1c3:
            installs_est = st.number_input(
                "Estimated Installs (for Engagement calc)", 1, 100_000_000, 50_000, 1000
            )
            chosen_mdl = st.radio(
                "ML Model", ["Random Forest", "Logistic Regression"], horizontal=True
            )

        st.markdown("---")
        submit = st.form_submit_button("Predict Now", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if submit:
        # Encode categorical inputs
        le_cat = label_encoders["Category"]
        le_cr  = label_encoders["Content Rating"]

        cat_enc = le_cat.transform([category])[0] if category in le_cat.classes_ else 0
        cr_enc  = le_cr.transform([content_r])[0] if content_r in le_cr.classes_ else 0

        paid_app   = 1 if price > 0 else 0
        engagement = reviews / (installs_est + 1e-6)
        cat_df     = df_clean[df_clean["Category"] == category]
        cat_avg    = cat_df["Installs"].mean()

        # Supervised category features — computed live from dataset
        cat_high_rate  = float((cat_df["Install_Tier"] == "High").mean()) if len(cat_df) > 0 else 0.20
        cat_app_count  = float(len(cat_df))
        max_cat_count  = float(df_clean["Category"].value_counts().max())
        cat_saturation = cat_app_count / max_cat_count if max_cat_count > 0 else 0.20

        input_dict = {
            "Rating":                rating,
            "Reviews":               reviews,
            "Size_MB":               size_mb,
            "Price":                 price,
            "Paid_App":              paid_app,
            "Engagement":            round(engagement, 6),
            "Category_Avg_Installs": cat_avg,
            "Category_enc":          cat_enc,
            "Content Rating_enc":    cr_enc,
            # Supervised category features (NEW)
            "Category_High_Rate":    cat_high_rate,
            "Category_Saturation":   cat_saturation,
            "Category_App_Count":    cat_app_count,
            # Pass content rating name for rule engine
            "_content_rating_name":  content_r,
        }

        tier, probs = predict_success(
            input_dict, models, scaler, feat_names,
            label_encoders["Install_Tier"], chosen_mdl,
        )

        st.markdown("---")
        st.markdown("### Prediction Result")

        css_map   = {"High": "badge-high", "Medium": "badge-medium", "Low": "badge-low"}

        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.markdown(
                f'<div class="{css_map[tier]}">'
                f'Predicted Tier: <strong>{tier}</strong>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # ── Probability breakdown ────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### Success Probability Breakdown")
        st.caption("Estimated probability of each success tier from your full feature set.")

        col_h, col_m, col_l = st.columns(3)

        p_high = probs.get("High", 0.0)
        p_med  = probs.get("Medium", 0.0)
        p_low  = probs.get("Low", 0.0)

        with col_h:
            st.markdown(
                f"<div class='result-card'>"
                f"<div style='font-size:0.86rem;font-weight:600;color:#A9B9D4;'>High Tier</div>"
                f"<div style='height:8px; background:#2A3A55; border-radius:8px; margin:10px 0 12px 0;'>"
                f"<div style='width:{p_high*100:.1f}%;height:100%;background:#10B981;border-radius:8px;'></div>"
                f"</div>"
                f"<div style='font-size:1.8rem;font-weight:700;color:#A7F3D0;'>{p_high*100:.1f}%</div>"
                f"<div style='font-size:0.8rem;color:#8EA0BC;'>&gt; 1 Million installs</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with col_m:
            st.markdown(
                f"<div class='result-card'>"
                f"<div style='font-size:0.86rem;font-weight:600;color:#A9B9D4;'>Medium Tier</div>"
                f"<div style='height:8px; background:#2A3A55; border-radius:8px; margin:10px 0 12px 0;'>"
                f"<div style='width:{p_med*100:.1f}%;height:100%;background:#F59E0B;border-radius:8px;'></div>"
                f"</div>"
                f"<div style='font-size:1.8rem;font-weight:700;color:#FCD58A;'>{p_med*100:.1f}%</div>"
                f"<div style='font-size:0.8rem;color:#8EA0BC;'>10K to 1M installs</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with col_l:
            st.markdown(
                f"<div class='result-card'>"
                f"<div style='font-size:0.86rem;font-weight:600;color:#A9B9D4;'>Low Tier</div>"
                f"<div style='height:8px; background:#2A3A55; border-radius:8px; margin:10px 0 12px 0;'>"
                f"<div style='width:{p_low*100:.1f}%;height:100%;background:#EF4444;border-radius:8px;'></div>"
                f"</div>"
                f"<div style='font-size:1.8rem;font-weight:700;color:#FECACA;'>{p_low*100:.1f}%</div>"
                f"<div style='font-size:0.8rem;color:#8EA0BC;'>&lt; 10K installs</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        # Visual stacked bar
        st.markdown("<br>", unsafe_allow_html=True)
        bar_html = (
            f"<div style='display:flex;height:18px;border-radius:999px;overflow:hidden;"
            f"border:1px solid #2A3A55;background:#111A2B;'>"
            f"<div style='width:{p_high*100:.1f}%;background:#10B981;'></div>"
            f"<div style='width:{p_med*100:.1f}%;background:#F59E0B;'></div>"
            f"<div style='width:{p_low*100:.1f}%;background:#EF4444;'></div>"
            f"</div>"
        )
        st.markdown(bar_html, unsafe_allow_html=True)
        st.caption("Overall tier distribution for this prediction.")

        # What factors drove this
        st.markdown("---")
        st.markdown("### Key Factors Influencing Your Prediction")

        factors = []
        if rating >= 4.5:
            factors.append(("Strong", "Excellent rating (>= 4.5)", "Strong positive signal"))
        elif rating >= 4.0:
            factors.append(("Strong", f"Good rating ({rating})", "Positive signal"))
        elif rating >= 3.5:
            factors.append(("Neutral", f"Average rating ({rating})", "Neutral - try to push above 4.0"))
        elif rating >= 2.0:
            factors.append(("Risk", f"Below-average rating ({rating})", "Pulling prediction down"))
        else:
            factors.append(("Risk", f"Very low rating ({rating})", "Strong negative signal - major concern"))

        if price == 0:
            factors.append(("Strong", "Free app", "Free apps get far more installs"))
        elif price <= 2.0:
            factors.append(("Neutral", f"Low price (${price:.2f})", "Slight negative vs free"))
        else:
            factors.append(("Risk", f"Paid app (${price:.2f})", "Significantly limits installs"))

        if reviews >= 500_000:
            factors.append(("Strong", f"{reviews:,} reviews", "Very high social proof"))
        elif reviews >= 50_000:
            factors.append(("Strong", f"{reviews:,} reviews", "Good social proof"))
        elif reviews >= 5_000:
            factors.append(("Neutral", f"{reviews:,} reviews", "Moderate - more reviews help"))
        else:
            factors.append(("Risk", f"{reviews:,} reviews", "Low reviews reduce credibility"))

        if "Everyone" in content_r:
            factors.append(("Strong", f"Content: {content_r}", "Broadest possible audience"))
        elif "Teen" in content_r:
            factors.append(("Neutral", f"Content: {content_r}", "Decent audience reach"))
        else:
            factors.append(("Risk", f"Content: {content_r}", "Restricted audience"))

        tone_class = {"Strong": "factor-strong", "Neutral": "factor-neutral", "Risk": "factor-risk"}
        for tone, label, note in factors:
            st.markdown(
                f"<div class='result-card' style='margin-bottom:10px;'>"
                f"<span class='factor-chip {tone_class[tone]}'>{tone}</span>"
                f"<strong>{label}</strong><br>"
                f"<span style='color:#A9B9D4; font-size:0.92rem;'>{note}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown("### What This Means")
        if tier == "High":
            st.success(
                "**High Success Potential.** Your app profile matches high-install apps. "
                "Invest in ASO (App Store Optimisation), social media campaigns, and "
                "regular updates to maintain momentum after launch."
            )
        elif tier == "Medium":
            st.warning(
                "**Medium Success Potential.** You have a solid foundation. "
                "Focus on increasing reviews (in-app prompts), improving your rating, "
                "and targeting a category with higher avg installs to break into the High tier."
            )
        else:
            st.error(
                "**Low Success Potential.** Consider making the app **free**, "
                "reducing the size, improving the rating, and choosing a more popular category. "
                "A/B test your store listing — icon, screenshots, and description matter enormously."
            )

        # Input summary
        with st.expander("View Input Summary"):
            summary = pd.DataFrame({
                "Feature": ["Rating", "Reviews", "Price ($)", "Size (MB)",
                            "Category", "Content Rating", "Paid App",
                            "Engagement", "Category Avg Installs",
                            "Category High Rate", "Category Saturation"],
                "Value":   [rating, f"{reviews:,}", f"${price:.2f}", size_mb,
                            category, content_r,
                            "Yes" if paid_app else "No",
                            f"{engagement:.4f}", f"{cat_avg/1e6:.2f}M",
                            f"{cat_high_rate*100:.1f}% of apps reach High",
                            f"{cat_saturation*100:.1f}% crowded ({int(cat_app_count)} apps)"],
            })
            st.dataframe(summary, use_container_width=True)
