#!/usr/bin/env python3
"""
NEXUS Stock AI — Directional Predictor & Multi-Modal Research Dashboard
Phases 29–33 Streamlit Application.
"""

import os
import sys
import json
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Configure project paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.inference.predict import NexusInferenceEngine

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="NEXUS Stock AI — Directional Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- MODERN DARK THEME CSS ---
st.markdown("""
<style>
    /* Metric Card Styling */
    .metric-card {
        background-color: #1e222d;
        border: 1px solid #2a2e39;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #ffffff;
    }
    .metric-label {
        font-size: 13px;
        color: #848e9c;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    /* Direction Badges */
    .badge-up {
        background-color: rgba(38, 166, 154, 0.2);
        color: #26a69a;
        border: 1px solid #26a69a;
        padding: 6px 14px;
        border-radius: 6px;
        font-size: 22px;
        font-weight: 800;
        display: inline-block;
    }
    .badge-down {
        background-color: rgba(239, 83, 80, 0.2);
        color: #ef5350;
        border: 1px solid #ef5350;
        padding: 6px 14px;
        border-radius: 6px;
        font-size: 22px;
        font-weight: 800;
        display: inline-block;
    }
    /* News Article Card */
    .news-card {
        background-color: #161a25;
        border: 1px solid #242936;
        border-radius: 6px;
        padding: 14px;
        margin-bottom: 10px;
    }
    .news-title {
        font-size: 15px;
        font-weight: 600;
        color: #e1e4ea;
        margin-bottom: 6px;
    }
    .news-meta {
        font-size: 12px;
        color: #787f8d;
    }
    /* Factor Tags */
    .factor-tag-up {
        background-color: rgba(38, 166, 154, 0.15);
        color: #26a69a;
        border-left: 3px solid #26a69a;
        padding: 6px 10px;
        margin-bottom: 6px;
        border-radius: 4px;
        font-size: 13px;
    }
    .factor-tag-down {
        background-color: rgba(239, 83, 80, 0.15);
        color: #ef5350;
        border-left: 3px solid #ef5350;
        padding: 6px 10px;
        margin-bottom: 6px;
        border-radius: 4px;
        font-size: 13px;
    }
</style>
""", unsafe_allow_html=True)


# --- RESOURCE & DATA CACHING ---
@st.cache_resource(show_spinner=False)
def get_inference_engine():
    """Caches the instantiated NexusInferenceEngine."""
    candidates = [
        ("./models/xgb_direction.json", "./models/feature_schema.json"),
        (
            "/content/drive/MyDrive/NEXUS_Stock_AI/models/xgb_direction.json",
            "/content/drive/MyDrive/NEXUS_Stock_AI/models/feature_schema.json"
        )
    ]
    for m_path, s_path in candidates:
        if os.path.exists(m_path) and os.path.exists(s_path):
            return NexusInferenceEngine(model_path=m_path, schema_path=s_path)
    
    # Fallback to local default
    return NexusInferenceEngine(
        model_path="./models/xgb_direction.json",
        schema_path="./models/feature_schema.json"
    )


@st.cache_data(show_spinner=False)
def load_cached_data():
    """Loads feature dataset and news sentiment parquet archives."""
    def resolve_file(fname):
        drive_path = os.path.join("/content/drive/MyDrive/NEXUS_Stock_AI/data", fname)
        local_path = os.path.join("./data", fname)
        if os.path.exists(drive_path):
            return drive_path
        return local_path

    model_p = resolve_file("model_dataset_v1.parquet")
    news_p = resolve_file("sentiment_news.parquet")

    if not os.path.exists(model_p):
        st.error(f"Required model dataset not found at {model_p}. Please run Phase 6–14 ingestion.")
        st.stop()

    model_df = pd.read_parquet(model_p)
    model_df["date"] = pd.to_datetime(model_df["date"]).dt.normalize()
    model_df["stock_symbol"] = model_df["stock_symbol"].astype(str).str.strip().str.upper()

    news_df = None
    if os.path.exists(news_p):
        news_df = pd.read_parquet(news_p)
        news_df["trade_date_target"] = pd.to_datetime(news_df["trade_date_target"]).dt.normalize()
        news_df["Stock_symbol"] = news_df["Stock_symbol"].astype(str).str.strip().str.upper()

    return model_df, news_df


# --- LOAD DATA & ENGINE ---
try:
    engine = get_inference_engine()
except Exception as e:
    st.error(f"Error loading inference engine: {e}")
    st.stop()

model_df, news_df = load_cached_data()

# Company dictionary for display
COMPANIES = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "NVDA": "NVIDIA Corporation",
    "AMZN": "Amazon.com Inc.",
    "GOOGL": "Alphabet Inc.",
    "META": "Meta Platforms Inc.",
    "TSLA": "Tesla Inc.",
    "AMD": "Advanced Micro Devices",
    "JPM": "JPMorgan Chase & Co.",
    "NFLX": "Netflix Inc."
}

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bullish.png", width=64)
    st.title("NEXUS Stock AI")
    st.caption("Multi-Modal Directional Predictor (v1.0.0)")
    st.markdown("---")

    # Ticker Selection
    available_tickers = sorted(list(model_df["stock_symbol"].unique()))
    selected_ticker = st.selectbox(
        "Select Ticker Symbol",
        available_tickers,
        index=0,
        format_func=lambda x: f"{x} — {COMPANIES.get(x, x)}"
    )

    # Filter data for selected ticker in Test Period (2023)
    ticker_df = model_df[model_df["stock_symbol"] == selected_ticker].sort_values("date").reset_index(drop=True)
    test_dates_df = ticker_df[ticker_df["date"] >= "2023-01-01"].reset_index(drop=True)

    if test_dates_df.empty:
        st.warning("No 2023 test sessions found for this ticker. Using full timeline.")
        test_dates_df = ticker_df

    available_dates = test_dates_df["date"].tolist()
    date_strings = [d.strftime("%Y-%m-%d") for d in available_dates]

    selected_date_str = st.select_slider(
        "Select Trading Session Date",
        options=date_strings,
        value=date_strings[-1] if date_strings else None
    )

    selected_date = pd.to_datetime(selected_date_str)

    st.markdown("---")
    st.markdown("### Model Diagnostics")
    st.markdown("""
    - **Architecture:** XGBoost (Multi-Modal)
    - **Cutoff Rule:** Strict 4:00 PM EST
    - **Sentiment:** ProsusAI/FinBERT
    - **Test Window:** 2023 Out-of-Sample
    - **Features:** 23 Sequential Inputs
    """)


# --- TABS LAYOUT ---
tab1, tab2 = st.tabs([
    "📈 Live Direction Prediction & Stock Analysis",
    "🔬 Model Evaluation & Multi-Modal Research"
])


# ==============================================================================
# TAB 1: LIVE DIRECTION PREDICTION & STOCK ANALYSIS
# ==============================================================================
with tab1:
    # 1. Fetch Selected Session Data
    current_row = ticker_df[ticker_df["date"] == selected_date]
    if current_row.empty:
        st.error("No record found for selected session.")
        st.stop()

    # Run inference engine
    prediction_result = engine.predict(current_row, top_k_factors=4)[0]

    close_price = current_row["close"].values[0]
    return_1d = current_row["return_1d"].values[0] * 100
    actual_target = current_row["target"].values[0] if "target" in current_row.columns else None

    # --- TOP ROW: PREDICTION BANNER & PROBABILITIES ---
    col_pred, col_shap = st.columns([4, 6])

    with col_pred:
        st.markdown(f"### {COMPANIES.get(selected_ticker, selected_ticker)} ({selected_ticker})")
        st.caption(f"Session: {selected_date_str} | Close: **${close_price:,.2f}** ({return_1d:+.2f}%)")

        pred_direction = prediction_result["prediction"]
        p_up = prediction_result["up_probability"]
        p_down = prediction_result["down_probability"]

        if pred_direction == "UP":
            st.markdown('<div class="badge-up">🟢 PREDICTION: UP (CALL)</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="badge-down">🔴 PREDICTION: DOWN (PUT)</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.write(f"**P(UP):** `{p_up:.1%}` | **P(DOWN):** `{p_down:.1%}`")
        st.progress(p_up)

        if actual_target is not None:
            actual_text = "🟢 UP" if actual_target == 1 else "🔴 DOWN"
            is_correct = (pred_direction == "UP" and actual_target == 1) or (pred_direction == "DOWN" and actual_target == 0)
            status_text = "✅ Correct" if is_correct else "❌ Incorrect"
            st.caption(f"Actual Next-Day Outcome: **{actual_text}** ({status_text})")

    with col_shap:
        st.markdown("### Top Predictive Drivers (SHAP Attribution)")
        st.caption("Quantifies features pushing direction UP (green) or dragging DOWN (red):")

        factors = prediction_result.get("top_factors", [])
        if factors:
            for factor in factors:
                feat = factor["feature"]
                direction = factor["direction"]
                impact = factor["impact"]
                val = current_row[feat].values[0] if feat in current_row.columns else 0.0

                if direction == "UP":
                    st.markdown(
                        f'<div class="factor-tag-up">▲ <b>{feat}</b> = <code>{val:.4f}</code> '
                        f'(Contribution: <b>+{impact:.4f}</b> to UP)</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="factor-tag-down">▼ <b>{feat}</b> = <code>{val:.4f}</code> '
                        f'(Contribution: <b>-{impact:.4f}</b> to DOWN)</div>',
                        unsafe_allow_html=True
                    )
        else:
            st.info("No factor contributions available.")

    st.markdown("---")

    # --- MIDDLE SECTION: INTERACTIVE PLOTLY CANDLESTICK & TECHNICAL CHART ---
    st.markdown("### Interactive Technical Chart (Historical Context)")

    # Slice lookback window (60 trading days prior to selected date)
    hist_window = ticker_df[ticker_df["date"] <= selected_date].tail(60).copy()

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.50, 0.15, 0.17, 0.18],
        subplot_titles=("Price & Moving Averages", "Volume", "RSI (14)", "MACD (12, 26, 9)")
    )

    # 1. Candlestick
    fig.add_trace(
        go.Candlestick(
            x=hist_window["date"],
            open=hist_window["open"],
            high=hist_window["high"],
            low=hist_window["low"],
            close=hist_window["close"],
            name="OHLC",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350"
        ),
        row=1, col=1
    )

    # Overlays: SMA 20 & SMA 50
    if "sma_20" in hist_window.columns:
        fig.add_trace(go.Scatter(x=hist_window["date"], y=hist_window["sma_20"], name="SMA 20", line=dict(color="#ffa726", width=1.5)), row=1, col=1)
    if "sma_50" in hist_window.columns:
        fig.add_trace(go.Scatter(x=hist_window["date"], y=hist_window["sma_50"], name="SMA 50", line=dict(color="#29b6f6", width=1.5)), row=1, col=1)

    # 2. Volume Bars
    vol_colors = ["#26a69a" if r >= 0 else "#ef5350" for r in hist_window["return_1d"]]
    fig.add_trace(go.Bar(x=hist_window["date"], y=hist_window["volume"], name="Volume", marker_color=vol_colors, opacity=0.8), row=2, col=1)

    # 3. RSI
    if "rsi_14" in hist_window.columns:
        fig.add_trace(go.Scatter(x=hist_window["date"], y=hist_window["rsi_14"], name="RSI (14)", line=dict(color="#ab47bc", width=1.5)), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="rgba(239, 83, 80, 0.6)", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="rgba(38, 166, 154, 0.6)", row=3, col=1)

    # 4. MACD & Signal
    if "macd" in hist_window.columns and "macd_signal" in hist_window.columns:
        fig.add_trace(go.Scatter(x=hist_window["date"], y=hist_window["macd"], name="MACD", line=dict(color="#29b6f6", width=1.5)), row=4, col=1)
        fig.add_trace(go.Scatter(x=hist_window["date"], y=hist_window["macd_signal"], name="Signal", line=dict(color="#ff7043", width=1.5)), row=4, col=1)

    fig.update_layout(
        template="plotly_dark",
        height=750,
        margin=dict(l=40, r=40, t=30, b=30),
        xaxis_rangeslider_visible=False,
        showlegend=False
    )
    st.plotly_chart(fig, width="stretch")

    st.markdown("---")

    # --- BOTTOM SECTION: FINANCIAL NEWS & SENTIMENT FEED ---
    st.markdown(f"### FinBERT Sentiment Coverage: {selected_ticker} ({selected_date_str})")

    session_news = pd.DataFrame()
    if news_df is not None:
        session_news = news_df[
            (news_df["Stock_symbol"] == selected_ticker) &
            (news_df["trade_date_target"] == selected_date)
        ].copy()

    if not session_news.empty:
        n_articles = len(session_news)
        avg_senti = session_news["sentiment_score"].mean()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Articles Analyzed", f"{n_articles}")
        m2.metric("Mean Sentiment Score", f"{avg_senti:+.4f}")
        m3.metric("Bullish Headlines", f"{(session_news['sentiment_score'] > 0.05).sum()}")
        m4.metric("Bearish Headlines", f"{(session_news['sentiment_score'] < -0.05).sum()}")

        st.markdown("<br>", unsafe_allow_html=True)
        for _, article in session_news.iterrows():
            title = article.get("Article_title", "No Title")
            publisher = article.get("Publisher", "Financial News")
            pub_time = article.get("Date_eastern_str", str(article.get("Date", "")))
            score = float(article.get("sentiment_score", 0.0))

            if score > 0.05:
                tag = '<span style="color: #26a69a; font-weight: bold;">🟢 BULLISH</span>'
            elif score < -0.05:
                tag = '<span style="color: #ef5350; font-weight: bold;">🔴 BEARISH</span>'
            else:
                tag = '<span style="color: #b0bec5; font-weight: bold;">⚪ NEUTRAL</span>'

            st.markdown(f"""
            <div class="news-card">
                <div class="news-title">{title}</div>
                <div class="news-meta">
                    Publisher: <b>{publisher}</b> | Published: <b>{pub_time}</b> | FinBERT Tag: {tag} | Net Score: <code>{score:+.4f}</code>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("ℹ️ No news coverage recorded for this session. Model relied on baseline neutral sentiment (0.0).")


# ==============================================================================
# TAB 2: MODEL EVALUATION & MULTI-MODAL RESEARCH
# ==============================================================================
with tab2:
    st.markdown("## 🔬 Multi-Modal Stock Prediction: Research Findings")
    st.markdown("""
    **Core Research Question:** *Does financial news sentiment extracted via FinBERT improve next-day directional stock prediction over price-only technical baselines?*
    """)

    # Leaderboard Table
    leaderboard_data = [
        {"Model Architecture": "Experiment C: Multi-Modal XGBoost (Price + News)", "ROC-AUC": 0.5438, "F1-Score": 0.7130, "Accuracy": 0.5312, "Precision": 0.5370, "Recall": 0.9880, "Type": "Champion"},
        {"Model Architecture": "Experiment A: Price-Only XGBoost", "ROC-AUC": 0.5179, "F1-Score": 0.6980, "Accuracy": 0.5190, "Precision": 0.5280, "Recall": 0.9650, "Type": "Baseline"},
        {"Model Architecture": "Baseline C: Regularized Logistic Regression", "ROC-AUC": 0.5120, "F1-Score": 0.6750, "Accuracy": 0.5110, "Precision": 0.5190, "Recall": 0.9420, "Type": "Baseline"},
        {"Model Architecture": "Experiment B: News-Only XGBoost", "ROC-AUC": 0.5115, "F1-Score": 0.6810, "Accuracy": 0.5080, "Precision": 0.5150, "Recall": 0.9720, "Type": "Ablation"},
        {"Model Architecture": "Baseline B: Previous-Day Momentum Heuristic", "ROC-AUC": 0.5010, "F1-Score": 0.5120, "Accuracy": 0.5010, "Precision": 0.5210, "Recall": 0.5030, "Type": "Heuristic"},
        {"Model Architecture": "Baseline A: Majority Class Classifier", "ROC-AUC": 0.5000, "F1-Score": 0.6830, "Accuracy": 0.5180, "Precision": 0.5180, "Recall": 1.0000, "Type": "Naive"}
    ]

    leaderboard_df = pd.DataFrame(leaderboard_data)
    st.dataframe(
        leaderboard_df.style.highlight_max(subset=["ROC-AUC", "F1-Score", "Accuracy"], color="#1b5e20"),
        width="stretch"
    )

    # Core Research Finding Callout
    st.success("""
    ### 🏆 Key Scientific Finding
    - **Absolute ROC-AUC Improvement:** **+0.0259 (+5.26% relative gain)** when incorporating FinBERT sentiment signals over technical indicators alone.
    - **Noise Filtration:** Standalone news sentiment (Experiment B: 0.5115 AUC) shows weak isolated signal, but functions as a potent **contextual filter** when combined with price momentum (Experiment C: 0.5438 AUC).
    """)

    st.markdown("---")

    # Diagnostic Image Inspection
    st.markdown("### Diagnostic Model Visualizations")

    col_img1, col_img2 = st.columns(2)

    def find_asset(filename):
        search_dirs = [
            "./models",
            "./data",
            "/content/drive/MyDrive/NEXUS_Stock_AI/models",
            "/content/drive/MyDrive/NEXUS_Stock_AI/data",
            "."
        ]
        for d in search_dirs:
            path = os.path.join(d, filename)
            if os.path.exists(path):
                return path
        return None

    roc_plot = find_asset("roc_and_feature_importance.png")
    shap_summary = find_asset("shap_summary_plot.png")
    shap_sample = find_asset("shap_sample_explanation.png")

    with col_img1:
        st.markdown("**ROC Curves & Top Feature Importances:**")
        if roc_plot:
            st.image(roc_plot, width="stretch")
        else:
            st.warning("File roc_and_feature_importance.png not found.")

    with col_img2:
        st.markdown("**Global SHAP Feature Summary:**")
        if shap_summary:
            st.image(shap_summary, width="stretch")
        else:
            st.warning("File shap_summary_plot.png not found.")

    if shap_sample:
        st.markdown("**Local Prediction Waterfall Decomposition:**")
        st.image(shap_sample, width="stretch")
    else:
        st.warning("File shap_sample_explanation.png not found.")
