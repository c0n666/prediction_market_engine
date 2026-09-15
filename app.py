"""
Streamlit Dashboard for Real-Time Prediction Market Mispricing Engine.
"""

import asyncio
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from src.data_ingestion import PolymarketDataIngestion
from src.math_engine import BreedenLitzenbergerEngine
from src.anomaly_detector import MispricingAnomalyDetector

st.set_page_config(page_title="Prediction Market Mispricing Engine", layout="wide")

st.title("📈 Prediction Market Mispricing Engine")
st.caption("Real-Time Quantitative Analytics & Anomaly Detection Dashboard")

# Sidebar Controls
st.sidebar.header("Control Panel")
market_limit = st.sidebar.slider("Number of Markets to Scan", 5, 50, 20)
score_threshold = st.sidebar.slider("Anomaly Threshold", 0.0, 1.0, 0.5)

@st.cache_data(ttl=60)
def load_and_process_data(limit):
    ingester = PolymarketDataIngestion()
    events = asyncio.run(ingester.fetch_active_events(limit=limit))
    df = ingester.process_market_data(events)
    
    if df.empty:
        return pd.DataFrame()

    math_engine = BreedenLitzenbergerEngine()
    fair_prices, deltas = [], []

    for _, row in df.iterrows():
        strikes = np.array([0.0, 0.5, 1.0])
        prices = np.array([0.0, row["price_yes"], 1.0])
        grid_k, pdf, delta = math_engine.extract_density_and_delta(strikes, prices)
        
        fair_prob = math_engine.compute_fair_probability(row["price_yes"], pdf, grid_k) if len(pdf) > 0 else row["price_yes"]
        avg_delta = float(np.mean(delta)) if len(delta) > 0 else 0.5
        
        fair_prices.append(fair_prob)
        deltas.append(avg_delta)

    df["fair_price"] = fair_prices
    df["delta"] = deltas

    detector = MispricingAnomalyDetector()
    return detector.fit_predict(df)

with st.spinner("Fetching market data and running quantitative models..."):
    df_results = load_and_process_data(market_limit)

if df_results.empty:
    st.warning("No market data available.")
else:
    # Metrics Overview
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Scanned Markets", len(df_results))
    anomalies_count = len(df_results[df_results["anomaly_score"] >= score_threshold])
    col2.metric("Detected Anomalies", anomalies_count)
    col3.metric("Avg Liquidity (USD)", f"${df_results['liquidity'].mean():,.2f}")

    # Anomaly Table
    st.subheader("Detected Mispricings")
    filtered_df = df_results[df_results["anomaly_score"] >= score_threshold].sort_values(
        by="anomaly_score", ascending=False
    )
    
    st.dataframe(
        filtered_df[[
            "question", "price_yes", "fair_price", "anomaly_score", "delta", "volume", "liquidity"
        ]],
        use_container_width=True
    )

    # Risk-Neutral Density Plot (Visualization)
    st.subheader("Risk-Neutral Probability Density (Breeden-Litzenberger)")
    selected_market = st.selectbox("Select Market to Inspect", df_results["question"].unique())
    
    market_data = df_results[df_results["question"] == selected_market].iloc[0]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Market Price (YES)", "Fair Price (Model)"],
        y=[market_data["price_yes"], market_data["fair_price"]],
        marker_color=["#ef553b", "#00cc96"]
    ))
    fig.update_layout(title=f"Price Divergence: {selected_market}", yaxis_range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)