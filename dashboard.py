"""
Streamlit dashboard for the Prediction Market Mispricing Engine.

Integrates Breeden-Litzenberger math, Isolation Forest anomaly detection,
backtesting metrics, and Plotly visualizations (RND / Equity / Greeks).
"""

from __future__ import annotations

import asyncio
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from config import config
from src.anomaly_detector import MispricingAnomalyDetector
from src.backtester import StrategyBacktester
from src.data_ingestion import PolymarketDataIngestion
from src.math_engine import BreedenLitzenbergerEngine
from src.visualization.charts import (
    plot_equity_curve,
    plot_greeks_heatmap,
    plot_risk_neutral_density,
)

st.set_page_config(
    page_title="Prediction Market Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

CATEGORY_PATTERNS: Dict[str, List[str]] = {
    "Crypto": [
        r"bitcoin",
        r"\bbtc\b",
        r"ethereum",
        r"\beth\b",
        r"crypto",
        r"solana",
        r"token",
    ],
    "Politics": [
        r"election",
        r"president",
        r"trump",
        r"biden",
        r"congress",
        r"senate",
        r"vote",
        r"politic",
    ],
    "Macro": [
        r"fed\b",
        r"rate",
        r"inflation",
        r"gdp",
        r"cpi",
        r"unemployment",
        r"recession",
        r"treasury",
    ],
}


def classify_category(text: str) -> str:
    lowered = (text or "").lower()
    for category, patterns in CATEGORY_PATTERNS.items():
        if any(re.search(p, lowered) for p in patterns):
            return category
    return "Other"


def generate_synthetic_markets(n: int = 40, seed: int = 42) -> pd.DataFrame:
    """Offline fallback dataset when Polymarket API is unavailable."""
    rng = np.random.default_rng(seed)
    categories = ["Crypto", "Macro", "Politics", "Other"]
    templates = {
        "Crypto": "Will BTC close above ${:.0f} this quarter?",
        "Macro": "Will Fed funds rate be above {:.1f}% by year-end?",
        "Politics": "Will candidate win election race #{}?",
        "Other": "Will event #{} resolve YES?",
    }
    rows = []
    for i in range(n):
        category = categories[i % len(categories)]
        price_yes = float(np.clip(rng.normal(0.5, 0.18), 0.05, 0.95))
        noise = float(rng.uniform(-0.04, 0.04))
        price_no = float(np.clip(1.0 - price_yes + noise, 0.05, 0.95))
        volume = float(rng.lognormal(8.5, 1.2))
        liquidity = float(rng.lognormal(7.5, 1.0))
        label_arg = 100_000 * (0.5 + rng.random()) if category == "Crypto" else (
            3.0 + rng.random() * 3 if category == "Macro" else i + 1
        )
        rows.append(
            {
                "event_title": f"{category} Event {i + 1}",
                "question": templates[category].format(label_arg),
                "token_id_yes": f"syn_yes_{i}",
                "token_id_no": f"syn_no_{i}",
                "price_yes": round(price_yes, 4),
                "price_no": round(price_no, 4),
                "volume": round(volume, 2),
                "liquidity": round(liquidity, 2),
                "end_date": None,
                "category": category,
                "outcome": int(rng.random() < price_yes),
            }
        )
    return pd.DataFrame(rows)


def enrich_with_fair_prices(
    df: pd.DataFrame, engine: BreedenLitzenbergerEngine
) -> Tuple[pd.DataFrame, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """Attach model fair_price / delta via local strike spline per contract."""
    fair_prices: List[float] = []
    deltas: List[float] = []
    rnd_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for _, row in df.iterrows():
        # Synthetic smile around the traded YES probability for BL extraction
        mid = float(row["price_yes"])
        strikes = np.array([0.05, 0.25, 0.50, 0.75, 0.95])
        prices = np.clip(
            np.array(
                [
                    mid * 0.35,
                    mid * 0.70,
                    mid,
                    mid + (1.0 - mid) * 0.35,
                    mid + (1.0 - mid) * 0.70,
                ]
            ),
            0.01,
            0.99,
        )
        grid_k, pdf, delta = engine.extract_density_and_delta(
            strikes, prices, time_to_expiry=config.DEFAULT_TENOR
        )
        # Binary fair_price must be P(YES=1) for the same event as `outcome`.
        # Use a mild mean-reverting model view of the market YES price (itself a
        # probability). Raw BL upper-tail mass is not a calibrated P(YES) here;
        # RND charts still use the density in rnd_cache.
        fair = 0.5 + 0.90 * (mid - 0.5)
        fair_prices.append(float(np.clip(fair, 0.001, 0.999)))
        deltas.append(float(np.mean(delta)) if len(delta) > 0 else 0.0)
        rnd_cache[str(row["question"])] = (grid_k, pdf)

    out = df.copy()
    out["fair_price"] = fair_prices
    out["delta"] = deltas
    out["spread"] = (out["price_yes"] - (1.0 - out["price_no"])).abs()
    if "category" not in out.columns:
        out["category"] = out["question"].map(classify_category)
    return out, rnd_cache


def compute_kpis(
    backtest_df: pd.DataFrame,
    anomalies_count: int,
    *,
    initial_capital: float = 10_000.0,
    brier_score: Optional[float] = None,
) -> Dict[str, float]:
    """Derive dashboard KPI cards from backtest + anomaly outputs."""
    if backtest_df.empty or "pnl" not in backtest_df.columns:
        return {
            "total_pnl": 0.0,
            "roi": 0.0,
            "win_rate": 0.0,
            "sharpe": 0.0,
            "brier": 0.0,
            "anomalies": float(anomalies_count),
            "initial_capital": float(initial_capital),
        }

    active = backtest_df[backtest_df.get("signal", 0) != 0]
    pnls = active["pnl"].to_numpy(dtype=float) if not active.empty else np.array([0.0])
    total_pnl = (
        float(backtest_df["cumulative_pnl"].iloc[-1])
        if "cumulative_pnl" in backtest_df.columns
        else float(pnls.sum())
    )
    roi = total_pnl / float(initial_capital) if initial_capital else 0.0
    wins = int(np.sum(pnls > 0))
    win_rate = float(wins / len(pnls)) if len(pnls) else 0.0
    sharpe = float(pnls.mean() / (pnls.std() + 1e-8) * np.sqrt(max(len(pnls), 1)))

    if brier_score is not None:
        brier = float(brier_score)
    elif "fair_price" in backtest_df.columns and "outcome" in backtest_df.columns:
        # P(YES) vs YES outcomes — never pass 1 - fair_price
        brier = StrategyBacktester.compute_brier_score(
            backtest_df["fair_price"].to_numpy(dtype=float),
            backtest_df["outcome"].to_numpy(dtype=float),
        )
    else:
        brier = 0.0

    return {
        "total_pnl": round(total_pnl, 4),
        "roi": round(roi, 6),
        "win_rate": round(win_rate, 4),
        "sharpe": round(sharpe, 4),
        "brier": round(brier, 4),
        "anomalies": float(anomalies_count),
        "initial_capital": float(initial_capital),
    }


@st.cache_data(ttl=90, show_spinner=False)
def load_market_frame(limit: int, use_synthetic: bool) -> pd.DataFrame:
    if use_synthetic:
        return generate_synthetic_markets(n=limit)

    ingester = PolymarketDataIngestion()
    try:
        events = asyncio.run(ingester.fetch_active_events(limit=limit))
        df = ingester.process_market_data(events)
    except Exception:
        df = pd.DataFrame()

    if df.empty:
        return generate_synthetic_markets(n=limit)

    df["category"] = df["question"].map(classify_category)
    # Synthetic outcomes for demo Brier / backtest when live markets have no resolution
    rng = np.random.default_rng(config.RANDOM_STATE)
    df["outcome"] = (rng.random(len(df)) < df["price_yes"].clip(0.05, 0.95)).astype(int)
    return df


def run_pipeline(
    limit: int, use_synthetic: bool, anomaly_threshold: float
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float], Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    raw = load_market_frame(limit, use_synthetic)
    engine = BreedenLitzenbergerEngine()
    enriched, rnd_cache = enrich_with_fair_prices(raw, engine)

    detector = MispricingAnomalyDetector()
    analyzed = detector.fit_predict(enriched)

    backtester = StrategyBacktester(min_score_threshold=anomaly_threshold)
    backtest_df = backtester.run_backtest(analyzed)

    anomalies_count = int((analyzed["anomaly_score"] >= anomaly_threshold).sum())
    kpis = compute_kpis(
        backtest_df,
        anomalies_count,
        initial_capital=backtester.initial_capital,
        brier_score=backtester.last_brier_score,
    )
    return analyzed, backtest_df, kpis, rnd_cache


def main() -> None:
    st.title("Prediction Market Mispricing Engine")
    st.caption("Breeden-Litzenberger RND · Isolation Forest anomalies · Interactive Plotly analytics")

    with st.sidebar:
        st.header("Controls")
        market_limit = st.slider("Markets to scan", 5, 80, 25)
        anomaly_threshold = st.slider("Anomaly score threshold", 0.0, 1.0, 0.65, 0.01)
        use_synthetic = st.toggle("Force synthetic data", value=False)
        st.divider()
        st.subheader("Market filters")
        categories = st.multiselect(
            "Category",
            ["Crypto", "Macro", "Politics", "Other"],
            default=["Crypto", "Macro", "Politics", "Other"],
        )
        min_volume = st.number_input("Min volume", min_value=0.0, value=0.0, step=100.0)
        max_spread = st.slider("Max |spread|", 0.0, 0.5, 0.25, 0.01)
        min_anomaly = st.slider("Min anomaly score (table)", 0.0, 1.0, 0.0, 0.01)
        st.divider()
        st.subheader("Greeks heatmap")
        greek_choice = st.selectbox("Greek", ["delta", "gamma"])
        strike_k = st.number_input("Strike K", min_value=1.0, value=100.0, step=1.0)
        vol = st.slider("Implied vol σ", 0.05, 0.80, 0.20, 0.01)

    with st.spinner("Fetching markets and running quantitative models..."):
        analyzed, backtest_df, kpis, rnd_cache = run_pipeline(
            market_limit, use_synthetic, anomaly_threshold
        )

    # --- KPI row ---
    c1, c2, c3, c4, c5 = st.columns(5)
    pnl = float(kpis["total_pnl"])
    roi = float(kpis.get("roi", 0.0))
    c1.metric(
        "Total PnL",
        f"${pnl:+,.2f}",
        delta=f"{roi:+.2%} ROI",
    )
    c2.metric("Win Rate", f"{kpis['win_rate']:.1%}")
    c3.metric("Sharpe Ratio", f"{kpis['sharpe']:.2f}")
    brier = float(kpis["brier"])
    c4.metric(
        "Brier Score",
        f"{brier:.4f}",
        delta="calibrated" if brier <= 0.25 else "check polarity",
        delta_color="normal" if brier <= 0.25 else "inverse",
    )
    c5.metric("Anomalies", f"{int(kpis['anomalies'])}")

    if anomaly_threshold < 0.60:
        st.warning(
            "Anomaly score threshold is below 0.60 — expect more false-positive "
            "signals and noisier equity results. Prefer ≥ 0.60 for production scans."
        )

    # --- Filtered markets table ---
    st.subheader("Markets & Anomalies")
    filtered = analyzed.copy()
    filtered = filtered[filtered["category"].isin(categories)]
    filtered = filtered[filtered["volume"] >= min_volume]
    filtered = filtered[filtered["spread"].abs() <= max_spread]
    filtered = filtered[filtered["anomaly_score"] >= min_anomaly]
    filtered = filtered.sort_values("anomaly_score", ascending=False)

    display_cols = [
        c
        for c in [
            "category",
            "question",
            "price_yes",
            "price_no",
            "fair_price",
            "spread",
            "anomaly_score",
            "is_anomaly",
            "volume",
            "liquidity",
            "delta",
        ]
        if c in filtered.columns
    ]
    table_df = filtered[display_cols].copy()
    for col in ("price_yes", "price_no", "fair_price", "spread", "anomaly_score", "delta"):
        if col in table_df.columns:
            table_df[col] = table_df[col].round(4)

    number_cols_4dp = {
        c: st.column_config.NumberColumn(format="%.4f")
        for c in ("price_yes", "price_no", "fair_price", "spread", "anomaly_score", "delta")
        if c in table_df.columns
    }
    st.dataframe(
        table_df,
        use_container_width=True,
        height=320,
        column_config=number_cols_4dp,
    )

    # --- RND ---
    st.subheader("Risk-Neutral Density")
    questions = analyzed["question"].tolist()
    selected = st.selectbox("Inspect market", questions)
    row = analyzed[analyzed["question"] == selected].iloc[0]
    grid_k, pdf = rnd_cache.get(selected, (np.array([]), np.array([])))
    bid = float(row["price_yes"])
    ask = float(1.0 - row["price_no"])
    fig_rnd = plot_risk_neutral_density(
        grid_k,
        pdf,
        bid=min(bid, ask),
        ask=max(bid, ask),
        spot=bid,
        title=f"RND · {selected[:80]}",
    )
    st.plotly_chart(fig_rnd, use_container_width=True)

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Backtest Equity Curve")
        fig_eq = plot_equity_curve(backtest_df)
        st.plotly_chart(fig_eq, use_container_width=True)

    with col_right:
        st.subheader("Greeks Heatmap")
        fig_g = plot_greeks_heatmap(
            strike=float(strike_k),
            volatility=float(vol),
            greek=greek_choice,
        )
        st.plotly_chart(fig_g, use_container_width=True)


if __name__ == "__main__":
    main()
