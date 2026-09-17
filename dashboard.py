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
) -> Dict[str, Optional[float]]:
    """Derive dashboard KPI cards from backtest + anomaly outputs."""
    empty = {
        "total_pnl": 0.0,
        "roi": 0.0,
        "win_rate": None,
        "sharpe": None,
        "brier": None,
        "anomalies": float(anomalies_count),
        "initial_capital": float(initial_capital),
        "n_markets": 0,
        "n_trades": 0,
    }
    if backtest_df is None or backtest_df.empty or "pnl" not in backtest_df.columns:
        return empty

    signal_col = backtest_df["signal"] if "signal" in backtest_df.columns else pd.Series(0, index=backtest_df.index)
    active = backtest_df[signal_col != 0]
    n_trades = int(len(active))

    if n_trades == 0:
        total_pnl = 0.0
        win_rate: Optional[float] = 0.0
        sharpe: Optional[float] = 0.0
    else:
        pnls = active["pnl"].to_numpy(dtype=float)
        total_pnl = (
            float(backtest_df["cumulative_pnl"].iloc[-1])
            if "cumulative_pnl" in backtest_df.columns
            else float(pnls.sum())
        )
        wins = int(np.sum(pnls > 0))
        win_rate = float(wins / n_trades)
        pnl_std = float(np.std(pnls, ddof=0))
        sharpe = float(pnls.mean() / (pnl_std + 1e-8) * np.sqrt(n_trades))

    roi = total_pnl / float(initial_capital) if initial_capital else 0.0

    if brier_score is not None:
        brier: Optional[float] = float(brier_score)
    elif "fair_price" in backtest_df.columns and "outcome" in backtest_df.columns:
        brier = StrategyBacktester.compute_brier_score(
            backtest_df["fair_price"].to_numpy(dtype=float),
            backtest_df["outcome"].to_numpy(dtype=float),
        )
    else:
        brier = None

    return {
        "total_pnl": round(total_pnl, 4),
        "roi": round(roi, 6),
        "win_rate": None if win_rate is None else round(win_rate, 4),
        "sharpe": None if sharpe is None else round(float(sharpe), 4),
        "brier": None if brier is None else round(float(brier), 4),
        "anomalies": float(anomalies_count),
        "initial_capital": float(initial_capital),
        "n_markets": int(len(backtest_df)),
        "n_trades": n_trades,
    }


def apply_market_filters(
    df: pd.DataFrame,
    *,
    categories: List[str],
    min_volume: float,
    max_spread: float,
    min_anomaly_score: float,
) -> pd.DataFrame:
    """Apply sidebar market filters; safe on missing columns / empty frames."""
    if df is None or df.empty:
        return pd.DataFrame()

    filtered = df.copy()
    if "category" in filtered.columns and categories:
        filtered = filtered[filtered["category"].isin(categories)]
    if "volume" in filtered.columns:
        filtered = filtered[filtered["volume"] >= float(min_volume)]
    if "spread" in filtered.columns:
        filtered = filtered[filtered["spread"].abs() <= float(max_spread)]
    if "anomaly_score" in filtered.columns:
        filtered = filtered[filtered["anomaly_score"] >= float(min_anomaly_score)]

    if filtered.empty:
        return filtered.reset_index(drop=True)

    if "anomaly_score" in filtered.columns:
        filtered = filtered.sort_values("anomaly_score", ascending=False)
    return filtered.reset_index(drop=True)


@st.cache_data(ttl=90, show_spinner=False)
def load_market_frame(
    limit: int, use_synthetic: bool, categories_key: Tuple[str, ...]
) -> pd.DataFrame:
    categories = list(categories_key)
    if use_synthetic:
        df = generate_synthetic_markets(n=limit)
        if categories:
            df = df[df["category"].isin(categories)]
        return df.reset_index(drop=True)

    ingester = PolymarketDataIngestion()
    try:
        # Fetch per selected category via API tag_slug (union), not post-filter of top-N
        async def _fetch_union():
            if not categories or set(categories) >= {"Crypto", "Macro", "Politics", "Other"}:
                return await ingester.fetch_active_events(limit=limit, category=None)
            tasks = [
                ingester.fetch_active_events(limit=limit, category=cat)
                for cat in categories
                if cat != "Other"
            ]
            if not tasks:
                return await ingester.fetch_active_events(limit=limit, category=None)
            batches = await asyncio.gather(*tasks)
            merged = {}
            for batch in batches:
                for ev in batch:
                    key = str(ev.get("id") or ev.get("slug") or ev.get("title"))
                    merged[key] = ev
            return list(merged.values())

        events = asyncio.run(_fetch_union())
        df = ingester.process_market_data(events)
    except Exception:
        df = pd.DataFrame()

    if df.empty:
        df = generate_synthetic_markets(n=limit)
        if categories:
            df = df[df["category"].isin(categories)].reset_index(drop=True)
        return df

    if "category" not in df.columns:
        df["category"] = df["question"].map(classify_category)
    if categories:
        df = df[df["category"].isin(categories)].reset_index(drop=True)

    rng = np.random.default_rng(config.RANDOM_STATE)
    if not df.empty:
        df["outcome"] = (rng.random(len(df)) < df["price_yes"].clip(0.05, 0.95)).astype(int)
    return df


def run_pipeline(
    limit: int, use_synthetic: bool, categories: List[str]
) -> Tuple[pd.DataFrame, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """Load markets, attach fair prices, run anomaly detection (pre-filter)."""
    raw = load_market_frame(limit, use_synthetic, tuple(sorted(categories)))
    engine = BreedenLitzenbergerEngine()
    enriched, rnd_cache = enrich_with_fair_prices(raw, engine)
    detector = MispricingAnomalyDetector()
    analyzed = detector.fit_predict(enriched)
    return analyzed, rnd_cache


def metrics_from_filtered(
    filtered: pd.DataFrame,
    *,
    anomaly_threshold: float,
    initial_capital: float = 10_000.0,
    sizing_mode: str = "fixed",
    stake_amount: float = 200.0,
    scale_with_anomaly: bool = True,
    slippage_pct: float = 0.5,
    trading_fee_pct: float = 0.2,
    cross_the_spread: bool = True,
    auto_frictions: bool = False,
) -> Tuple[Dict[str, Optional[float]], pd.DataFrame]:
    """Backtest + KPIs on the already-filtered market frame."""
    if filtered.empty:
        kpis = compute_kpis(
            pd.DataFrame(),
            anomalies_count=0,
            initial_capital=initial_capital,
        )
        return kpis, pd.DataFrame()

    mode = "percent" if sizing_mode == "percent" else "fixed"
    backtester = StrategyBacktester(
        min_score_threshold=anomaly_threshold,
        initial_capital=initial_capital,
        sizing_mode=mode,
        stake_amount=float(stake_amount),
        scale_with_anomaly=bool(scale_with_anomaly),
        slippage_pct=float(slippage_pct),
        trading_fee_pct=float(trading_fee_pct),
        cross_the_spread=bool(cross_the_spread),
        auto_frictions=bool(auto_frictions),
    )
    backtest_df = backtester.run_backtest(filtered)
    anomalies_count = (
        int((filtered["anomaly_score"] >= anomaly_threshold).sum())
        if "anomaly_score" in filtered.columns
        else 0
    )
    kpis = compute_kpis(
        backtest_df,
        anomalies_count,
        initial_capital=backtester.initial_capital,
        brier_score=backtester.last_brier_score,
    )
    kpis["max_drawdown_pct"] = float(backtester.max_drawdown_pct)
    if not backtest_df.empty and "slippage_pct" in backtest_df.columns:
        active = backtest_df[backtest_df.get("signal", 0) != 0]
        kpis["avg_slippage_pct"] = (
            float(active["slippage_pct"].mean()) if not active.empty else 0.0
        )
    else:
        kpis["avg_slippage_pct"] = float(backtester.slippage_pct)
    return kpis, backtest_df


def _fmt_pct(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{value:.1%}"


def _fmt_num(value: Optional[float], digits: int = 2) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def main() -> None:
    st.title("Prediction Market Mispricing Engine")
    st.caption("Breeden-Litzenberger RND · Isolation Forest anomalies · Interactive Plotly analytics")

    with st.sidebar:
        st.header("Controls")
        market_limit = st.slider("Markets to scan", 5, 80, 25)
        anomaly_threshold = st.slider("Anomaly score threshold", 0.0, 1.0, 0.65, 0.01)
        use_synthetic = st.toggle("Force synthetic data", value=False)

        st.divider()
        st.subheader("Position sizing")
        sizing_label = st.radio(
            "Sizing method",
            options=["Fixed Amount ($)", "Percentage of Equity (%)"],
            index=0,
        )
        sizing_mode = "fixed" if sizing_label.startswith("Fixed") else "percent"
        if sizing_mode == "fixed":
            stake_amount = st.number_input(
                "Stake amount ($)",
                min_value=1.0,
                max_value=float(getattr(config, "INITIAL_CAPITAL", 10_000.0)),
                value=float(getattr(config, "DEFAULT_STAKE_AMOUNT", 200.0)),
                step=50.0,
            )
        else:
            stake_amount = st.number_input(
                "Stake amount (% of equity)",
                min_value=0.1,
                max_value=100.0,
                value=2.0,
                step=0.5,
            )
        scale_with_anomaly = st.toggle(
            "Scale size with anomaly score",
            value=bool(getattr(config, "DEFAULT_SCALE_WITH_ANOMALY", True)),
            help="Higher anomaly_score → linearly larger position (score × base stake).",
        )

        st.divider()
        st.subheader("Execution frictions")
        auto_frictions = st.toggle(
            "Auto-Calculate Frictions (Dynamic Market Impact)",
            value=False,
            help="Slippage from order-book depth / liquidity vs stake; "
            "Polymarket protocol fee 0%; Cross the Spread forced ON.",
        )
        if auto_frictions:
            slippage_pct = 0.0
            trading_fee_pct = float(getattr(config, "POLYMARKET_PROTOCOL_FEE_PCT", 0.0))
            cross_the_spread = True
            st.caption(
                "Auto mode: slippage = book VWAP impact (or √(stake/liquidity) heuristic) · "
                f"fee = {trading_fee_pct:.2f}% · Cross the Spread = ON"
            )
        else:
            slippage_pct = st.slider("Slippage (%)", 0.0, 2.0, 0.5, 0.05)
            trading_fee_pct = st.slider("Trading Fee (%)", 0.0, 1.0, 0.2, 0.05)
            cross_the_spread = st.toggle(
                "Cross the Spread",
                value=True,
                help="Enter at ask / exit at bid. If off, mid is used before slippage.",
            )

        st.divider()
        st.subheader("Market filters")
        categories = st.multiselect(
            "Category",
            ["Crypto", "Macro", "Politics", "Other"],
            default=["Crypto", "Macro", "Politics", "Other"],
        )
        min_volume = st.number_input("Min volume", min_value=0.0, value=0.0, step=100.0)
        max_spread = st.slider("Max |spread|", 0.0, 0.5, 0.25, 0.01)
        min_anomaly_score = st.slider("Min anomaly score", 0.0, 1.0, 0.0, 0.01)
        st.divider()
        st.subheader("Greeks heatmap")
        greek_choice = st.selectbox("Greek", ["delta", "gamma"])
        strike_k = st.number_input("Strike K", min_value=1.0, value=100.0, step=1.0)
        vol = st.slider("Implied vol σ", 0.05, 0.80, 0.20, 0.01)

    with st.spinner("Fetching markets and running quantitative models..."):
        analyzed, rnd_cache = run_pipeline(market_limit, use_synthetic, categories)

    # 1) Filter first — KPI / table / charts all share this frame
    filtered = apply_market_filters(
        analyzed,
        categories=categories,
        min_volume=min_volume,
        max_spread=max_spread,
        min_anomaly_score=min_anomaly_score,
    )

    # 2) Recalculate metrics on the filtered subset (incl. position sizing)
    initial_capital = float(getattr(config, "INITIAL_CAPITAL", 10_000.0))
    kpis, backtest_df = metrics_from_filtered(
        filtered,
        anomaly_threshold=anomaly_threshold,
        initial_capital=initial_capital,
        sizing_mode=sizing_mode,
        stake_amount=float(stake_amount),
        scale_with_anomaly=scale_with_anomaly,
        slippage_pct=float(slippage_pct),
        trading_fee_pct=float(trading_fee_pct),
        cross_the_spread=cross_the_spread,
        auto_frictions=auto_frictions,
    )

    # --- KPI row (driven by filtered data) ---
    c1, c2, c3, c4, c5 = st.columns(5)
    if filtered.empty:
        c1.metric("Total PnL", "N/A", delta="N/A")
        c2.metric("Win Rate", "N/A")
        c3.metric("Sharpe Ratio", "N/A")
        c4.metric("Brier Score", "N/A")
        c5.metric("Anomalies", "0")
        st.info("No markets match the current filters. Relax Min volume / Max |spread| / Min anomaly score.")
    else:
        pnl = float(kpis["total_pnl"] or 0.0)
        roi = float(kpis.get("roi") or 0.0)
        c1.metric("Total PnL", f"${pnl:+,.2f}", delta=f"{roi:+.2%} ROI")
        c2.metric("Win Rate", _fmt_pct(kpis.get("win_rate")))
        c3.metric("Sharpe Ratio", _fmt_num(kpis.get("sharpe"), 2))
        brier = kpis.get("brier")
        if brier is None:
            c4.metric("Brier Score", "N/A")
        else:
            c4.metric(
                "Brier Score",
                f"{float(brier):.4f}",
                delta="calibrated" if float(brier) <= 0.25 else "check polarity",
                delta_color="normal" if float(brier) <= 0.25 else "inverse",
            )
        c5.metric("Anomalies", f"{int(kpis.get('anomalies') or 0)}")
        stake_desc = (
            f"${stake_amount:,.0f} fixed"
            if sizing_mode == "fixed"
            else f"{stake_amount:.1f}% of equity"
        )
        if scale_with_anomaly:
            stake_desc += " × anomaly score"
        max_dd = kpis.get("max_drawdown_pct")
        dd_txt = f"{float(max_dd):.4f}%" if max_dd is not None else "N/A"
        if auto_frictions:
            avg_slip = float(kpis.get("avg_slippage_pct") or 0.0)
            friction_desc = (
                f"auto-slip **{avg_slip:.3f}%** avg · fee **{trading_fee_pct:.2f}%** · cross-spread on"
            )
        else:
            friction_desc = (
                f"slip **{slippage_pct:.2f}%** · fee **{trading_fee_pct:.2f}%** · "
                f"cross-spread={'on' if cross_the_spread else 'off'}"
            )
        st.caption(
            f"KPIs for **{int(kpis.get('n_markets') or 0)}** filtered markets "
            f"({int(kpis.get('n_trades') or 0)} trades) · "
            f"sizing: **{stake_desc}** · Max DD: **{dd_txt}** · {friction_desc}"
        )

    if anomaly_threshold < 0.60:
        st.warning(
            "Anomaly score threshold is below 0.60 — expect more false-positive "
            "signals and noisier equity results. Prefer ≥ 0.60 for production scans."
        )

    # --- Markets table (same filtered frame) ---
    st.subheader("Markets & Anomalies")
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
    if filtered.empty or not display_cols:
        st.dataframe(pd.DataFrame(), use_container_width=True, height=120)
    else:
        table_df = filtered[display_cols].copy()
        if not backtest_df.empty and "position_size" in backtest_df.columns:
            table_df["position_size"] = backtest_df["position_size"].to_numpy()
            if "slippage_pct" in backtest_df.columns:
                table_df["slippage_pct"] = backtest_df["slippage_pct"].to_numpy()
            table_df["entry_price"] = backtest_df["entry_price"].to_numpy()
            table_df["exit_price"] = backtest_df["exit_price"].to_numpy()
            table_df["fee_entry"] = backtest_df["fee_entry"].to_numpy()
            table_df["fee_exit"] = backtest_df["fee_exit"].to_numpy()
            table_df["pnl"] = backtest_df["pnl"].to_numpy()
        for col in (
            "price_yes",
            "price_no",
            "fair_price",
            "spread",
            "anomaly_score",
            "delta",
            "position_size",
            "slippage_pct",
            "entry_price",
            "exit_price",
            "fee_entry",
            "fee_exit",
            "pnl",
        ):
            if col in table_df.columns:
                table_df[col] = table_df[col].round(4)
        number_cols_4dp = {
            c: st.column_config.NumberColumn(format="%.4f")
            for c in (
                "price_yes",
                "price_no",
                "fair_price",
                "spread",
                "anomaly_score",
                "delta",
                "position_size",
                "slippage_pct",
                "entry_price",
                "exit_price",
                "fee_entry",
                "fee_exit",
                "pnl",
            )
            if c in table_df.columns
        }
        st.dataframe(
            table_df,
            use_container_width=True,
            height=320,
            column_config=number_cols_4dp,
        )

    # --- RND (filtered markets only) ---
    st.subheader("Risk-Neutral Density")
    if filtered.empty:
        st.caption("RND unavailable — filtered dataset is empty.")
    else:
        questions = filtered["question"].tolist()
        selected = st.selectbox("Inspect market", questions)
        row = filtered[filtered["question"] == selected].iloc[0]
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
        if backtest_df.empty:
            st.caption("No trades / markets under current filters.")
        else:
            fig_eq = plot_equity_curve(backtest_df, initial_capital=initial_capital)
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
