"""
main.py — End-to-end quantitative pipeline for the practice assignment.

Steps:
1. REST collection / synthetic historical time series
2. Fair-price (Breeden-Litzenberger + Black-Scholes binaries)
3. Isolation Forest anomaly detection
4. BL vs ML probability accuracy comparison
5. Chronological + cross-sectional backtests
6. Chart / metrics reports
"""

from __future__ import annotations

import asyncio
import logging
import os

import numpy as np
import pandas as pd

from config import config, ensure_data_dirs
from src.anomaly_detector import MispricingAnomalyDetector
from src.backtester import QuantitativeBacktester, TimeSeriesBacktester
from src.comparison import ProbabilityModelComparator
from src.data_ingestion import PolymarketDataIngestion
from src.math_engine import BlackScholesBinaryEngine, BreedenLitzenbergerEngine
from src.visualizer import QuantVisualizer

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _attach_fair_prices(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Attach BL/BS fair prices and return a representative RND curve."""
    bl_engine = BreedenLitzenbergerEngine()
    bs_engine = BlackScholesBinaryEngine()
    fair_prices = []
    deltas = []
    grid_k, pdf_rnd = np.array([]), np.array([])

    has_bs_fields = {"underlying_spot", "strike_price", "tenor_years"}.issubset(df.columns)

    for _, row in df.iterrows():
        if has_bs_fields:
            S, K, T = row["underlying_spot"], row["strike_price"], row["tenor_years"]
            fair = bs_engine.price_binary_call(
                S, K, T, config.RISK_FREE_RATE, config.DEFAULT_VOLATILITY
            )
            greeks = bs_engine.compute_binary_greeks(
                S, K, T, config.RISK_FREE_RATE, config.DEFAULT_VOLATILITY
            )
            # Blend BS digital price with market-anchored BL estimate
            bl_fair, gk, pdf = bl_engine.estimate_binary_fair_price(row["price_yes"])
            fair = float(np.clip(0.5 * fair + 0.5 * bl_fair, 0.001, 0.999))
            delta = greeks["delta"]
            if len(pdf) > 0:
                grid_k, pdf_rnd = gk, pdf
        else:
            fair, gk, pdf = bl_engine.estimate_binary_fair_price(row["price_yes"])
            delta = 0.0
            if len(pdf) > 0:
                grid_k, pdf_rnd = gk, pdf

        fair_prices.append(round(fair, 4))
        deltas.append(round(float(delta), 6))

    out = df.copy()
    out["fair_price"] = fair_prices
    out["delta"] = deltas
    if "price_no" not in out.columns:
        out["price_no"] = (1.0 - out["price_yes"]).clip(0.01, 0.99)
    return out, grid_k, pdf_rnd


def _enrich_timeseries(ts: pd.DataFrame) -> pd.DataFrame:
    """Apply fair-price + anomaly scores on each timeseries row."""
    bl = BreedenLitzenbergerEngine()
    fair = []
    for _, row in ts.iterrows():
        f, _, _ = bl.estimate_binary_fair_price(row["price_yes"])
        fair.append(f)
    enriched = ts.copy()
    enriched["fair_price"] = fair
    detector = MispricingAnomalyDetector()
    return detector.fit_predict(enriched)


def run_quant_pipeline(num_markets: int = 50, use_api: bool = True) -> dict:
    ensure_data_dirs()
    logger.info("Initializing Quantitative Engine Pipeline...")

    ingester = PolymarketDataIngestion()
    api_paths = {}

    # 1) Data collection
    if use_api:
        try:
            api_paths = asyncio.run(ingester.collect_and_persist(active_limit=num_markets))
            logger.info("REST snapshots: %s", api_paths)
        except Exception as exc:
            logger.warning("API collection failed (%s); continuing with synthetic data.", exc)

    df = ingester.generate_synthetic_market_data(num_markets=num_markets)
    ts_path = os.path.join(config.PROCESSED_DATA_DIR, "historical_timeseries.csv")
    if os.path.exists(ts_path):
        ts_df = pd.read_csv(ts_path)
    else:
        ts_df = ingester.generate_historical_timeseries()

    # 2–3) Fair prices + RND + anomalies
    df_fair, grid_k, pdf_rnd = _attach_fair_prices(df)
    detector = MispricingAnomalyDetector()
    df_analyzed = detector.fit_predict(df_fair)

    # 4) Compare BL vs ML vs market
    comparator = ProbabilityModelComparator()
    comparison = comparator.compare(df_analyzed)
    comparison_path = comparator.save_report(comparison)
    logger.info("BL vs ML comparison:\n%s", comparison.to_string(index=False))
    logger.info("Conclusion: %s", comparator.last_report.get("conclusion"))

    # 5a) Cross-sectional backtest
    backtester = QuantitativeBacktester()
    backtest_results = backtester.run_backtest(df_analyzed)

    # 5b) Chronological time-series backtest
    ts_enriched = _enrich_timeseries(ts_df)
    ts_bt = TimeSeriesBacktester().run_timeseries_backtest(ts_enriched)

    # 6) Visual reports
    logger.info("Rendering visual analytics and backtest dashboard...")
    visualizer = QuantVisualizer(output_dir=os.path.join(config.REPORTS_DIR, "charts"))
    path_bt = visualizer.plot_backtest_dashboard(backtest_results)
    spot = float(df_analyzed["underlying_spot"].mean()) if "underlying_spot" in df_analyzed else 100.0
    path_rnd = visualizer.plot_breeden_litzenberger_rnd(grid_k, pdf_rnd, spot_price=spot)
    path_anom = visualizer.plot_anomaly_detection(df_analyzed)

    # Persist analyzed panel
    analyzed_path = os.path.join(config.PROCESSED_DATA_DIR, "analyzed_markets.csv")
    df_analyzed.to_csv(analyzed_path, index=False)

    summary = {
        "api_paths": api_paths,
        "comparison_path": comparison_path,
        "best_model": comparator.last_report.get("best_model"),
        "cross_section": {
            "total_pnl": backtest_results["total_pnl"],
            "roi_percent": backtest_results["roi_percent"],
            "sharpe_ratio": backtest_results["sharpe_ratio"],
            "brier_score": backtest_results.get("brier_score"),
            "max_drawdown_pct": backtest_results.get("max_drawdown_pct"),
        },
        "timeseries": {
            "total_pnl": ts_bt["total_pnl"],
            "roi_percent": ts_bt["roi_percent"],
            "sharpe_ratio": ts_bt["sharpe_ratio"],
            "n_observations": ts_bt.get("n_observations"),
            "max_drawdown_pct": ts_bt.get("max_drawdown_pct"),
        },
        "charts": {"backtest": path_bt, "rnd": path_rnd, "anomalies": path_anom},
        "analyzed_path": analyzed_path,
    }

    logger.info(
        "Pipeline Completed | PnL: $%s | ROI: %s%% | Sharpe: %s | Best model: %s",
        backtest_results["total_pnl"],
        backtest_results["roi_percent"],
        backtest_results["sharpe_ratio"],
        summary["best_model"],
    )
    logger.info("Time-series backtest | PnL: $%s | obs: %s", ts_bt["total_pnl"], ts_bt.get("n_observations"))
    return summary


if __name__ == "__main__":
    run_quant_pipeline(num_markets=50, use_api=True)
