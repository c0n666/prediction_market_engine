"""
Prediction Market Mispricing Engine - Main Pipeline Orchestrator.
"""

import asyncio
import logging
import pandas as pd
import numpy as np

from config import config
from src.data_ingestion import PolymarketDataIngestion
from src.math_engine import BreedenLitzenbergerEngine
from src.anomaly_detector import MispricingAnomalyDetector

# Setup Structured Logging
logging.basicConfig(
    level=config.LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("MainOrchestrator")


async def run_pipeline():
    logger.info("Initializing Prediction Market Engine Pipeline...")

    # Step 1: Data Ingestion
    ingester = PolymarketDataIngestion()
    logger.info("Fetching real-time market data from Polymarket API...")
    events = await ingester.fetch_active_events(limit=30)
    df_raw = ingester.process_market_data(events)

    if df_raw.empty:
        logger.warning("No active markets retrieved. Exiting pipeline.")
        return

    logger.info(f"Retrieved {len(df_raw)} active binary contracts.")

    # Step 2: Quantitative Math Processing (Breeden-Litzenberger)
    math_engine = BreedenLitzenbergerEngine()
    fair_prices = []
    deltas = []

    for idx, row in df_raw.iterrows():
        # Synthetic strike grid for demonstration using binary outcomes
        strikes = np.array([0.0, 0.5, 1.0])
        prices = np.array([0.0, row["price_yes"], 1.0])

        grid_k, pdf, delta = math_engine.extract_density_and_delta(strikes, prices)
        
        if len(pdf) > 0:
            fair_prob = math_engine.compute_fair_probability(row["price_yes"], pdf, grid_k)
            avg_delta = float(np.mean(delta)) if len(delta) > 0 else 0.5
        else:
            fair_prob = row["price_yes"]
            avg_delta = 0.5

        fair_prices.append(fair_prob)
        deltas.append(avg_delta)

    df_raw["fair_price"] = fair_prices
    df_raw["delta"] = deltas

    # Step 3: Machine Learning Anomaly Detection
    logger.info("Running Isolation Forest Anomaly Detection...")
    detector = MispricingAnomalyDetector()
    processed_df = detector.fit_predict(df_raw)

    # Filter High-Score Mispricings
    anomalies = processed_df[processed_df["is_anomaly"]].sort_values(
        by="anomaly_score", ascending=False
    )

    logger.info(f"Analysis Complete. Identified {len(anomalies)} mispriced opportunities.")

    # Step 4: Output Top Signals
    print("\n" + "="*80)
    print("TOP PREDICTION MARKET MISPRICING OPPORTUNITIES")
    print("="*80)
    
    output_cols = ["question", "price_yes", "fair_price", "anomaly_score", "volume"]
    if not anomalies.empty:
        print(anomalies[output_cols].head(10).to_string(index=False))
    else:
        print("No significant mispricings detected under current parameters.")
    print("="*80 + "\n")


if __name__ == "__main__":
    asyncio.run(run_pipeline())