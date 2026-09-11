"""
Backtesting and Evaluation Engine.
Evaluates model precision using statistical scoring rules (Brier Score, LogLoss)
and simulates financial execution performance.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict
from sklearn.metrics import brier_score_loss, log_loss

logger = logging.getLogger(__name__)


class StrategyBacktester:
    """
    Simulates trading execution on historical prediction market events
    and evaluates probabilistic forecast accuracy.
    """

    def __init__(self, execution_fee: float = 0.002, min_score_threshold: float = 0.65):
        self.fee = execution_fee
        self.threshold = min_score_threshold

    def evaluate_forecast_accuracy(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
        """
        Computes standard probabilistic evaluation metrics.
        """
        if len(y_true) == 0:
            return {"brier_score": 0.0, "log_loss": 0.0}

        # Clip probabilities to avoid infinity in log_loss
        y_prob_clipped = np.clip(y_prob, 1e-5, 1.0 - 1e-5)

        brier = float(brier_score_loss(y_true, y_prob))
        lloss = float(log_loss(y_true, y_prob_clipped))

        return {
            "brier_score": round(brier, 4),
            "log_loss": round(lloss, 4)
        }

    def run_backtest(self, historical_df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes paper trading simulation on historical signals.
        
        Strategy Logic:
        - If anomaly_score >= threshold and fair_price > market_price -> BUY YES
        - If anomaly_score >= threshold and fair_price < market_price -> BUY NO
        """
        df = historical_df.copy()
        trades = []
        cumulative_pnl = 0.0

        for idx, row in df.iterrows():
            signal = 0  # 0: HOLD, 1: BUY YES, -1: BUY NO
            pnl = 0.0

            if row.get("anomaly_score", 0.0) >= self.threshold:
                if row["fair_price"] > row["price_yes"]:
                    signal = 1
                    # Profit if target event outcome == 1
                    actual_outcome = row.get("outcome", 1)
                    pnl = (actual_outcome - row["price_yes"]) - self.fee
                elif row["fair_price"] < row["price_yes"]:
                    signal = -1
                    actual_outcome = row.get("outcome", 0)
                    pnl = ((1 - actual_outcome) - row["price_no"]) - self.fee

            cumulative_pnl += pnl
            trades.append({
                "trade_id": idx,
                "signal": signal,
                "pnl": round(pnl, 4),
                "cumulative_pnl": round(cumulative_pnl, 4)
            })

        trades_df = pd.DataFrame(trades)
        return pd.concat([df.reset_index(drop=True), trades_df], axis=1)