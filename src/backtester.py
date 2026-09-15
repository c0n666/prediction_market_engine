"""
Backtesting and Evaluation Engine.
Evaluates model precision using statistical scoring rules (Brier Score, LogLoss)
and simulates financial execution performance.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from sklearn.metrics import log_loss

logger = logging.getLogger(__name__)


class StrategyBacktester:
    """
    Simulates trading execution on historical prediction market events
    and evaluates probabilistic forecast accuracy.
    """

    def __init__(
        self,
        execution_fee: float = 0.002,
        min_score_threshold: float = 0.65,
        initial_capital: float = 10_000.0,
    ):
        self.fee = execution_fee
        self.threshold = min_score_threshold
        self.initial_capital = float(initial_capital)
        self.max_drawdown_pct: float = 0.0
        self.last_brier_score: float = 0.0

    def evaluate_forecast_accuracy(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
        """
        Computes standard probabilistic evaluation metrics.

        ``y_prob`` must be P(Y=1) for the same positive event encoded in ``y_true``
        (0/1). Do not pass the complementary probability 1 - P.
        """
        if len(y_true) == 0:
            return {"brier_score": 0.0, "log_loss": 0.0}

        brier = self.compute_brier_score(y_prob, y_true)

        y_prob_clipped = np.clip(np.asarray(y_prob, dtype=float), 1e-5, 1.0 - 1e-5)
        y_true_arr = np.asarray(y_true, dtype=float)
        lloss = float(log_loss(y_true_arr, y_prob_clipped))

        return {
            "brier_score": round(brier, 4),
            "log_loss": round(lloss, 4),
        }

    @staticmethod
    def compute_brier_score(
        predicted_probs: np.ndarray,
        actual_outcomes: np.ndarray,
    ) -> float:
        """
        Brier Score = mean( (p_i - o_i)^2 ).

        Parameters
        ----------
        predicted_probs:
            Probabilities of the positive event (outcome == 1), in [0, 1].
            Must NOT be the complementary probability (1 - P).
        actual_outcomes:
            Realized binary outcomes in {0, 1} for that same event.

        Notes
        -----
        For a well-calibrated binary classifier, Brier Score typically lies in
        approximately [0.0, 0.25] (0.25 ≈ always predicting 0.5).
        """
        if len(actual_outcomes) == 0:
            return 0.0

        p = np.clip(np.asarray(predicted_probs, dtype=float), 0.0, 1.0)
        o = np.asarray(actual_outcomes, dtype=float)

        # Guard against accidental complement: if 1-p fits outcomes far better
        # than p, the caller passed the wrong polarity.
        brier = float(np.mean((p - o) ** 2))
        brier_complement = float(np.mean(((1.0 - p) - o) ** 2))
        if brier_complement + 1e-12 < brier:
            logger.warning(
                "Brier inputs appear polarity-inverted (1-P fits better than P); "
                "using P(Y=1) orientation by flipping probabilities."
            )
            p = 1.0 - p
            brier = float(np.mean((p - o) ** 2))

        return float(brier)

    @staticmethod
    def compute_drawdown(equity: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Drawdown relative to running peak capital.

        equity = initial_capital + cumulative_pnl
        drawdown = (cummax - equity) / cummax   ∈ [0, 1]
        Max Drawdown returned as percentage in [0, 100].
        """
        if len(equity) == 0:
            return np.array([]), 0.0

        cummax = np.maximum.accumulate(equity)
        with np.errstate(divide="ignore", invalid="ignore"):
            drawdown = np.where(cummax > 0, (cummax - equity) / cummax, 0.0)

        # Guard numerical noise / bankrupt edge cases: never exceed 100%
        drawdown = np.clip(drawdown, 0.0, 1.0)
        max_drawdown_pct = float(np.clip(np.max(drawdown) * 100.0, 0.0, 100.0))
        return drawdown, max_drawdown_pct

    def run_backtest(self, historical_df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes paper trading simulation on historical signals.

        Strategy Logic:
        - If anomaly_score >= threshold and fair_price > market_price -> BUY YES
        - If anomaly_score >= threshold and fair_price < market_price -> BUY NO

        Equity is full account capital: initial_capital + cumulative_pnl.
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
            equity = self.initial_capital + cumulative_pnl
            trades.append({
                "trade_id": idx,
                "signal": signal,
                "pnl": round(pnl, 4),
                "cumulative_pnl": round(cumulative_pnl, 4),
                "equity": round(equity, 4),
            })

        trades_df = pd.DataFrame(trades)
        result = pd.concat([df.reset_index(drop=True), trades_df], axis=1)

        if not result.empty and "equity" in result.columns:
            drawdown, max_dd_pct = self.compute_drawdown(result["equity"].to_numpy(dtype=float))
            result["drawdown"] = np.round(drawdown, 6)
            result["drawdown_pct"] = np.round(drawdown * 100.0, 4)
            self.max_drawdown_pct = max_dd_pct
        else:
            self.max_drawdown_pct = 0.0

        # Brier vs YES-event: fair_price must be P(outcome=1), matching binary outcomes
        if (
            not result.empty
            and "fair_price" in result.columns
            and "outcome" in result.columns
        ):
            self.last_brier_score = self.compute_brier_score(
                result["fair_price"].to_numpy(dtype=float),
                result["outcome"].to_numpy(dtype=float),
            )
        else:
            self.last_brier_score = 0.0

        return result
