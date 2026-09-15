"""
Backtesting and Evaluation Engine.
Supports cross-sectional strategy simulation and chronological time-series backtests.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from config import config

logger = logging.getLogger(__name__)


class StrategyBacktester:
    """
    Simulates trading execution on prediction market events
    and evaluates probabilistic forecast accuracy.
    """

    def __init__(
        self,
        execution_fee: float = config.EXECUTION_FEE,
        min_score_threshold: float = config.ANOMALY_SCORE_THRESHOLD,
        initial_capital: float = config.INITIAL_CAPITAL,
    ):
        self.fee = execution_fee
        self.threshold = min_score_threshold
        self.initial_capital = float(initial_capital)
        self.max_drawdown_pct: float = 0.0
        self.last_brier_score: float = 0.0

    def evaluate_forecast_accuracy(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
        if len(y_true) == 0:
            return {"brier_score": 0.0, "log_loss": 0.0}

        brier = self.compute_brier_score(y_prob, y_true)
        y_prob_clipped = np.clip(np.asarray(y_prob, dtype=float), 1e-5, 1.0 - 1e-5)
        y_true_arr = np.asarray(y_true, dtype=float)
        lloss = float(log_loss(y_true_arr, y_prob_clipped))
        return {"brier_score": round(brier, 4), "log_loss": round(lloss, 4)}

    @staticmethod
    def compute_brier_score(
        predicted_probs: np.ndarray,
        actual_outcomes: np.ndarray,
    ) -> float:
        """Brier Score = mean((p_i - o_i)^2) for P(Y=1) forecasts."""
        if len(actual_outcomes) == 0:
            return 0.0

        p = np.clip(np.asarray(predicted_probs, dtype=float), 0.0, 1.0)
        o = np.asarray(actual_outcomes, dtype=float)

        brier = float(np.mean((p - o) ** 2))
        brier_complement = float(np.mean(((1.0 - p) - o) ** 2))
        if brier_complement + 1e-12 < brier:
            logger.warning(
                "Brier inputs appear polarity-inverted (1-P fits better than P); flipping."
            )
            p = 1.0 - p
            brier = float(np.mean((p - o) ** 2))
        return float(brier)

    @staticmethod
    def compute_drawdown(equity: np.ndarray) -> Tuple[np.ndarray, float]:
        if len(equity) == 0:
            return np.array([]), 0.0
        cummax = np.maximum.accumulate(equity)
        with np.errstate(divide="ignore", invalid="ignore"):
            drawdown = np.where(cummax > 0, (cummax - equity) / cummax, 0.0)
        drawdown = np.clip(drawdown, 0.0, 1.0)
        max_drawdown_pct = float(np.clip(np.max(drawdown) * 100.0, 0.0, 100.0))
        return drawdown, max_drawdown_pct

    def run_backtest(self, historical_df: pd.DataFrame) -> pd.DataFrame:
        df = historical_df.copy()
        trades = []
        cumulative_pnl = 0.0

        for idx, row in df.iterrows():
            signal = 0
            pnl = 0.0

            if row.get("anomaly_score", 0.0) >= self.threshold:
                if row["fair_price"] > row["price_yes"]:
                    signal = 1
                    actual_outcome = int(row.get("outcome", 1))
                    pnl = (actual_outcome - row["price_yes"]) - self.fee
                elif row["fair_price"] < row["price_yes"]:
                    signal = -1
                    actual_outcome = int(row.get("outcome", 0))
                    pnl = ((1 - actual_outcome) - row["price_no"]) - self.fee

            cumulative_pnl += pnl
            equity = self.initial_capital + cumulative_pnl
            trades.append(
                {
                    "trade_id": idx,
                    "signal": signal,
                    "pnl": round(pnl, 4),
                    "cumulative_pnl": round(cumulative_pnl, 4),
                    "equity": round(equity, 4),
                }
            )

        trades_df = pd.DataFrame(trades)
        result = pd.concat([df.reset_index(drop=True), trades_df], axis=1)

        if not result.empty and "equity" in result.columns:
            drawdown, max_dd_pct = self.compute_drawdown(result["equity"].to_numpy(dtype=float))
            result["drawdown"] = np.round(drawdown, 6)
            result["drawdown_pct"] = np.round(drawdown * 100.0, 4)
            self.max_drawdown_pct = max_dd_pct
        else:
            self.max_drawdown_pct = 0.0

        if not result.empty and "fair_price" in result.columns and "outcome" in result.columns:
            self.last_brier_score = self.compute_brier_score(
                result["fair_price"].to_numpy(dtype=float),
                result["outcome"].to_numpy(dtype=float),
            )
        else:
            self.last_brier_score = 0.0

        return result

    def summarize(self, result_df: pd.DataFrame) -> Dict[str, float]:
        """Aggregate performance metrics for reports / visualizer."""
        if result_df.empty or "pnl" not in result_df.columns:
            return {
                "total_pnl": 0.0,
                "roi_percent": 0.0,
                "sharpe_ratio": 0.0,
                "win_rate_percent": 0.0,
                "max_drawdown_pct": 0.0,
                "brier_score": 0.0,
                "initial_capital": self.initial_capital,
            }

        active = result_df[result_df["signal"] != 0]
        pnls = active["pnl"].to_numpy(dtype=float) if not active.empty else np.array([0.0])
        total_pnl = float(result_df["cumulative_pnl"].iloc[-1])
        wins = int(np.sum(pnls > 0))
        win_rate = float(wins / len(pnls)) if len(pnls) else 0.0
        sharpe = float(pnls.mean() / (pnls.std() + 1e-8) * np.sqrt(max(len(pnls), 1)))
        roi = total_pnl / self.initial_capital * 100.0

        return {
            "total_pnl": round(total_pnl, 4),
            "roi_percent": round(roi, 4),
            "sharpe_ratio": round(sharpe, 4),
            "win_rate_percent": round(win_rate * 100.0, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct, 4),
            "brier_score": round(self.last_brier_score, 4),
            "initial_capital": self.initial_capital,
        }


class QuantitativeBacktester(StrategyBacktester):
    """
    Extended backtester returning a rich result dict compatible with QuantVisualizer
    and practice-report metrics (equity curve, trades, Sharpe, ROI).
    """

    def run_backtest(self, historical_df: pd.DataFrame) -> Dict:
        trades_df = super().run_backtest(historical_df)
        summary = self.summarize(trades_df)
        equity = (
            trades_df["equity"].to_numpy(dtype=float)
            if not trades_df.empty and "equity" in trades_df.columns
            else np.array([self.initial_capital])
        )
        return {
            **summary,
            "equity_curve": equity.tolist(),
            "trades_detail": trades_df,
            "max_drawdown_pct": self.max_drawdown_pct,
        }


class TimeSeriesBacktester(StrategyBacktester):
    """
    Chronological walk-forward backtest on daily contract time series.

    For each day t, uses prices available at t and settles against final outcome.
    """

    def run_timeseries_backtest(
        self,
        ts_df: pd.DataFrame,
        *,
        timestamp_col: str = "timestamp",
        score_col: str = "anomaly_score",
        fair_col: str = "fair_price",
    ) -> Dict:
        if ts_df.empty:
            return self.run_backtest(pd.DataFrame())

        df = ts_df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df = df.sort_values([timestamp_col, "contract_id"]).reset_index(drop=True)

        # Take last observation per contract as entry (end-of-series signal)
        # and also evaluate mid-path daily signals for equity path
        daily = []
        for (day, contract), g in df.groupby([df[timestamp_col].dt.floor("D"), "contract_id"]):
            row = g.iloc[-1].copy()
            row["timestamp"] = day
            daily.append(row)
        panel = pd.DataFrame(daily).sort_values("timestamp").reset_index(drop=True)

        # Ensure required columns
        if score_col not in panel.columns:
            panel[score_col] = panel.get("anomaly_score", 0.5)
        if fair_col not in panel.columns:
            panel[fair_col] = panel["price_yes"]

        # Rename for StrategyBacktester
        panel = panel.rename(columns={score_col: "anomaly_score", fair_col: "fair_price"})
        result = QuantitativeBacktester(
            execution_fee=self.fee,
            min_score_threshold=self.threshold,
            initial_capital=self.initial_capital,
        ).run_backtest(panel)
        result["mode"] = "timeseries"
        result["n_observations"] = int(len(panel))
        return result
