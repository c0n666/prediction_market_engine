"""
Backtesting and Evaluation Engine.
Supports cross-sectional / time-series simulation with position sizing and
real-world execution frictions (bid/ask, slippage, fees on entry & exit).
"""

from __future__ import annotations

import logging
from typing import Dict, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from config import config

logger = logging.getLogger(__name__)

SizingMode = Literal["fixed", "percent"]


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
        sizing_mode: SizingMode = "fixed",
        stake_amount: float = config.DEFAULT_STAKE_AMOUNT,
        scale_with_anomaly: bool = config.DEFAULT_SCALE_WITH_ANOMALY,
        slippage_pct: float = config.DEFAULT_SLIPPAGE_PCT,
        trading_fee_pct: float = config.DEFAULT_TRADING_FEE_PCT,
        cross_the_spread: bool = config.DEFAULT_CROSS_THE_SPREAD,
        auto_frictions: bool = False,
    ):
        # Legacy flat fee kept for backward compat; prefer trading_fee_pct
        self.fee = execution_fee
        self.threshold = min_score_threshold
        self.initial_capital = float(initial_capital)
        self.sizing_mode: SizingMode = "percent" if sizing_mode == "percent" else "fixed"
        self.stake_amount = float(stake_amount)
        self.scale_with_anomaly = bool(scale_with_anomaly)
        self.auto_frictions = bool(auto_frictions)
        if self.auto_frictions:
            # Polymarket protocol fee ≈ 0%; market orders cross the spread
            self.slippage_pct = 0.0  # overwritten per-trade dynamically
            self.trading_fee_pct = float(config.POLYMARKET_PROTOCOL_FEE_PCT)
            self.cross_the_spread = True
        else:
            self.slippage_pct = float(np.clip(slippage_pct, 0.0, 100.0))
            self.trading_fee_pct = float(np.clip(trading_fee_pct, 0.0, 100.0))
            self.cross_the_spread = bool(cross_the_spread)
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

    def compute_position_size(self, equity: float, anomaly_score: float) -> float:
        """Dollar stake for the next trade (fixed $ or % of equity)."""
        equity = max(float(equity), 0.0)
        if self.sizing_mode == "percent":
            base = equity * (float(self.stake_amount) / 100.0)
        else:
            base = float(self.stake_amount)

        base = float(np.clip(base, 0.0, equity if equity > 0 else 0.0))
        if self.scale_with_anomaly:
            base *= float(np.clip(anomaly_score, 0.0, 1.0))
        return round(max(base, 0.0), 4)

    @staticmethod
    def _half_spread(row: pd.Series, mid: float) -> float:
        if "spread" in row.index and pd.notna(row.get("spread")):
            return abs(float(row["spread"])) / 2.0
        if "price_no" in row.index and pd.notna(row.get("price_no")):
            implied = abs(mid - (1.0 - float(row["price_no"])))
            return implied / 2.0
        return 0.0

    def quote_bid_ask(self, row: pd.Series, *, side: Literal["yes", "no"]) -> Tuple[float, float]:
        """
        Build (bid, ask) for YES or NO.

        Never uses ``fair_price`` for fills. When ``cross_the_spread`` is False,
        bid = ask = mid (slippage still applied later).
        """
        if side == "yes":
            mid = float(row["price_yes"])
            # Complementary book: YES bid ≈ 1 - NO ask, YES ask ≈ price_yes
            comp = 1.0 - float(row["price_no"]) if "price_no" in row.index else mid
            raw_ask = max(mid, comp)
            raw_bid = min(mid, comp)
        else:
            mid = float(row["price_no"])
            comp = 1.0 - float(row["price_yes"]) if "price_yes" in row.index else mid
            raw_ask = max(mid, comp)
            raw_bid = min(mid, comp)

        mid = float(np.clip(mid, 0.01, 0.99))
        if not self.cross_the_spread:
            return mid, mid

        half = self._half_spread(row, mid)
        ask = float(np.clip(max(raw_ask, mid + half), 0.01, 0.99))
        bid = float(np.clip(min(raw_bid, mid - half), 0.01, 0.99))
        if ask < bid:
            return mid, mid
        return bid, ask

    def _exec_prices(
        self, bid: float, ask: float, *, slippage_pct: Optional[float] = None
    ) -> Tuple[float, float]:
        """Buy entry at ask*(1+slip); sell/exit at bid*(1-slip)."""
        slip = (self.slippage_pct if slippage_pct is None else float(slippage_pct)) / 100.0
        entry = float(np.clip(ask * (1.0 + slip), 0.001, 1.0))
        exit_px = float(np.clip(bid * (1.0 - slip), 0.0, 1.0))
        return entry, exit_px

    @staticmethod
    def _parse_book_levels(raw_levels) -> list:
        """Normalize CLOB book levels to [(price, size), ...] ascending by price."""
        levels = []
        if raw_levels is None:
            return levels
        if isinstance(raw_levels, str):
            try:
                import json

                raw_levels = json.loads(raw_levels)
            except Exception:
                return levels
        if not isinstance(raw_levels, (list, tuple)):
            return levels
        for lvl in raw_levels:
            try:
                if isinstance(lvl, dict):
                    px = float(lvl.get("price", lvl.get("p", 0)))
                    sz = float(lvl.get("size", lvl.get("s", lvl.get("quantity", 0))))
                elif isinstance(lvl, (list, tuple)) and len(lvl) >= 2:
                    px, sz = float(lvl[0]), float(lvl[1])
                else:
                    continue
                if px > 0 and sz > 0:
                    levels.append((px, sz))
            except (TypeError, ValueError):
                continue
        levels.sort(key=lambda x: x[0])
        return levels

    @classmethod
    def slippage_from_order_book(cls, asks_raw, stake_usd: float) -> Optional[float]:
        """
        Walk the ask book until ``stake_usd`` is filled; return VWAP slippage % vs best ask.
        """
        asks = cls._parse_book_levels(asks_raw)
        if not asks or stake_usd <= 0:
            return None
        best = asks[0][0]
        remaining = float(stake_usd)
        cost = 0.0
        shares = 0.0
        for px, sz in asks:
            level_notional = px * sz
            take = min(remaining, level_notional)
            qty = take / px
            cost += take
            shares += qty
            remaining -= take
            if remaining <= 1e-9:
                break
        if shares <= 0 or best <= 0:
            return None
        # Unfilled remainder: penalize at last px + 50% adverse
        if remaining > 1e-9:
            last_px = asks[-1][0]
            pen_px = min(last_px * 1.5, 0.99)
            cost += remaining
            shares += remaining / max(pen_px, 1e-6)
        vwap = cost / shares
        return float(np.clip((vwap / best - 1.0) * 100.0, 0.0, 2.0))

    def estimate_dynamic_slippage_pct(self, row: pd.Series, stake: float) -> float:
        """
        Dynamic market-impact slippage (%) from order-book depth when available,
        else sqrt(participation) heuristic using liquidity / volume.
        """
        # Prefer explicit book columns if present (asks / order_book_asks)
        for key in ("order_book_asks", "asks", "book_asks"):
            if key in row.index and row.get(key) not in (None, "", [], {}):
                book_slip = self.slippage_from_order_book(row.get(key), stake)
                if book_slip is not None:
                    return float(book_slip)

        liquidity = float(row.get("liquidity", 0) or 0)
        volume = float(row.get("volume", 0) or 0)
        depth = max(liquidity, volume * 0.05, 1.0)
        participation = float(stake) / depth

        spread = abs(float(row.get("spread", 0) or 0))
        if spread <= 0 and "price_yes" in row.index and "price_no" in row.index:
            spread = abs(float(row["price_yes"]) - (1.0 - float(row["price_no"])))
        half_spread_pct = 100.0 * (spread / 2.0) / max(float(row.get("price_yes", 0.5)), 0.05)

        impact = 100.0 * float(config.AUTO_IMPACT_COEFF) * np.sqrt(max(participation, 0.0))
        slip = half_spread_pct + impact
        return float(np.clip(slip, 0.0, 2.0))

    def _realize_long(
        self,
        *,
        stake: float,
        entry_price: float,
        exit_price: float,
    ) -> Tuple[float, float, float, float]:
        """
        Buy ``stake`` dollars at entry_price, exit at exit_price.
        Trading fees deducted on entry notional and exit notional.
        """
        if stake <= 0 or entry_price <= 0:
            return 0.0, 0.0, 0.0, 0.0
        fee_r = self.trading_fee_pct / 100.0
        shares = stake / entry_price
        fee_entry = fee_r * stake
        exit_value = shares * exit_price
        fee_exit = fee_r * abs(exit_value)
        pnl = exit_value - stake - fee_entry - fee_exit
        return float(pnl), float(shares), float(fee_entry), float(fee_exit)

    def run_backtest(self, historical_df: pd.DataFrame) -> pd.DataFrame:
        df = historical_df.copy()
        trades = []
        cumulative_pnl = 0.0
        equity = float(self.initial_capital)

        for idx, row in df.iterrows():
            signal = 0
            pnl = 0.0
            position_size = 0.0
            entry_price = 0.0
            exit_price = 0.0
            fee_entry = 0.0
            fee_exit = 0.0
            anomaly_score = float(row.get("anomaly_score", 0.0) or 0.0)

            if anomaly_score >= self.threshold and equity > 0:
                position_size = self.compute_position_size(equity, anomaly_score)
                if position_size > 0:
                    trade_slip = (
                        self.estimate_dynamic_slippage_pct(row, position_size)
                        if self.auto_frictions
                        else float(self.slippage_pct)
                    )
                    # Signal from model vs mid — fills never use fair_price
                    if row["fair_price"] > row["price_yes"]:
                        signal = 1
                        bid, ask = self.quote_bid_ask(row, side="yes")
                        entry_price, exit_mkt = self._exec_prices(
                            bid, ask, slippage_pct=trade_slip
                        )
                        face = float(int(row.get("outcome", 1)))
                        exit_price = float(np.clip(face * (1.0 - trade_slip / 100.0), 0.0, 1.0))
                        if "outcome" not in row.index or pd.isna(row.get("outcome")):
                            exit_price = exit_mkt
                        pnl, _, fee_entry, fee_exit = self._realize_long(
                            stake=position_size,
                            entry_price=entry_price,
                            exit_price=exit_price,
                        )
                    elif row["fair_price"] < row["price_yes"]:
                        signal = -1
                        bid, ask = self.quote_bid_ask(row, side="no")
                        entry_price, exit_mkt = self._exec_prices(
                            bid, ask, slippage_pct=trade_slip
                        )
                        face = float(1 - int(row.get("outcome", 0)))
                        exit_price = float(np.clip(face * (1.0 - trade_slip / 100.0), 0.0, 1.0))
                        if "outcome" not in row.index or pd.isna(row.get("outcome")):
                            exit_price = exit_mkt
                        pnl, _, fee_entry, fee_exit = self._realize_long(
                            stake=position_size,
                            entry_price=entry_price,
                            exit_price=exit_price,
                        )
                    else:
                        trade_slip = 0.0
                        position_size = 0.0
                else:
                    trade_slip = 0.0
            else:
                trade_slip = 0.0

            cumulative_pnl += pnl
            equity = self.initial_capital + cumulative_pnl
            trades.append(
                {
                    "trade_id": idx,
                    "signal": signal,
                    "position_size": round(position_size, 4),
                    "slippage_pct": round(float(trade_slip), 4),
                    "entry_price": round(entry_price, 6),
                    "exit_price": round(exit_price, 6),
                    "fee_entry": round(fee_entry, 4),
                    "fee_exit": round(fee_exit, 4),
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
    """Rich result dict for QuantVisualizer / practice reports."""

    def run_backtest(self, historical_df: pd.DataFrame) -> Dict:
        trades_df = StrategyBacktester.run_backtest(self, historical_df)
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
    """Chronological walk-forward backtest on daily contract time series."""

    def run_timeseries_backtest(
        self,
        ts_df: pd.DataFrame,
        *,
        timestamp_col: str = "timestamp",
        score_col: str = "anomaly_score",
        fair_col: str = "fair_price",
    ) -> Dict:
        bt_kwargs = dict(
            execution_fee=self.fee,
            min_score_threshold=self.threshold,
            initial_capital=self.initial_capital,
            sizing_mode=self.sizing_mode,
            stake_amount=self.stake_amount,
            scale_with_anomaly=self.scale_with_anomaly,
            slippage_pct=self.slippage_pct,
            trading_fee_pct=self.trading_fee_pct,
            cross_the_spread=self.cross_the_spread,
            auto_frictions=self.auto_frictions,
        )
        if ts_df.empty:
            empty = QuantitativeBacktester(**bt_kwargs).run_backtest(pd.DataFrame())
            empty["mode"] = "timeseries"
            empty["n_observations"] = 0
            return empty

        df = ts_df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df = df.sort_values([timestamp_col, "contract_id"]).reset_index(drop=True)

        daily = []
        for (day, _contract), g in df.groupby([df[timestamp_col].dt.floor("D"), "contract_id"]):
            row = g.iloc[-1].copy()
            row["timestamp"] = day
            daily.append(row)
        panel = pd.DataFrame(daily).sort_values("timestamp").reset_index(drop=True)

        if score_col not in panel.columns:
            panel[score_col] = panel.get("anomaly_score", 0.5)
        if fair_col not in panel.columns:
            panel[fair_col] = panel["price_yes"]

        panel = panel.rename(columns={score_col: "anomaly_score", fair_col: "fair_price"})
        result = QuantitativeBacktester(**bt_kwargs).run_backtest(panel)
        result["mode"] = "timeseries"
        result["n_observations"] = int(len(panel))
        return result
