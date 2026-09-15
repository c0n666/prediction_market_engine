"""
Model comparison: Breeden-Litzenberger fair probabilities vs ML anomaly-aware
forecasts vs raw market prices — Brier / LogLoss / MAE on resolved outcomes.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd

from config import config, ensure_data_dirs
from src.backtester import StrategyBacktester

logger = logging.getLogger(__name__)


class ProbabilityModelComparator:
    """
    Compares probability estimation accuracy across:
    - market: traded YES price
    - breeden_litzenberger: model fair_price (RND-calibrated P(YES))
    - ml_hybrid: fair_price adjusted when Isolation Forest flags mispricing
    """

    def __init__(self):
        self.last_report: Dict = {}

    @staticmethod
    def _ml_hybrid_probs(df: pd.DataFrame) -> np.ndarray:
        """
        Hybrid forecast: when anomaly_score is high, trust fair_price more;
        otherwise blend toward market price_yes.
        """
        market = df["price_yes"].to_numpy(dtype=float)
        fair = df["fair_price"].to_numpy(dtype=float)
        score = df.get("anomaly_score", pd.Series(np.zeros(len(df)))).to_numpy(dtype=float)
        weight = np.clip(score, 0.0, 1.0)
        return np.clip((1.0 - weight) * market + weight * fair, 0.001, 0.999)

    def compare(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return a metrics table (one row per model).

        Requires columns: outcome, price_yes, fair_price; anomaly_score optional.
        """
        required = {"outcome", "price_yes", "fair_price"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns for comparison: {missing}")

        y = df["outcome"].to_numpy(dtype=float)
        models = {
            "market": df["price_yes"].to_numpy(dtype=float),
            "breeden_litzenberger": df["fair_price"].to_numpy(dtype=float),
            "ml_hybrid": self._ml_hybrid_probs(df),
        }

        rows = []
        for name, probs in models.items():
            metrics = StrategyBacktester().evaluate_forecast_accuracy(y, probs)
            mae = float(np.mean(np.abs(probs - y)))
            rows.append(
                {
                    "model": name,
                    "brier_score": metrics["brier_score"],
                    "log_loss": metrics["log_loss"],
                    "mae": round(mae, 4),
                    "n_samples": int(len(y)),
                }
            )

        report = pd.DataFrame(rows).sort_values("brier_score").reset_index(drop=True)
        best = report.iloc[0]["model"]
        self.last_report = {
            "best_model": best,
            "metrics": report.to_dict(orient="records"),
            "conclusion": (
                f"Найнижчий Brier Score має модель «{best}» "
                f"({report.iloc[0]['brier_score']:.4f}). "
                "Breeden-Litzenberger дає ризик-нейтральну оцінку P(YES); "
                "ML-hybrid підсилює її при високому anomaly score."
            ),
        }
        return report

    def save_report(self, report_df: pd.DataFrame, filename: str = "bl_vs_ml_comparison.json") -> str:
        ensure_data_dirs()
        csv_path = os.path.join(config.REPORTS_DIR, "metrics", filename.replace(".json", ".csv"))
        json_path = os.path.join(config.REPORTS_DIR, "metrics", filename)
        report_df.to_csv(csv_path, index=False)
        payload = self.last_report or {
            "metrics": report_df.to_dict(orient="records"),
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        logger.info("Saved comparison report: %s", json_path)
        return json_path
