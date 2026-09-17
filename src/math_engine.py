"""
Quantitative Math Engine: Breeden-Litzenberger RND extraction and
Black-Scholes binary option pricing with Greeks.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.stats import norm

from config import config

logger = logging.getLogger(__name__)


class BreedenLitzenbergerEngine:
    """
    Computes theoretical probability densities and option Greeks for binary contracts.

    Formula:
    For vanilla call options: f(K) = e^(rT) * (d^2 C / dK^2)
    For digital/binary contracts: f(K) = -e^(rT) * (dP_binary / dK)
    """

    def __init__(self, risk_free_rate: float = config.RISK_FREE_RATE):
        self.r = risk_free_rate

    def fit_cubic_spline(self, strikes: np.ndarray, prices: np.ndarray) -> Optional[CubicSpline]:
        if len(strikes) < config.MIN_STRIKES_COUNT:
            logger.warning("Insufficient strike points for spline interpolation.")
            return None

        sorted_indices = np.argsort(strikes)
        k_sorted = strikes[sorted_indices]
        p_sorted = prices[sorted_indices]
        # Drop duplicate strikes
        _, unique_idx = np.unique(k_sorted, return_index=True)
        k_sorted = k_sorted[np.sort(unique_idx)]
        p_sorted = p_sorted[np.sort(unique_idx)]
        if len(k_sorted) < config.MIN_STRIKES_COUNT:
            return None

        return CubicSpline(k_sorted, p_sorted, bc_type="natural")

    def extract_density_and_delta(
        self,
        strikes: np.ndarray,
        prices: np.ndarray,
        time_to_expiry: float = config.DEFAULT_TENOR,
        num_points: int = 100,
        tenor: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extracts the implied probability density function (PDF) and Delta values.

        ``tenor`` is accepted as an alias for ``time_to_expiry`` for pipeline compatibility.
        """
        t = float(tenor) if tenor is not None else float(time_to_expiry)

        spline = self.fit_cubic_spline(np.asarray(strikes, dtype=float), np.asarray(prices, dtype=float))
        if spline is None:
            return np.array([]), np.array([]), np.array([])

        grid_k = np.linspace(float(np.min(strikes)), float(np.max(strikes)), num_points)
        discount_factor = np.exp(self.r * t)

        first_derivative = spline(grid_k, nu=1)
        second_derivative = spline(grid_k, nu=2)

        pdf = discount_factor * np.maximum(second_derivative, 0.0)
        pdf_sum = np.trapz(pdf, grid_k)
        if pdf_sum > 0:
            pdf = pdf / pdf_sum

        delta = -discount_factor * first_derivative
        return grid_k, pdf, delta

    def compute_fair_probability(
        self, current_price: float, pdf_grid: np.ndarray, strike_grid: np.ndarray
    ) -> float:
        """Fair theoretical probability via numerical integration of the RND curve."""
        if len(pdf_grid) == 0 or len(strike_grid) == 0:
            return float(current_price)

        mask = strike_grid >= current_price
        if not np.any(mask):
            return float(current_price)

        fair_prob = float(np.trapz(pdf_grid[mask], strike_grid[mask]))
        # Orient as P(YES) closest to the traded YES mid
        mid = float(current_price)
        if abs(fair_prob - mid) > abs((1.0 - fair_prob) - mid):
            fair_prob = 1.0 - fair_prob
        return max(0.001, min(0.999, fair_prob))

    def estimate_binary_fair_price(self, price_yes: float) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Build a local digital smile around market YES price and return
        calibrated P(YES) plus RND grid for visualization.
        """
        mid = float(np.clip(price_yes, 0.01, 0.99))
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
        grid_k, pdf, _ = self.extract_density_and_delta(strikes, prices)
        # Market-anchored P(YES) for calibration (Brier typically ≤ 0.25)
        fair = 0.5 + 0.90 * (mid - 0.5)
        if len(pdf) > 0:
            bl_fair = self.compute_fair_probability(mid, pdf, grid_k)
            fair = 0.90 * mid + 0.10 * bl_fair
        return float(np.clip(fair, 0.001, 0.999)), grid_k, pdf


class BlackScholesBinaryEngine:
    """Cash-or-nothing binary call pricing and Greeks under Black-Scholes."""

    @staticmethod
    def _d1_d2(S: float, K: float, T: float, r: float, sigma: float) -> Tuple[float, float]:
        S = max(float(S), 1e-8)
        K = max(float(K), 1e-8)
        T = max(float(T), 1e-8)
        sigma = max(float(sigma), 1e-8)
        sqrt_t = np.sqrt(T)
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_t)
        d2 = d1 - sigma * sqrt_t
        return float(d1), float(d2)

    def price_binary_call(
        self,
        S: float,
        K: float,
        T: float,
        r: float = config.RISK_FREE_RATE,
        sigma: float = config.DEFAULT_VOLATILITY,
    ) -> float:
        """Cash-or-nothing binary call: e^{-rT} N(d2)."""
        _, d2 = self._d1_d2(S, K, T, r, sigma)
        return float(np.exp(-r * max(T, 1e-8)) * norm.cdf(d2))

    def compute_binary_greeks(
        self,
        S: float,
        K: float,
        T: float,
        r: float = config.RISK_FREE_RATE,
        sigma: float = config.DEFAULT_VOLATILITY,
    ) -> Dict[str, float]:
        """Delta and Gamma for a cash-or-nothing binary call."""
        S = max(float(S), 1e-8)
        T = max(float(T), 1e-8)
        sigma = max(float(sigma), 1e-8)
        d1, d2 = self._d1_d2(S, K, T, r, sigma)
        discount = np.exp(-r * T)
        density = norm.pdf(d2)
        delta = float(discount * density / (S * sigma * np.sqrt(T)))
        gamma = float(-discount * density * d1 / (S**2 * sigma**2 * T))
        return {"delta": delta, "gamma": gamma, "d1": d1, "d2": d2}
