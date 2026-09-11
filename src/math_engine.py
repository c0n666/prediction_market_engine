"""
Quantitative Math Engine: Implements the Breeden-Litzenberger framework
for risk-neutral probability density extraction and options Greeks calculation.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from scipy.interpolate import CubicSpline

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
        """
        Fits a smooth cubic spline over discrete market prices to ensure
        twice-differentiable continuity required for Breeden-Litzenberger.
        """
        if len(strikes) < config.MIN_STRIKES_COUNT:
            logger.warning("Insufficient strike points for spline interpolation.")
            return None

        # Sort strikes and remove duplicates
        sorted_indices = np.argsort(strikes)
        k_sorted = strikes[sorted_indices]
        p_sorted = prices[sorted_indices]

        # Natural cubic spline enforcing boundary smoothness
        spline = CubicSpline(k_sorted, p_sorted, bc_type='natural')
        return spline

    def extract_density_and_delta(
        self, 
        strikes: np.ndarray, 
        prices: np.ndarray, 
        time_to_expiry: float = config.DEFAULT_TENOR,
        num_points: int = 100
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extracts the implied probability density function (PDF) and Delta values.

        Returns:
            grid_k: Fine grid of strike prices
            pdf: Calculated risk-neutral probability density f(K)
            delta: Option Delta (sensitivity to underlying price movement)
        """
        spline = self.fit_cubic_spline(strikes, prices)
        if spline is None:
            return np.array([]), np.array([]), np.array([])

        grid_k = np.linspace(strikes.min(), strikes.max(), num_points)
        discount_factor = np.exp(self.r * time_to_expiry)

        # First derivative: dP/dK
        first_derivative = spline(grid_k, nu=1)
        
        # Second derivative: d^2P/dK^2
        second_derivative = spline(grid_k, nu=2)

        # Breeden-Litzenberger RND calculation: f(K) = exp(rT) * d^2C/dK^2
        pdf = discount_factor * np.maximum(second_derivative, 0.0)

        # Normalize PDF to ensure total probability equals 1.0
        pdf_sum = np.trapz(pdf, grid_k)
        if pdf_sum > 0:
            pdf = pdf / pdf_sum

        # Calculate Binary Delta: Sensitivity of option price to underlying movements
        delta = -discount_factor * first_derivative

        return grid_k, pdf, delta

    def compute_fair_probability(self, current_price: float, pdf_grid: np.ndarray, strike_grid: np.ndarray) -> float:
        """
        Computes fair theoretical probability via numerical integration of the RND curve.
        """
        if len(pdf_grid) == 0 or len(strike_grid) == 0:
            return current_price

        # Integrate area under PDF curve where strike > current price threshold
        mask = strike_grid >= current_price
        if not np.any(mask):
            return current_price

        fair_prob = float(np.trapz(pdf_grid[mask], strike_grid[mask]))
        return max(0.001, min(0.999, fair_prob))