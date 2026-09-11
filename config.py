"""
Centralized Configuration Module.
Defines environment settings, API endpoints, mathematical constants, and ML parameters.
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class SystemConfig:
    # API Settings
    GAMMA_API_URL: str = "https://gamma-api.polymarket.com/events"
    CLOB_API_URL: str = "https://clob.polymarket.com/book"
    HTTP_TIMEOUT: int = 10  # seconds

    # Mathematical & Quantitative Constants
    RISK_FREE_RATE: float = 0.05  # 5% annual risk-free rate (r)
    DEFAULT_TENOR: float = 0.25   # Time to expiration in years (T)
    MIN_STRIKES_COUNT: int = 3    # Minimum points required for spline fitting

    # Market Filtering Constraints
    MIN_LIQUIDITY_USD: float = 500.0
    MAX_SPREAD_RATIO: float = 0.15  # 15% maximum bid-ask spread

    # Machine Learning Parameters
    ANOMALY_CONTAMINATION: float = 0.05  # Expected percentage of anomalies
    RANDOM_STATE: int = 42

    # System Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")


config = SystemConfig()