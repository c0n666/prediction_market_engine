"""
Centralized Configuration Module.
Defines environment settings, API endpoints, mathematical constants, and ML parameters.
"""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SystemConfig:
    # API Settings
    GAMMA_API_URL: str = "https://gamma-api.polymarket.com/events"
    CLOB_API_URL: str = "https://clob.polymarket.com/book"
    HTTP_TIMEOUT: int = 10  # seconds

    # Mathematical & Quantitative Constants
    RISK_FREE_RATE: float = 0.05  # 5% annual risk-free rate (r)
    DEFAULT_TENOR: float = 0.25  # Time to expiration in years (T)
    DEFAULT_VOLATILITY: float = 0.20  # Implied vol for binary BS pricing
    MIN_STRIKES_COUNT: int = 3  # Minimum points required for spline fitting

    # Market Filtering Constraints
    MIN_LIQUIDITY_USD: float = 500.0
    MAX_SPREAD_RATIO: float = 0.15  # 15% maximum bid-ask spread

    # Machine Learning Parameters
    ANOMALY_CONTAMINATION: float = 0.05  # Expected percentage of anomalies
    RANDOM_STATE: int = 42

    # Persistence
    DATA_DIR: str = os.getenv("DATA_DIR", "data")
    RAW_DATA_DIR: str = os.path.join(DATA_DIR, "raw")
    PROCESSED_DATA_DIR: str = os.path.join(DATA_DIR, "processed")
    REPORTS_DIR: str = os.getenv("REPORTS_DIR", "reports")

    # Backtest defaults
    INITIAL_CAPITAL: float = 10_000.0
    EXECUTION_FEE: float = 0.002
    ANOMALY_SCORE_THRESHOLD: float = 0.65
    DEFAULT_SIZING_MODE: str = "fixed"  # "fixed" | "percent"
    DEFAULT_STAKE_AMOUNT: float = 200.0  # $200 or 2.0 if percent mode
    DEFAULT_SCALE_WITH_ANOMALY: bool = True
    DEFAULT_SLIPPAGE_PCT: float = 0.5
    DEFAULT_TRADING_FEE_PCT: float = 0.2
    DEFAULT_CROSS_THE_SPREAD: bool = True
    POLYMARKET_PROTOCOL_FEE_PCT: float = 0.0  # taker/protocol default for auto-frictions
    AUTO_IMPACT_COEFF: float = 0.15  # sqrt(participation) market-impact factor

    # System Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")


config = SystemConfig()


def ensure_data_dirs() -> None:
    """Create local data / report directories if missing."""
    for path in (
        config.RAW_DATA_DIR,
        config.PROCESSED_DATA_DIR,
        os.path.join(config.REPORTS_DIR, "charts"),
        os.path.join(config.REPORTS_DIR, "metrics"),
    ):
        Path(path).mkdir(parents=True, exist_ok=True)
