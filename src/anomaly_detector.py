"""
Machine Learning Anomaly Detection Module.
Identifies structural mispricing signals using Isolation Forest and feature engineering.
"""

import logging
import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from config import config

logger = logging.getLogger(__name__)


class MispricingAnomalyDetector:
    """
    ML Pipeline for detecting price anomalies in binary prediction contracts.
    Combines quantitative math outputs with order book microstructure features.
    """

    def __init__(self, contamination: float = config.ANOMALY_CONTAMINATION):
        self.contamination = contamination
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=config.RANDOM_STATE,
            n_estimators=100
        )
        self.is_fitted = False

    def build_feature_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Constructs features from market data and mathematical model outputs.
        """
        features = pd.DataFrame()

        # 1. Price divergence (Theoretical Fair Price vs Actual Market Price)
        features["price_delta"] = np.abs(df["price_yes"] - df["fair_price"])

        # 2. Relative spread ratio
        features["spread"] = np.abs(df["price_yes"] - (1.0 - df["price_no"]))

        # 3. Liquidity & Volume indicators (Log-transformed to handle skewness)
        features["log_volume"] = np.log1p(df["volume"])
        features["log_liquidity"] = np.log1p(df["liquidity"])

        # 4. Interaction term: Divergence weighted by liquidity factor
        features["divergence_liquidity_ratio"] = (
            features["price_delta"] * (1.0 + features["log_liquidity"])
        )

        return features.fillna(0.0)

    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fits Isolation Forest on the feature space and appends anomaly scores.
        """
        if df.empty:
            logger.warning("Empty DataFrame passed to anomaly detector.")
            df["is_anomaly"] = False
            df["anomaly_score"] = 0.0
            return df

        X = self.build_feature_matrix(df)
        X_scaled = self.scaler.fit_transform(X)

        # Fit and predict (-1 for anomalies, 1 for inliers)
        predictions = self.model.fit_predict(X_scaled)
        
        # Decision function: lower values mean more anomalous
        raw_scores = self.model.decision_function(X_scaled)

        # Normalize score into a range [0, 1] (where 1 is highly anomalous)
        min_s, max_s = raw_scores.min(), raw_scores.max()
        normalized_scores = (
            1.0 - (raw_scores - min_s) / (max_s - min_s + 1e-8)
        )

        result_df = df.copy()
        result_df["is_anomaly"] = predictions == -1
        result_df["anomaly_score"] = np.round(normalized_scores, 4)
        self.is_fitted = True

        return result_df