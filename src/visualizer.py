import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from typing import Dict, Any

plt.style.use('dark_background')

class QuantVisualizer:
    def __init__(self, output_dir: str = "reports/charts"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_backtest_dashboard(self, backtest_results: Dict[str, Any], filename: str = "backtest_performance.png") -> str:

        equity_curve = np.array(backtest_results.get("equity_curve", []))
        trades_df = backtest_results.get("trades_detail", pd.DataFrame())

        if len(equity_curve) == 0:
            return ""

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={'height_ratios': [2, 1]}, dpi=200)

        # 1. Equity Curve & Drawdown Area
        trades_idx = np.arange(len(equity_curve))
        ax1.plot(trades_idx, equity_curve, color='#00e5ff', linewidth=2.2, 
                 label=f"Portfolio Equity (Initial: ${backtest_results.get('initial_capital', 10000):,.0f})")
        
        cummax = np.maximum.accumulate(equity_curve)
        ax1.plot(trades_idx, cummax, color='#00ffaa', linestyle='--', linewidth=1.0, alpha=0.7, label="High Water Mark")
        ax1.fill_between(trades_idx, cummax, equity_curve, color='#ff0055', alpha=0.25, label="Drawdown Area")

        ax1.set_title(
            f"Strategy Backtest Performance | ROI: {backtest_results.get('roi_percent', 0)}% | "
            f"Sharpe: {backtest_results.get('sharpe_ratio', 0)} | Win Rate: {backtest_results.get('win_rate_percent', 0)}%", 
            fontsize=12, fontweight='bold', color='#ffffff', pad=12
        )
        ax1.set_ylabel("Account Equity ($)", fontsize=10, color='#a0aec0')
        ax1.legend(loc='upper left', fontsize=8.5, framealpha=0.4, facecolor='#1a202c')
        ax1.grid(True, linestyle=':', alpha=0.2, color='#4a5568')

        # 2. Individual Trade PnL Waterfall
        if not trades_df.empty and "pnl" in trades_df.columns:
            pnl_series = trades_df["pnl"].values
            colors = ['#00ffaa' if p > 0 else '#ff0055' for p in pnl_series]
            ax2.bar(np.arange(1, len(pnl_series) + 1), pnl_series, color=colors, alpha=0.85, width=0.7)
            ax2.axhline(0, color='#ffffff', linestyle='-', linewidth=0.8, alpha=0.5)
            ax2.set_ylabel("Trade PnL ($)", fontsize=10, color='#a0aec0')
            ax2.set_xlabel("Trade Sequence Number", fontsize=10, color='#a0aec0')
            ax2.grid(True, linestyle=':', alpha=0.2, color='#4a5568')

        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=200, bbox_inches='tight', facecolor='#0b0f19')
        plt.close()
        return filepath

    def plot_breeden_litzenberger_rnd(
        self, grid_k: np.ndarray, pdf_rnd: np.ndarray, spot_price: float = 100.0, filename: str = "breeden_litzenberger_rnd.png"
    ) -> str:
        """Візуалізація ризиково-нейтральної щільності ймовірності (RND)"""
        fig, ax = plt.subplots(figsize=(9, 5), dpi=200)

        bs_pdf = stats.norm.pdf(grid_k, loc=spot_price, scale=spot_price * 0.12)
        ax.plot(grid_k, bs_pdf, color='#00ffaa', linestyle='--', linewidth=1.5, label='Black-Scholes Benchmark')
        ax.plot(grid_k, pdf_rnd, color='#00e5ff', linewidth=2.2, label='Breeden-Litzenberger Implied RND')
        ax.fill_between(grid_k, 0, pdf_rnd, color='#00e5ff', alpha=0.18)
        ax.axvline(x=spot_price, color='#ff007f', linestyle=':', linewidth=1.2, label=f'Current Spot (${spot_price:.2f})')

        ax.set_title("Implied Risk-Neutral Probability Density q(K)", fontsize=12, fontweight='bold', color='#ffffff', pad=12)
        ax.set_xlabel("Strike Price K ($)", fontsize=10, color='#a0aec0')
        ax.set_ylabel("Probability Density", fontsize=10, color='#a0aec0')
        ax.legend(loc='upper right', fontsize=8.5, framealpha=0.4, facecolor='#1a202c')
        ax.grid(True, linestyle=':', alpha=0.2, color='#4a5568')

        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=200, bbox_inches='tight', facecolor='#0b0f19')
        plt.close()
        return filepath

    def plot_anomaly_detection(self, df: pd.DataFrame, filename: str = "mispricing_anomalies.png") -> str:
        """Візуалізація розходження цін та знайдених ML-аномалій"""
        fig, ax = plt.subplots(figsize=(10, 5), dpi=200)

        indices = np.arange(len(df))
        ax.plot(indices, df["price_yes"], color='#38bdf8', marker='o', markersize=4, linestyle='-', linewidth=1.2, label='Market Price', alpha=0.8)
        ax.plot(indices, df["fair_price"], color='#00ffaa', marker='s', markersize=4, linestyle='--', linewidth=1.2, label='BS Fair Model Price', alpha=0.8)

        anomalies = df[df["is_anomaly"] == True] if "is_anomaly" in df.columns else pd.DataFrame()
        if not anomalies.empty:
            ax.scatter(anomalies.index.values, anomalies["price_yes"], color='#ff0055', s=90, zorder=5, label='ML Isolation Forest Anomaly', edgecolor='#ffffff')

        ax.set_title("Market Mispricing & Unsupervised Anomaly Detection", fontsize=12, fontweight='bold', color='#ffffff', pad=12)
        ax.set_xlabel("Contract Index", fontsize=10, color='#a0aec0')
        ax.set_ylabel("Probability / Price ($)", fontsize=10, color='#a0aec0')
        ax.legend(loc='upper left', fontsize=8.5, framealpha=0.4, facecolor='#1a202c')
        ax.grid(True, linestyle=':', alpha=0.2, color='#4a5568')

        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=200, bbox_inches='tight', facecolor='#0b0f19')
        plt.close()
        return filepath