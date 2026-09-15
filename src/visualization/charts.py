"""
Plotly chart builders: Risk-Neutral Density, Equity Curve, and Greeks Heatmap.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

from config import config


def plot_risk_neutral_density(
    strikes: np.ndarray,
    pdf: np.ndarray,
    *,
    bid: Optional[float] = None,
    ask: Optional[float] = None,
    spot: Optional[float] = None,
    title: str = "Risk-Neutral Density (Breeden-Litzenberger)",
) -> go.Figure:
    """
    Plot implied RND f(K) with optional bid/ask spread markers and spot line.
    """
    fig = go.Figure()

    if len(strikes) == 0 or len(pdf) == 0:
        fig.update_layout(
            title=title,
            annotations=[
                {
                    "text": "Insufficient strike data for RND extraction",
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.5,
                    "y": 0.5,
                    "showarrow": False,
                }
            ],
        )
        return fig

    fig.add_trace(
        go.Scatter(
            x=strikes,
            y=pdf,
            mode="lines",
            name="f(K)",
            line={"color": "#00bcd4", "width": 2.5},
            fill="tozeroy",
            fillcolor="rgba(0, 188, 212, 0.18)",
            hovertemplate="K=%{x:.4f}<br>f(K)=%{y:.4f}<extra></extra>",
        )
    )

    if spot is not None:
        fig.add_vline(
            x=spot,
            line_dash="dot",
            line_color="#ff6f00",
            annotation_text=f"Spot {spot:.3f}",
            annotation_position="top",
        )

    if bid is not None and ask is not None:
        mid = 0.5 * (bid + ask)
        fig.add_vrect(
            x0=min(bid, ask),
            x1=max(bid, ask),
            fillcolor="rgba(255, 193, 7, 0.18)",
            line_width=0,
            annotation_text=f"Bid/Ask [{bid:.3f}, {ask:.3f}]",
            annotation_position="top left",
        )
        fig.add_vline(x=mid, line_dash="dash", line_color="#ffc107", opacity=0.7)

    fig.update_layout(
        title=title,
        xaxis_title="Strike K",
        yaxis_title="Probability Density f(K)",
        template="plotly_white",
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
        margin={"l": 50, "r": 30, "t": 60, "b": 50},
    )
    return fig


def _max_drawdown_series(equity: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Drawdown vs running peak capital.

    Returns drawdown fractions in [0, 1] and Max DD as percentage in [0, 100].
    """
    if len(equity) == 0:
        return np.array([]), 0.0
    cummax = np.maximum.accumulate(equity)
    with np.errstate(divide="ignore", invalid="ignore"):
        drawdown = np.where(cummax > 0, (cummax - equity) / cummax, 0.0)
    drawdown = np.clip(drawdown, 0.0, 1.0)
    max_dd_pct = float(np.clip(np.max(drawdown) * 100.0, 0.0, 100.0)) if len(drawdown) else 0.0
    return drawdown, max_dd_pct


def plot_equity_curve(
    trades_df: pd.DataFrame,
    *,
    pnl_col: str = "pnl",
    equity_col: str = "equity",
    initial_capital: float = 10_000.0,
    title: str = "Backtest Equity Curve",
) -> go.Figure:
    """
    Equity curve + trade PnL waterfall + max drawdown subplot.

    Prefers full capital column ``equity`` (= initial_capital + cumulative_pnl).
    Falls back to reconstructing equity from cumulative PnL when needed.
    """
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        row_heights=[0.45, 0.3, 0.25],
        subplot_titles=("Portfolio Equity", "Trade PnL Waterfall", "Drawdown (%)"),
    )

    if trades_df is None or trades_df.empty:
        fig.update_layout(
            title=title,
            showlegend=False,
            margin=dict(t=100, l=50, r=40, b=40),
            title_pad=dict(b=20),
        )
        return fig

    df = trades_df.copy()
    if equity_col not in df.columns:
        if "cumulative_pnl" in df.columns:
            df[equity_col] = float(initial_capital) + df["cumulative_pnl"].astype(float)
        elif pnl_col in df.columns:
            df[equity_col] = float(initial_capital) + df[pnl_col].cumsum()
        else:
            fig.update_layout(
                title=title,
                showlegend=False,
                margin=dict(t=100, l=50, r=40, b=40),
                title_pad=dict(b=20),
            )
            return fig

    x = np.arange(len(df))
    equity = df[equity_col].to_numpy(dtype=float)
    pnl = df[pnl_col].to_numpy(dtype=float) if pnl_col in df.columns else np.diff(equity, prepend=equity[0])
    if "drawdown" in df.columns:
        drawdown = np.clip(df["drawdown"].to_numpy(dtype=float), 0.0, 1.0)
        max_dd_pct = float(np.clip(np.max(drawdown) * 100.0, 0.0, 100.0))
    else:
        drawdown, max_dd_pct = _max_drawdown_series(equity)

    fig.add_trace(
        go.Scatter(
            x=x,
            y=equity,
            mode="lines",
            name="Equity",
            line={"color": "#1565c0", "width": 2.2},
            hovertemplate="Trade %{x}<br>Equity=%{y:,.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    cummax = np.maximum.accumulate(equity)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=cummax,
            mode="lines",
            name="High Water Mark",
            line={"color": "#43a047", "width": 1, "dash": "dash"},
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )

    colors = ["#2e7d32" if p >= 0 else "#c62828" for p in pnl]
    fig.add_trace(
        go.Bar(
            x=x,
            y=pnl,
            name="Trade PnL",
            marker_color=colors,
            hovertemplate="Trade %{x}<br>PnL=%{y:.4f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    drawdown_pct = drawdown * 100.0
    fig.add_trace(
        go.Scatter(
            x=x,
            y=-drawdown_pct,
            mode="lines",
            name="Drawdown",
            fill="tozeroy",
            line={"color": "#c62828", "width": 1.5},
            hovertemplate="Trade %{x}<br>DD=%{customdata:.2f}%<extra></extra>",
            customdata=drawdown_pct,
        ),
        row=3,
        col=1,
    )

    fig.update_layout(
        title={
            "text": f"{title} | Max DD: {max_dd_pct:.2f}%",
            "y": 0.98,
            "x": 0.0,
            "xanchor": "left",
            "yanchor": "top",
            "pad": dict(b=20),
        },
        template="plotly_white",
        # Subplot titles already label each panel — hide legend to avoid overlap
        showlegend=False,
        margin=dict(t=100, l=50, r=40, b=40),
        title_pad=dict(b=20),
        height=760,
    )

    # Subplot annotation (title) styling — keep clear of main title
    for annotation in fig.layout.annotations:
        annotation.font.size = 12
        annotation.yshift = 4

    fig.update_yaxes(title_text="Capital ($)", row=1, col=1)
    fig.update_yaxes(title_text="Trade PnL", row=2, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=3, col=1, range=[-100, 5])
    fig.update_xaxes(title_text="Trade #", row=3, col=1)
    return fig


def _binary_call_greeks(
    S: np.ndarray,
    K: float,
    T: np.ndarray,
    r: float,
    sigma: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cash-or-nothing binary call Greeks on a mesh of spot S and tenor T.
    Delta = e^{-rT} n(d2) / (S σ √T)
    Gamma = -e^{-rT} n(d2) d1 / (S² σ² T)
    """
    S = np.maximum(S, 1e-8)
    T = np.maximum(T, 1e-8)
    sqrt_t = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    discount = np.exp(-r * T)
    density = norm.pdf(d2)

    delta = discount * density / (S * sigma * sqrt_t)
    gamma = -discount * density * d1 / (S**2 * sigma**2 * T)
    return delta, gamma


def plot_greeks_heatmap(
    *,
    spot_range: Tuple[float, float] = (50.0, 150.0),
    tenor_range: Tuple[float, float] = (0.05, 1.0),
    strike: float = 100.0,
    rate: float = config.RISK_FREE_RATE,
    volatility: float = 0.20,
    n_spot: int = 40,
    n_tenor: int = 30,
    greek: str = "delta",
    title: Optional[str] = None,
) -> go.Figure:
    """
    Interactive heatmap of binary-option Delta or Gamma vs underlying price and T.
    """
    spots = np.linspace(spot_range[0], spot_range[1], n_spot)
    tenors = np.linspace(tenor_range[0], tenor_range[1], n_tenor)
    S_grid, T_grid = np.meshgrid(spots, tenors)

    delta, gamma = _binary_call_greeks(S_grid, strike, T_grid, rate, volatility)
    greek_key = greek.lower().strip()
    z = delta if greek_key == "delta" else gamma
    label = "Δ (Delta)" if greek_key == "delta" else "Γ (Gamma)"
    chart_title = title or f"Binary Option {label} Heatmap | K={strike:.1f}, σ={volatility:.0%}"

    fig = go.Figure(
        data=go.Heatmap(
            x=spots,
            y=tenors,
            z=z,
            colorscale="RdYlBu_r",
            colorbar={"title": label},
            hovertemplate=(
                "S=%{x:.2f}<br>T=%{y:.3f}y<br>"
                + label
                + "=%{z:.6f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=chart_title,
        xaxis_title="Underlying Price S",
        yaxis_title="Time to Expiry T (years)",
        template="plotly_white",
        margin={"l": 60, "r": 30, "t": 60, "b": 50},
        height=520,
    )
    return fig
