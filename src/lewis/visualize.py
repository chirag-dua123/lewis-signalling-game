"""
visualize.py — Plotly chart helpers for the Streamlit dashboard.

All functions return go.Figure objects ready for st.plotly_chart().
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------

def plot_training_curves(metrics_log: list[dict]) -> go.Figure:
    """
    Plot accuracy, loss, and temperature over epochs on a 3-row subplot.

    Parameters
    ----------
    metrics_log : list of dicts with keys epoch, accuracy, loss, temperature
    """
    if not metrics_log:
        return go.Figure()

    epochs = [m["epoch"] for m in metrics_log]
    accs = [m["accuracy"] for m in metrics_log]
    losses = [m["loss"] for m in metrics_log]
    temps = [m["temperature"] for m in metrics_log]

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=("Accuracy", "Loss", "Gumbel-Softmax Temperature"),
        vertical_spacing=0.08,
    )

    fig.add_trace(
        go.Scatter(x=epochs, y=accs, mode="lines", name="Accuracy",
                   line=dict(color="#00b4d8", width=2)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=losses, mode="lines", name="Loss",
                   line=dict(color="#f77f00", width=2)),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=temps, mode="lines", name="Temperature",
                   line=dict(color="#a8dadc", width=2, dash="dot")),
        row=3, col=1,
    )

    fig.update_yaxes(title_text="Accuracy", range=[0, 1], row=1, col=1)
    fig.update_yaxes(title_text="Loss", row=2, col=1)
    fig.update_yaxes(title_text="τ", row=3, col=1)
    fig.update_xaxes(title_text="Epoch", row=3, col=1)

    fig.update_layout(
        height=500,
        showlegend=False,
        template="plotly_dark",
        margin=dict(l=60, r=20, t=50, b=40),
    )
    return fig


def plot_topsim_curve(metrics_log: list[dict]) -> go.Figure:
    """Plot TopSim over epochs (only epochs where TopSim was computed)."""
    ts_points = [m for m in metrics_log if "topsim" in m and m["topsim"] is not None]
    if not ts_points:
        fig = go.Figure()
        fig.update_layout(
            title="TopSim not yet computed",
            template="plotly_dark",
            height=250,
        )
        return fig

    epochs = [m["epoch"] for m in ts_points]
    topsims = [m["topsim"] for m in ts_points]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=epochs, y=topsims, mode="lines+markers",
        name="TopSim",
        line=dict(color="#90e0ef", width=2),
        marker=dict(size=6),
    ))
    fig.add_hrect(y0=0.5, y1=1.0, fillcolor="green", opacity=0.05,
                  annotation_text="Compositional (>0.5)", annotation_position="top left")
    fig.add_hrect(y0=0.2, y1=0.5, fillcolor="yellow", opacity=0.05,
                  annotation_text="Structured (0.2–0.5)", annotation_position="top left")

    fig.update_layout(
        title="Topographic Similarity (TopSim) over Training",
        xaxis_title="Epoch",
        yaxis_title="TopSim (Spearman ρ)",
        yaxis=dict(range=[-0.1, 1.0]),
        template="plotly_dark",
        height=300,
        margin=dict(l=60, r=20, t=50, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# Language analysis
# ---------------------------------------------------------------------------

def plot_symbol_heatmap(freq: np.ndarray) -> go.Figure:
    """
    Heatmap of symbol usage frequency.
    rows = message positions, cols = symbol indices.
    """
    msg_len, vocab_size = freq.shape

    fig = go.Figure(data=go.Heatmap(
        z=freq,
        x=[f"sym {i}" for i in range(vocab_size)],
        y=[f"pos {i}" for i in range(msg_len)],
        colorscale="Blues",
        zmin=0,
        zmax=freq.max() if freq.max() > 0 else 1,
        colorbar=dict(title="Frequency"),
    ))
    fig.update_layout(
        title="Symbol Usage Frequency (per message position)",
        xaxis_title="Symbol",
        yaxis_title="Position",
        template="plotly_dark",
        height=280,
        margin=dict(l=60, r=20, t=50, b=40),
    )
    return fig


def plot_entropy_bars(entropies: np.ndarray, max_entropy: float) -> go.Figure:
    """Bar chart of per-position entropy vs. maximum possible entropy."""
    positions = [f"pos {i}" for i in range(len(entropies))]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=positions,
        y=entropies,
        name="Actual entropy",
        marker_color="#00b4d8",
    ))
    fig.add_hline(
        y=max_entropy,
        line_dash="dot",
        line_color="#f77f00",
        annotation_text=f"Max entropy ({max_entropy:.2f} bits)",
        annotation_position="top right",
    )
    fig.update_layout(
        title="Symbol Entropy per Message Position",
        yaxis_title="Entropy (bits)",
        template="plotly_dark",
        height=260,
        margin=dict(l=60, r=20, t=50, b=40),
        showlegend=False,
    )
    return fig


def plot_message_scatter(rows: list[dict], n_values: int) -> go.Figure:
    """
    Scatter-like table: objects coloured by their first emitted symbol,
    giving a visual sense of how the Sender partitions the object space.
    """
    if not rows:
        return go.Figure()

    labels = [r["object_str"] for r in rows]
    symbols = [r["first_symbol"] for r in rows]
    messages = [r["message_str"] for r in rows]

    # Use first symbol as color
    fig = px.scatter(
        x=list(range(len(rows))),
        y=[0] * len(rows),
        color=[str(s) for s in symbols],
        hover_name=labels,
        hover_data={"message": messages},
        color_discrete_sequence=px.colors.qualitative.Plotly,
        title="Object Space — Colour = First Symbol Emitted by Sender",
    )
    fig.update_traces(marker=dict(size=14, line=dict(width=1, color="white")))
    fig.update_yaxes(visible=False)
    fig.update_xaxes(title_text="Object index")
    fig.update_layout(
        template="plotly_dark",
        height=260,
        margin=dict(l=60, r=20, t=50, b=40),
        legend_title="First symbol",
    )
    return fig
