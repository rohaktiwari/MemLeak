from typing import Dict

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_probabilities(df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        df,
        x=df.index,
        y="membership_prob",
        color="split",
        symbol="attack",
        hover_data=["text", "loss", "confidence"],
        title="Membership probability per sample",
    )
    fig.update_layout(height=450)
    return fig


def plot_heatmap(df: pd.DataFrame) -> go.Figure:
    pivot = df.pivot_table(values="membership_prob", index="attack", columns="split", aggfunc="mean")
    fig = px.imshow(
        pivot,
        text_auto=".2f",
        aspect="auto",
        title="Average membership probability by attack/split",
        color_continuous_scale="OrRd",
        labels={"color": "Probability"},
    )
    return fig


def plot_attack_curves(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for attack, group in df.groupby("attack"):
        sorted_scores = sorted(group["membership_prob"].tolist(), reverse=True)
        fig.add_trace(go.Scatter(y=sorted_scores, mode="lines", name=attack))
    fig.update_layout(
        title="Attack score curves (higher is more member-like)",
        xaxis_title="Sorted sample index",
        yaxis_title="Membership probability",
        height=400,
    )
    return fig

