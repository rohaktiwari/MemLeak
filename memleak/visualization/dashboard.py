import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def render_dashboard(df: pd.DataFrame, risk_score: float) -> go.Figure:
    by_attack = df.groupby("attack")["membership_prob"].mean().reset_index()
    by_split = df.groupby("split")["membership_prob"].mean().reset_index()

    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "bar"}, {"type": "indicator"}]])

    fig.add_trace(
        go.Bar(
            x=by_attack["attack"],
            y=by_attack["membership_prob"],
            marker_color="indianred",
            name="Attack mean",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=risk_score,
            number={"suffix": " / 100"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "darkred" if risk_score > 70 else "gold" if risk_score > 40 else "green"},
                "steps": [
                    {"range": [0, 40], "color": "#e8f5e9"},
                    {"range": [40, 70], "color": "#fff9c4"},
                    {"range": [70, 100], "color": "#ffebee"},
                ],
            },
            title={"text": "Privacy Risk Score"},
        ),
        row=1,
        col=2,
    )

    fig.update_layout(title="MemLeak Dashboard", height=380, showlegend=False)
    return fig


def plot_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["attack", "split"]).agg(
        avg_prob=("membership_prob", "mean"),
        avg_conf=("confidence", "mean"),
        count=("text", "count"),
    )

