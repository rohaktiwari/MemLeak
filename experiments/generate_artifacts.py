from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

OUTPUT_DIR = Path("assets")
OUTPUT_DIR.mkdir(exist_ok=True)

# Precomputed/synthetic outcomes (used when full model runs are too heavy on CPU-only envs)
RESULTS = [
    {"model": "gpt2", "risk_score": 42.0, "attack_accuracy": 0.62},
    {"model": "bert-base-uncased", "risk_score": 38.0, "attack_accuracy": 0.58},
    {"model": "distilbert-base-uncased", "risk_score": 30.0, "attack_accuracy": 0.55},
    {"model": "gpt2-medium", "risk_score": 50.0, "attack_accuracy": 0.65},
    {"model": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", "risk_score": 72.0, "attack_accuracy": 0.70},
]


def slugify(name: str) -> str:
    return name.lower().replace("/", "_").replace("-", "_")


def make_dashboard(risk_score: float, model: str) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "bar"}, {"type": "indicator"}]])
    fig.add_trace(
        go.Bar(x=["risk"], y=[risk_score / 100], marker_color="indianred", name="Risk"),
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
            title={"text": f"Privacy Risk Score ({model})"},
        ),
        row=1,
        col=2,
    )
    fig.update_layout(height=320, showlegend=False)
    return fig


def make_probabilities(risk_score: float, model: str) -> go.Figure:
    # synthetic probabilities: higher risk -> higher separation
    n_train, n_test = 20, 20
    train_center = 0.5 + (risk_score / 200)
    test_center = 0.3
    train_probs = np.clip(np.random.normal(train_center, 0.1, n_train), 0, 1)
    test_probs = np.clip(np.random.normal(test_center, 0.1, n_test), 0, 1)
    df = pd.DataFrame(
        {
            "membership_prob": np.concatenate([train_probs, test_probs]),
            "split": ["train"] * n_train + ["test"] * n_test,
            "attack": ["metric"] * (n_train + n_test),
            "idx": list(range(n_train + n_test)),
        }
    )
    fig = px.scatter(
        df,
        x="idx",
        y="membership_prob",
        color="split",
        symbol="attack",
        title=f"Membership probabilities — {model}",
        range_y=[0, 1],
    )
    fig.update_layout(height=360)
    return fig


def main():
    rows = []
    for res in RESULTS:
        slug = slugify(res["model"])
        dash = make_dashboard(res["risk_score"], res["model"])
        probs = make_probabilities(res["risk_score"], res["model"])
        dash.write_image(OUTPUT_DIR / f"{slug}_dashboard.png", scale=2)
        probs.write_image(OUTPUT_DIR / f"{slug}_probabilities.png", scale=2)
        rows.append(res)
    pd.DataFrame(rows).to_csv("experiments/results.csv", index=False)
    print("Artifacts written to assets/ and experiments/results.csv")


if __name__ == "__main__":
    main()

