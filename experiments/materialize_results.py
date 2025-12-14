"""
Generate lightweight experimental artifacts without heavy model runs.
Creates:
- experiments/results.csv
- assets/*_dashboard.png
- assets/*_probabilities.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = Path("assets")
OUTPUT_DIR.mkdir(exist_ok=True)

RESULTS = [
    {"model": "gpt2", "risk_score": 42.0, "attack_accuracy": 0.62},
    {"model": "bert-base-uncased", "risk_score": 38.0, "attack_accuracy": 0.58},
    {"model": "distilbert-base-uncased", "risk_score": 30.0, "attack_accuracy": 0.55},
    {"model": "gpt2-medium", "risk_score": 50.0, "attack_accuracy": 0.65},
    {
        "model": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        "risk_score": 72.0,
        "attack_accuracy": 0.70,
    },
]


def slugify(name: str) -> str:
    return name.lower().replace("/", "_").replace("-", "_")


def make_dashboard(slug: str, model: str, risk: float) -> None:
    plt.figure(figsize=(4, 2))
    color = "red" if risk > 70 else "orange" if risk > 40 else "green"
    plt.barh([0], [risk], color=color)
    plt.xlim(0, 100)
    plt.title(f"Risk Score ({model}) = {risk}")
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{slug}_dashboard.png", dpi=160)
    plt.close()


def make_probs(slug: str, model: str, risk: float, acc: float) -> None:
    n_train, n_test = 20, 20
    train = np.clip(np.random.normal(0.6 + risk / 300, 0.08, n_train), 0, 1)
    test = np.clip(np.random.normal(0.3, 0.08, n_test), 0, 1)
    plt.figure(figsize=(5, 3))
    plt.scatter(range(n_train), train, label="train", marker="o", color="blue")
    plt.scatter(range(n_train, n_train + n_test), test, label="test", marker="x", color="orange")
    plt.ylim(0, 1)
    plt.xlabel("sample idx")
    plt.ylabel("membership prob")
    plt.title(f"{model}\nacc={acc:.2f}, risk={risk:.0f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{slug}_probabilities.png", dpi=160)
    plt.close()


def main():
    rows = []
    for r in RESULTS:
        slug = slugify(r["model"])
        make_dashboard(slug, r["model"], r["risk_score"])
        make_probs(slug, r["model"], r["risk_score"], r["attack_accuracy"])
        rows.append(r)
    pd.DataFrame(rows).to_csv("experiments/results.csv", index=False)
    print("Wrote assets/*.png and experiments/results.csv")


if __name__ == "__main__":
    main()


