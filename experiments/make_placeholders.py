from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path("assets")
OUTPUT_DIR.mkdir(exist_ok=True)

RESULTS = [
    ("gpt2", 42.0, 0.62),
    ("bert-base-uncased", 38.0, 0.58),
    ("distilbert-base-uncased", 30.0, 0.55),
    ("gpt2-medium", 50.0, 0.65),
    ("microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", 72.0, 0.70),
]


def make_figs():
    for model, risk, acc in RESULTS:
        slug = model.lower().replace("/", "_").replace("-", "_")

        # Dashboard-like gauge bar
        plt.figure(figsize=(4, 2))
        plt.barh([0], [risk], color="red" if risk > 70 else "orange" if risk > 40 else "green")
        plt.xlim(0, 100)
        plt.title(f"Risk Score ({model}) = {risk}")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{slug}_dashboard.png", dpi=180)
        plt.close()

        # Probabilities scatter
        n_train, n_test = 20, 20
        train = np.clip(np.random.normal(0.6 + risk / 300, 0.08, n_train), 0, 1)
        test = np.clip(np.random.normal(0.3, 0.08, n_test), 0, 1)
        plt.figure(figsize=(5, 3))
        plt.scatter(range(n_train), train, label="train", marker="o", color="blue")
        plt.scatter(range(n_train, n_train + n_test), test, label="test", marker="x", color="orange")
        plt.ylim(0, 1)
        plt.xlabel("sample idx")
        plt.ylabel("membership prob")
        plt.title(f"Membership probabilities — {model}\n(acc={acc:.2f})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{slug}_probabilities.png", dpi=180)
        plt.close()


if __name__ == "__main__":
    make_figs()
    print("Placeholder figures written to assets/")

