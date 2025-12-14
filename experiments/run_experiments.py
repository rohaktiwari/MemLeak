import os
from pathlib import Path
from typing import Dict, List

import pandas as pd

from memleak import MembershipTester


OUTPUT_DIR = Path("assets")
OUTPUT_DIR.mkdir(exist_ok=True)


def attack_accuracy(df: pd.DataFrame, threshold: float = 0.5) -> float:
    labels = df["split"].apply(lambda x: 1 if x == "train" else 0).values
    preds = (df["membership_prob"].values > threshold).astype(int)
    return float((preds == labels).mean())


def slugify(name: str) -> str:
    return name.lower().replace("/", "_").replace("-", "_")


def run_one(model_id: str, train: List[str], test: List[str], max_length: int = 96, batch_size: int = 1) -> Dict:
    tester = MembershipTester(model=model_id, max_length=max_length, batch_size=batch_size)
    report = tester.run_attacks(train, test)
    figs = report.visualize()

    slug = slugify(model_id)
    figs["dashboard"].write_image(OUTPUT_DIR / f"{slug}_dashboard.png", scale=2)
    figs["probabilities"].write_image(OUTPUT_DIR / f"{slug}_probabilities.png", scale=2)

    metric_df = report.attack_frames["metric"]
    acc = attack_accuracy(metric_df)

    tester.close()
    return {
        "model": model_id,
        "risk_score": report.risk_score,
        "attack_accuracy": acc,
        "dashboard_path": f"assets/{slug}_dashboard.png",
        "prob_path": f"assets/{slug}_probabilities.png",
    }


def main():
    experiments = [
        {
            "model": "gpt2",
            "train": [
                "The patient was diagnosed with a rare condition.",
                "Quantum computing is still in its infancy.",
                "Alice loves to bake sourdough bread on Sundays.",
                "Bob plays guitar on weekends.",
            ],
            "test": [
                "This sentence was not in training.",
                "Large language models can memorize data.",
                "A new sample unseen in training.",
            ],
            "max_length": 64,
        },
        {
            "model": "bert-base-uncased",
            "train": [
                "Privacy-preserving machine learning protects sensitive data.",
                "Neural networks can overfit small datasets.",
                "Membership inference attacks exploit confidence scores.",
            ],
            "test": [
                "This is a new example not seen during training.",
                "Regularization reduces overfitting.",
            ],
            "max_length": 96,
        },
        {
            "model": "distilbert-base-uncased",
            "train": [
                "Short texts may leak less information.",
                "Distilled models trade accuracy for speed.",
            ],
            "test": [
                "Holdout sample for evaluation.",
                "Another unseen sentence.",
            ],
            "max_length": 96,
        },
        {
            "model": "gpt2-medium",
            "train": [
                "Medium models can memorize specific phrases.",
                "Training data sometimes repeats in outputs.",
            ],
            "test": [
                "Unseen phrasing should get lower confidence.",
                "Membership signals can be subtle.",
            ],
            "max_length": 64,
        },
        {
            "model": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
            "train": [
                "Patient presents with mild fever and headache.",
                "Diagnosis: Type II diabetes with HbA1c of 8.1%.",
                "Prescription: amoxicillin 500mg twice daily.",
            ],
            "test": [
                "Follow-up visit scheduled in two weeks.",
                "No past medical history noted.",
            ],
            "max_length": 96,
        },
    ]

    rows = []
    for exp in experiments:
        result = run_one(
            model_id=exp["model"],
            train=exp["train"],
            test=exp["test"],
            max_length=exp.get("max_length", 96),
            batch_size=1,
        )
        rows.append(result)
        print(f"[{result['model']}] risk_score={result['risk_score']} attack_accuracy={result['attack_accuracy']:.2f}")

    df = pd.DataFrame(rows)
    df.to_csv("experiments/results.csv", index=False)
    print("\nSaved results to experiments/results.csv")


if __name__ == "__main__":
    main()

