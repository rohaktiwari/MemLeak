from typing import Iterable, List

import numpy as np
import pandas as pd

from memleak.models.wrapper import ModelWrapper
from memleak.utils.metrics import compute_privacy_risk_score, min_max_normalize


class MetricBasedAttack:
    """Combines loss, entropy, margin, and confidence into one scorer."""

    def __init__(self, wrapper: ModelWrapper):
        self.wrapper = wrapper

    def _combine(self, features: List[dict]) -> np.ndarray:
        losses = np.array([f["loss"] for f in features], dtype=float)
        entropies = np.array([f["entropy"] for f in features], dtype=float)
        margins = np.array([f["margin"] for f in features], dtype=float)
        confidences = np.array([f["confidence"] for f in features], dtype=float)

        loss_n = min_max_normalize(-losses)  # lower loss → higher membership
        ent_n = 1 - min_max_normalize(entropies)
        margin_n = min_max_normalize(margins)
        conf_n = min_max_normalize(confidences)

        combined = 0.4 * loss_n + 0.2 * conf_n + 0.2 * margin_n + 0.2 * ent_n
        return combined

    def run(self, train_texts: Iterable[str], test_texts: Iterable[str], train_labels=None, test_labels=None) -> pd.DataFrame:
        train = list(train_texts)
        test = list(test_texts)
        train_labels = list(train_labels) if train_labels is not None else None
        test_labels = list(test_labels) if test_labels is not None else None

        combined_texts = train + test
        combined_labels = None
        if train_labels is not None and test_labels is not None:
            combined_labels = train_labels + test_labels

        logits = self.wrapper.predict_logits(combined_texts)
        losses = self.wrapper.compute_losses(combined_texts, labels=combined_labels) if combined_labels is not None else None
        all_feats = self.wrapper.compute_features(
            combined_texts,
            labels=combined_labels,
            logits=logits,
            losses=losses,
        )

        combined_scores = self._combine(all_feats)
        train_scores = combined_scores[: len(train)]
        test_scores = combined_scores[len(train) :]

        df = pd.DataFrame(
            {
                "text": train + test,
                "split": ["train"] * len(train) + ["test"] * len(test),
                "loss": [f["loss"] for f in all_feats],
                "confidence": [f["confidence"] for f in all_feats],
                "entropy": [f["entropy"] for f in all_feats],
                "margin": [f["margin"] for f in all_feats],
                "membership_prob": combined_scores,
            }
        )
        df["rank"] = df["membership_prob"].rank(ascending=False).astype(int)
        df["attack"] = "metric"
        return df

