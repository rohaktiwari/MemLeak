from typing import Iterable, List

import numpy as np
import pandas as pd

from memleak.models.wrapper import ModelWrapper
from memleak.utils.metrics import min_max_normalize


class LossBasedAttack:
    """Membership inference by contrasting loss distributions."""

    def __init__(self, wrapper: ModelWrapper):
        self.wrapper = wrapper

    def _score(self, losses: List[float]) -> np.ndarray:
        normalized = min_max_normalize([-l for l in losses])  # lower loss → higher membership
        return normalized

    def run(self, train_texts: Iterable[str], test_texts: Iterable[str], train_labels=None, test_labels=None) -> pd.DataFrame:
        train = list(train_texts)
        test = list(test_texts)

        is_classification = self.wrapper.model_type == "sequence-classification"
        if is_classification:
            if train_labels is None or test_labels is None:
                raise ValueError(
                    "LossBasedAttack requires both train_labels and test_labels when model_type='sequence-classification'."
                )
            train_losses = self.wrapper.compute_losses(train, labels=list(train_labels))
            test_losses = self.wrapper.compute_losses(test, labels=list(test_labels))
        else:
            # causal LM: labels derived from input_ids internally
            train_losses = self.wrapper.compute_losses(train, labels=None)
            test_losses = self.wrapper.compute_losses(test, labels=None)

        train_scores = self._score(train_losses)
        test_scores = self._score(test_losses)

        df = pd.DataFrame(
            {
                "text": train + test,
                "split": ["train"] * len(train) + ["test"] * len(test),
                "loss": train_losses + test_losses,
                "membership_prob": np.concatenate([train_scores, test_scores]),
                "confidence": np.concatenate([train_scores, test_scores]),
            }
        )
        df["rank"] = df["membership_prob"].rank(ascending=False).astype(int)
        df["attack"] = "loss"
        return df

