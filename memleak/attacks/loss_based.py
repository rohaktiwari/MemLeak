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

    def run(self, train_texts: Iterable[str], test_texts: Iterable[str]) -> pd.DataFrame:
        train = list(train_texts)
        test = list(test_texts)
        train_losses = self.wrapper.compute_losses(train)
        test_losses = self.wrapper.compute_losses(test)

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

