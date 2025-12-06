from typing import Iterable, List

import numpy as np
import pandas as pd
import torch.nn.functional as F

from memleak.models.wrapper import ModelWrapper
from memleak.utils.metrics import min_max_normalize


class ConfidenceBasedAttack:
    """Membership inference via maximum softmax probability."""

    def __init__(self, wrapper: ModelWrapper):
        self.wrapper = wrapper

    def run(self, train_texts: Iterable[str], test_texts: Iterable[str]) -> pd.DataFrame:
        train = list(train_texts)
        test = list(test_texts)

        logits_train = self.wrapper.predict_logits(train)
        logits_test = self.wrapper.predict_logits(test)

        conf_train = F.softmax(logits_train, dim=-1).max(dim=-1).values.cpu().numpy()
        conf_test = F.softmax(logits_test, dim=-1).max(dim=-1).values.cpu().numpy()

        scores = min_max_normalize(np.concatenate([conf_train, conf_test]))
        train_scores = scores[: len(train)]
        test_scores = scores[len(train) :]

        df = pd.DataFrame(
            {
                "text": train + test,
                "split": ["train"] * len(train) + ["test"] * len(test),
                "confidence": np.concatenate([conf_train, conf_test]),
                "membership_prob": np.concatenate([train_scores, test_scores]),
            }
        )
        df["loss"] = np.nan
        df["rank"] = df["membership_prob"].rank(ascending=False).astype(int)
        df["attack"] = "confidence"
        return df

