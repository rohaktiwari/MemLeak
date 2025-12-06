from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from memleak.attacks.confidence_based import ConfidenceBasedAttack
from memleak.attacks.loss_based import LossBasedAttack
from memleak.attacks.metric_based import MetricBasedAttack
from memleak.models.wrapper import ModelWrapper
from memleak.utils.metrics import compute_privacy_risk_score, mitigation_recommendations
from memleak.visualization.dashboard import render_dashboard
from memleak.visualization.plots import plot_attack_curves, plot_heatmap, plot_probabilities


@dataclass
class AttackReport:
    """Container for attack outputs and helper visualizations."""

    attack_frames: Dict[str, pd.DataFrame]
    risk_score: float
    metadata: Dict[str, Any]

    @property
    def summary(self) -> pd.DataFrame:
        frames = []
        for name, df in self.attack_frames.items():
            tmp = df.copy()
            tmp["attack"] = name
            frames.append(tmp)
        return pd.concat(frames, ignore_index=True)

    def visualize(self) -> Dict[str, Any]:
        df = self.summary
        figs = {
            "probabilities": plot_probabilities(df),
            "heatmap": plot_heatmap(df),
            "curves": plot_attack_curves(df),
            "dashboard": render_dashboard(df, self.risk_score),
        }
        return figs

    def recommendations(self) -> List[str]:
        return mitigation_recommendations(self.risk_score, self.metadata)


class MembershipTester:
    """High-level API to run membership inference attacks."""

    def __init__(
        self,
        model: str = "bert-base-uncased",
        device: Optional[str] = None,
        max_length: int = 256,
        batch_size: int = 4,
    ) -> None:
        self.wrapper = ModelWrapper(model, device=device, max_length=max_length, batch_size=batch_size)

    def run_attacks(
        self,
        train_data: Iterable[str],
        test_data: Iterable[str],
    ) -> AttackReport:
        train_list = list(train_data)
        test_list = list(test_data)

        loss_attack = LossBasedAttack(self.wrapper)
        conf_attack = ConfidenceBasedAttack(self.wrapper)
        metric_attack = MetricBasedAttack(self.wrapper)

        loss_df = loss_attack.run(train_list, test_list)
        conf_df = conf_attack.run(train_list, test_list)
        metric_df = metric_attack.run(train_list, test_list)

        combined = pd.concat([loss_df, conf_df, metric_df], ignore_index=True)
        risk_score = compute_privacy_risk_score(combined["membership_prob"])
        metadata = {
            "train_size": len(train_list),
            "test_size": len(test_list),
            "model_name": self.wrapper.model_name,
        }
        return AttackReport(
            attack_frames={
                "loss": loss_df,
                "confidence": conf_df,
                "metric": metric_df,
            },
            risk_score=risk_score,
            metadata=metadata,
        )

    def close(self) -> None:
        self.wrapper.close()

