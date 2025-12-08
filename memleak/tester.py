from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
from sklearn import metrics

from memleak.attacks.confidence_based import ConfidenceBasedAttack
from memleak.attacks.loss_based import LossBasedAttack
from memleak.attacks.metric_based import MetricBasedAttack
from memleak.models.wrapper import ModelWrapper
from memleak.utils.metrics import mitigation_recommendations
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

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _validate_inputs(self, train: Iterable[str], test: Iterable[str]) -> None:
        if not train or not test:
            raise ValueError("Need non-empty train and test sets for membership inference.")
        overlap = set(train) & set(test)
        if overlap:
            warnings.warn(f"Found {len(overlap)} overlapping samples in train and test.", UserWarning)

    def _membership_labels(self, n_train: int, n_test: int) -> np.ndarray:
        return np.concatenate([np.ones(n_train, dtype=int), np.zeros(n_test, dtype=int)])

    def run_attacks(
        self,
        train_data: Iterable[str],
        test_data: Iterable[str],
        train_labels: Optional[Iterable[int]] = None,
        test_labels: Optional[Iterable[int]] = None,
    ) -> AttackReport:
        train_list = list(train_data)
        test_list = list(test_data)
        self._validate_inputs(train_list, test_list)

        loss_attack = LossBasedAttack(self.wrapper)
        conf_attack = ConfidenceBasedAttack(self.wrapper)
        metric_attack = MetricBasedAttack(self.wrapper)

        have_labels = train_labels is not None and test_labels is not None

        loss_df = loss_attack.run(train_list, test_list, train_labels=train_labels, test_labels=test_labels) if have_labels else None
        conf_df = conf_attack.run(train_list, test_list)
        metric_df = metric_attack.run(train_list, test_list, train_labels=train_labels, test_labels=test_labels) if have_labels else None

        attack_frames = {"confidence": conf_df}
        if loss_df is not None:
            attack_frames["loss"] = loss_df
        if metric_df is not None:
            attack_frames["metric"] = metric_df

        combined = pd.concat(list(attack_frames.values()), ignore_index=True)
        membership_labels = self._membership_labels(len(train_list), len(test_list))

        attack_metrics = []
        for name, df in attack_frames.items():
            scores = df["membership_prob"].to_numpy()
            auc = metrics.roc_auc_score(membership_labels, scores) if len(np.unique(membership_labels)) > 1 else 0.5
            preds = (scores >= 0.5).astype(int)
            acc = metrics.accuracy_score(membership_labels, preds)
            attack_metrics.append({"attack": name, "auc": auc, "accuracy": acc})

        # random baseline
        attack_metrics.append({"attack": "random", "auc": 0.5, "accuracy": 0.5})

        best_auc = max(m["auc"] for m in attack_metrics if m["attack"] != "random")
        risk_score = round(float(best_auc * 100), 2)
        metadata = {
            "train_size": len(train_list),
            "test_size": len(test_list),
            "model_name": self.wrapper.model_name,
        }
        report = AttackReport(
            attack_frames=attack_frames,
            risk_score=risk_score,
            metadata=metadata,
        )
        # attach metrics for convenience
        report.metrics = pd.DataFrame(attack_metrics)
        return report

    def close(self) -> None:
        self.wrapper.close()

