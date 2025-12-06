from typing import Dict, Iterable, List

import numpy as np
import torch


def compute_entropy(probs: torch.Tensor) -> float:
    probs = probs + 1e-12
    entropy = -torch.sum(probs * torch.log(probs), dim=-1)
    return float(entropy.detach().cpu().item())


def compute_margin(probs: torch.Tensor) -> float:
    top2 = torch.topk(probs, k=min(2, probs.shape[-1]), dim=-1).values
    if top2.shape[-1] == 1:
        return 0.0
    return float((top2[..., 0] - top2[..., 1]).detach().cpu().item())


def min_max_normalize(values: Iterable[float]) -> np.ndarray:
    arr = np.array(list(values), dtype=float)
    if arr.size == 0:
        return arr
    min_v, max_v = arr.min(), arr.max()
    if max_v - min_v < 1e-8:
        return np.zeros_like(arr)
    return (arr - min_v) / (max_v - min_v)


def compute_privacy_risk_score(probabilities: Iterable[float]) -> float:
    arr = np.array(list(probabilities), dtype=float)
    if arr.size == 0:
        return 0.0
    base = np.clip(arr.mean(), 0, 1)
    return round(float(base * 100), 2)


def mitigation_recommendations(risk_score: float, metadata: Dict[str, object]) -> List[str]:
    tips = []
    if risk_score >= 70:
        tips.append("Enable differential privacy: target ε in [1, 5] with clipping and noise multiplier tuned to validation utility.")
        tips.append("Increase dropout to 0.2–0.5 and weight decay to >=0.01 to reduce memorization.")
        tips.append("Augment data with paraphrasing or back-translation for sensitive samples.")
    elif risk_score >= 40:
        tips.append("Consider moderate DP training: target ε in [5, 8] with gradient clipping.")
        tips.append("Apply mixup or word dropout on text inputs.")
    else:
        tips.append("Risk appears low; monitor during further training and consider DP-SGD for high-sensitivity domains.")

    tips.append(f"Model analyzed: {metadata.get('model_name', 'unknown')}; train size={metadata.get('train_size')}, test size={metadata.get('test_size')}.")
    return tips

