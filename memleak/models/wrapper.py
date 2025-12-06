from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from memleak.models.loader import load_model_tokenizer
from memleak.utils.metrics import compute_entropy, compute_margin


class ModelWrapper:
    """Standardizes access to HF models for membership inference."""

    def __init__(self, model_name_or_path: str, device: str | None = None, max_length: int = 256, batch_size: int = 4):
        self.model, self.tokenizer, self.model_type = load_model_tokenizer(model_name_or_path, device=device)
        self.device = self.model.device
        self.max_length = max_length
        self.batch_size = batch_size
        self.model_name = model_name_or_path

    def _tokenize(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def predict_logits(self, texts: List[str]) -> torch.Tensor:
        logits_list = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            inputs = self._tokenize(batch).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            logits = outputs.logits
            if logits.dim() == 3 and self.model_type == "causal-lm":
                # take last token
                logits = logits[:, -1, :]
            logits_list.append(logits.cpu())
        return torch.cat(logits_list, dim=0)

    def compute_losses(self, texts: List[str]) -> List[float]:
        losses: List[float] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            inputs = self._tokenize(batch).to(self.device)
            with torch.no_grad():
                if self.model_type == "sequence-classification":
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    probs = F.softmax(logits, dim=-1)
                    # pseudo-label with model prediction
                    pseudo_labels = probs.argmax(dim=-1)
                    loss = F.cross_entropy(logits, pseudo_labels, reduction="none")
                else:
                    labels = inputs["input_ids"]
                    outputs = self.model(**inputs, labels=labels)
                    # average token-level loss per example
                    loss = outputs.loss.detach()
                    if loss.dim() == 0:
                        loss = loss.repeat(len(batch))
                losses.extend(loss.cpu().tolist())
        return losses

    def compute_features(self, texts: List[str]) -> List[Dict[str, float]]:
        logits = self.predict_logits(texts)
        probs = F.softmax(logits, dim=-1)
        losses = self.compute_losses(texts)
        features = []
        for idx in range(len(texts)):
            prob = probs[idx]
            loss = losses[idx]
            entropy = compute_entropy(prob)
            margin = compute_margin(prob)
            confidence = float(prob.max().item())
            features.append(
                {
                    "loss": float(loss),
                    "entropy": float(entropy),
                    "margin": float(margin),
                    "confidence": confidence,
                }
            )
        return features

    def close(self) -> None:
        del self.model
        torch.cuda.empty_cache()

