from typing import Dict, Iterable, List, Optional, Tuple

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

    def compute_losses(self, texts: List[str], labels: Optional[List[int]] = None) -> List[float]:
        """Compute per-sample losses. For classification, true labels are required."""
        losses: List[float] = []
        if self.model_type == "sequence-classification" and labels is None:
            raise ValueError("compute_losses requires labels for classification models.")

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            inputs = self._tokenize(batch).to(self.device)
            with torch.no_grad():
                if self.model_type == "sequence-classification":
                    batch_labels = torch.tensor(labels[i : i + len(batch)], device=self.device)
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    loss = F.cross_entropy(logits, batch_labels, reduction="none")
                else:
                    input_ids = inputs["input_ids"]
                    # next-token prediction: shift inputs by one
                    labels_tokens = input_ids[:, 1:].contiguous()
                    logits = self.model(**inputs).logits[:, :-1, :].contiguous()
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        labels_tokens.view(-1),
                        reduction="none",
                    )
                    # reshape back to per-example mean loss
                    loss = loss.view(logits.size(0), -1).mean(dim=1)
            losses.extend(loss.cpu().tolist())
        return losses

    def compute_features(
        self,
        texts: List[str],
        labels: Optional[List[int]] = None,
        logits: Optional[torch.Tensor] = None,
        losses: Optional[List[float]] = None,
    ) -> List[Dict[str, float]]:
        logits = logits if logits is not None else self.predict_logits(texts)
        probs = F.softmax(logits, dim=-1)
        losses = losses if losses is not None else self.compute_losses(texts, labels=labels)
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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

