from typing import Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


def load_model_tokenizer(model_name_or_path: str, device: str | None = None) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase, str]:
    """Load a HF model + tokenizer, preferring sequence classification heads."""
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = None
    model_type = "sequence-classification"

    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        model_type = "causal-lm"

    # Ensure padding tokens exist for causal LM models (e.g., GPT-2).
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if model_type == "causal-lm":
        tokenizer.padding_side = "left"

    torch_device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(torch_device)
    model.eval()
    return model, tokenizer, model_type

