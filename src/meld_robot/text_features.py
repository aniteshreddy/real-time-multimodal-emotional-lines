"""Text modality: frozen DistilRoBERTa sentence embeddings (~82M params)."""
from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from . import config


class TextEncoder:
    """Mean-pooled last-hidden-state embeddings from DistilRoBERTa.

    The encoder is frozen in the lean build: only the fusion module is
    trained, which keeps MELD-scale fine-tuning stable and the parameter
    accounting simple.
    """

    def __init__(self, model_id: str = config.TEXT_ENCODER_ID, device: torch.device | None = None):
        self.device = device or torch.device("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """(N, 768) mean-pooled embeddings."""
        out = np.zeros((len(texts), config.TEXT_EMB_DIM), dtype=np.float32)
        for i in range(0, len(texts), batch_size):
            batch = [t if t else " " for t in texts[i : i + batch_size]]
            enc = self.tokenizer(
                batch, padding=True, truncation=True, max_length=128, return_tensors="pt"
            ).to(self.device)
            hidden = self.model(**enc).last_hidden_state  # (B, T, H)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1)
            out[i : i + len(batch)] = pooled.cpu().numpy()
        return out

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters())
