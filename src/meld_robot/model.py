"""Fusion model: the only component trained from scratch (~2M params).

Four modality tokens (text, audio, vision, prosody) are projected to a shared
dimension, augmented with type embeddings, passed through a small
transformer, mean-pooled, and classified over the 7 MELD emotions.

Missing modalities are handled with learned absence tokens; during training,
whole modalities are randomly dropped (modality dropout) so the model is
robust when a modality is unavailable at inference time.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from . import config

MODALITY_NAMES = ("text", "audio", "vision", "prosody")


class FusionModel(nn.Module):
    def __init__(
        self,
        n_classes: int = len(config.EMOTIONS),
        dim: int = config.FUSION_DIM,
        n_layers: int = config.FUSION_LAYERS,
        n_heads: int = config.FUSION_HEADS,
        ffn: int = config.FUSION_FFN,
        dropout: float = config.DROPOUT,
    ):
        super().__init__()
        self.n_modalities = len(MODALITY_NAMES)
        self.dim = dim

        self.proj = nn.ModuleList(
            [
                nn.Linear(config.TEXT_EMB_DIM, dim),     # text
                nn.Linear(config.AUDIO_EMB_DIM, dim),    # audio
                nn.Linear(config.VISION_FEAT_DIM, dim),  # vision
                nn.Linear(config.PROSODY_FEAT_DIM, dim),  # prosody
            ]
        )
        self.type_embed = nn.Embedding(self.n_modalities, dim)
        self.absent_token = nn.Parameter(torch.zeros(self.n_modalities, dim))
        nn.init.normal_(self.absent_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=ffn,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(dim), nn.Dropout(dropout), nn.Linear(dim, n_classes)
        )

    def forward(
        self,
        text: torch.Tensor,       # (B, 768)
        audio: torch.Tensor,      # (B, 768)
        vision: torch.Tensor,     # (B, 37)
        prosody: torch.Tensor,    # (B, 8)
        present: torch.Tensor,     # (B, 4) bool
    ) -> torch.Tensor:
        B = text.shape[0]
        feats = [text, audio, vision, prosody]
        tokens = torch.stack(
            [self.proj[i](feats[i]) for i in range(self.n_modalities)], dim=1
        )  # (B, 4, D)
        tokens = tokens + self.type_embed.weight.unsqueeze(0)

        absent = self.absent_token.unsqueeze(0).expand(B, -1, -1)  # (B, 4, D)
        mask = (~present).unsqueeze(-1)  # True where missing
        tokens = torch.where(mask, absent, tokens)

        h = self.encoder(tokens)  # (B, 4, D)
        pooled = h.mean(dim=1)
        return self.head(pooled)

    @torch.no_grad()
    def predict_probs(
        self, text, audio, vision, prosody, present
    ) -> torch.Tensor:
        return torch.softmax(self.forward(text, audio, vision, prosody, present), dim=-1)


def apply_modality_dropout(
    present: torch.Tensor, p: float = config.MODALITY_DROPOUT
) -> torch.Tensor:
    """Randomly drop whole modalities per sample (training robustness).

    Always keeps at least one modality present per sample.
    """
    if p <= 0:
        return present
    drop = torch.bernoulli(torch.full_like(present, p, dtype=torch.float)).bool()
    dropped = present & ~drop
    all_missing = ~dropped.any(dim=-1)
    if all_missing.any():
        # restore one random modality for those samples: keep text if dropped rand
        restore_idx = torch.randint(0, present.shape[1], (int(all_missing.sum()),))
        dropped[all_missing] = present[all_missing].scatter(
            1, restore_idx.unsqueeze(1), True
        )
    return dropped
