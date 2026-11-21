"""Audio modality: frozen Wav2Vec2 embeddings (~94M params) plus zero-parameter
prosody features (librosa) that double as human-readable evidence."""
from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModel, Wav2Vec2FeatureExtractor

from . import config


class AudioEncoder:
    """Mean-over-time last hidden states from Wav2Vec2-base.

    wav2vec2-base uses group norm (not layer norm), so it does not support
    attention masking reliably: clips are encoded individually.
    """

    def __init__(self, model_id: str = config.AUDIO_ENCODER_ID, device: torch.device | None = None):
        self.device = device or torch.device("cpu")
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def encode(self, waveforms: list[np.ndarray]) -> np.ndarray:
        """waveforms: list of float32 mono arrays at 16 kHz. Returns (N, 768)."""
        out = np.zeros((len(waveforms), config.AUDIO_EMB_DIM), dtype=np.float32)
        for i, wav in enumerate(waveforms):
            if wav.ndim != 1 or len(wav) == 0:
                wav = np.zeros(1600, dtype=np.float32)
            inputs = self.processor(
                wav.astype(np.float32), sampling_rate=16000, return_tensors="pt"
            ).input_values.to(self.device)
            hidden = self.model(inputs).last_hidden_state  # (1, T, H)
            out[i] = hidden.mean(dim=1).squeeze(0).cpu().numpy()
        return out

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters())


# ---------------------------------------------------------------------------
# Handcrafted prosody features (0 learned parameters)
# ---------------------------------------------------------------------------
def prosody_features(wav: np.ndarray, sr: int = 16000) -> np.ndarray:
    """(8,) prosody vector: duration, loudness, pitch and spectral stats."""
    import librosa  # heavy import, deferred

    wav = wav.astype(np.float32)
    if len(wav) == 0:
        return np.zeros(config.PROSODY_FEAT_DIM, dtype=np.float32)
    rms = librosa.feature.rms(y=wav)[0]
    f0 = librosa.yin(wav, fmin=65, fmax=400, sr=sr, frame_length=1024)
    voiced = f0[(f0 > 65) & (rms > 1e-4)]
    centroid = librosa.feature.spectral_centroid(y=wav, sr=sr)[0]
    zcr = librosa.feature.zero_crossing_rate(wav)[0]
    return np.array(
        [
            len(wav) / sr,                      # duration s
            float(rms.mean()),                  # loudness mean
            float(rms.std()),                   # loudness variability
            float(voiced.mean()) if len(voiced) else 0.0,      # pitch mean Hz
            float(voiced.std()) if len(voiced) else 0.0,       # pitch variability
            float(np.ptp(voiced)) if len(voiced) else 0.0,      # pitch range
            float(centroid.mean()),             # spectral centroid
            float(zcr.mean()),                  # noisiness proxy
        ],
        dtype=np.float32,
    )


PROSODY_NAMES = [
    "duration_s", "rms_mean", "rms_std", "f0_mean_hz", "f0_std_hz",
    "f0_range_hz", "centroid_hz", "zcr",
]


def prosody_evidence(p: np.ndarray) -> list[str]:
    """Turn a prosody vector into short human-readable evidence strings.

    Thresholds are coarse heuristics for evidence grounding only; the fusion
    model uses the raw vector, not these strings.
    """
    (dur, rms_m, rms_s, f0_m, f0_s, f0_r, cent, _zcr) = [float(x) for x in p[:8]]
    ev: list[str] = []
    if f0_m > 210:
        ev.append(f"elevated mean pitch ({f0_m:.0f} Hz)")
    elif 0 < f0_m < 130:
        ev.append(f"low mean pitch ({f0_m:.0f} Hz)")
    if f0_m > 0 and f0_s / max(f0_m, 1) > 0.25:
        ev.append(f"strong pitch variation (+/-{f0_s:.0f} Hz)")
    if rms_m > 0.05 or (dur > 0.3 and rms_s > 0.04):
        ev.append(f"raised loudness (rms {rms_m:.3f})")
    if rms_m < 0.015 and rms_m > 0:
        ev.append(f"flat, low-energy delivery (rms {rms_m:.3f})")
    if dur < 1.0:
        ev.append(f"short burst ({dur:.1f}s)")
    return ev or [f"even prosody (pitch {f0_m:.0f} Hz, rms {rms_m:.3f})"]
