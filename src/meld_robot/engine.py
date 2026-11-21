"""Real-time interaction engine.

Turn-based interaction model (the definition of "real-time" for this
prototype, stated in the README):

  * While a user turn is in progress, the caller streams partial text,
    audio chunks, and camera frames into the engine (observe_* methods).
  * When the turn ends (decided by the caller / VAD in a live deployment),
    end_turn() runs the fused emotion prediction and returns structured
    state + evidence. Target: p95 <= 500 ms.
  * If a responder is attached, its reply tokens are streamed afterwards.
    Target: p95 <= 2000 ms for the full reply.

Missing modalities are first-class: each observe_* is optional, and absent
modalities are routed through the fusion model's learned absence token.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from . import config
from .audio_features import prosody_evidence, prosody_features
from .model import MODALITY_NAMES
from .vision_features import vision_evidence


class UtteranceEngine:
    def __init__(self, text_encoder, audio_encoder, face_extractor, fusion_model, responder=None,
                 device: torch.device | None = None):
        self.device = device if device is not None else next(fusion_model.parameters()).device
        self.text_encoder = text_encoder
        self.audio_encoder = audio_encoder
        self.face_extractor = face_extractor
        self.fusion = fusion_model.eval()
        self.responder = responder

    # ------------------------------------------------------------------
    # streaming observation API
    # ------------------------------------------------------------------
    def start_turn(self) -> None:
        """Reset buffers: a new user turn begins."""
        self._text = ""
        self._wav_chunks: list[np.ndarray] = []
        self._frames: list[np.ndarray] = []
        self._t0 = time.perf_counter()

    def observe_text(self, chunk: str) -> None:
        self._text += chunk

    def observe_audio(self, chunk: np.ndarray, sr: int = 16000) -> None:
        """Append a mono float32 audio chunk; resampled to 16 kHz if needed."""
        if sr != 16000:
            import librosa
            chunk = librosa.resample(chunk.astype(np.float32), orig_sr=sr, target_sr=16000)
        self._wav_chunks.append(chunk)

    def observe_frame(self, frame_bgr: np.ndarray) -> None:
        self._frames.append(frame_bgr)

    # ------------------------------------------------------------------
    # turn end: perception -> fusion -> structured state
    # ------------------------------------------------------------------
    def end_turn(self) -> dict:
        """Run the full multimodal pipeline on the buffered turn."""
        from .vision_features import clip_geometry_safe

        timings = {}

        # assemble inputs
        t = time.perf_counter()
        text = self._text.strip() or " "
        wav = (np.concatenate(self._wav_chunks) if self._wav_chunks
               else np.zeros(1600, dtype=np.float32))
        frames = self._frames[-config.MAX_VISION_FRAMES:]
        timings["buffering_ms"] = (time.perf_counter() - t) * 1000

        # features
        t = time.perf_counter()
        text_emb = self.text_encoder.encode([text])
        timings["text_ms"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        audio_emb = (self.audio_encoder.encode([wav])
                     if self._wav_chunks else np.zeros((1, config.AUDIO_EMB_DIM), np.float32))
        timings["audio_ms"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        prosody = prosody_features(wav) if self._wav_chunks else np.zeros(config.PROSODY_FEAT_DIM, np.float32)
        timings["prosody_ms"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        geo = clip_geometry_safe(self.face_extractor, frames) if frames else []
        if geo:
            G = np.stack(geo)
            vis_feat = np.concatenate(
                [G.mean(0), G.std(0) if len(G) > 1 else G[0] * 0, G.max(0),
                 [len(geo) / max(len(frames), 1)]]
            ).astype(np.float32)
            vision_present = True
        else:
            vis_feat = np.zeros(config.VISION_FEAT_DIM, np.float32)
            vision_present = False
        timings["vision_ms"] = (time.perf_counter() - t) * 1000

        # fusion
        present = np.array([[1, 1 if self._wav_chunks else 0, 1 if vision_present else 0,
                             1 if self._wav_chunks else 0]], dtype=bool)
        t = time.perf_counter()
        with torch.no_grad():
            probs = self.fusion.predict_probs(
                torch.from_numpy(text_emb).to(self.device),
                torch.from_numpy(audio_emb).to(self.device),
                torch.from_numpy(vis_feat[None, :]).to(self.device),
                torch.from_numpy(prosody[None, :]).to(self.device),
                torch.from_numpy(present).to(self.device),
            )[0].cpu().numpy()
        timings["fusion_ms"] = (time.perf_counter() - t) * 1000
        timings["total_ms"] = sum(timings.values()) - timings["buffering_ms"]

        order = np.argsort(probs)[::-1]
        top = [(config.EMOTIONS[i], float(probs[i])) for i in order[:3]]
        emotion, confidence = top[0]

        evidence = {
            "text": f'"{text[:120]}"',
            "audio": prosody_evidence(prosody) if self._wav_chunks else ["no audio received"],
            "face": vision_evidence(vis_feat) if vision_present else ["no face detected"],
        }

        state = {
            "emotion": emotion,
            "confidence": round(confidence, 3),
            "top3": [(e, round(p, 3)) for e, p in top],
            "evidence": evidence,
            "modalities_present": {name: bool(present[0][i]) for i, name in enumerate(MODALITY_NAMES)},
            "latency_ms": {k: round(v, 1) for k, v in timings.items()},
        }
        # expose for the responder
        self._last_state = state
        return state

    # ------------------------------------------------------------------
    def respond(self, state: dict):
        """Stream the responder reply tokens for a given state."""
        if self.responder is None:
            raise RuntimeError("No responder attached.")
        yield from self.responder.generate_stream(state)
