# MELD Robot: Real-Time Multimodal Emotion Understanding

A focused end-to-end prototype for an emotion-aware companion robot, built on
[MELD](https://affective-meld.github.io/). It fuses **text, audio and vision**
into a MELD emotion tag plus evidence, and generates a short grounded reply,
turn by turn, in real time. **All inference is local.**

**Track:** tri-modal (text + audio + vision), i.e. the optional standout
extension is the core design; the notebook also demonstrates graceful
degradation when modalities are missing.

## Architecture

```
            text ──► DistilRoBERTa (frozen, 82M) ──────────────┐
                                                              │
 utterance  audio ──► Wav2Vec2-base (frozen, 94M) ────────────┤
 streamed           + prosody vector (librosa, 0 params) ───┤
 over time                                                   ├─► 2-layer fusion transformer
            video ──► MediaPipe landmarks (~2M est.) ───────┘    + absence tokens (trained here, ~1.5M)
                      + 12-dim face geometry (0 params)           │
                                                                  ├─► 7-way emotion head ─► structured tags + evidence
                                                                  └───────────────┐
                                                        Qwen2.5-1.5B-Instruct (1.54B) ◄─ prompt = tags + evidence
                                                                       └─► streamed reply text
```

## Quickstart

Requirements: Python 3.11 (pinned), [uv](https://docs.astral.sh/uv/), `ffmpeg` on PATH,
a CUDA GPU (>= 8 GB VRAM recommended; runs on smaller GPUs/CPU but slower).
Disk: ~11 GB for MELD + ~4 GB for model weights.

```bash
# 1. install deps (creates .venv and uv.lock on first run)
uv sync

# 2. run the prototype notebook (data download runs inside it)
uv run jupyter lab notebooks/prototype.ipynb
```

The notebook is the single orchestration surface; all reusable logic lives in
`src/meld_robot/`. Expected first-run costs (single consumer GPU):

| Step | Expected time |
|---|---|
| MELD download (10.9 GB, HF mirror `declare-lab/MELD`) | 10-60 min |
| wav extraction (ffmpeg, idempotent) | ~2 min |
| Featurize all splits (~13K utterances; vision is the bottleneck) | 1-2 h |
| Fusion training (only ~1.5M params trained) | ~5-10 min |
| Real-time demo turn (tags) | ~0.1-0.5 s |

Set `USE_SUBSET = True` in the notebook for a fast smoke pass.

## Parameter budget (hard constraint: 6e9 total, all local inference)

| Component | ~Params | Source |
|---|---|---|
| DistilRoBERTa text encoder (frozen) | 82M | pretrained HF |
| Wav2Vec2-base audio encoder (frozen) | 94M | pretrained HF |
| MediaPipe FaceLandmarker (frozen) | ~2M (upper-bound est.) | pretrained |
| Fusion transformer + emotion head | ~1.5M | **trained here** |
| Qwen2.5-1.5B-Instruct responder | 1.54B | pretrained HF |
| Prosody + face geometry features | 0 | handcrafted |
| **Total** | **~1.72B (29% of budget)** | |

Every learned parameter in the inference path is counted (total, not
trainable). The MediaPipe task file (3.7 MB fp16) has no published parameter
count, so a conservative upper bound is counted in full; this is disclosed
rather than assumed away. No remote inference is used at runtime.

## Definition of real time

Interaction is turn-based. During a user turn the caller streams text chunks,
audio windows and camera frames into the engine (`observe_text` /
`observe_audio` / `observe_frame`). At turn end, `end_turn()` returns the
structured emotion state, targeting **p95 <= 500 ms**; the generated reply
then streams token by token, targeting **p95 <= 2 s** for the full reply.
Per-stage latency is measured on every turn and included in the state dict.

## Repo structure

```
├── pyproject.toml            # uv-managed environment (uv sync installs pinned deps)
├── notebooks/prototype.ipynb  # core orchestration: data → features → train → demo → budget
└── src/meld_robot/
    ├── config.py             # paths, model IDs, hyperparameters, budget constants
    ├── data.py               # MELD download/load/locate + ffmpeg wav extraction (no torch)
    ├── text_features.py      # frozen DistilRoBERTa embeddings
    ├── audio_features.py     # frozen Wav2Vec2 embeddings + 0-param prosody evidence
    ├── vision_features.py    # MediaPipe landmark geometry (evidence + pooled features)
    ├── model.py              # fusion transformer + learned absence tokens (only trained part)
    ├── train.py              # feature caching, modality dropout loop, weighted-F1 eval
    ├── engine.py             # streaming turn-based inference API
    ├── responder.py          # Qwen2.5-1.5B-Instruct grounded reply generator
    └── utils.py              # seeding, device pick, parameter accounting, latency, hardware
```

## Design decisions and trade-offs

1. **Frozen encoders + tiny trained fusion.** MELD has ~10K training
   utterances; fine-tuning 82M+ encoders invites overfitting and complicates
   honest accounting. Cross-modal interaction is learned where it matters, in
   a ~1.5M-parameter fusion transformer.
2. **Modality dropout** (p=0.2) during training: whole modalities are randomly
   replaced by learned absence tokens, so the system degrades gracefully when
   the camera is off or ASR fails.
3. **Interpretable by design.** Prosody and face-geometry features double as
   human-readable evidence strings, so the structured state is auditable:
   you can see *why* the model said "anger".
4. **The responder never sees raw signals**, only the structured state. That
   is what makes "grounded in the multimodal input" enforceable, keeps the
   prompt tiny, and keeps the LLM small (1.5B).

## Limitations and scope (stated honestly)

- MELD audio has canned laughter and background music; models can partially
  learn shortcuts (laughter -> joy) that will not transfer to a robot mic.
- Vision features are expressive but shallow: pooled statistics, no
  fine-grained temporal dynamics.
- ASR is out of scope for the core path (transcripts come with MELD); a live
  microphone demo would add ASR parameters to the budget and is documented
  as the main extension path.
- Live mic/camera capture, VAD endpointing, RL fine-tuning, and fine-grained
  video temporal modeling are intentionally left out; see the notebook's final
  section for the extension map.

## Status

- [x] Tri-modal pipeline: streaming ingest → fusion → tags/evidence → streamed reply
- [x] Trained fusion component with class weighting + modality dropout
- [x] Missing-modality abatements in the demo
- [x] Parameter budget table + hardware report (generated in the notebook)
- [ ] Live microphone/camera capture (VAD, ASR) — planned extension
- [ ] RL on one component — planned extension

## Data and attribution

MELD: Poria et al., "MELD: A Multimodal Multi-Party Dataset for Emotion
Recognition in Conversation", 2018. Downloaded from the official Hugging Face
mirror (`declare-lab/MELD`). Pretrained components: DistilRoBERTa, Wav2Vec2
(META AI), Qwen2.5 (Alibaba, Apache-2.0), MediaPipe (Google, Apache-2.0).
Their outputs are our outputs: we own the combined-system behavior and report
it as measured.
# real-time-multimodal-emotional-lines
