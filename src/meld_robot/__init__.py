"""meld_robot: real-time tri-modal emotion understanding for a companion robot.

Lean build, all local inference, total ~1.72B learned parameters
(budget: 6B). Modules are independent: the notebook orchestrates them.

  config           - paths, model IDs, hyperparameters, budget constants
  data             - MELD download/load/audio extraction (no torch needed)
  text_features    - frozen DistilRoBERTa encodings
  audio_features   - frozen Wav2Vec2 encodings + zero-param prosody evidence
  vision_features  - MediaPipe landmark geometry (evidence + pooled features)
  model            - fusion transformer + absence tokens (trained here)
  train            - featurization cache, training loop, weighted-F1 evaluation
  engine           - streaming turn-based inference API
  responder        - Qwen2.5-1.5B-Instruct grounded response generator
  utils            - seeding, device pick, parameter accounting, latency
"""
__version__ = "0.1.0"
