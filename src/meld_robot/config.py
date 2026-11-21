"""Central configuration for the MELD tri-modal prototype (lean build).

All paths, model IDs, and hyperparameters live here so the notebook and every
module share one source of truth. Assumptions are stated inline where they
affect the challenge constraints (parameter budget, real-time targets).
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "MELD.Raw"          # extracted MELD.Raw.tar.gz
PROCESSED_DIR = DATA_DIR / "processed"   # extracted 16 kHz mono wavs
FEATURE_CACHE = DATA_DIR / "feature_cache"  # cached embeddings per split
MODELS_DIR = DATA_DIR / "models"         # MediaPipe task file etc.
ARTIFACTS_DIR = ROOT / "artifacts"       # checkpoints, reports, plots
CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoints"

for _p in (DATA_DIR, PROCESSED_DIR, FEATURE_CACHE, MODELS_DIR, ARTIFACTS_DIR, CHECKPOINT_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# MELD dataset
# ---------------------------------------------------------------------------
MELD_HF_REPO = "declare-lab/MELD"        # official mirror listed on affective-meld.github.io
MELD_RAW_ARCHIVE = "MELD.Raw.tar.gz"     # ~10.9 GB: CSVs + mp4 clips

SPLITS = ("train", "dev", "test")

# CSV file names inside MELD.Raw (made robust in data.load_split via glob)
SPLIT_CSV_PATTERNS = {
    "train": "train_sent_emo.csv",
    "dev": "dev_sent_emo.csv",
    "test": "test_sent_emo.csv",
}
# mp4 clip directories inside MELD.Raw
SPLIT_CLIP_DIRS = {
    "train": RAW_DIR / "train" / "train_splits",
    "dev": RAW_DIR / "dev" / "dev_splits_complete",
    "test": RAW_DIR / "test" / "output_repeated_splits_test",
}

EMOTIONS = ["neutral", "joy", "surprise", "anger", "sadness", "disgust", "fear"]
SENTIMENTS = ["neutral", "positive", "negative"]
EMOTION_TO_IDX = {e: i for i, e in enumerate(EMOTIONS)}

# ---------------------------------------------------------------------------
# Pretrained components (each one counts toward the 6B budget)
# ---------------------------------------------------------------------------
TEXT_ENCODER_ID = "distilroberta-base"          # ~82M params
TEXT_EMB_DIM = 768
AUDIO_ENCODER_ID = "facebook/wav2vec2-base"     # ~94M params
AUDIO_EMB_DIM = 768
RESPONDER_ID = "Qwen/Qwen2.5-1.5B-Instruct"     # ~1.54B params

# MediaPipe FaceLandmarker task file (fp16, ~3.7 MB). Upper-bound estimate of
# learned parameters, counted in full toward the budget. See README.
FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
FACE_LANDMARKER_PARAMS = 2_000_000  # conservative upper bound
FACE_LANDMARKER_PATH = MODELS_DIR / "face_landmarker.task"

# ---------------------------------------------------------------------------
# Handcrafted feature dimensions (zero learned parameters)
# ---------------------------------------------------------------------------
# Per-frame geometric landmarks -> pooled stats. 12 dims x {mean, std, max} + face-presence
VISION_FEAT_DIM = 12 * 3 + 1
PROSODY_FEAT_DIM = 8  # librosa: duration, rms, f0, spectral stats
MAX_VISION_FRAMES = 8  # frames sampled per clip

# ---------------------------------------------------------------------------
# Fusion model (the only component we train from scratch)
# ---------------------------------------------------------------------------
FUSION_DIM = 256
FUSION_LAYERS = 2
FUSION_HEADS = 4
FUSION_FFN = 512
DROPOUT = 0.1
MODALITY_DROPOUT = 0.20  # fraction of training samples with a modality dropped
NUM_MODALITIES = 4  # text, audio, vision, prosody

LR = 5e-4
WEIGHT_DECAY = 0.01
BATCH_SIZE = 64
MAX_EPOCHS = 40
PATIENCE = 6

# ---------------------------------------------------------------------------
# Responder generation
# ---------------------------------------------------------------------------
MAX_NEW_TOKENS = 90
RESPONSE_TEMPERATURE = 0.7

# ---------------------------------------------------------------------------
# Real-time definition (stated in the README and demo)
# Interaction is turn-based. Targets are p95, measured per utterance end.
# ---------------------------------------------------------------------------
TARGET_TAG_LATENCY_MS = 500
TARGET_RESPONSE_LATENCY_MS = 2000

# ---------------------------------------------------------------------------
# Hard budget from the assignment: 6e9 total learned parameters, all components
# ---------------------------------------------------------------------------
PARAM_BUDGET = 6_000_000_000

SEED = 42
