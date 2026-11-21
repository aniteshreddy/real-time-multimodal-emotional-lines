"""MELD dataset handling: download, locate, load, and audio extraction.

Light pandas logic is importable without torch/transformers so the data
layer can be smoke-tested independently of the GPU stack.
"""
from __future__ import annotations

import shutil
import subprocess
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import config


@dataclass
class Utterance:
    uid: str  # e.g. "dia0_utt0"
    split: str
    text: str
    speaker: str
    emotion: str
    sentiment: str
    video_path: Optional[Path]
    wav_path: Path  # extracted 16 kHz mono wav (created by ensure_audio_wavs)


# ---------------------------------------------------------------------------
# Download / extract
# ---------------------------------------------------------------------------
def download_and_extract_meld(force: bool = False) -> Path:
    """Download MELD.Raw.tar.gz (~10.9 GB) from the official Hugging Face
    mirror and extract it into data/MELD.Raw. Skips if already present."""
    if (config.RAW_DIR / "train").exists() and not force:
        print(f"MELD already extracted at {config.RAW_DIR}, skipping download.")
        return config.RAW_DIR

    from huggingface_hub import hf_hub_download  # heavy dep, imported lazily

    archive = hf_hub_download(
        repo_id=config.MELD_HF_REPO,
        filename=config.MELD_RAW_ARCHIVE,
        repo_type="dataset",
        local_dir=config.DATA_DIR,
    )
    print(f"Extracting {archive} ...")
    with tarfile.open(archive) as tf:
        tf.extractall(config.DATA_DIR)
    return config.RAW_DIR


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _find_csv(split: str) -> Path:
    """MELD raw layout differs slightly across releases; glob defensively."""
    for pattern in (
        config.SPLIT_CSV_PATTERNS[split],
        f"{split}*_sent_emo.csv",
        f"{split}*.csv",
    ):
        hits = sorted(config.RAW_DIR.rglob(pattern))
        if hits:
            return hits[0]
    raise FileNotFoundError(
        f"Could not find a CSV for split '{split}' under {config.RAW_DIR}. "
        "Run download_and_extract_meld() first."
    )


def clip_dir_for(split: str) -> Optional[Path]:
    cand = config.SPLIT_CLIP_DIRS.get(split)
    if cand and cand.exists():
        return cand
    for sub in sorted(config.RAW_DIR.rglob("*.mp4")):
        return sub.parent  # first mp4 found defines the clip dir
    return None


def load_split(split: str) -> pd.DataFrame:
    """Load a MELD split CSV with normalized labels."""
    if split not in config.SPLITS:
        raise ValueError(f"split must be one of {config.SPLITS}")
    df = pd.read_csv(_find_csv(split))
    df.columns = [c.strip() for c in df.columns]
    df["Utterance"] = df["Utterance"].astype(str).str.strip()
    df["emotion"] = df["Emotion"].astype(str).str.strip().str.lower()
    df["sentiment"] = df["Sentiment"].astype(str).str.strip().str.lower()
    unknown = set(df["emotion"].unique()) - set(config.EMOTIONS)
    if unknown:
        n_before = len(df)
        df = df[df["emotion"].isin(config.EMOTIONS)].reset_index(drop=True)
        print(f"Dropped {n_before - len(df)} rows with unexpected/missing labels {unknown}")
    df["label"] = df["emotion"].map(config.EMOTION_TO_IDX)
    return df


def utterances_from_split(split: str) -> list[Utterance]:
    """Materialize Utterance records with resolved media paths."""
    df = load_split(split)
    clips = clip_dir_for(split)
    out: list[Utterance] = []
    for _, row in df.iterrows():
        uid = f"dia{int(row['Dialogue_ID'])}_utt{int(row['Utterance_ID'])}"
        video = clips / f"{uid}.mp4" if clips else None
        if video is not None and not video.exists():
            video = None
        wav = config.PROCESSED_DIR / split / f"{uid}.wav"
        out.append(
            Utterance(
                uid=uid,
                split=split,
                text=row["Utterance"],
                speaker=str(row.get("Speaker", "unknown")),
                emotion=row["emotion"],
                sentiment=row["sentiment"],
                video_path=video,
                wav_path=wav,
            )
        )
    return out


def class_weights(labels: list[int]) -> np.ndarray:
    """Inverse-frequency class weights for the imbalanced 7-class problem."""
    counts = np.bincount(labels, minlength=len(config.EMOTIONS)).astype(np.float64)
    w = counts.sum() / (len(counts) * np.maximum(counts, 1.0))
    return w


# ---------------------------------------------------------------------------
# Audio extraction (ffmpeg, system dependency; see README)
# ---------------------------------------------------------------------------
def _extract_wav(video_path: Path, wav_path: Path) -> None:
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "16000", str(wav_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def ensure_audio_wavs(utterances: list[Utterance], workers: int = 8) -> list[Utterance]:
    """Extract 16 kHz mono wavs for every utterance that has a video clip.
    Idempotent: existing wavs are skipped, so it is safe to re-run."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH. Install it first (README has notes).")
    todo = [
        u for u in utterances
        if u.video_path is not None and not u.wav_path.exists()
    ]
    if not todo:
        return utterances
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_extract_wav, u.video_path, u.wav_path) for u in todo]
        for fut in futures:
            fut.result()
    print(f"Extracted {len(todo)} wavs to {config.PROCESSED_DIR}")
    return utterances
