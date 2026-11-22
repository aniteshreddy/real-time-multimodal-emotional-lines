"""MELD dataset handling: download, locate, load, and audio extraction.

Light pandas logic is importable without torch/transformers so the data
layer can be smoke-tested independently of the GPU stack.

Audio extraction is pure Python: PyAV (wheel-bundled FFmpeg libraries)
decodes mp4 audio in-process; soundfile writes 16 kHz mono wavs. No system
ffmpeg, no subprocess, no sudo. Corrupt source clips (MELD ships a few) are
skipped and reported; the training pipeline treats them as missing audio.
"""
from __future__ import annotations

import tarfile
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


@dataclass
class Utterance:
    uid: str
    split: str
    text: str
    speaker: str
    emotion: str
    sentiment: str
    wav_path: Path
    video_path: Optional[Path] = None


def download_and_extract_meld(force: bool = False) -> Path:
    if sorted(config.RAW_DIR.glob("*_sent_emo.csv")) and not force:
        print(f"MELD already extracted at {config.RAW_DIR}, skipping download.")
        return config.RAW_DIR

    from huggingface_hub import hf_hub_download

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


def _find_csv(split: str) -> Path:
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


CLIP_DIR_FALLBACKS = {
    "train": ("train_splits",),
    "dev": ("dev_splits_complete", "dev_splits"),
    "test": ("output_repeated_splits_test",),
}


def clip_dir_for(split: str) -> Optional[Path]:
    cand = config.SPLIT_CLIP_DIRS.get(split)
    if cand and cand.exists():
        return cand
    for name in CLIP_DIR_FALLBACKS.get(split, ()):
        hits = sorted(p for p in config.RAW_DIR.rglob(name) if p.is_dir())
        if hits:
            return hits[0]
    return None


def load_split(split: str) -> pd.DataFrame:
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
    counts = np.bincount(labels, minlength=len(config.EMOTIONS)).astype(np.float64)
    w = counts.sum() / (len(counts) * np.maximum(counts, 1.0))
    return w


def extract_audio(video_path: Path, wav_path: Path, sr: int = 16000) -> bool:
    """Decode an mp4's audio track in-process with PyAV and write a 16 kHz
    mono wav. Returns False when the clip is undecodable or has no audio."""
    import av
    import soundfile as sf

    wav_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        container = av.open(str(video_path))
    except Exception:
        return False  # corrupt/truncated file (e.g. missing moov atom)
    try:
        if not container.streams.audio:
            return False
        resampler = av.AudioResampler(format="s16", layout="mono", rate=sr)
        chunks = []
        for frame in container.decode(audio=0):
            for resampled in resampler.resample(frame):
                arr = resampled.to_ndarray()
                chunks.append(np.asarray(arr).reshape(-1))
        if not chunks:
            return False
        raw = np.concatenate(chunks).astype(np.float32) / 32768.0
        sf.write(str(wav_path), raw, sr)
        return True
    except Exception:
        return False
    finally:
        container.close()


def ensure_audio_wavs(utterances: list[Utterance], workers: int = 8) -> list[Utterance]:
    """Extract 16 kHz mono wavs. Idempotent and fault-tolerant: corrupt
    source clips are reported and treated as missing audio downstream."""
    todo = [
        u for u in utterances
        if u.video_path is not None and not u.wav_path.exists()
    ]
    if not todo:
        return utterances
    from concurrent.futures import ThreadPoolExecutor

    failed = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = ex.map(lambda u: extract_audio(u.video_path, u.wav_path), todo)
        for u, ok in zip(todo, results):
            if not ok:
                failed.append(u.uid)
    print(f"Extracted {len(todo) - len(failed)}/{len(todo)} wavs to {config.PROCESSED_DIR}")
    if failed:
        print(
            f"NOTE: {len(failed)} clips are unreadable (corrupt in the source "
            f"data); they will be treated as missing audio. First few: {failed[:10]}"
        )
    return utterances
