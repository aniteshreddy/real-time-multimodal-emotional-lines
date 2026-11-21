"""Featurization, training, and evaluation for the fusion model.

Featurization runs the three frozen encoders over a MELD split once and
caches the results as .npz, so training iterations afterward are fast and
deterministic.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from . import config, utils
from .model import MODALITY_NAMES, FusionModel, apply_modality_dropout


# ---------------------------------------------------------------------------
# Featurize a split and cache to disk
# ---------------------------------------------------------------------------
def featurize_split(split: str, text_encoder, audio_encoder, face_extractor, utterances) -> Path:
    """Compute and cache text/audio/prosody/vision features for a split."""
    cache_path = config.FEATURE_CACHE / f"{split}.npz"
    if cache_path.exists():
        print(f"{split}: features cached at {cache_path}")
        return cache_path

    from .audio_features import prosody_features
    from .vision_features import clip_vision_features

    texts = [u.text for u in utterances]
    print(f"{split}: encoding {len(texts)} utterances (text) ...")
    text_emb = text_encoder.encode(texts)

    wavs = []
    import soundfile as sf
    for u in tqdm(utterances, desc=f"{split}: loading wavs"):
        if u.wav_path.exists():
            wav, sr = sf.read(str(u.wav_path), dtype="float32")
            if wav.ndim > 1:  # safety: mix to mono
                wav = wav.mean(axis=1)
            if sr != 16000:
                import librosa
                wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=16000)
            wavs.append(np.asarray(wav, dtype=np.float32))
        else:
            wavs.append(np.zeros(1600, dtype=np.float32))
    # resample safeguard: librosa handles other rates
    print(f"{split}: encoding audio ...")
    audio_emb = audio_encoder.encode(wavs)

    print(f"{split}: prosody features ...")
    pros = np.stack([prosody_features(w) for w in tqdm(wavs, leave=False)])

    print(f"{split}: vision features ...")
    vis = np.stack(
        [clip_vision_features(u.video_path, face_extractor) for u in tqdm(utterances, leave=False)]
    )

    np.savez_compressed(
        cache_path,
        uid=np.array([u.uid for u in utterances]),
        text=np.asarray(text_emb, dtype=np.float32),
        audio=np.asarray(audio_emb, dtype=np.float32),
        vision=np.asarray(vis, dtype=np.float32),
        prosody=np.asarray(pros, dtype=np.float32),
    )
    print(f"{split}: cached features -> {cache_path}")
    return cache_path


class MELDFusionDataset(Dataset):
    """Loads cached features + labels. All four modalities are marked present;
    modality dropout is applied inside the training loop."""

    def __init__(self, cache_path: Path, labels: dict[str, int]):
        data = np.load(cache_path)
        self.uids = [str(u) for u in data["uid"]]
        self.text = data["text"]
        self.audio = data["audio"]
        self.vision = data["vision"]
        self.prosody = data["prosody"]
        self.present = np.ones((len(self.uids), len(MODALITY_NAMES)), dtype=bool)
        self.labels = np.array(
            [labels.get(u, 0) for u in self.uids], dtype=np.int64
        )
        found = sum(1 for u in self.uids if u in labels)
        print(f"{Path(cache_path).name}: matched {found}/{len(self.uids)} label rows")

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, i):
        return (
            torch.from_numpy(self.text[i]),
            torch.from_numpy(self.audio[i]),
            torch.from_numpy(self.vision[i]),
            torch.from_numpy(self.prosody[i]),
            torch.from_numpy(self.present[i]),
            torch.tensor(self.labels[i]),
        )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def evaluate(model: FusionModel, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    losses, preds, gold = [], [], []
    loss_fn = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for text, audio, vision, prosody, present, label in loader:
            text, audio, vision, prosody = (
                text.to(device), audio.to(device), vision.to(device), prosody.to(device)
            )
            present, label = present.to(device), label.to(device)
            logits = model(text, audio, vision, prosody, present)
            losses.append(loss_fn(logits, label).item())
            preds.extend(logits.argmax(-1).cpu().tolist())
            gold.extend(label.cpu().tolist())
    preds, gold = np.array(preds), np.array(gold)
    return {
        "loss": float(np.mean(losses)),
        "wf1": float(f1_score(gold, preds, average="weighted")),
        "macro_f1": float(f1_score(gold, preds, average="macro")),
        "accuracy": float((preds == gold).mean()),
        "confusion": confusion_matrix(gold, preds, labels=range(len(config.EMOTIONS))).tolist(),
        "report": classification_report(
            gold, preds, labels=range(len(config.EMOTIONS)),
            target_names=config.EMOTIONS, zero_division=0, output_dict=True,
        ),
        "preds": preds.tolist(),
        "gold": gold.tolist(),
    }


def train_model(
    train_ds: MELDFusionDataset,
    val_ds: MELDFusionDataset,
    device: torch.device | None = None,
    epochs: int = config.MAX_EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    lr: float = config.LR,
    patience: int = config.PATIENCE,
    modality_dropout: float = config.MODALITY_DROPOUT,
) -> tuple[FusionModel, dict]:
    device = device or utils.get_device()
    model = FusionModel().to(device)

    from .data import class_weights
    weights = torch.tensor(class_weights(train_ds.labels.tolist()), dtype=torch.float32).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    def make_loader(ds, shuffle):
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    train_loader, val_loader = make_loader(train_ds, True), make_loader(val_ds, False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    history = {"train_loss": [], "val_loss": [], "val_wf1": []}
    best_wf1, best_state, bad_epochs = -1.0, None, 0

    for epoch in range(1, epochs + 1):
        model.train()
        running = []
        for text, audio, vision, prosody, present, label in train_loader:
            text, audio, vision, prosody = (
                text.to(device), audio.to(device), vision.to(device), prosody.to(device)
            )
            present, label = present.to(device), label.to(device)
            if modality_dropout > 0:
                present = apply_modality_dropout(present, modality_dropout)
            logits = model(text, audio, vision, prosody, present)
            loss = loss_fn(logits, label)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running.append(loss.item())
        sched.step()

        val = evaluate(model, val_loader, device)
        history["train_loss"].append(float(np.mean(running)))
        history["val_loss"].append(val["loss"])
        history["val_wf1"].append(val["wf1"])
        star = ""
        if val["wf1"] > best_wf1:
            best_wf1, best_state, bad_epochs = val["wf1"], {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
            star = " *"
        else:
            bad_epochs += 1
        print(
            f"epoch {epoch:3d} | train_loss {np.mean(running):.4f} | "
            f"val_loss {val['loss']:.4f} | val_WF1 {val['wf1']:.4f}{star}"
        )
        if bad_epochs >= patience:
            print(f"early stop at epoch {epoch} (best val_WF1 {best_wf1:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    ckpt = config.CHECKPOINT_DIR / "fusion_best.pt"
    torch.save(model.state_dict(), ckpt)
    print(f"best val_WF1 {best_wf1:.4f} -> saved {ckpt}")
    return model, history
