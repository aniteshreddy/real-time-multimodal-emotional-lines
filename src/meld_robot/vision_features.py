"""Vision modality: MediaPipe FaceLandmarker geometry features.

Zero learned parameters beyond the MediaPipe task model itself
(~2M, upper-bound estimate, counted in the budget). We convert per-frame
landmarks into 12 interpretable facial-geometry dimensions, then pool
(mean/std/max) over sampled frames of the clip.
"""
from __future__ import annotations

import math
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from . import config

GEOMETRY_NAMES = [
    "eye_aspect_l", "eye_aspect_r", "brow_raise_l", "brow_raise_r",
    "mouth_open", "mouth_width", "corner_pull_l", "corner_pull_r",
    "nostril_widen", "head_pitch", "head_yaw", "head_roll",
]


def ensure_landmarker_model() -> Path:
    if not config.FACE_LANDMARKER_PATH.exists():
        config.FACE_LANDMARKER_PATH.parent.mkdir(parents=True, exist_ok=True)
        print("Downloading MediaPipe FaceLandmarker task file ...")
        urllib.request.urlretrieve(config.FACE_LANDMARKER_URL, config.FACE_LANDMARKER_PATH)
    return config.FACE_LANDMARKER_PATH


class FaceGeometryExtractor:
    """Runs MediaPipe FaceLandmarker in VIDEO mode over sampled frames."""

    def __init__(self):
        import mediapipe as mp  # heavy import, deferred
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        self._mp = mp
        model_path = ensure_landmarker_model()
        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,
            output_facial_transformation_matrixes=True,
        )
        self.landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    # ------------------------------------------------------------------
    def _frame_geometry(self, lm, matrix: np.ndarray | None) -> np.ndarray:
        """Landmark list -> 12-dim geometry vector. Coordinates are normalized
        to [0,1]; x is aspect-corrected so distances are isotropic."""

        def pt(idx):
            p = lm[idx]
            return np.array([p.x, p.y])

        def dist(a, b):
            return float(np.linalg.norm(pt(a) - pt(b)))

        face_h = max(dist(10, 152), 1e-6)  # forehead to chin

        eye_aspect_l = (dist(159, 145) + dist(158, 153)) / (2 * max(dist(33, 133), 1e-6))
        eye_aspect_r = (dist(386, 374) + dist(385, 380)) / (2 * max(dist(362, 263), 1e-6))
        brow_raise_l = (pt(105)[1] - pt(33)[1]) / face_h  # larger = raised brow
        brow_raise_r = (pt(334)[1] - pt(263)[1]) / face_h
        mouth_open = dist(13, 14) / face_h
        mouth_width = dist(61, 291) / face_h
        corner_pull_l = (pt(13)[1] - pt(61)[1]) / face_h  # positive = corner above lip center
        corner_pull_r = (pt(13)[1] - pt(291)[1]) / face_h
        nostril_widen = dist(129, 358) / face_h

        pitch = yaw = roll = 0.0
        if matrix is not None:
            R = np.asarray(matrix)[:3, :3]
            sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
            if sy > 1e-6:
                pitch = math.atan2(R[2, 1], R[2, 2])
                yaw = math.atan2(-R[2, 0], sy)
                roll = math.atan2(R[1, 0], R[0, 0])
            else:
                pitch = math.atan2(-R[1, 2], R[1, 1])
                yaw = math.atan2(-R[2, 0], sy)
            pitch, yaw, roll = (max(-1.0, min(1.0, v)) for v in (pitch, yaw, roll))

        return np.array(
            [eye_aspect_l, eye_aspect_r, brow_raise_l, brow_raise_r, mouth_open,
             mouth_width, corner_pull_l, corner_pull_r, nostril_widen,
             pitch, yaw, roll],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    def clip_geometry(self, frames_bgr: list[np.ndarray]) -> list[np.ndarray]:
        """Per-frame geometry vectors for frames where a face is detected."""
        from mediapipe.tasks.python import vision as mp_vision

        out = []
        for i, frame in enumerate(frames_bgr):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
            result = self.landmarker.detect_for_video(mp_image, int(i * 1000))
            if result.face_landmarks:
                matrix = None
                if result.facial_transformation_matrixes:
                    matrix = result.facial_transformation_matrixes[0]
                out.append(self._frame_geometry(result.face_landmarks[0], matrix))
        return out


def sample_frames(video_path: Path, max_frames: int = config.MAX_VISION_FRAMES) -> list[np.ndarray]:
    """Evenly sample up to max_frames frames from an mp4 clip."""
    cap = cv2.VideoCapture(str(video_path))
    frames: list[np.ndarray] = []
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            return frames
        idxs = np.linspace(0, total - 1, num=min(max_frames, total)).astype(int)
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if ok and frame is not None:
                frames.append(frame)
    finally:
        cap.release()
    return frames


def clip_vision_features(video_path: Path | None, extractor: FaceGeometryExtractor | None) -> np.ndarray:
    """(37,) pooled vision features: per-geometry {mean, std, max} + presence.

    No face or no video -> presence 0 and zeros elsewhere; the fusion model
    has a learned missing-modality token, so this is handled gracefully.
    """
    if video_path is None or not Path(video_path).exists() or extractor is None:
        return np.zeros(config.VISION_FEAT_DIM, dtype=np.float32)
    frames = sample_frames(video_path)
    if not frames:
        return np.zeros(config.VISION_FEAT_DIM, dtype=np.float32)
    geo = clip_geometry_safe(extractor, frames)
    if not geo:
        return np.zeros(config.VISION_FEAT_DIM, dtype=np.float32)
    G = np.stack(geo)  # (F, 12), F >= 1
    feats = np.concatenate([G.mean(0), G.std(0) if len(G) > 1 else G[0] * 0, G.max(0)])
    presence = np.array([len(geo) / len(frames)], dtype=np.float32)
    return np.concatenate([feats, presence]).astype(np.float32)


def clip_geometry_safe(extractor: FaceGeometryExtractor, frames: list[np.ndarray]) -> list[np.ndarray]:
    try:
        return extractor.clip_geometry(frames)
    except Exception as exc:  # mediapipe can throw on corrupt frames
        print(f"vision extraction warning: {exc}")
        return []


def vision_evidence(v: np.ndarray) -> list[str]:
    """Human-readable face-movement evidence from the pooled vision vector."""
    if len(v) < config.VISION_FEAT_DIM or v[-1] <= 0:
        return ["no face detected"]
    mean_part = v[0:12]
    std_part = v[12:24]
    ev: list[str] = []
    brow = float(np.mean([mean_part[2], mean_part[3]]))
    brow_var = float(np.mean([std_part[2], std_part[3]]))
    mouth_open = float(mean_part[4])
    corner_pull = float(np.mean([mean_part[6], mean_part[7]]))
    if brow_var > 0.004:
        ev.append("visible brow movement")
    if mouth_open > 0.06:
        ev.append("mouth open while speaking")
    if corner_pull > 0.01:
        ev.append("lip corners pulled up (smile-like)")
    elif corner_pull < -0.01:
        ev.append("lip corners pulled down")
    pitch = abs(float(mean_part[9]))
    if pitch > 0.2:
        ev.append("marked head tilt")
    return ev or ["relaxed, neutral face"]
