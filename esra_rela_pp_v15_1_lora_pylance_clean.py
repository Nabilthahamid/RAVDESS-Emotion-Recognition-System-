"""
ESRA-RELA++ V15.1-LoRA-Publication-Final Pylance-Clean: Publication-ready multimodal RAVDESS pipeline
====================================================================
Subject-wise 5-fold audio-visual emotion recognition on RAVDESS.

Core architecture
-----------------
1. Hybrid audio branch:
   frozen SSL audio embedding + handcrafted acoustic/prosodic descriptors.
2. Optional LoRA audio branch:
   fold-safe parameter-efficient fine-tuning of Wav2Vec2/WavLM-style audio
   classifiers.  It is added as a separate OOF modality, never trained on
   outer-test actors.
3. Video branch:
   OpenFace-guided emotion-saliency frame selection, landmark-based face crop,
   MobileNetV3 frame embeddings, and temporal pooling.
4. AU branches:
   OpenFace AU global statistics plus AU dynamic trajectory/FACS-group features.
5. Fusion:
   leakage-safe actor-group OOF stacking, posterior-temperature calibration,
   reliability features, dual fusion with calibrated average probabilities,
   weak-pair specialists, optional gender-stratified specialists, Sad-vs-rest
   specialist, and uncertainty logging.

Publication safety
------------------
- The outer test fold is never used for feature/model selection.
- LoRA models are trained only inside each outer-train fold and inside each
  inner OOF split for meta-training.
- RAVDESS intensity/statement/repetition metadata are excluded by default.
- Retracted work is not used as methodological support or benchmark evidence.

Recommended install
-------------------
pip install torch torchvision torchaudio transformers librosa opencv-python             pandas scikit-learn scipy tqdm matplotlib

Optional LoRA install
---------------------
pip install peft accelerate

Example full run without LoRA:
python esra_rela_pp_v15_lora_publication_final.py --data_dir /path/RAVDESS     --openface_dir /path/OpenFace_CSVs --run_all_folds

Example full run with LoRA audio branch:
python esra_rela_pp_v15_lora_publication_final.py --data_dir /path/RAVDESS     --openface_dir /path/OpenFace_CSVs --run_all_folds --use_lora_audio
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    classification_report, confusion_matrix, f1_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from torch.utils.data import Dataset as TorchDataset

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ──────────────────────────────────────────────────────────────────
# Global model caches
# ──────────────────────────────────────────────────────────────────
_AUDIO_EXTRACTOR: Any = None
_AUDIO_MODEL: Any = None
_AUDIO_MODEL_NAME: Optional[str] = None
_VIDEO_MODEL: Any = None
_VIDEO_TRANSFORM: Any = None
_VIDEO_EMBED_DIM: Optional[int] = None
FEATURE_CACHE_VERSION = "esra_rela_pp_v15_1_lora_pylance_clean_1"

# ──────────────────────────────────────────────────────────────────
# RAVDESS metadata
# ──────────────────────────────────────────────────────────────────
EMOTIONS: Dict[int, str] = {
    1: "Neutral", 2: "Calm",    3: "Happy",    4: "Sad",
    5: "Angry",   6: "Fearful", 7: "Disgust",  8: "Surprised",
}
LABEL_NAMES: List[str] = [EMOTIONS[i] for i in range(1, 9)]
NAME_TO_LABEL: Dict[str, int] = {n: i for i, n in enumerate(LABEL_NAMES)}

# Paper-1 subject-wise 5-fold splits
SUBJECT5_FOLDS: List[List[int]] = [
    [2, 5, 14, 15, 16],
    [3, 6, 7,  13, 18],
    [10, 11, 12, 19, 20],
    [8, 17, 21, 23, 24],
    [1, 4, 9,  22],
]

# RAVDESS actor gender mapping (odd actor IDs = male, even = female)
MALE_ACTORS   = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23}
FEMALE_ACTORS = {2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24}

# OpenFace AU column names (intensity suffix _r)
AU_LIST = [
    "AU01_r","AU02_r","AU04_r","AU05_r","AU06_r","AU07_r",
    "AU09_r","AU10_r","AU12_r","AU14_r","AU15_r","AU17_r",
    "AU20_r","AU23_r","AU25_r","AU26_r","AU45_r",
]
AU_INT_LIST = [1,2,4,5,6,7,9,10,12,14,15,17,20,23,25,26,45]

# Weak confusion pairs — from error analysis across all three reference papers
# V11 adds Neutral/Calm because it is a frequent low-arousal confusion pair.
SPECIALIST_PAIRS = [
    ("Fearful",  "Surprised"),
    ("Sad",      "Disgust"),
    ("Sad",      "Fearful"),
    ("Angry",    "Disgust"),
    ("Neutral",  "Calm"),      # NEW V11
]

# ──────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────
@dataclass
class Sample:
    path: Path
    filename: str
    actor: int
    label: int        # 0-based emotion index
    emotion: str
    intensity: int
    statement: int
    repetition: int
    gender: str       # "male" | "female"  — NEW V11


@dataclass
class Config:
    data_dir: str
    openface_dir: str
    cache_dir: str         = "esra_rela_v15_lora_cache"
    results_dir: str       = "esra_rela_v15_lora_results"

    split: str             = "subject5"
    run_all_folds: bool    = False
    fold: int              = 0
    max_files: int         = 0
    rebuild_cache: bool    = False
    save_diagnostics: bool = True

    use_audio: bool        = True
    use_video: bool        = True
    use_au: bool           = True
    use_au_dynamic: bool   = True    # ESRA-RELA++: AU temporal trajectory branch

    # "none" = handcrafted only (no SSL model download)
    audio_model: str       = "facebook/wav2vec2-base-960h"
    audio_sr: int          = 16000
    audio_seconds: float   = 5.5

    # Optional LoRA audio branch. It is OFF by default because it is slow.
    # When enabled, it is trained fold-safely and added as another OOF modality.
    use_lora_audio: bool   = False
    lora_audio_model: str  = "facebook/wav2vec2-base-960h"
    lora_epochs: int       = 8
    lora_batch_size: int   = 2
    lora_grad_accum: int   = 2
    lora_lr: float         = 1e-4
    lora_weight_decay: float = 0.01
    lora_r: int            = 8
    lora_alpha: int        = 16
    lora_dropout: float    = 0.10
    lora_max_train_samples: int = 0
    lora_target_modules: str = "q_proj,k_proj,v_proj,out_proj"
    lora_patience: int     = 2

    video_frames: int      = 10
    saliency_top_pool: int = 24

    base_model: str        = "logreg"   # logreg | svm
    meta_model: str        = "logreg"
    base_C: float          = 0.45
    meta_C: float          = 0.55
    pca_audio: int         = 128
    pca_video: int         = 128
    pca_au: int            = 64
    pca_au_dynamic: int    = 64
    inner_splits: int      = 4

    enable_specialists: bool           = True
    enable_gender_specialists: bool    = True    # NEW V11
    specialist_threshold: float        = 0.38
    specialist_margin_threshold: float = 0.12
    specialist_blend: float            = 0.30
    specialist_pca: int                = 128

    # ESRA-RELA++: meta output is blended with calibrated simple-average fusion.
    # This preserves the strong simple-fusion baseline while keeping reliability-aware stacking.
    dual_fusion_blend: float           = 0.25
    dual_fusion_weighted: bool         = True

    # ESRA-RELA++: Sad-vs-rest correction, because Sad was the weakest class in V10.
    enable_sad_specialist: bool        = True
    sad_specialist_threshold: float    = 0.42
    sad_specialist_blend: float        = 0.18
    sad_min_base_prob: float           = 0.05

    # NEW V11: temperature calibration before meta-learner
    calibrate_temperatures: bool       = True
    temp_init: float                   = 1.5

    # NEW V11: uncertainty logging only; predictions are not changed
    uncertainty_threshold: float       = 0.30

    # Publication-safe default: do not append RAVDESS metadata covariates
    # such as intensity/statement/repetition to the reliability vector.
    # Enable only for an explicitly reported ablation.
    include_metadata_covariates: bool  = False

    random_state: int      = 42


# ──────────────────────────────────────────────────────────────────
# General utilities
# ──────────────────────────────────────────────────────────────────
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _softmax_np(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - x.max()
    e = np.exp(x)
    return e / (float(np.sum(e)) + 1e-9)


def entropy_1d(p: np.ndarray) -> float:
    # Pylance-safe: ensure probability vector is floating, not bool/object.
    p_float = np.asarray(p, dtype=np.float32)
    p_float = np.clip(p_float, 1e-9, 1.0)
    return float(-np.sum(p_float * np.log(p_float)) / math.log(len(p_float)))


def entropy_mat(P: np.ndarray) -> np.ndarray:
    """Normalised entropy column for each row of a probability matrix."""
    P = np.clip(P, 1e-9, 1.0)
    return (-np.sum(P * np.log(P), axis=1, keepdims=True) /
            math.log(P.shape[1])).astype(np.float32)


def margin_mat(P: np.ndarray) -> np.ndarray:
    """Top-1 minus top-2 probability margin column."""
    sp = np.sort(P, axis=1)
    return (sp[:, -1:] - sp[:, -2:-1]).astype(np.float32)


def normalize01(x: np.ndarray) -> np.ndarray:
    mn, mx = float(x.min()), float(x.max())
    if mx - mn < 1e-8:
        return np.zeros_like(x, np.float32)
    return ((x - mn) / (mx - mn)).astype(np.float32)


def save_npz(path: Path, **kw: Any) -> None:
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **kw)
    tmp.replace(path)


# ──────────────────────────────────────────────────────────────────
# RAVDESS file discovery
# ──────────────────────────────────────────────────────────────────
def parse_ravdess_filename(path: Path) -> Optional[Sample]:
    parts = path.stem.split("-")
    if len(parts) != 7:
        return None
    try:
        modality, channel, emotion_id, intensity, statement, repetition, actor = map(int, parts)
    except ValueError:
        return None
    if modality != 1 or channel != 1:   # full-AV speech only (matches Paper 1)
        return None
    if emotion_id not in EMOTIONS:
        return None
    if actor in MALE_ACTORS:
        gender = "male"
    elif actor in FEMALE_ACTORS:
        gender = "female"
    else:
        warnings.warn(f"Unknown RAVDESS actor ID {actor}; setting gender='unknown'.")
        gender = "unknown"
    return Sample(
        path=path, filename=path.name, actor=actor,
        label=emotion_id - 1, emotion=EMOTIONS[emotion_id],
        intensity=intensity, statement=statement, repetition=repetition,
        gender=gender,
    )


def discover_samples(data_dir: Path, max_files: int = 0) -> List[Sample]:
    samples: List[Sample] = []
    for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
        for p in data_dir.rglob(ext):
            s = parse_ravdess_filename(p)
            if s is not None:
                samples.append(s)
    samples = sorted(samples, key=lambda s: str(s.path.resolve()).lower())
    if max_files > 0:
        samples = samples[:max_files]
    if not samples:
        raise FileNotFoundError(f"No RAVDESS full-AV speech files under {data_dir}")
    return samples


# ──────────────────────────────────────────────────────────────────
# OpenFace CSV helpers
# ──────────────────────────────────────────────────────────────────
def find_openface_csv(sample: Sample, openface_dir: Path) -> Optional[Path]:
    if not str(openface_dir) or not openface_dir.exists():
        return None
    direct = openface_dir / f"{sample.path.stem}.csv"
    if direct.exists():
        return direct
    hits = list(openface_dir.rglob(f"{sample.path.stem}.csv"))
    return hits[0] if hits else None


def read_openface(sample: Sample, openface_dir: Path) -> Optional[pd.DataFrame]:
    p = find_openface_csv(sample, openface_dir)
    if p is None:
        return None
    try:
        df = pd.read_csv(p)
        df.columns = [c.strip() for c in df.columns]
        return df
    except Exception:
        return None


def _of_meta(sample: Sample, openface_dir: Path) -> Dict[str, Any]:
    p = find_openface_csv(sample, openface_dir)
    if p is None:
        return {"csv_path": None, "csv_size": 0, "csv_mtime": 0.0}
    try:
        return {"csv_path": str(p.resolve()),
                "csv_size": int(p.stat().st_size),
                "csv_mtime": float(p.stat().st_mtime)}
    except OSError:
        return {"csv_path": str(p), "csv_size": 0, "csv_mtime": 0.0}




def openface_dataset_fingerprint(samples: Sequence[Sample], openface_dir: Optional[Path]) -> Dict[str, Any]:
    """Matrix-cache fingerprint for OpenFace CSV coverage/content.

    If OpenFace CSVs are regenerated, the feature-matrix cache should rebuild,
    because video saliency, face crop, and AU features can change.
    """
    if openface_dir is None or not str(openface_dir) or not openface_dir.exists():
        return {"csv_found": 0, "csv_size_sum": 0, "csv_mtime_sum": 0}
    found = 0
    size_sum = 0
    mtime_sum = 0
    for s in samples:
        p = find_openface_csv(s, openface_dir)
        if p is not None:
            try:
                st = p.stat()
                found += 1
                size_sum += int(st.st_size)
                mtime_sum += int(st.st_mtime)
            except OSError:
                pass
    return {"csv_found": found, "csv_size_sum": size_sum, "csv_mtime_sum": mtime_sum}

# ──────────────────────────────────────────────────────────────────
# Cache key
# ──────────────────────────────────────────────────────────────────
def cache_key(sample: Sample, cfg: Config, kind: str) -> str:
    obj: Dict[str, Any] = {
        "kind": kind, "version": FEATURE_CACHE_VERSION,
        "path": str(sample.path.resolve()),
        "mtime": os.path.getmtime(sample.path),
        "size": os.path.getsize(sample.path),
        "audio_model": cfg.audio_model,
        "audio_sr": cfg.audio_sr,
        "audio_seconds": cfg.audio_seconds,
        "video_frames": cfg.video_frames,
        "saliency_top_pool": cfg.saliency_top_pool,
    }
    if kind in {"video", "au", "au_dynamic"}:
        if cfg.openface_dir:
            obj["openface"] = _of_meta(sample, Path(cfg.openface_dir))
        else:
            obj["openface"] = {"csv_path": None, "csv_size": 0, "csv_mtime": 0.0}
    return hashlib.sha1(json.dumps(obj, sort_keys=True).encode()).hexdigest()


# ──────────────────────────────────────────────────────────────────
# AU saliency (merged from ESRA-CMT)
# ──────────────────────────────────────────────────────────────────
def _row_au_intensities(df: pd.DataFrame, row_id: int) -> Dict[int, float]:
    row_id = int(np.clip(row_id, 0, len(df) - 1))
    row    = df.iloc[row_id]
    out: Dict[int, float] = {}
    for au in AU_INT_LIST:
        r_col = f"AU{au:02d}_r"
        c_col = f"AU{au:02d}_c"
        intensity = 0.0
        presence  = 0.0
        try:
            if r_col in df.columns and not pd.isna(row[r_col]):
                intensity = max(0.0, min(float(row[r_col]) / 5.0, 1.0))
        except Exception:
            pass
        try:
            if c_col in df.columns and not pd.isna(row[c_col]):
                presence = max(0.0, min(float(row[c_col]), 1.0))
        except Exception:
            pass
        out[au] = max(intensity, 0.5 * presence)
    return out


def au_saliency_scores(
    df: Optional[pd.DataFrame], frame_indices: Sequence[int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (au_intensity, au_delta, fer_entropy) per candidate frame.

    Merged from ESRA-CMT openface_saliency_scores / au_rule_based_fer_entropy.
    Fixed: now a proper top-level function (was an internal alias in V3).
    """
    n = len(frame_indices)
    if df is None or len(df) == 0:
        return (np.zeros(n, np.float32),
                np.zeros(n, np.float32),
                np.ones(n, np.float32))

    intensity_l: List[float] = []
    delta_l:     List[float] = []
    entropy_l:   List[float] = []

    for fi in frame_indices:
        row_id   = int(np.clip(fi, 0, len(df) - 1))
        au       = _row_au_intensities(df, row_id)
        au_prev  = _row_au_intensities(df, max(0, row_id - 1))
        active   = np.array([au.get(a, 0.0) for a in AU_INT_LIST], np.float32)
        delta    = np.array([abs(au.get(a, 0.0) - au_prev.get(a, 0.0)) for a in AU_INT_LIST], np.float32)
        intensity_l.append(float(active.mean()))
        delta_l.append(float(delta.mean()))
        # FACS rule-based logits → entropy (low entropy = high AU confidence)
        logits = np.array([
            0.35 - 0.65 * float(active.mean()),
            0.25 - 0.35 * float(delta.mean()),
            1.20 * au.get(6,0) + 1.60 * au.get(12,0) + 0.25 * au.get(25,0),
            1.10 * au.get(1,0) + 1.00 * au.get(4,0)  + 1.25 * au.get(15,0),
            1.30 * au.get(4,0) + 0.95 * au.get(5,0)  + 1.05 * au.get(7,0) + 1.00 * au.get(23,0),
            0.90 * au.get(1,0) + 0.90 * au.get(2,0)  + 0.80 * au.get(4,0)
              + 1.05 * au.get(5,0) + 1.00 * au.get(20,0) + 0.85 * au.get(26,0),
            1.35 * au.get(9,0) + 1.20 * au.get(10,0) + 0.80 * au.get(17,0),
            1.25 * au.get(1,0) + 1.25 * au.get(2,0)  + 1.35 * au.get(5,0) + 1.25 * au.get(26,0),
        ], np.float32)
        probs = _softmax_np(logits)
        entropy_l.append(max(0.0, min(entropy_1d(probs), 1.0)))

    return (normalize01(np.array(intensity_l, np.float32)),
            normalize01(np.array(delta_l,     np.float32)),
            np.array(entropy_l, np.float32))


def openface_row_scores(df: Optional[pd.DataFrame]) -> Optional[np.ndarray]:
    """Fast per-row saliency score for frame selection (no frame_indices needed)."""
    if df is None or len(df) == 0:
        return None
    def _to_float(col: str) -> np.ndarray:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").fillna(0).to_numpy(np.float32)
        return np.ones(len(df), np.float32)
    success = _to_float("success")
    conf    = _to_float("confidence")
    au_cols = [c for c in AU_LIST if c in df.columns]
    score   = np.zeros(len(df), np.float32)
    if au_cols:
        au      = df[au_cols].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(np.float32)
        au_int  = au.mean(axis=1)
        au_del  = np.zeros_like(au_int)
        au_del[1:] = np.mean(np.abs(au[1:] - au[:-1]), axis=1)
        score  += 0.65 * au_int + 0.35 * au_del
    return score * np.clip(conf, 0, 1) * np.clip(success, 0, 1)


def select_frame_indices(video_path: Path, df: Optional[pd.DataFrame],
                          n_frames: int, pool: int) -> List[int]:
    cap   = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if total <= 0:
        total = max(n_frames, 1)

    scores = openface_row_scores(df)
    if scores is None or len(scores) < 2 or float(np.max(scores)) <= 0:
        idxs = np.linspace(int(0.12 * max(total - 1, 1)),
                           int(0.90 * max(total - 1, 1)), n_frames)
        return [int(np.clip(round(float(x)), 0, max(total - 1, 0))) for x in idxs]

    m           = min(len(scores), total)
    row_to_frm  = np.linspace(0, max(total - 1, 0), m)
    top_rows    = sorted(np.argsort(-scores[:m])[:min(pool, m)].tolist())
    step        = max(1, len(top_rows) // n_frames)
    final_rows  = top_rows[::step][:n_frames]
    while len(final_rows) < n_frames:
        uni = int(np.linspace(0, max(total - 1, 0), n_frames)[len(final_rows)])
        final_rows.append(uni)
    return [int(np.clip(round(float(row_to_frm[r])), 0, max(total - 1, 0)))
            for r in final_rows[:n_frames]]


# ──────────────────────────────────────────────────────────────────
# Audio feature extraction  (V11: hybrid SSL + handcrafted)
# ──────────────────────────────────────────────────────────────────
def _load_handcrafted(path: Path, sr: int, seconds: float) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (waveform, handcrafted_vector)."""
    try:
        import librosa
    except ImportError as exc:
        raise RuntimeError("pip install librosa") from exc

    y, _ = librosa.load(str(path), sr=sr, mono=True, duration=seconds)
    max_len = int(sr * seconds)
    y = np.pad(y, (0, max(0, max_len - len(y))))[:max_len]

    mfcc   = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    d_mfcc = librosa.feature.delta(mfcc)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    mel    = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw       = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spec_rolloff  = librosa.feature.spectral_rolloff(y=y, sr=sr)
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    rms    = float(np.sqrt(np.mean(y ** 2)) + 1e-8)
    zcr    = float(librosa.feature.zero_crossing_rate(y)[0].mean())

    # ESRA-RELA++: add prosody-inspired descriptors.  pyin can fail on short/noisy clips;
    # in that case we add zeros rather than crashing the pipeline.
    try:
        f0, _, _ = librosa.pyin(y, fmin=float(librosa.note_to_hz("C2")), fmax=float(librosa.note_to_hz("C7")), sr=sr)
        f0_clean = f0[np.isfinite(f0)] if f0 is not None else np.array([], dtype=np.float32)
        if len(f0_clean) > 0:
            f0_stats = np.array([
                float(np.mean(f0_clean)), float(np.std(f0_clean)),
                float(np.min(f0_clean)), float(np.max(f0_clean)),
                float(np.median(f0_clean)), float(len(f0_clean) / max(len(f0), 1)),
            ], dtype=np.float32)
        else:
            f0_stats = np.zeros(6, dtype=np.float32)
    except Exception:
        f0_stats = np.zeros(6, dtype=np.float32)

    hand   = np.concatenate([
        mfcc.mean(1), mfcc.std(1), d_mfcc.mean(1), d_mfcc.std(1),
        chroma.mean(1), chroma.std(1),
        np.log(mel + 1e-6).mean(1), np.log(mel + 1e-6).std(1),
        spec_centroid.mean(1), spec_centroid.std(1),
        spec_bw.mean(1), spec_bw.std(1),
        spec_rolloff.mean(1), spec_rolloff.std(1),
        spec_contrast.mean(1), spec_contrast.std(1),
        f0_stats,
        [rms, zcr],
    ]).astype(np.float32)
    return y.astype(np.float32), hand


def _ssl_embed(y: np.ndarray, sr: int, model_name: str) -> np.ndarray:
    global _AUDIO_EXTRACTOR, _AUDIO_MODEL, _AUDIO_MODEL_NAME
    import torch
    from transformers import AutoFeatureExtractor, AutoModel
    if _AUDIO_MODEL is None or _AUDIO_MODEL_NAME != model_name:
        print(f"[INFO] Loading SSL audio model: {model_name}")
        _AUDIO_EXTRACTOR = AutoFeatureExtractor.from_pretrained(model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _AUDIO_MODEL = AutoModel.from_pretrained(model_name).to(device).eval()
        _AUDIO_MODEL_NAME = model_name

    extractor = _AUDIO_EXTRACTOR
    model     = _AUDIO_MODEL
    device    = next(model.parameters()).device

    with torch.no_grad():
        inp = extractor(y, sampling_rate=sr, return_tensors="pt", padding=True)
        inp = {k: v.to(device) for k, v in inp.items()}
        h   = model(**inp).last_hidden_state.squeeze(0).detach().float().cpu().numpy()
    return np.concatenate([h.mean(0), h.std(0), h.max(0)]).astype(np.float32)


def extract_audio_feature(sample: Sample, cfg: Config) -> Tuple[np.ndarray, Dict[str, float]]:
    cdir  = Path(cfg.cache_dir) / "audio"
    ensure_dir(cdir)
    cfile = cdir / f"{cache_key(sample, cfg, 'audio')}.npz"
    if cfile.exists() and not cfg.rebuild_cache:
        d = np.load(cfile, allow_pickle=True)
        return d["feat"].astype(np.float32), json.loads(str(d["quality"].item()))

    y, hand = _load_handcrafted(sample.path, cfg.audio_sr, cfg.audio_seconds)
    quality: Dict[str, float] = {
        "audio_rms": float(np.sqrt(np.mean(y ** 2))),
        "audio_zcr": float(np.mean(np.diff(np.sign(y)) != 0)),
    }

    if cfg.audio_model.lower() != "none":
        ssl  = _ssl_embed(y, cfg.audio_sr, cfg.audio_model)
        feat = np.concatenate([ssl, hand]).astype(np.float32)   # SSL + handcrafted
    else:
        feat = hand.copy()

    quality["audio_feat_norm"] = float(np.linalg.norm(feat))
    save_npz(cfile, feat=feat, quality=json.dumps(quality))
    return feat, quality


# ──────────────────────────────────────────────────────────────────
# Video feature extraction  (MobileNetV3-Small + saliency selection)
# ──────────────────────────────────────────────────────────────────
def _face_crop_from_of(frame_rgb: np.ndarray, df: Optional[pd.DataFrame],
                        frame_idx: int) -> np.ndarray:
    """Crop the face using all 68 OpenFace landmark points.

    Important fix:
    OpenFace columns x_0..x_67/y_0..y_67 are landmark coordinates, not a
    ready-made bounding box. Earlier code used only x_1/x_2/y_1/y_2, which
    can create a tiny/incorrect crop. This version builds a real landmark
    bounding box from all available points and adds a margin.
    """
    if df is None or len(df) == 0:
        return frame_rgb

    row_id = int(np.clip(frame_idx, 0, len(df) - 1))
    row = df.iloc[row_id]
    h, w = frame_rgb.shape[:2]

    xs: List[float] = []
    ys: List[float] = []
    for i in range(68):
        xc, yc = f"x_{i}", f"y_{i}"
        if xc in df.columns and yc in df.columns:
            try:
                x_val = float(row[xc])
                y_val = float(row[yc])
                if np.isfinite(x_val) and np.isfinite(y_val):
                    xs.append(x_val)
                    ys.append(y_val)
            except Exception:
                pass

    if len(xs) < 10 or len(ys) < 10:
        return frame_rgb

    x1 = max(0, int(min(xs)))
    x2 = min(w, int(max(xs)))
    y1 = max(0, int(min(ys)))
    y2 = min(h, int(max(ys)))
    bw, bh = x2 - x1, y2 - y1
    if bw < 20 or bh < 20:
        return frame_rgb

    pad = int(0.35 * max(bw, bh))
    x1 = max(0, x1 - pad)
    x2 = min(w, x2 + pad)
    y1 = max(0, y1 - pad)
    y2 = min(h, y2 + pad)

    crop = frame_rgb[y1:y2, x1:x2]
    return crop if crop.size > 0 else frame_rgb

def _load_video_model() -> Tuple[Any, Any]:
    global _VIDEO_MODEL, _VIDEO_TRANSFORM, _VIDEO_EMBED_DIM
    if _VIDEO_MODEL is None:
        import torch
        import torch.nn as nn
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
        weights  = MobileNet_V3_Small_Weights.DEFAULT
        base     = mobilenet_v3_small(weights=weights)
        # MobileNetV3-Small feature dimension is normally 576, but read it
        # dynamically from the classifier so the fallback feature size stays valid
        # if the backbone is changed later.
        try:
            first_classifier_layer: Any = base.classifier[0]
            _VIDEO_EMBED_DIM = int(getattr(first_classifier_layer, "in_features", 576))
        except Exception:
            _VIDEO_EMBED_DIM = 576
        device   = "cuda" if torch.cuda.is_available() else "cpu"
        _VIDEO_MODEL = nn.Sequential(
            base.features, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()
        ).to(device).eval()
        _VIDEO_TRANSFORM = weights.transforms()
    return _VIDEO_MODEL, _VIDEO_TRANSFORM


def extract_video_feature(sample: Sample, cfg: Config,
                           df: Optional[pd.DataFrame]) -> Tuple[np.ndarray, Dict[str, float]]:
    cdir  = Path(cfg.cache_dir) / "video"
    ensure_dir(cdir)
    cfile = cdir / f"{cache_key(sample, cfg, 'video')}.npz"
    if cfile.exists() and not cfg.rebuild_cache:
        d = np.load(cfile, allow_pickle=True)
        return d["feat"].astype(np.float32), json.loads(str(d["quality"].item()))

    import torch
    from PIL import Image as PILImage
    model, transform = _load_video_model()
    device           = next(model.parameters()).device

    idxs = select_frame_indices(sample.path, df, cfg.video_frames, cfg.saliency_top_pool)
    cap  = cv2.VideoCapture(str(sample.path))
    imgs = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb  = _face_crop_from_of(rgb, df, idx)
        imgs.append(transform(PILImage.fromarray(rgb)))
    cap.release()

    quality = {"video_frames_used": float(len(imgs)), "video_saliency": float(df is not None)}
    if not imgs:
        dim = int(_VIDEO_EMBED_DIM or 576)
        # V11 video pooling uses mean/std/max/delta/first/last = 6 blocks.
        feat = np.zeros(dim * 6, np.float32)
        save_npz(cfile, feat=feat, quality=json.dumps(quality))
        return feat, quality

    x   = torch.stack(imgs).to(device)
    with torch.no_grad():
        emb = model(x).detach().float().cpu().numpy()   # (n_frames, feature_dim)

    delta  = np.diff(emb, axis=0)
    d_mean = np.abs(delta).mean(0) if len(delta) > 0 else np.zeros(emb.shape[1], np.float32)
    # V11 keeps temporal-order cues by appending first and last frame embeddings.
    feat   = np.concatenate([emb.mean(0), emb.std(0), emb.max(0), d_mean, emb[0], emb[-1]]).astype(np.float32)
    quality["video_feat_norm"] = float(np.linalg.norm(feat))
    save_npz(cfile, feat=feat, quality=json.dumps(quality))
    return feat, quality


# ──────────────────────────────────────────────────────────────────
# AU feature extraction  (8 statistics × N AUs + reliability header)
# ──────────────────────────────────────────────────────────────────
def extract_au_feature(sample: Sample, cfg: Config,
                        df: Optional[pd.DataFrame]) -> Tuple[np.ndarray, Dict[str, float]]:
    cdir  = Path(cfg.cache_dir) / "au"
    ensure_dir(cdir)
    cfile = cdir / f"{cache_key(sample, cfg, 'au')}.npz"
    if cfile.exists() and not cfg.rebuild_cache:
        d = np.load(cfile, allow_pickle=True)
        return d["feat"].astype(np.float32), json.loads(str(d["quality"].item()))

    fixed_dim = len(AU_LIST) * 8 + 4
    q: Dict[str, float] = {"au_found": 0.0, "au_conf_mean": 0.0, "au_success_rate": 0.0}

    if df is None or len(df) == 0:
        feat = np.zeros(fixed_dim, np.float32)
        save_npz(cfile, feat=feat, quality=json.dumps(q))
        return feat, q

    q["au_found"] = 1.0
    if "confidence" in df.columns:
        q["au_conf_mean"] = float(pd.to_numeric(df["confidence"], errors="coerce").fillna(0).mean())
    if "success" in df.columns:
        q["au_success_rate"] = float(pd.to_numeric(df["success"], errors="coerce").fillna(0).mean())

    cols = [c for c in AU_LIST if c in df.columns]
    if not cols:
        feat = np.zeros(fixed_dim, np.float32)
        save_npz(cfile, feat=feat, quality=json.dumps(q))
        return feat, q

    x  = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(np.float32)
    dx = np.diff(x, axis=0)
    if len(dx) == 0:
        dx = np.zeros_like(x)

    # 8 statistics per AU
    stats = [
        x.mean(0), x.std(0), x.max(0), x.min(0),
        np.median(x, 0),
        np.quantile(x, 0.75, 0) - np.quantile(x, 0.25, 0),
        np.abs(dx).mean(0),
        x[-1] - x[0],
    ]
    raw  = np.concatenate(stats).astype(np.float32)
    body = np.zeros(len(AU_LIST) * 8, np.float32)
    body[:min(len(body), len(raw))] = raw[:len(body)]
    rel  = np.array([q["au_found"], q["au_conf_mean"], q["au_success_rate"],
                     float(np.linalg.norm(body))], np.float32)
    feat = np.concatenate([body, rel]).astype(np.float32)
    save_npz(cfile, feat=feat, quality=json.dumps(q))
    return feat, q



# ──────────────────────────────────────────────────────────────────
# ESRA-RELA++: AU dynamic / FACS trajectory features
# ──────────────────────────────────────────────────────────────────
def extract_au_dynamic_feature(sample: Sample, cfg: Config,
                               df: Optional[pd.DataFrame]) -> Tuple[np.ndarray, Dict[str, float]]:
    """OpenFace AU sequence branch.

    The normal AU branch stores global statistics.  This V11++ branch keeps
    coarse temporal shape: segment-wise means/maxima, velocity, peak timing,
    and FACS group activity.  It is still a fixed vector, so it remains stable
    on small RAVDESS folds and can be used with the same OOF stacking.
    """
    cdir  = Path(cfg.cache_dir) / "au_dynamic"
    ensure_dir(cdir)
    cfile = cdir / f"{cache_key(sample, cfg, 'au_dynamic')}.npz"
    if cfile.exists() and not cfg.rebuild_cache:
        d = np.load(cfile, allow_pickle=True)
        return d["feat"].astype(np.float32), json.loads(str(d["quality"].item()))

    n_au = len(AU_LIST)
    n_segments = 5
    # Fixed vector layout:
    #   n_au * n_segments     segment means
    # + n_au * n_segments     segment maxima
    # + n_au                  mean AU velocity
    # + n_au                  max AU velocity
    # + n_au                  normalized peak timing
    # + 18                    FACS group trajectory statistics
    # + 4                     reliability header
    fixed_dim = n_au * (n_segments * 2 + 3) + 18 + 4
    q: Dict[str, float] = {"au_dyn_found": 0.0, "au_dyn_conf_mean": 0.0, "au_dyn_success_rate": 0.0}

    if df is None or len(df) == 0:
        feat = np.zeros(fixed_dim, np.float32)
        save_npz(cfile, feat=feat, quality=json.dumps(q))
        return feat, q

    cols = [c for c in AU_LIST if c in df.columns]
    if not cols:
        feat = np.zeros(fixed_dim, np.float32)
        save_npz(cfile, feat=feat, quality=json.dumps(q))
        return feat, q

    q["au_dyn_found"] = 1.0
    if "confidence" in df.columns:
        q["au_dyn_conf_mean"] = float(pd.to_numeric(df["confidence"], errors="coerce").fillna(0).mean())
    if "success" in df.columns:
        q["au_dyn_success_rate"] = float(pd.to_numeric(df["success"], errors="coerce").fillna(0).mean())

    x_raw = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(np.float32)
    x = np.zeros((len(x_raw), n_au), np.float32)
    for j, c in enumerate(cols):
        if c in AU_LIST:
            x[:, AU_LIST.index(c)] = x_raw[:, j]

    if len(x) == 0:
        feat = np.zeros(fixed_dim, np.float32)
        save_npz(cfile, feat=feat, quality=json.dumps(q))
        return feat, q

    seg_means: List[np.ndarray] = []
    seg_maxs: List[np.ndarray] = []
    boundaries = np.linspace(0, len(x), n_segments + 1, dtype=int)
    for k in range(n_segments):
        a, b = int(boundaries[k]), int(boundaries[k + 1])
        seg = x[a:b] if b > a else x[max(0, min(a, len(x) - 1)):max(0, min(a, len(x) - 1)) + 1]
        seg_means.append(seg.mean(axis=0))
        seg_maxs.append(seg.max(axis=0))

    dx = np.diff(x, axis=0)
    if len(dx) == 0:
        dx = np.zeros_like(x)
    vel_mean = np.abs(dx).mean(axis=0)
    vel_max  = np.abs(dx).max(axis=0)
    peak_pos = np.argmax(x, axis=0).astype(np.float32) / max(float(len(x) - 1), 1.0)

    # FACS group trajectory features.  Groups are coarse and interpretable.
    group_map = {
        "brow":  ["AU01_r", "AU02_r", "AU04_r"],
        "eye":   ["AU05_r", "AU06_r", "AU07_r", "AU45_r"],
        "nose":  ["AU09_r", "AU10_r"],
        "mouth": ["AU12_r", "AU14_r", "AU15_r", "AU17_r", "AU20_r", "AU23_r", "AU25_r", "AU26_r"],
    }
    group_feats: List[float] = []
    for names in group_map.values():
        ids = [AU_LIST.index(c) for c in names if c in AU_LIST]
        if ids:
            gx = x[:, ids].mean(axis=1)
            gdx = np.abs(np.diff(gx)) if len(gx) > 1 else np.zeros(1, np.float32)
            group_feats.extend([
                float(gx.mean()), float(gx.std()), float(gx.max()),
                float(gx[-1] - gx[0]), float(gdx.mean()),
            ])
        else:
            group_feats.extend([0.0] * 5)
    # Pairwise interpretable differences: mouth-eye, brow-mouth, eye-brow, nose-mouth.
    while len(group_feats) < 20:
        group_feats.append(0.0)
    gf = np.array(group_feats[:18], np.float32)

    body = np.concatenate(seg_means + seg_maxs + [vel_mean, vel_max, peak_pos, gf]).astype(np.float32)
    feat = np.zeros(fixed_dim, np.float32)
    feat[:min(len(feat) - 4, len(body))] = body[:min(len(feat) - 4, len(body))]
    feat[-4:] = np.array([
        q["au_dyn_found"], q["au_dyn_conf_mean"], q["au_dyn_success_rate"], float(np.linalg.norm(body))
    ], np.float32)
    save_npz(cfile, feat=feat, quality=json.dumps(q))
    return feat, q

# ──────────────────────────────────────────────────────────────────
# Feature matrix builder
# ──────────────────────────────────────────────────────────────────
def build_feature_matrix(
    samples: List[Sample], cfg: Config
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray]:
    cdir      = Path(cfg.cache_dir)
    ensure_dir(cdir)
    mat_file  = cdir / "esra_rela_v15_lora_matrix.npz"
    meta_file = cdir / "esra_rela_v15_lora_matrix_meta.json"
    of_dir: Optional[Path] = Path(cfg.openface_dir) if cfg.openface_dir else None

    meta = {
        "cache_version": FEATURE_CACHE_VERSION,
        "files": [str(s.path.resolve()) for s in samples],
        "openface_fingerprint": openface_dataset_fingerprint(samples, of_dir),
        "cfg": {
            "audio_model": cfg.audio_model, "audio_sr": cfg.audio_sr,
            "audio_seconds": cfg.audio_seconds,
            "video_frames": cfg.video_frames,
            "saliency_top_pool": cfg.saliency_top_pool,
            "use_audio": cfg.use_audio, "use_video": cfg.use_video,
            "use_au": cfg.use_au, "use_au_dynamic": cfg.use_au_dynamic,
            "include_metadata_covariates": cfg.include_metadata_covariates,
            "version": "v15_lora_publication_final",
        },
    }
    if mat_file.exists() and meta_file.exists() and not cfg.rebuild_cache:
        try:
            if json.loads(meta_file.read_text("utf-8")) == meta:
                print(f"[INFO] Loading feature matrix: {mat_file}")
                d = np.load(mat_file)
                return (d["audio"], d["video"], d["au"], d["au_dyn"], d["rel"],
                        d["y"], d["groups"], d["genders"])
        except Exception:
            pass

    try:
        from tqdm import tqdm as _tqdm
        iter_fn = lambda it, **kw: _tqdm(it, **kw)
    except ImportError:
        iter_fn = lambda it, **kw: it

    audio_l: List[np.ndarray] = []
    video_l: List[np.ndarray] = []
    au_l:    List[np.ndarray] = []
    au_dyn_l: List[np.ndarray] = []
    rel_l:   List[np.ndarray] = []
    y_l:     List[int]        = []
    groups_l: List[int]       = []
    genders_l: List[int]      = []

    for s in iter_fn(samples, desc="Features"):
        df = (read_openface(s, of_dir)
              if (cfg.use_au or cfg.use_video) and of_dir is not None and of_dir.exists() else None)
        aq: Dict[str, float] = {}
        vq: Dict[str, float] = {}
        uq: Dict[str, float] = {}
        af = np.zeros(1, np.float32)
        vf = np.zeros(1, np.float32)
        uf = np.zeros(1, np.float32)
        udf = np.zeros(1, np.float32)

        if cfg.use_audio:
            af, aq = extract_audio_feature(s, cfg)
        if cfg.use_video:
            vf, vq = extract_video_feature(s, cfg, df)
        if cfg.use_au:
            uf, uq = extract_au_feature(s, cfg, df)
            if cfg.use_au_dynamic:
                udf, udq = extract_au_dynamic_feature(s, cfg, df)
                uq.update(udq)

        rel_values = [
            aq.get("audio_rms", 0.0),
            aq.get("audio_zcr", 0.0),
            aq.get("audio_feat_norm", 0.0),
            vq.get("video_frames_used", 0.0) / max(1.0, float(cfg.video_frames)),
            vq.get("video_saliency", 0.0),
            vq.get("video_feat_norm", 0.0),
            uq.get("au_found", 0.0),
            uq.get("au_conf_mean", 0.0),
            uq.get("au_success_rate", 0.0),
            uq.get("au_dyn_found", 0.0),
            uq.get("au_dyn_conf_mean", 0.0),
            uq.get("au_dyn_success_rate", 0.0),
        ]
        if cfg.include_metadata_covariates:
            # Explicit ablation only; disabled by default for publication safety.
            rel_values.extend([float(s.intensity), float(s.statement), float(s.repetition)])
        r = np.array(rel_values, np.float32)

        audio_l.append(af);   video_l.append(vf);   au_l.append(uf); au_dyn_l.append(udf)
        rel_l.append(r);      y_l.append(s.label);  groups_l.append(s.actor)
        genders_l.append(0 if s.gender == "male" else (1 if s.gender == "female" else -1))

    audio   = np.vstack(audio_l).astype(np.float32)
    video   = np.vstack(video_l).astype(np.float32)
    au      = np.vstack(au_l).astype(np.float32)
    au_dyn  = np.vstack(au_dyn_l).astype(np.float32)
    rel     = np.vstack(rel_l).astype(np.float32)
    y       = np.array(y_l,      np.int64)
    groups  = np.array(groups_l, np.int64)
    genders = np.array(genders_l,np.int64)

    save_npz(mat_file, audio=audio, video=video, au=au, au_dyn=au_dyn,
             rel=rel, y=y, groups=groups, genders=genders)
    meta_file.write_text(json.dumps(meta, indent=2), "utf-8")
    return audio, video, au, au_dyn, rel, y, groups, genders


# ──────────────────────────────────────────────────────────────────
# Classifier helpers
# ──────────────────────────────────────────────────────────────────
def _make_pipeline(kind: str, C: float, pca_dim: int,
                   n_features: int, seed: int) -> Pipeline:
    """Build a StandardScaler → (optional PCA) → classifier pipeline.

    Bug fix from ESRA-CMT: old code used the string 'passthrough' as a step
    value, which fails in sklearn < 1.1. Now we conditionally include PCA.
    """
    steps: List[Tuple[str, Any]] = [("scaler", StandardScaler())]
    if pca_dim > 0 and n_features > pca_dim + 5:
        steps.append(("pca", PCA(n_components=min(pca_dim, n_features - 1),
                                  random_state=seed)))
    if kind == "svm":
        base = LinearSVC(C=C, class_weight="balanced",
                          random_state=seed, max_iter=5000)
        clf  = CalibratedClassifierCV(base, cv=3)
    else:
        clf = LogisticRegression(C=C, class_weight="balanced",
                                  max_iter=2500, solver="lbfgs",
                                  random_state=seed)
    steps.append(("clf", clf))
    return Pipeline(steps)


def aligned_proba(model: Pipeline, X: np.ndarray) -> np.ndarray:
    """Return (N, 8) probability matrix even if classifier saw only a subset of classes.

    Bug fix: classifiers trained on inner folds can miss a rare emotion entirely.
    aligned_proba fills the missing columns with zeros so the shape is always (N,8).
    """
    raw     = model.predict_proba(X)
    classes = model.named_steps["clf"].classes_
    out     = np.zeros((len(X), len(LABEL_NAMES)), np.float32)
    for j, c in enumerate(classes):
        out[:, int(c)] = raw[:, j]
    return out


# ──────────────────────────────────────────────────────────────────
# NEW V11: posterior-temperature calibration
# ──────────────────────────────────────────────────────────────────
def fit_temperature(probs_oof: np.ndarray, y_oof: np.ndarray,
                    T_init: float = 1.5) -> float:
    """Grid-search a scalar smoothing/sharpening temperature on OOF posteriors.

    Because sklearn base models expose probabilities rather than neural logits,
    this uses softmax(log(p) / T), implemented as p**(1/T) then renormalized.
    The calibrated OOF probabilities are returned to the meta-learner together
    with calibrated test probabilities, so train/test meta inputs are matched.
    """
    best_T, best_nll = T_init, float("inf")
    for T in np.linspace(0.5, 5.0, 46):
        p = np.clip(probs_oof, 1e-9, 1.0) ** (1.0 / T)
        p = p / p.sum(axis=1, keepdims=True)
        nll = -float(np.mean(np.log(p[np.arange(len(y_oof)), y_oof.astype(int)])))
        if nll < best_nll:
            best_nll, best_T = nll, float(T)
    return best_T


def apply_temperature(probs: np.ndarray, T: float) -> np.ndarray:
    T = max(T, 1e-3)
    p = np.clip(probs, 1e-9, 1.0) ** (1.0 / T)
    p = p / p.sum(axis=1, keepdims=True)
    return p.astype(np.float32)


# ──────────────────────────────────────────────────────────────────
# OOF stacking
# ──────────────────────────────────────────────────────────────────
def oof_and_test_probs(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray,
    train_idx: np.ndarray, test_idx: np.ndarray,
    cfg: Config, pca_dim: int, seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    Xtr, ytr, gtr = X[train_idx], y[train_idx], groups[train_idx]
    Xt = X[test_idx]
    n_splits = min(cfg.inner_splits, len(np.unique(gtr)))
    if n_splits < 2:
        raise RuntimeError("Need ≥2 actor groups for inner OOF stacking.")
    if n_splits < cfg.inner_splits:
        warnings.warn(
            f"Reduced inner_splits from {cfg.inner_splits} to {n_splits} because only "
            f"{len(np.unique(gtr))} train actor groups are available."
        )

    oof = np.zeros((len(train_idx), len(LABEL_NAMES)), np.float32)
    for tr_l, va_l in GroupKFold(n_splits=n_splits).split(Xtr, ytr, gtr):
        m = _make_pipeline(cfg.base_model, cfg.base_C, pca_dim, Xtr.shape[1], seed)
        m.fit(Xtr[tr_l], ytr[tr_l])
        oof[va_l] = aligned_proba(m, Xtr[va_l])

    final = _make_pipeline(cfg.base_model, cfg.base_C, pca_dim, Xtr.shape[1], seed + 99)
    final.fit(Xtr, ytr)
    test_probs = aligned_proba(final, Xt)

    # NEW V11: temperature calibration
    if cfg.calibrate_temperatures:
        T          = fit_temperature(oof, ytr, cfg.temp_init)
        oof        = apply_temperature(oof, T)
        test_probs = apply_temperature(test_probs, T)

    return oof, test_probs


# ──────────────────────────────────────────────────────────────────
# Pair specialists  (from RELA-HLF, expanded)
# ──────────────────────────────────────────────────────────────────
def _build_specialist(X: np.ndarray, yy: np.ndarray, seed: int, max_pca: int = 128) -> Pipeline:
    # Xfull can be high-dimensional because it concatenates audio, video, AU,
    # AU-dynamic, and reliability vectors. A slightly larger specialist PCA
    # preserves more AU-dynamic information than the older 96-D cap.
    n_pca = min(int(max_pca), X.shape[1] - 1, len(yy) - 2)
    if n_pca >= 5:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_pca, random_state=seed)),
            ("clf", LogisticRegression(C=0.40, class_weight="balanced",
                                        max_iter=1500, random_state=seed)),
        ])
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=0.40, class_weight="balanced",
                                    max_iter=1500, random_state=seed)),
    ])


def train_pair_specialists(
    Xfull: np.ndarray, y: np.ndarray, train_idx: np.ndarray, cfg: Config, fold_id: int
) -> Dict[Tuple[int, int], Pipeline]:
    out: Dict[Tuple[int, int], Pipeline] = {}
    for pair_i, (a_name, b_name) in enumerate(SPECIALIST_PAIRS):
        a, b = NAME_TO_LABEL[a_name], NAME_TO_LABEL[b_name]
        idx  = train_idx[np.isin(y[train_idx], [a, b])]
        if len(np.unique(y[idx])) < 2 or len(idx) < 10:
            continue
        yy = (y[idx] == b).astype(int)
        m  = _build_specialist(Xfull[idx], yy, cfg.random_state + fold_id * 1000 + pair_i, cfg.specialist_pca)
        m.fit(Xfull[idx], yy)
        out[(a, b)] = m
    return out


def train_gender_pair_specialists(
    Xfull: np.ndarray, y: np.ndarray, genders: np.ndarray,
    train_idx: np.ndarray, cfg: Config, fold_id: int,
) -> Dict[Tuple[int, int, int], Pipeline]:
    """V11: separate binary classifiers per actor-gender group for each weak pair."""
    out: Dict[Tuple[int, int, int], Pipeline] = {}
    for g in (0, 1):
        g_idx = train_idx[genders[train_idx] == g]
        for pair_i, (a_name, b_name) in enumerate(SPECIALIST_PAIRS):
            a, b = NAME_TO_LABEL[a_name], NAME_TO_LABEL[b_name]
            idx  = g_idx[np.isin(y[g_idx], [a, b])]
            if len(np.unique(y[idx])) < 2 or len(idx) < 8:
                continue
            yy = (y[idx] == b).astype(int)
            seed = cfg.random_state + fold_id * 1000 + pair_i + 100 * g
            m  = _build_specialist(Xfull[idx], yy, seed, cfg.specialist_pca)
            m.fit(Xfull[idx], yy)
            out[(a, b, g)] = m
    return out


def apply_specialists(
    probs: np.ndarray,
    Xfull_test: np.ndarray,
    specialists: Dict[Tuple[int, int], Pipeline],
    genders_test: Optional[np.ndarray],
    gender_specialists: Optional[Dict[Tuple[int, int, int], Pipeline]],
    cfg: Config,
) -> np.ndarray:
    """Apply pair specialists with batched predict_proba calls.

    V11 change: V8 called predict_proba once per sample per specialist.
    Here, each specialist predicts the whole test fold once, then the same
    pair-trigger logic is applied sample-by-sample. This is much faster and
    produces the same type of blended probabilities.
    """
    if not specialists:
        return probs

    new_probs = probs.copy()
    top3      = np.argsort(-probs, axis=1)[:, :3]
    blend     = float(np.clip(cfg.specialist_blend, 0, 1))

    specialist_probs: Dict[Tuple[int, int], np.ndarray] = {
        key: model.predict_proba(Xfull_test) for key, model in specialists.items()
    }
    gender_specialist_probs: Dict[Tuple[int, int, int], np.ndarray] = {}
    if gender_specialists is not None:
        gender_specialist_probs = {
            key: model.predict_proba(Xfull_test) for key, model in gender_specialists.items()
        }

    for i in range(len(probs)):
        g = int(genders_test[i]) if genders_test is not None else -1
        for (a, b), sp_all in specialist_probs.items():
            pair_prob   = float(probs[i, a] + probs[i, b])
            pair_in_top = (a in top3[i]) and (b in top3[i])
            if not pair_in_top and pair_prob < cfg.specialist_threshold:
                continue
            if (abs(float(probs[i, a] - probs[i, b])) > cfg.specialist_margin_threshold
                    and not pair_in_top):
                continue

            pa, pb = float(sp_all[i, 0]), float(sp_all[i, 1])

            # V11: blend gender-specific specialist when available.
            g_key = (a, b, g)
            if g_key in gender_specialist_probs:
                gsp = gender_specialist_probs[g_key]
                pa  = 0.6 * pa + 0.4 * float(gsp[i, 0])
                pb  = 0.6 * pb + 0.4 * float(gsp[i, 1])

            total = max(pair_prob, cfg.specialist_threshold)
            new_probs[i, a] = (1 - blend) * new_probs[i, a] + blend * total * pa
            new_probs[i, b] = (1 - blend) * new_probs[i, b] + blend * total * pb
            s = float(new_probs[i].sum())
            if s > 1e-8:
                new_probs[i] /= s
    return new_probs


# ──────────────────────────────────────────────────────────────────
# ESRA-RELA++: Sad-vs-rest specialist correction
# ──────────────────────────────────────────────────────────────────
def train_sad_specialist(
    Xfull: np.ndarray, y: np.ndarray, train_idx: np.ndarray, cfg: Config, fold_id: int
) -> Optional[Pipeline]:
    """Binary Sad-vs-rest classifier fitted only on the outer-train fold."""
    sad = NAME_TO_LABEL["Sad"]
    yy = (y[train_idx] == sad).astype(int)
    if len(np.unique(yy)) < 2 or int(yy.sum()) < 8:
        return None
    seed = cfg.random_state + 9000 + fold_id
    m = _build_specialist(Xfull[train_idx], yy, seed, cfg.specialist_pca)
    m.fit(Xfull[train_idx], yy)
    return m


def apply_sad_specialist(
    probs: np.ndarray, Xfull_test: np.ndarray, sad_model: Optional[Pipeline], cfg: Config
) -> np.ndarray:
    """Boost Sad only when the binary Sad specialist is confident.

    This targets the largest V10 failure: Sad was distributed across Neutral,
    Fearful, Disgust, and Surprised.  The update is conservative and normalized.
    """
    if sad_model is None:
        return probs
    sad = NAME_TO_LABEL["Sad"]
    confusing = {NAME_TO_LABEL[n] for n in ["Neutral", "Fearful", "Disgust", "Surprised", "Calm"]}
    new_probs = probs.copy()
    sp = sad_model.predict_proba(Xfull_test)
    sad_binary = sp[:, 1] if sp.shape[1] > 1 else np.zeros(len(probs), np.float32)
    top3 = np.argsort(-probs, axis=1)[:, :3]
    blend = float(np.clip(cfg.sad_specialist_blend, 0.0, 1.0))
    for i in range(len(probs)):
        top1 = int(np.argmax(probs[i]))
        trigger = (
            sad_binary[i] >= cfg.sad_specialist_threshold
            and (sad in top3[i] or top1 in confusing)
        )
        # Avoid over-correcting confident non-Sad predictions when the
        # 8-class meta-learner has essentially no Sad evidence.
        if not trigger or float(probs[i, sad]) < float(cfg.sad_min_base_prob):
            continue
        current_sad = float(new_probs[i, sad])
        sad_score = float(sad_binary[i])
        if sad_score > current_sad:
            # Positive correction: boost Sad only when the binary specialist
            # is more confident than the 8-class meta prediction.
            new_probs[i, sad] = (1.0 - blend) * current_sad + blend * sad_score
        elif sad_score < 0.50 and current_sad > sad_score:
            # Negative correction: if the Sad specialist is explicitly doubtful,
            # gently suppress Sad.  This makes the specialist bidirectional
            # instead of boost-only.
            suppress_blend = 0.5 * blend
            new_probs[i, sad] = (1.0 - suppress_blend) * current_sad + suppress_blend * sad_score
        s = float(new_probs[i].sum())
        if s > 1e-8:
            new_probs[i] /= s
    return new_probs.astype(np.float32)


# ──────────────────────────────────────────────────────────────────
# ESRA-RELA++ V12: fold-safe LoRA audio branch
# ──────────────────────────────────────────────────────────────────
def _load_lora_waveform(sample: Sample, cfg: Config) -> np.ndarray:
    """Cache waveform for the optional LoRA audio classifier."""
    cdir = Path(cfg.cache_dir) / "lora_waveforms"
    ensure_dir(cdir)
    cfile = cdir / f"{cache_key(sample, cfg, 'lora_waveform')}.npz"
    if cfile.exists() and not cfg.rebuild_cache:
        return np.load(cfile)["wave"].astype(np.float32)
    try:
        import librosa
    except ImportError as exc:
        raise RuntimeError("LoRA audio branch needs librosa. Install: pip install librosa") from exc
    y, _ = librosa.load(str(sample.path), sr=cfg.audio_sr, mono=True, duration=cfg.audio_seconds)
    max_len = int(cfg.audio_sr * cfg.audio_seconds)
    y = np.pad(y, (0, max(0, max_len - len(y))))[:max_len].astype(np.float32)
    save_npz(cfile, wave=y)
    return y


class _WaveDataset(TorchDataset):
    """Minimal torch Dataset wrapper used by the LoRA audio branch."""
    def __init__(self, waves: Sequence[np.ndarray], labels: Optional[Any] = None):
        self.waves = list(waves)
        self.labels = None if labels is None else np.asarray(labels, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.waves)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item: Dict[str, Any] = {"wave": self.waves[idx]}
        if self.labels is not None:
            item["label"] = int(self.labels[idx])
        return item


def _make_audio_collate(extractor: Any, cfg: Config):
    def collate(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        import torch
        waves = [b["wave"] for b in batch]
        inputs = extractor(waves, sampling_rate=cfg.audio_sr, return_tensors="pt", padding=True)
        has_labels = all("label" in b for b in batch)
        if has_labels:
            inputs["labels"] = torch.tensor([int(b["label"]) for b in batch], dtype=torch.long)
        return inputs
    return collate


def _build_lora_audio_model(cfg: Config) -> Tuple[Any, Any]:
    """Create a PEFT/LoRA audio classifier. Imports are local so base code works without PEFT."""
    import torch
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
    try:
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError as exc:
        raise RuntimeError(
            "--use_lora_audio requires PEFT. Install it with: pip install peft accelerate"
        ) from exc

    extractor = AutoFeatureExtractor.from_pretrained(cfg.lora_audio_model)
    model = AutoModelForAudioClassification.from_pretrained(
        cfg.lora_audio_model,
        num_labels=len(LABEL_NAMES),
        ignore_mismatched_sizes=True,
    )
    targets = [t.strip() for t in cfg.lora_target_modules.split(",") if t.strip()]
    try:
        lora_cfg = LoraConfig(
            r=int(cfg.lora_r),
            lora_alpha=int(cfg.lora_alpha),
            lora_dropout=float(cfg.lora_dropout),
            bias="none",
            target_modules=targets,
            task_type=TaskType.SEQ_CLS,
        )
        model = get_peft_model(model, lora_cfg)
    except Exception as first_error:
        # Some PEFT versions are stricter about TaskType for audio models.
        # Retry with a generic LoRA config while keeping the same target modules.
        lora_cfg = LoraConfig(
            r=int(cfg.lora_r),
            lora_alpha=int(cfg.lora_alpha),
            lora_dropout=float(cfg.lora_dropout),
            bias="none",
            target_modules=targets,
        )
        try:
            model = get_peft_model(model, lora_cfg)
        except Exception as second_error:
            raise RuntimeError(
                "Could not attach LoRA adapters to the selected audio model. "
                "Try --lora_target_modules q_proj,k_proj,v_proj,out_proj or use "
                "--lora_audio_model microsoft/wavlm-base-plus. "
                f"Original error: {first_error}; retry error: {second_error}"
            ) from second_error

    # Make sure the newly initialized emotion head can learn even if PEFT freezes base weights.
    for name, param in model.named_parameters():
        lname = name.lower()
        if any(k in lname for k in ["classifier", "projector", "score"]):
            param.requires_grad = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return extractor, model


def _train_lora_audio_model(
    train_waves: Sequence[np.ndarray],
    train_y: np.ndarray,
    cfg: Config,
    seed: int,
    train_groups: Optional[np.ndarray] = None,
) -> Tuple[Any, Any]:
    """Train one LoRA audio classifier on the provided fold-safe training set.

    V14 change: if lora_patience > 0 and actor/group IDs are provided, early
    stopping uses a small actor-safe validation split carved only from the
    current fold's training actors.  No outer-test actor can enter this split.
    If such a split cannot be formed, the model trains for fixed epochs and
    reports that early stopping was disabled.
    """
    import random
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Important: the seed is set before _build_lora_audio_model(), because PEFT
    # initializes LoRA adapter weights when get_peft_model() is called.
    # np.random.default_rng(seed) below is independent from np.random.seed(seed),
    # but both are explicitly seeded for reproducibility.
    if cfg.lora_max_train_samples and cfg.lora_max_train_samples > 0 and len(train_waves) > cfg.lora_max_train_samples:
        rng = np.random.default_rng(seed)
        chosen = rng.choice(len(train_waves), size=cfg.lora_max_train_samples, replace=False)
        train_waves = [train_waves[int(i)] for i in chosen]
        train_y = train_y[chosen]
        if train_groups is not None:
            train_groups = np.asarray(train_groups)[chosen]

    # Actor-safe mini-validation for LoRA early stopping.
    # This is only used for selecting the LoRA epoch inside the training fold.
    train_positions = np.arange(len(train_waves), dtype=np.int64)
    val_positions: np.ndarray = np.array([], dtype=np.int64)
    patience = max(0, int(cfg.lora_patience))
    if patience > 0 and train_groups is not None:
        groups_arr = np.asarray(train_groups)
        unique_groups = np.unique(groups_arr)
        if len(unique_groups) >= 3:
            rng = np.random.default_rng(seed + 12345)
            n_val_groups = max(1, int(round(0.15 * len(unique_groups))))
            val_groups = set(rng.choice(unique_groups, size=n_val_groups, replace=False).tolist())
            val_mask = np.array([g in val_groups for g in groups_arr], dtype=bool)
            candidate_train = train_positions[~val_mask]
            candidate_val = train_positions[val_mask]
            # Keep the validation split only if both sides contain enough samples and labels.
            # The validation side also needs at least two classes; otherwise the
            # early-stop loss is not a meaningful multi-class generalization signal.
            if (len(candidate_train) >= 16 and len(candidate_val) >= 8
                    and len(np.unique(train_y[candidate_train])) >= 2
                    and len(np.unique(train_y[candidate_val])) >= 2):
                train_positions = candidate_train
                val_positions = candidate_val
            else:
                print("        [LoRA-audio] actor-safe validation split was too small or single-class; fixed-epoch training used.")
        else:
            print("        [LoRA-audio] fewer than 3 actor groups; fixed-epoch training used.")

    extractor, model = _build_lora_audio_model(cfg)
    collate = _make_audio_collate(extractor, cfg)
    train_loader = DataLoader(
        _WaveDataset([train_waves[int(i)] for i in train_positions], train_y[train_positions].astype(int).tolist()),
        batch_size=max(1, int(cfg.lora_batch_size)),
        shuffle=True,
        collate_fn=collate,
        num_workers=0,
    )
    val_loader = None
    if len(val_positions) > 0:
        val_loader = DataLoader(
            _WaveDataset([train_waves[int(i)] for i in val_positions], train_y[val_positions].astype(int).tolist()),
            batch_size=max(1, int(cfg.lora_batch_size)),
            shuffle=False,
            collate_fn=collate,
            num_workers=0,
        )
        print(f"        [LoRA-audio] actor-safe early-stop split: train={len(train_positions)}, val={len(val_positions)}")

    device = next(model.parameters()).device
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg.lora_lr),
        weight_decay=float(cfg.lora_weight_decay),
    )
    counts = np.bincount(train_y[train_positions].astype(int), minlength=len(LABEL_NAMES)).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / max(float(weights.mean()), 1e-8)
    class_weight = torch.tensor(weights, dtype=torch.float32, device=device)

    def evaluate_loss(loader: Any) -> float:
        total = 0.0
        n_batches = 0
        model.eval()
        with torch.no_grad():
            for batch in loader:
                labels = batch.pop("labels").to(device)
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(**batch)
                # Validation loss is intentionally unweighted so early stopping
                # measures validation generalization rather than the outer train
                # class distribution used for the training objective.
                loss_val = F.cross_entropy(out.logits, labels)
                total += float(loss_val.detach().cpu())
                n_batches += 1
        # Re-enable training mode so the outer loop continues with LoRA dropout active.
        model.train()
        return total / max(1, n_batches)

    accum = max(1, int(cfg.lora_grad_accum))
    best_metric = float("inf")
    best_trainable_state: Optional[Dict[str, Any]] = None
    bad_epochs = 0

    for epoch in range(max(1, int(cfg.lora_epochs))):
        model.train()
        opt.zero_grad(set_to_none=True)
        running = 0.0
        for step, batch in enumerate(train_loader, start=1):
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            raw_loss = F.cross_entropy(out.logits, labels, weight=class_weight)
            loss_for_backward = raw_loss / accum
            loss_for_backward.backward()
            running += float(raw_loss.detach().cpu())
            if step % accum == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)

        avg_train_loss = running / max(1, len(train_loader))
        if val_loader is not None:
            monitor_loss = evaluate_loss(val_loader)
            monitor_name = "val"
        else:
            monitor_loss = avg_train_loss
            monitor_name = "train"
        print(
            f"        [LoRA-audio] epoch {epoch+1:02d}/{cfg.lora_epochs} | "
            f"train_loss={avg_train_loss:.4f} | {monitor_name}_loss={monitor_loss:.4f}"
        )

        if val_loader is not None and patience > 0:
            if monitor_loss < best_metric - 1e-4:
                best_metric = monitor_loss
                bad_epochs = 0
                # Store only trainable parameters, not the full frozen base model.
                best_trainable_state = {
                    name: param.detach().cpu().clone()
                    for name, param in model.named_parameters()
                    if param.requires_grad
                }
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    print(f"        [LoRA-audio] early stop: no actor-safe validation improvement for {patience} epochs")
                    break

    # For fixed-epoch runs val_loader is None, so best_trainable_state remains None
    # and this block intentionally does nothing.
    if best_trainable_state is not None:
        with torch.no_grad():
            named_params = dict(model.named_parameters())
            for name, value in best_trainable_state.items():
                if name in named_params:
                    named_params[name].copy_(value.to(device))

    model.eval()
    return extractor, model


def _predict_lora_audio_probs(
    extractor: Any,
    model: Any,
    waves: Sequence[np.ndarray],
    cfg: Config,
) -> np.ndarray:
    import torch
    from torch.utils.data import DataLoader

    if len(waves) == 0:
        return np.zeros((0, len(LABEL_NAMES)), np.float32)
    collate = _make_audio_collate(extractor, cfg)
    loader = DataLoader(
        _WaveDataset(waves, None),
        batch_size=max(1, int(cfg.lora_batch_size)),
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
    )
    device = next(model.parameters()).device
    all_probs: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            probs = torch.softmax(logits, dim=-1).detach().float().cpu().numpy()
            all_probs.append(probs.astype(np.float32))
    if not all_probs:
        return np.zeros((len(waves), len(LABEL_NAMES)), np.float32)
    return np.vstack(all_probs).astype(np.float32)


def lora_audio_oof_and_test_probs(
    samples: List[Sample], y: np.ndarray, groups: np.ndarray,
    train_idx: np.ndarray, test_idx: np.ndarray,
    cfg: Config, seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fold-safe LoRA audio OOF probabilities + outer-test probabilities.

    This is the correct way to use LoRA here: it is not a global feature extractor
    trained on all actors.  Each inner OOF model sees only inner-train actors, and
    the final model sees only outer-train actors before predicting outer-test actors.
    """
    train_idx = np.asarray(train_idx, dtype=np.int64)
    test_idx = np.asarray(test_idx, dtype=np.int64)
    gtr = groups[train_idx]
    n_splits = min(cfg.inner_splits, len(np.unique(gtr)))
    if n_splits < 2:
        raise RuntimeError("Need ≥2 actor groups for LoRA inner OOF stacking.")

    needed_ids = sorted(set(train_idx.tolist()) | set(test_idx.tolist()))
    waves_by_id = {int(i): _load_lora_waveform(samples[int(i)], cfg) for i in needed_ids}
    oof = np.zeros((len(train_idx), len(LABEL_NAMES)), np.float32)
    print("[INFO] LoRA audio OOF stacking. This is slow but leakage-safe.")
    for split_i, (tr_l, va_l) in enumerate(GroupKFold(n_splits=n_splits).split(train_idx, y[train_idx], gtr), start=1):
        print(f"    [LoRA-audio] inner split {split_i}/{n_splits}")
        inner_train_ids = train_idx[tr_l]
        inner_val_ids = train_idx[va_l]
        extractor, model = _train_lora_audio_model(
            [waves_by_id[int(i)] for i in inner_train_ids],
            y[inner_train_ids],
            cfg,
            seed + split_i,
            # V15: inner OOF LoRA models use all inner-train actors for a fixed
            # number of epochs. This avoids losing extra actors inside already
            # small subject-wise folds; the outer-test actors are still unseen.
            None,
        )
        oof[va_l] = _predict_lora_audio_probs(
            extractor, model,
            [waves_by_id[int(i)] for i in inner_val_ids],
            cfg,
        )
        try:
            import torch
            del model
            del extractor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    print("    [LoRA-audio] final outer-train model")
    extractor, model = _train_lora_audio_model(
        [waves_by_id[int(i)] for i in train_idx],
        y[train_idx],
        cfg,
        seed + 999,
        # V15: final outer-train LoRA model uses all outer-train actors.
        # Holding out actors here would permanently reduce training data before
        # predicting the untouched outer-test actors.
        None,
    )
    test_probs = _predict_lora_audio_probs(
        extractor, model,
        [waves_by_id[int(i)] for i in test_idx],
        cfg,
    )
    try:
        import torch
        del model
        del extractor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    if cfg.calibrate_temperatures:
        T = fit_temperature(oof, y[train_idx], cfg.temp_init)
        oof = apply_temperature(oof, T)
        test_probs = apply_temperature(test_probs, T)
    return oof.astype(np.float32), test_probs.astype(np.float32)


# ──────────────────────────────────────────────────────────────────
# Fold runner
# ──────────────────────────────────────────────────────────────────
def make_subject_folds(groups: np.ndarray) -> List[np.ndarray]:
    return [np.where(np.isin(groups, actors))[0] for actors in SUBJECT5_FOLDS]


def run_fold(
    fold_id: int,
    samples: List[Sample],
    audio: np.ndarray,
    video: np.ndarray,
    au: np.ndarray,
    au_dyn: np.ndarray,
    rel: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    genders: np.ndarray,
    cfg: Config,
) -> Dict[str, Any]:
    folds     = make_subject_folds(groups)
    test_idx  = folds[fold_id]
    all_idx = np.arange(len(y), dtype=np.int64)
    train_idx = all_idx[~np.isin(all_idx, test_idx)]

    print(f"\n{'─'*70}")
    print(f"Fold {fold_id}")
    print(f"  train actors: {sorted(np.unique(groups[train_idx]).tolist())}")
    print(f"  test  actors: {sorted(np.unique(groups[test_idx]).tolist())}")

    meta_tr: List[np.ndarray] = []
    meta_te: List[np.ndarray] = []
    modality_test: Dict[str, np.ndarray] = {}
    # Per-modality OOF UAR is computed on calibrated OOF probabilities.
    # These values are used only for weighted dual fusion and are saved for ablation.
    modality_oof_uar: Dict[str, float] = {}
    seed_base = cfg.random_state + 11 * fold_id

    if cfg.use_lora_audio:
        print("[INFO] LoRA-audio OOF stacking…")
        oof, tst = lora_audio_oof_and_test_probs(samples, y, groups, train_idx, test_idx, cfg, seed_base + 707)
        modality_test["audio_lora"] = tst
        modality_oof_uar["audio_lora"] = float(balanced_accuracy_score(y[train_idx], oof.argmax(axis=1)))
        meta_tr += [oof, entropy_mat(oof), margin_mat(oof)]
        meta_te += [tst, entropy_mat(tst), margin_mat(tst)]

    if cfg.use_audio:
        print("[INFO] Audio OOF stacking…")
        oof, tst = oof_and_test_probs(audio, y, groups, train_idx, test_idx, cfg,
                                       cfg.pca_audio, seed_base)
        modality_test["audio"] = tst
        modality_oof_uar["audio"] = float(balanced_accuracy_score(y[train_idx], oof.argmax(axis=1)))
        meta_tr += [oof, entropy_mat(oof), margin_mat(oof)]
        meta_te += [tst, entropy_mat(tst), margin_mat(tst)]

    if cfg.use_video:
        print("[INFO] Video OOF stacking…")
        oof, tst = oof_and_test_probs(video, y, groups, train_idx, test_idx, cfg,
                                       cfg.pca_video, seed_base + 101)
        modality_test["video"] = tst
        modality_oof_uar["video"] = float(balanced_accuracy_score(y[train_idx], oof.argmax(axis=1)))
        meta_tr += [oof, entropy_mat(oof), margin_mat(oof)]
        meta_te += [tst, entropy_mat(tst), margin_mat(tst)]

    if cfg.use_au:
        print("[INFO] AU OOF stacking…")
        oof, tst = oof_and_test_probs(au, y, groups, train_idx, test_idx, cfg,
                                       cfg.pca_au, seed_base + 202)
        modality_test["au"] = tst
        modality_oof_uar["au"] = float(balanced_accuracy_score(y[train_idx], oof.argmax(axis=1)))
        meta_tr += [oof, entropy_mat(oof), margin_mat(oof)]
        meta_te += [tst, entropy_mat(tst), margin_mat(tst)]

        if cfg.use_au_dynamic:
            print("[INFO] AU-dynamic OOF stacking…")
            oof, tst = oof_and_test_probs(au_dyn, y, groups, train_idx, test_idx, cfg,
                                           cfg.pca_au_dynamic, seed_base + 303)
            modality_test["au_dynamic"] = tst
            modality_oof_uar["au_dynamic"] = float(balanced_accuracy_score(y[train_idx], oof.argmax(axis=1)))
            meta_tr += [oof, entropy_mat(oof), margin_mat(oof)]
            meta_te += [tst, entropy_mat(tst), margin_mat(tst)]

    if not meta_tr:
        raise RuntimeError("Enable at least one modality (--use_audio / --use_video / --use_au).")

    Xmeta_tr = np.hstack(meta_tr + [rel[train_idx]]).astype(np.float32)
    Xmeta_te = np.hstack(meta_te + [rel[test_idx]]).astype(np.float32)
    meta_clf  = _make_pipeline(cfg.meta_model, cfg.meta_C, 0,
                                Xmeta_tr.shape[1], cfg.random_state + 1000 + fold_id)
    meta_clf.fit(Xmeta_tr, y[train_idx])
    probs = aligned_proba(meta_clf, Xmeta_te)

    true_for_stage = y[test_idx]
    stage_ablations: Dict[str, Dict[str, float]] = {}

    def _record_stage(stage_name: str, stage_probs: np.ndarray) -> None:
        stage_pred = stage_probs.argmax(axis=1)
        stage_ablations[stage_name] = {
            "accuracy": float(accuracy_score(true_for_stage, stage_pred)),
            "macro_f1": float(f1_score(true_for_stage, stage_pred, average="macro")),
            "uar": float(balanced_accuracy_score(true_for_stage, stage_pred)),
        }

    _record_stage("meta_only", probs)

    # ESRA-RELA++ dual fusion: meta-fusion + calibrated simple average.
    # V10 showed simple average was a very strong baseline, so V11 keeps part of it.
    if cfg.dual_fusion_blend > 0 and len(modality_test) >= 2:
        modality_names = list(modality_test.keys())
        stack = np.stack([modality_test[name] for name in modality_names], axis=0).astype(np.float32)
        if cfg.dual_fusion_weighted:
            raw_w = np.array([max(modality_oof_uar.get(name, 0.0), 1e-3) for name in modality_names], dtype=np.float32)
            weights = raw_w / max(float(raw_w.sum()), 1e-8)
            avg_probs = np.sum(stack * weights[:, None, None], axis=0).astype(np.float32)
        else:
            weights = np.ones(len(modality_names), dtype=np.float32) / float(len(modality_names))
            avg_probs = np.mean(stack, axis=0).astype(np.float32)
        print("[INFO] Dual-fusion modality weights: " + ", ".join(
            f"{name}={float(w):.3f}" for name, w in zip(modality_names, weights)
        ))
        b = float(np.clip(cfg.dual_fusion_blend, 0.0, 1.0))
        probs = ((1.0 - b) * probs + b * avg_probs).astype(np.float32)
        probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-8)
    _record_stage("meta_plus_dual_fusion", probs)

    # Specialists trained on full concatenated feature space
    Xfull = np.hstack([audio, video, au, au_dyn, rel]).astype(np.float32)
    gen_sp: Optional[Dict[Tuple[int,int,int], Pipeline]] = None

    if cfg.enable_specialists:
        plain_sp = train_pair_specialists(Xfull, y, train_idx, cfg, fold_id)
        print(f"[INFO] Pair specialists: "
              f"{[(LABEL_NAMES[a], LABEL_NAMES[b]) for a,b in plain_sp]}")

        if cfg.enable_gender_specialists:
            gen_sp = train_gender_pair_specialists(Xfull, y, genders, train_idx, cfg, fold_id)
            print(f"[INFO] Gender specialists: {len(gen_sp)} models")

        probs = apply_specialists(
            probs, Xfull[test_idx],
            plain_sp,
            genders[test_idx] if cfg.enable_gender_specialists else None,
            gen_sp,
            cfg,
        )
    _record_stage("after_pair_and_gender_specialists", probs)

    if cfg.enable_sad_specialist:
        sad_sp = train_sad_specialist(Xfull, y, train_idx, cfg, fold_id)
        if sad_sp is not None:
            print("[INFO] Sad-vs-rest specialist: enabled")
            probs = apply_sad_specialist(probs, Xfull[test_idx], sad_sp, cfg)
    _record_stage("final_after_sad_specialist", probs)

    pred = probs.argmax(axis=1)
    true = y[test_idx]
    cm   = confusion_matrix(true, pred, labels=np.arange(8))
    acc  = float(accuracy_score(true, pred))
    mf1  = float(f1_score(true, pred, average="macro"))
    uar  = float(balanced_accuracy_score(true, pred))

    # NEW V11: uncertainty logging only; final predictions are not changed
    max_p       = probs.max(axis=1)
    uncertain   = max_p < cfg.uncertainty_threshold
    n_uncertain = int(uncertain.sum())
    print(f"[INFO] Uncertain (p < {cfg.uncertainty_threshold}): {n_uncertain}/{len(true)}")

    # Ablation
    ablations: Dict[str, Dict[str, float]] = {}
    for name, pr in modality_test.items():
        pp = pr.argmax(axis=1)
        ablations[name] = {
            "accuracy": float(accuracy_score(true, pp)),
            "macro_f1": float(f1_score(true, pp, average="macro")),
            "uar":      float(balanced_accuracy_score(true, pp)),
        }
    if len(modality_test) >= 2:
        avg = np.mean(np.stack(list(modality_test.values())), axis=0)
        pp  = avg.argmax(axis=1)
        ablations["simple_avg_fusion"] = {
            "accuracy": float(accuracy_score(true, pp)),
            "macro_f1": float(f1_score(true, pp, average="macro")),
            "uar":      float(balanced_accuracy_score(true, pp)),
        }

    print(f"[RESULT] Fold {fold_id}: "
          f"Acc={acc*100:.2f}%  Macro-F1={mf1*100:.2f}%  UAR={uar*100:.2f}%")
    return {
        "fold": fold_id,
        "accuracy": acc, "macro_f1": mf1, "uar": uar,
        "cm": cm, "true": true, "pred": pred, "test_idx": test_idx,
        "test_actors": sorted(np.unique(groups[test_idx]).tolist()),
        "ablations": ablations,
        "stage_ablations": stage_ablations,
        "modality_oof_uar": modality_oof_uar,
        "n_uncertain": n_uncertain,
        "uncertain_idx": test_idx[uncertain].tolist(),
    }


# ──────────────────────────────────────────────────────────────────
# Diagnostics / plotting helpers
# ──────────────────────────────────────────────────────────────────
def _plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        return None


def _plot_bar(series: pd.Series, out: Path, title: str,
              xlabel: str, ylabel: str) -> None:
    plt = _plt()
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    series.plot(kind="bar", ax=ax, color="steelblue")
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def _plot_heatmap(df: pd.DataFrame, out: Path, title: str,
                   xlabel: str, ylabel: str, fmt: str = ".2f") -> None:
    plt = _plt()
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(max(8, len(df.columns)), max(5, len(df))))
    im = ax.imshow(df.values.astype(float), aspect="auto")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(df))); ax.set_yticklabels(df.index)
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    for i in range(len(df)):
        for j in range(len(df.columns)):
            ax.text(j, i, format(float(df.values[i, j]), fmt),
                    ha="center", va="center", fontsize=7)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def save_cm_plot(cm: np.ndarray, out: Path, title: str,
                  normalize: bool = False) -> None:
    plt = _plt()
    if plt is None:
        return
    mat = cm.astype(float)
    if normalize:
        mat = mat / np.maximum(mat.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(mat)
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(8)); ax.set_xticklabels(LABEL_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(8)); ax.set_yticklabels(LABEL_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    for i in range(8):
        for j in range(8):
            v = float(mat[i, j])
            label = f"{v:.2f}" if normalize else str(int(round(v)))
            ax.text(j, i, label, ha="center", va="center", fontsize=8,
                    color="white" if v > mat.max() * 0.6 else "black")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def save_diagnostics(
    samples: List[Sample], cfg: Config,
    audio: np.ndarray, video: np.ndarray,
    au: np.ndarray, au_dyn: np.ndarray, rel: np.ndarray,
    y: np.ndarray, groups: np.ndarray,
) -> None:
    diag  = Path(cfg.results_dir) / "diagnostics"
    ensure_dir(diag)
    of_dir: Optional[Path] = Path(cfg.openface_dir) if cfg.openface_dir else None

    sdf = pd.DataFrame({
        "file":       [s.filename  for s in samples],
        "actor":      [s.actor     for s in samples],
        "gender":     [s.gender    for s in samples],
        "emotion":    [s.emotion   for s in samples],
        "label":      [s.label     for s in samples],
        "intensity":  [s.intensity for s in samples],
        "statement":  [s.statement for s in samples],
        "repetition": [s.repetition for s in samples],
        "of_found":   [find_openface_csv(s, of_dir) is not None if of_dir is not None else False for s in samples],
    })
    sdf.to_csv(diag / "d00_sample_metadata.csv", index=False)

    counts = sdf["emotion"].value_counts().reindex(LABEL_NAMES, fill_value=0)
    counts.to_csv(diag / "d01_class_distribution.csv", header=["count"])
    _plot_bar(counts, diag / "g01_class_distribution.png",
              "Class Distribution", "Emotion", "Count")

    # NEW V11: gender × emotion breakdown
    g_e = pd.crosstab(sdf["emotion"], sdf["gender"]).reindex(index=LABEL_NAMES, fill_value=0)
    g_e.to_csv(diag / "d02_gender_emotion.csv")
    _plot_heatmap(g_e, diag / "g02_gender_emotion.png",
                  "Emotion × Gender Distribution", "Gender", "Emotion", fmt=".0f")

    of_cov = sdf.groupby("emotion")["of_found"].mean().reindex(LABEL_NAMES, fill_value=0)
    of_cov.to_csv(diag / "d03_openface_coverage.csv", header=["coverage"])
    _plot_bar(of_cov, diag / "g03_openface_coverage.png",
              "OpenFace Coverage", "Emotion", "Coverage Ratio")

    folds = make_subject_folds(groups)
    fold_rows = []
    for fid, idx in enumerate(folds):
        # Pylance-safe class counts: avoid pandas Hashable index typing in c.items().
        fold_counts = np.bincount(y[idx].astype(int), minlength=len(LABEL_NAMES))
        row: Dict[str, Any] = {
            "fold": fid,
            "actors": ",".join(map(str, SUBJECT5_FOLDS[fid])),
            "n": int(len(idx)),
        }
        for class_i, class_name in enumerate(LABEL_NAMES):
            row[class_name] = int(fold_counts[class_i])
        fold_rows.append(row)
    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(diag / "d04_fold_distribution.csv", index=False)
    _plot_heatmap(fold_df.set_index("fold")[LABEL_NAMES],
                  diag / "g04_fold_distribution.png",
                  "Test Class Distribution per Fold", "Emotion", "Fold", fmt=".0f")

    rel_names = [
        "audio_rms","audio_zcr","audio_feat_norm",
        "video_frame_ratio","video_saliency","video_feat_norm",
        "au_found","au_conf_mean","au_success_rate",
        "au_dyn_found","au_dyn_conf_mean","au_dyn_success_rate",
    ]
    if rel.shape[1] == 15:
        rel_names += ["intensity","statement","repetition"]
    elif rel.shape[1] != len(rel_names):
        rel_names = [f"rel_{i}" for i in range(rel.shape[1])]
    rel_df = pd.DataFrame(rel, columns=rel_names)
    rel_df["emotion"] = [LABEL_NAMES[int(i)] for i in y]
    rel_by = rel_df.groupby("emotion")[rel_names].mean().reindex(LABEL_NAMES).fillna(0)
    rel_by.to_csv(diag / "d05_reliability_by_emotion.csv")
    _plot_heatmap(rel_by, diag / "g05_reliability_by_emotion.png",
                  "Mean Reliability Features by Emotion", "Feature", "Emotion", fmt=".2f")

    corr = rel_df[rel_names].corr().fillna(0)
    corr.to_csv(diag / "d06_reliability_correlation.csv")
    _plot_heatmap(corr, diag / "g06_reliability_correlation.png",
                  "Reliability Correlation Matrix", "Feature", "Feature", fmt=".2f")

    norms = pd.DataFrame({
        "audio_norm": np.linalg.norm(audio, axis=1) if audio.ndim == 2 else np.zeros(len(y)),
        "video_norm": np.linalg.norm(video, axis=1) if video.ndim == 2 else np.zeros(len(y)),
        "au_norm":    np.linalg.norm(au,    axis=1) if au.ndim    == 2 else np.zeros(len(y)),
        "au_dynamic_norm": np.linalg.norm(au_dyn, axis=1) if au_dyn.ndim == 2 else np.zeros(len(y)),
        "emotion":    [LABEL_NAMES[int(i)] for i in y],
    })
    norms.to_csv(diag / "d07_feature_norms.csv", index=False)
    _plot_heatmap(
        norms.groupby("emotion")[["audio_norm","video_norm","au_norm","au_dynamic_norm"]].mean().reindex(LABEL_NAMES),
        diag / "g07_feature_norms_by_emotion.png",
        "Mean Feature Norms by Emotion", "Modality", "Emotion", fmt=".2f",
    )

    pd.DataFrame([{
        "matrix": m, "rows": int(arr.shape[0]),
        "cols": int(arr.shape[1] if arr.ndim > 1 else 1),
    } for m, arr in [("audio",audio),("video",video),("au",au),("au_dynamic",au_dyn),("reliability",rel)]
    ]).to_csv(diag / "d08_feature_dimensions.csv", index=False)

    print(f"[INFO] Diagnostics saved: {diag}")


def save_final_tables(
    results: List[Dict[str, Any]],
    yt: np.ndarray, yp: np.ndarray,
    cfg: Config,
) -> None:
    out = Path(cfg.results_dir) / "final_results"
    ensure_dir(out)

    cm = confusion_matrix(yt, yp, labels=np.arange(8))
    pd.DataFrame(cm, index=LABEL_NAMES, columns=LABEL_NAMES).to_csv(out / "r01_cm.csv")
    save_cm_plot(cm, out / "r01_cm.png", "Aggregated Confusion Matrix")
    save_cm_plot(cm, out / "r02_cm_norm.png", "Aggregated Normalised CM", normalize=True)
    cm_n = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    pd.DataFrame(cm_n, index=LABEL_NAMES, columns=LABEL_NAMES).to_csv(out / "r02_cm_norm.csv")

    pd.DataFrame(
        classification_report(yt, yp, target_names=LABEL_NAMES, output_dict=True)
    ).T.to_csv(out / "r03_classification_report.csv")

    fold_df = pd.DataFrame([{
        "fold":     r["fold"],
        "accuracy": r["accuracy"],
        "macro_f1": r["macro_f1"],
        "uar":      r["uar"],
    } for r in results])
    fold_df.to_csv(out / "r04_fold_metrics.csv", index=False)

    plt = _plt()
    if plt is not None:
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(fold_df)); w = 0.25
        ax.bar(x-w, fold_df["accuracy"]*100, w, label="Accuracy")
        ax.bar(x,   fold_df["macro_f1"]*100, w, label="Macro-F1")
        ax.bar(x+w, fold_df["uar"]*100,      w, label="UAR")
        ax.set_xticks(x); ax.set_xticklabels([f"Fold {int(v)}" for v in fold_df["fold"]])
        ax.set_ylabel("Score (%)"); ax.set_title("ESRA-RELA++ V15-LoRA-Publication-Final — Per-Fold Metrics")
        ax.legend(); ax.set_ylim(0, 105); fig.tight_layout()
        fig.savefig(out / "r05_fold_metrics.png", dpi=150); plt.close(fig)

    ablation_rows = []
    for r in results:
        for name, met in r.get("ablations", {}).items():
            ablation_rows.append({"fold": r["fold"], "model": name, **met})
    if ablation_rows:
        pd.DataFrame(ablation_rows).to_csv(out / "r06_ablation_metrics.csv", index=False)

    modality_rows = []
    for r in results:
        for name, value in r.get("modality_oof_uar", {}).items():
            modality_rows.append({"fold": r["fold"], "modality": name, "oof_uar": float(value)})
    if modality_rows:
        pd.DataFrame(modality_rows).to_csv(out / "r08_modality_oof_uar.csv", index=False)

    specialist_config = pd.DataFrame([{
        "specialist_threshold": cfg.specialist_threshold,
        "specialist_margin_threshold": cfg.specialist_margin_threshold,
        "specialist_blend": cfg.specialist_blend,
        "specialist_pca": cfg.specialist_pca,
        "dual_fusion_blend": cfg.dual_fusion_blend,
        "dual_fusion_weighted": cfg.dual_fusion_weighted,
        "sad_specialist_threshold": cfg.sad_specialist_threshold,
        "sad_specialist_blend": cfg.sad_specialist_blend,
        "sad_min_base_prob": cfg.sad_min_base_prob,
    }])
    specialist_config.to_csv(out / "r09_specialist_config.csv", index=False)

    stage_rows = []
    for r in results:
        for stage_name, met in r.get("stage_ablations", {}).items():
            stage_rows.append({"fold": r["fold"], "stage": stage_name, **met})
    if stage_rows:
        pd.DataFrame(stage_rows).to_csv(out / "r10_sequential_stage_ablation.csv", index=False)

    # NEW V11: uncertainty log
    uncertain_rows = [
        {"fold": r["fold"], "sample_idx": idx}
        for r in results for idx in r.get("uncertain_idx", [])
    ]
    if uncertain_rows:
        pd.DataFrame(uncertain_rows).to_csv(out / "r07_uncertain_predictions.csv", index=False)

    print(f"[INFO] Final result tables saved: {out}")


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────
def run(cfg: Config) -> None:
    ensure_dir(Path(cfg.results_dir))
    ensure_dir(Path(cfg.cache_dir))

    data_path = Path(cfg.data_dir)
    candidate_exts = ("*.mp4", "*.avi", "*.mov", "*.mkv")
    total_video_files = sum(1 for ext in candidate_exts for _ in data_path.rglob(ext))
    samples = discover_samples(data_path, cfg.max_files)
    print(f"[INFO] Video files found before RAVDESS full-AV speech filter: {total_video_files}")
    print(f"[INFO] Samples after full-AV speech filter: {len(samples)}")
    print(f"[INFO] Actors  : {sorted({s.actor for s in samples})}")
    print("[INFO] Class distribution:")
    print(pd.Series([s.emotion for s in samples]).value_counts()
            .reindex(LABEL_NAMES).to_string())

    of_dir: Optional[Path] = Path(cfg.openface_dir) if cfg.openface_dir else None
    if cfg.use_au or cfg.use_video:
        found = sum(1 for s in samples if of_dir is not None and find_openface_csv(s, of_dir) is not None)
        pct   = 100.0 * found / max(len(samples), 1)
        print(f"[INFO] OpenFace coverage: {found}/{len(samples)} = {pct:.1f}%")
    if (not cfg.use_au) and cfg.use_au_dynamic:
        print("[WARN] AU-dynamic branch is ignored because --no_au disables all AU-based features.")
    if cfg.use_lora_audio and cfg.audio_model != cfg.lora_audio_model:
        print("[INFO] Frozen SSL audio model and LoRA audio model are different; this is treated as an audio-backbone ensemble.")

    audio, video, au, au_dyn, rel, y, groups, genders = build_feature_matrix(samples, cfg)

    if cfg.save_diagnostics:
        save_diagnostics(samples, cfg, audio, video, au, au_dyn, rel, y, groups)

    fold_ids = list(range(5)) if cfg.run_all_folds else [cfg.fold]
    results: List[Dict[str, Any]] = []
    all_true: List[np.ndarray] = []
    all_pred: List[np.ndarray] = []

    for fid in fold_ids:
        r = run_fold(fid, samples, audio, video, au, au_dyn, rel, y, groups, genders, cfg)
        results.append(r)
        all_true.append(r["true"])
        all_pred.append(r["pred"])
        out_dir = Path(cfg.results_dir)
        pd.DataFrame(r["cm"], index=LABEL_NAMES, columns=LABEL_NAMES).to_csv(
            out_dir / f"cm_fold{fid}.csv")
        save_cm_plot(r["cm"], out_dir / f"cm_fold{fid}.png",
                     f"Fold {fid} Confusion Matrix")
        save_cm_plot(r["cm"], out_dir / f"cm_fold{fid}_norm.png",
                     f"Fold {fid} Normalised CM", normalize=True)
        pd.DataFrame({
            "filename": [samples[int(i)].filename for i in r["test_idx"]],
            "actor": [samples[int(i)].actor for i in r["test_idx"]],
            "true": [LABEL_NAMES[int(i)] for i in r["true"]],
            "pred": [LABEL_NAMES[int(i)] for i in r["pred"]],
            "idx":  r["test_idx"].tolist(),
        }).to_csv(out_dir / f"preds_fold{fid}.csv", index=False)

    yt = np.concatenate(all_true)
    yp = np.concatenate(all_pred)

    fold_df = pd.DataFrame([{
        "fold":       r["fold"],
        "test_actors": ",".join(map(str, r["test_actors"])),
        "accuracy":   r["accuracy"],
        "macro_f1":   r["macro_f1"],
        "uar":        r["uar"],
        "n_uncertain": r["n_uncertain"],
    } for r in results])
    fold_df.to_csv(Path(cfg.results_dir) / "fold_summary.csv", index=False)

    if cfg.save_diagnostics:
        save_final_tables(results, yt, yp, cfg)

    summary = {
        "config":          asdict(cfg),
        "accuracy_mean":   float(fold_df["accuracy"].mean()),
        "accuracy_std":    float(fold_df["accuracy"].std(ddof=0)),
        "macro_f1_mean":   float(fold_df["macro_f1"].mean()),
        "macro_f1_std":    float(fold_df["macro_f1"].std(ddof=0)),
        "uar_mean":        float(fold_df["uar"].mean()),
        "uar_std":         float(fold_df["uar"].std(ddof=0)),
        "classification_report": classification_report(
            yt, yp, target_names=LABEL_NAMES, output_dict=True),
        "reference_paper1_86.70_pct":   "Luna-Jiménez et al. Appl.Sci. 2022 subject-wise 5-CV",
        "reference_paper2_93.59_pct":   "John & Kawanishi ICPR 2021 trial-split (NOT subject-wise)",
        "method_note_fold_imbalance": "Subject-wise 5-fold uses four 5-actor folds and one 4-actor fold, matching the Paper-1 actor protocol.",
        "method_note_uncertainty": "Uncertainty threshold only logs low-confidence samples; it does not abstain or alter predictions.",
        "method_note_specialists": "Pair specialists and Sad-vs-rest specialist use independent StandardScaler/PCA fitted only on train-fold samples.",
        "method_note_metadata_covariates": "RAVDESS intensity/statement/repetition metadata are excluded from reliability features by default; use --include_metadata_covariates only for an explicit ablation.",
        "method_note_v15_lora_final": "Optional LoRA audio is added as a separate fold-safe OOF modality; inner/final LoRA models are deleted after use to reduce GPU memory pressure.",
        "method_note_lora_training": "V15 uses fold-safe fixed-epoch LoRA training for inner OOF and final outer-train models so no additional actors are held out inside small subject-wise folds. The optional validation-split helper remains available only if train_groups is passed explicitly.",
        "method_note_modality_weights": "Dual-fusion modality weights are computed from calibrated OOF UAR and saved in final_results/r08_modality_oof_uar.csv.",
        "method_note_v11pp": "ESRA-RELA++ adds AU dynamic trajectory branch, OOF-UAR-weighted dual fusion, richer acoustic prosody, and Sad-vs-rest correction.",
        "method_note_sad_specialist": "Sad-vs-rest correction requires a minimum baseline Sad probability floor to avoid over-correcting confident non-Sad predictions.",
        "method_note_specialist_hyperparams": "Specialist thresholds/blends are saved in final_results/r09_specialist_config.csv and should be accompanied by sensitivity ablations for publication.",
        "method_note_stage_ablation": "Sequential post-processing stages are saved in final_results/r10_sequential_stage_ablation.csv.",
    }
    (Path(cfg.results_dir) / "summary.json").write_text(
        json.dumps(summary, indent=2), "utf-8")

    print("\n" + "=" * 70)
    print("ESRA-RELA++ V15-LoRA-Publication-Final — FINAL SUMMARY  (subject-wise 5-fold, 8 emotions)")
    print("=" * 70)
    print(fold_df.to_string(index=False, formatters={
        "accuracy": lambda x: f"{x*100:.2f}%",
        "macro_f1": lambda x: f"{x*100:.2f}%",
        "uar":      lambda x: f"{x*100:.2f}%",
    }))
    print("-" * 70)
    print(f"Accuracy  : {summary['accuracy_mean']*100:.2f}% ± {summary['accuracy_std']*100:.2f}")
    print(f"Macro-F1  : {summary['macro_f1_mean']*100:.2f}% ± {summary['macro_f1_std']*100:.2f}")
    print(f"UAR       : {summary['uar_mean']*100:.2f}% ± {summary['uar_std']*100:.2f}")
    print()
    print("Reference — Paper 1 (subject-wise 5-CV, same setup):  86.70%")
    print("Reference — Paper 2 (trial split, NOT subject-wise):   93.59%")
    print(f"[INFO] Results saved: {cfg.results_dir}")
    print("=" * 70)


# ──────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────
def parse_args() -> Config:
    p = argparse.ArgumentParser(
        description="ESRA-RELA++ V15-LoRA-Publication-Final — RAVDESS Multimodal Emotion Recognition")
    p.add_argument("--data_dir",     required=True,
                   help="Root directory of RAVDESS .mp4 files")
    p.add_argument("--openface_dir", default="",
                   help="Directory of OpenFace CSV files (empty string = skip AU)")
    p.add_argument("--cache_dir",    default="esra_rela_v15_lora_cache")
    p.add_argument("--results_dir",  default="esra_rela_v15_lora_results")
    p.add_argument("--run_all_folds",action="store_true",
                   help="Run all 5 subject-wise folds sequentially")
    p.add_argument("--fold",         type=int, default=0, choices=range(5))
    p.add_argument("--max_files",    type=int, default=0,
                   help="Limit number of files (0 = all). For smoke tests only.")
    p.add_argument("--rebuild_cache",action="store_true")
    p.add_argument("--no_diagnostics",action="store_true")

    # Modality toggles
    p.add_argument("--no_audio", action="store_true")
    p.add_argument("--no_video", action="store_true")
    p.add_argument("--no_au",    action="store_true")
    p.add_argument("--no_au_dynamic", action="store_true",
                   help="Disable ESRA-RELA++ AU temporal trajectory branch")

    # Audio
    p.add_argument("--audio_model",   default="facebook/wav2vec2-base-960h",
                   help='HuggingFace model ID, or "none" for handcrafted features only')
    p.add_argument("--audio_sr",      type=int,   default=16000)
    p.add_argument("--audio_seconds", type=float, default=5.5)

    # Optional LoRA audio branch
    p.add_argument("--use_lora_audio", action="store_true",
                   help="Enable fold-safe LoRA-finetuned audio classifier as an extra OOF modality")
    p.add_argument("--lora_audio_model", default="facebook/wav2vec2-base-960h",
                   help="HF audio classification backbone for LoRA branch")
    p.add_argument("--lora_epochs", type=int, default=8)
    p.add_argument("--lora_batch_size", type=int, default=2)
    p.add_argument("--lora_grad_accum", type=int, default=2)
    p.add_argument("--lora_lr", type=float, default=1e-4)
    p.add_argument("--lora_weight_decay", type=float, default=0.01)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.10)
    p.add_argument("--lora_max_train_samples", type=int, default=0,
                   help="Smoke-test cap only. 0 uses all fold-safe train samples.")
    p.add_argument("--lora_target_modules", default="q_proj,k_proj,v_proj,out_proj")
    p.add_argument("--lora_patience", type=int, default=2,
                   help="Optional LoRA actor-safe validation patience when train_groups is explicitly passed; the default pipeline uses fixed-epoch LoRA for inner/final fold models. 0 disables.")

    # Video
    p.add_argument("--video_frames",       type=int, default=10)
    p.add_argument("--saliency_top_pool",  type=int, default=24)

    # Classifiers
    p.add_argument("--base_model",   default="logreg", choices=["logreg","svm"])
    p.add_argument("--meta_model",   default="logreg", choices=["logreg","svm"])
    p.add_argument("--base_C",       type=float, default=0.45)
    p.add_argument("--meta_C",       type=float, default=0.55)
    p.add_argument("--pca_audio",    type=int,   default=128)
    p.add_argument("--pca_video",    type=int,   default=128)
    p.add_argument("--pca_au",       type=int,   default=64)
    p.add_argument("--pca_au_dynamic", type=int, default=64)
    p.add_argument("--inner_splits", type=int,   default=4)

    # Specialists
    p.add_argument("--no_specialists",       action="store_true")
    p.add_argument("--no_gender_specialists",action="store_true")
    p.add_argument("--specialist_threshold", type=float, default=0.38)
    p.add_argument("--specialist_margin",    type=float, default=0.12)
    p.add_argument("--specialist_blend",     type=float, default=0.30)
    p.add_argument("--specialist_pca",       type=int, default=128,
                   help="Maximum PCA dimension used inside pair/gender/Sad specialists.")
    p.add_argument("--dual_fusion_blend", type=float, default=0.25,
                   help="Blend meta-fusion with calibrated average fusion. 0 disables.")
    p.add_argument("--no_dual_fusion_weighted", action="store_true",
                   help="Use equal modality weights in dual fusion instead of OOF-UAR weights.")
    p.add_argument("--no_sad_specialist", action="store_true",
                   help="Disable ESRA-RELA++ Sad-vs-rest specialist")
    p.add_argument("--sad_threshold", type=float, default=0.42)
    p.add_argument("--sad_blend", type=float, default=0.18)
    p.add_argument("--sad_min_base_prob", type=float, default=0.05,
                   help="Minimum 8-class Sad probability required before Sad-vs-rest specialist can boost Sad.")

    # NEW V11
    p.add_argument("--no_calibration",   action="store_true",
                   help="Disable posterior-temperature calibration (V11 feature)")
    p.add_argument("--temp_init",        type=float, default=1.5)
    p.add_argument("--uncertainty_threshold", type=float, default=0.30,
                   help="Log predictions below this max-posterior as uncertain; predictions are not changed.")
    p.add_argument("--abstain_threshold", dest="uncertainty_threshold", type=float,
                   help="Backward-compatible alias for --uncertainty_threshold.")
    p.add_argument("--include_metadata_covariates", action="store_true",
                   help="Ablation only: append RAVDESS intensity/statement/repetition metadata to reliability features.")

    p.add_argument("--random_state",     type=int, default=42)

    a = p.parse_args()
    return Config(
        data_dir=a.data_dir,
        openface_dir=a.openface_dir,
        cache_dir=a.cache_dir,
        results_dir=a.results_dir,
        run_all_folds=a.run_all_folds,
        fold=a.fold,
        max_files=a.max_files,
        rebuild_cache=a.rebuild_cache,
        save_diagnostics=not a.no_diagnostics,
        use_audio=not a.no_audio,
        use_video=not a.no_video,
        use_au=not a.no_au,
        use_au_dynamic=(not a.no_au and not a.no_au_dynamic),
        audio_model=a.audio_model,
        audio_sr=a.audio_sr,
        audio_seconds=a.audio_seconds,
        use_lora_audio=a.use_lora_audio,
        lora_audio_model=a.lora_audio_model,
        lora_epochs=a.lora_epochs,
        lora_batch_size=a.lora_batch_size,
        lora_grad_accum=a.lora_grad_accum,
        lora_lr=a.lora_lr,
        lora_weight_decay=a.lora_weight_decay,
        lora_r=a.lora_r,
        lora_alpha=a.lora_alpha,
        lora_dropout=a.lora_dropout,
        lora_max_train_samples=a.lora_max_train_samples,
        lora_target_modules=a.lora_target_modules,
        lora_patience=a.lora_patience,
        video_frames=a.video_frames,
        saliency_top_pool=a.saliency_top_pool,
        base_model=a.base_model,
        meta_model=a.meta_model,
        base_C=a.base_C,
        meta_C=a.meta_C,
        pca_audio=a.pca_audio,
        pca_video=a.pca_video,
        pca_au=a.pca_au,
        pca_au_dynamic=a.pca_au_dynamic,
        inner_splits=a.inner_splits,
        enable_specialists=not a.no_specialists,
        enable_gender_specialists=not a.no_gender_specialists,
        specialist_threshold=a.specialist_threshold,
        specialist_margin_threshold=a.specialist_margin,
        specialist_blend=a.specialist_blend,
        specialist_pca=a.specialist_pca,
        dual_fusion_blend=a.dual_fusion_blend,
        dual_fusion_weighted=not a.no_dual_fusion_weighted,
        enable_sad_specialist=not a.no_sad_specialist,
        sad_specialist_threshold=a.sad_threshold,
        sad_specialist_blend=a.sad_blend,
        sad_min_base_prob=a.sad_min_base_prob,
        calibrate_temperatures=not a.no_calibration,
        temp_init=a.temp_init,
        uncertainty_threshold=a.uncertainty_threshold,
        include_metadata_covariates=a.include_metadata_covariates,
        random_state=a.random_state,
    )


if __name__ == "__main__":
    run(parse_args())