"""src/utils/helpers.py — Shared math + serialisation helpers."""

import json
import math
import re
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pandas as pd

# ── Probability maths ─────────────────────────────────────────────────────────
RANDOM_SEED  = 42
PROB_FLOOR   = 0.05
PROB_CEILING = 0.95
LOG_EPS      = 1e-7


def vote_to_prob_array(vs: np.ndarray, k: float = 8.0) -> np.ndarray:
    """P = 1 / (1 + exp(-k*(vs - 0.5)))  — calibrated for Assam 2021 results."""
    return 1.0 / (1.0 + np.exp(-k * (vs - 0.5)))


def normalise(probs: np.ndarray, floor: float = 1e-4) -> np.ndarray:
    p = np.maximum(probs, floor)
    return p / p.sum()


def soft_flatten(probs: np.ndarray, strength: float = 0.15) -> np.ndarray:
    """P_flat = (1-s)*P + s*0.5  — prevents overconfidence."""
    flat = (1 - strength) * probs + strength * 0.5
    flat = np.maximum(flat, 1e-4)
    return flat / flat.sum()


def temperature_scale(p: np.ndarray, temp: float = 2.0) -> np.ndarray:
    p = np.clip(p, LOG_EPS, 1 - LOG_EPS)
    logit = np.log(p / (1 - p))
    return 1.0 / (1.0 + np.exp(-logit / temp))


def standardise(X: np.ndarray,
                mean: Optional[np.ndarray] = None,
                std: Optional[np.ndarray] = None,
                clip_sigma: float = 3.0):
    """(X - mean) / std, clipped to ±clip_sigma."""
    if mean is None: mean = X.mean(axis=0)
    if std  is None:
        std = X.std(axis=0)
        std[std == 0] = 1.0
    Xs = np.clip((X - mean) / std, -clip_sigma, clip_sigma)
    return Xs, mean, std


def expected_calibration_error(y_true: np.ndarray, y_pred: np.ndarray,
                                n_bins: int = 8) -> dict:
    bins = np.linspace(0, 1, n_bins + 1)
    ece, stats = 0.0, []
    n = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_pred >= lo) & (y_pred < hi)
        cnt  = int(mask.sum())
        if cnt == 0: continue
        mp = float(y_pred[mask].mean())
        fp = float(y_true[mask].mean())
        ece += (cnt / n) * abs(mp - fp)
        stats.append({"lo":round(lo,2),"hi":round(hi,2),"n":cnt,
                      "mean_pred":round(mp,3),"frac_pos":round(fp,3),
                      "error":round(abs(mp-fp),3)})
    return {"ece": round(ece, 4), "bins": stats}


def log_loss_safe(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, LOG_EPS, 1 - LOG_EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


# ── JSON ──────────────────────────────────────────────────────────────────────

def json_safe(obj: Any):
    """Convert numpy scalars so json.dump never raises TypeError."""
    if isinstance(obj, (np.bool_,)):    return bool(obj)
    if isinstance(obj, (np.integer,)):  return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray):     return obj.tolist()
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=json_safe)


# ── Text / NLP ────────────────────────────────────────────────────────────────

def normalise_text(text: str) -> str:
    if not text: return ""
    t = text.lower()
    t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"@\w+", " ", t)
    t = re.sub(r"#(\w+)", r"\1", t)
    t = re.sub(r"[^\w\s']", " ", t)
    return re.sub(r"\s+", " ", t).strip()


# ── Backwards-compatibility aliases (used by train.py, backtest.py) ──────────
def standardise_features(X, mean=None, std=None, clip_sigma=3.0):
    """Alias for standardise() — kept for module compatibility."""
    return standardise(X, mean, std, clip_sigma)

def temperature_scale_array(p, temp=2.0):
    """Alias for temperature_scale() — kept for module compatibility."""
    return temperature_scale(p, temp)


# ── Additional aliases used by predict.py ────────────────────────────────────
def soft_flatten_array(probs: np.ndarray, strength: float = 0.15) -> np.ndarray:
    return soft_flatten(probs, strength)

def clip_prob_array(probs: np.ndarray,
                    floor: float = PROB_FLOOR,
                    ceiling: float = PROB_CEILING) -> np.ndarray:
    return np.clip(probs, floor, ceiling)

def blend_probs(base: np.ndarray, upset: np.ndarray,
                alpha: float = 0.12) -> np.ndarray:
    """Blend base probabilities with upset adjustment."""
    blended = (1 - alpha) * base + alpha * upset
    blended = np.clip(blended, PROB_FLOOR, PROB_CEILING)
    return blended / blended.sum()
