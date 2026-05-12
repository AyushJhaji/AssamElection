"""
src/sentiment_utils.py  —  V13 Sentiment Analysis Engine
=========================================================
Self-contained NLP scorer. Zero external NLP dependencies.
Works with only stdlib (re, math) + numpy.

Implements a VADER-inspired lexicon-based approach tuned for
Indian political discourse (Assam 2026 context).

Public API:
    score_text(text)         → float in [-1, +1]
    score_headline(h, d)     → float in [-1, +1]
    score_batch(texts)       → list[float]
    aggregate(scores, w)     → float
"""

import re
import math
import logging
from typing import List, Optional

import numpy as np

log = logging.getLogger(__name__)

# ── Political sentiment lexicon ───────────────────────────────────────────────
_LEX = {
    # Strong positive
    "win":4.0,"victory":4.0,"landslide":4.5,"dominant":3.8,"surge":3.2,
    "leads":3.0,"ahead":3.0,"popular":2.8,"strong":2.5,"confident":2.5,
    "development":2.0,"progress":2.2,"growth":2.0,"momentum":2.8,"rising":2.5,
    "rally":2.5,"crowd":1.8,"overwhelming":3.5,"sweep":3.8,"excellent":3.5,
    "great":3.0,"best":3.2,"improved":2.0,"support":2.2,"united":2.0,
    "infrastructure":1.8,"jobs":2.0,"employment":1.8,"welfare":1.8,
    "good":2.0,"hopeful":1.8,"positive":2.0,"gained":1.8,"increase":1.5,
    "upward":1.5,"rise":1.8,"favourable":2.2,"winning":3.2,"triumph":3.5,
    # Mild positive
    "okay":0.6,"fine":0.7,"decent":0.8,"adequate":0.6,"fair":0.5,
    # Neutral / political
    "vote":0.2,"election":0.1,"campaign":0.3,"party":0.1,"rally":1.5,
    # Mild negative
    "concern":-0.8,"challenge":-0.8,"issue":-0.5,"struggle":-1.2,
    "difficult":-1.0,"uncertain":-1.0,"worry":-1.2,"problem":-1.2,
    # Moderate negative
    "loss":-3.0,"trailing":-2.5,"behind":-2.2,"weak":-2.0,"poor":-2.0,
    "bad":-2.2,"negative":-2.0,"fail":-2.5,"failure":-2.8,"scandal":-3.2,
    "protest":-1.8,"anger":-2.2,"dissatisfied":-2.2,"frustrated":-2.0,
    "oppose":-1.5,"rejected":-2.8,"dropped":-1.5,"decline":-1.8,
    # Strong negative
    "lose":-3.2,"defeat":-3.5,"collapse":-3.8,"crisis":-3.2,"corruption":-3.8,
    "fraud":-3.5,"scam":-3.2,"crime":-3.0,"violence":-3.5,"conflict":-2.8,
    "betrayal":-3.2,"boycott":-2.8,"discredited":-3.2,"riot":-3.0,
    # Assam-specific
    "himanta":1.8,"sarma":1.5,"tarun":0.8,"gogoi":0.8,"badruddin":0.5,
    "nda":1.2,"bjp":0.8,"inc":0.5,"aiudf":0.3,"agp":0.6,"uppl":0.4,
    "flood":-2.2,"erosion":-1.8,"nrc":-0.8,"caa":-0.5,
    "road":0.5,"bridge":0.6,"hospital":0.5,"school":0.5,"electricity":0.6,
    "irrigation":0.5,"farmer":0.3,"tea":0.2,"petroleum":0.2,
}

# Fix negative literal issue
_LEX.update({
    "concern":  -0.8, "challenge": -0.8, "issue":    -0.5,
    "struggle": -1.2, "difficult": -1.0, "uncertain":-1.0,
    "worry":    -1.2, "problem":   -1.2, "loss":     -3.0,
    "trailing": -2.5, "behind":    -2.2, "weak":     -2.0,
    "poor":     -2.0, "bad":       -2.2, "negative": -2.0,
    "fail":     -2.5, "failure":   -2.8, "scandal":  -3.2,
    "protest":  -1.8, "anger":     -2.2, "dissatisfied":-2.2,
    "frustrated":-2.0,"oppose":   -1.5, "rejected": -2.8,
    "dropped":  -1.5, "decline":  -1.8, "lose":     -3.2,
    "defeat":   -3.5, "collapse": -3.8, "crisis":   -3.2,
    "corruption":-3.8,"fraud":    -3.5, "scam":     -3.2,
    "crime":    -3.0, "violence": -3.5, "conflict": -2.8,
    "betrayal": -3.2, "boycott":  -2.8, "discredited":-3.2,
    "riot":     -3.0, "nrc":      -0.8, "caa":      -0.5,
    "flood":    -2.2, "erosion":  -1.8,
})

_BOOSTS    = {"very":1.3,"extremely":1.5,"absolutely":1.4,"hugely":1.4,
               "strongly":1.3,"highly":1.3,"completely":1.4,"totally":1.3,
               "significantly":1.2,"clearly":1.1,"really":1.2,"quite":1.1}
_DAMPENERS = {"somewhat":0.7,"slightly":0.6,"might":0.8,"may":0.8,
               "could":0.8,"possibly":0.75,"perhaps":0.7,"a bit":0.65}
_NEGATIONS = {"not","no","never","neither","nobody","nothing","nor",
               "cannot","can't","won't","doesn't","don't","didn't",
               "isn't","aren't","wasn't","weren't","without","lack",
               "against","deny","denied","fail","lacks","missing"}


def _tokenize(text: str) -> List[str]:
    return re.sub(r"[^a-z0-9\s']","", text.lower()).split()


def score_text(text: str) -> float:
    """Score a single text string → float in [-1, +1]."""
    if not text or not text.strip():
        return 0.0
    tokens = _tokenize(text)
    if not tokens:
        return 0.0
    scores = []
    for i, tok in enumerate(tokens):
        if tok not in _LEX:
            continue
        base = _LEX[tok]
        # Negation: look back 3 words
        if any(tokens[j] in _NEGATIONS for j in range(max(0,i-3), i)):
            base = -base * 0.74
        # Amplifier/dampener: word before
        if i > 0:
            prev = tokens[i-1]
            if prev in _BOOSTS:    base *= _BOOSTS[prev]
            elif prev in _DAMPENERS: base *= _DAMPENERS[prev]
        scores.append(base)
    if not scores:
        return 0.0
    raw = sum(scores)
    compound = raw / math.sqrt(raw * raw + 15.0)   # VADER normalisation
    # Punctuation boost
    excl = text.count("!")
    if excl > 0:
        sign = 1 if compound >= 0 else -1
        compound += sign * min(excl * 0.027, 0.27)
    return float(max(-1.0, min(1.0, compound)))


def score_headline(headline: str, description: str = "") -> float:
    return score_text(headline) * 0.65 + score_text(description) * 0.35


def score_batch(texts: List[str]) -> List[float]:
    return [score_text(t) for t in texts]


def aggregate(scores: List[float],
              weights: Optional[List[float]] = None) -> float:
    if not scores:
        return 0.0
    arr = np.array(scores, dtype=float)
    if weights:
        w = np.array(weights, dtype=float)
        w = w / w.sum()
        return float(np.dot(arr, w))
    return float(arr.mean())
