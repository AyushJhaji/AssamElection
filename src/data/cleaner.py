"""
src/data/cleaner.py
===================
Text cleaning for social media and news DataFrames.
Every removal is logged with its reason.
"""

import hashlib
import logging
import re
from typing import Set

import pandas as pd

log = logging.getLogger(__name__)

MIN_CHARS  = 20
MAX_JACCARD = 0.85

_SPAM_RE = [
    re.compile(r"^(.{5,})\1{2,}$"),
    re.compile(r"follow\s+back",       re.I),
    re.compile(r"click\s+here",        re.I),
    re.compile(r"buy\s+now",           re.I),
    re.compile(r"free\s+offer",        re.I),
    re.compile(r"[\U0001F600-\U0001F9FF]{4,}"),
]


def _norm(text: str) -> str:
    if not text: return ""
    t = text.lower()
    t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"@\w+",          " ", t)
    t = re.sub(r"#(\w+)",      r"\1", t)
    t = re.sub(r"[^\w\s']",     " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _fp(text: str) -> str:
    return hashlib.md5(_norm(text).encode()).hexdigest()


def _jaccard(a: str, b: str) -> float:
    def bigrams(s: str) -> Set:
        w = s.split()
        return set(zip(w, w[1:])) if len(w) >= 2 else set(w)
    A, B = bigrams(a), bigrams(b)
    if not A or not B: return 0.0
    return len(A & B) / len(A | B)


def clean(df: pd.DataFrame,
          text_col: str = "tweet_text",
          source_col: str = "source") -> pd.DataFrame:
    """
    Full cleaning pipeline: length filter → spam → exact dedup → near-dedup.
    Returns cleaned DataFrame with source audit logged.
    """
    if df.empty: return df

    # Resolve text column
    if text_col not in df.columns:
        for alt in ["post_text", "title", "tweet_text", "text"]:
            if alt in df.columns:
                text_col = alt; break
        else:
            log.warning("  clean: no text column found — skipping")
            return df

    n_orig = len(df)
    df = df.copy()
    df["_n"] = df[text_col].fillna("").apply(_norm)

    # 1. Length
    short = df["_n"].str.len() < MIN_CHARS
    if short.any():
        log.info(f"  clean: dropped {short.sum()} short texts (<{MIN_CHARS} chars)")
    df = df[~short].copy()

    # 2. Spam
    spam = df["_n"].apply(lambda t: any(p.search(t) for p in _SPAM_RE))
    if spam.any():
        log.info(f"  clean: dropped {spam.sum()} spam items")
    df = df[~spam].copy()

    # 3. Exact dedup
    df["_fp"] = df["_n"].apply(_fp)
    before = len(df)
    df = df.drop_duplicates(subset=["_fp"])
    if (before - len(df)):
        log.info(f"  clean: dropped {before - len(df)} exact duplicates")
    df = df.drop(columns=["_fp"])

    # 4. Near-dedup (only on real/live items; skip if too many rows)
    if len(df) <= 300:
        real_idx = df[df.get(source_col, pd.Series("x", index=df.index)) != "synthetic"].index.tolist()
        keep     = set(df.index)
        texts    = df["_n"].to_dict()
        nd = 0
        for i in range(len(real_idx)):
            if real_idx[i] not in keep: continue
            for j in range(i + 1, len(real_idx)):
                if real_idx[j] not in keep: continue
                if _jaccard(texts[real_idx[i]], texts[real_idx[j]]) >= MAX_JACCARD:
                    keep.discard(real_idx[j]); nd += 1
        if nd:
            log.info(f"  clean: dropped {nd} near-duplicates (Jaccard≥{MAX_JACCARD})")
            df = df[df.index.isin(keep)]

    df = df.drop(columns=["_n"], errors="ignore").reset_index(drop=True)

    # Source audit
    if source_col in df.columns:
        counts  = df[source_col].value_counts().to_dict()
        real_n  = sum(v for k, v in counts.items() if k not in ("synthetic","historical"))
        pct     = real_n / len(df) * 100 if len(df) else 0
        log.info(f"  clean audit: {len(df)}/{n_orig} kept | "
                 f"real={real_n} ({pct:.0f}%) | {counts}")
        if pct < 5 and real_n < 3:
            log.warning(f"  ⚠ Very low real data ({pct:.1f}%) — check API connectivity")

    return df
