"""
src/sentiment/fusion.py
=======================
Fuses Meta(30%) + Twitter(25%) + News(20%) + GTrends(25%) into a single
party-level sentiment signal with trend, volatility, momentum, and
search_momentum features.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

PARTIES = ["BJP", "INC", "AIUDF", "AGP", "UPPL", "BPF"]

# Local constituency offsets (2021-calibrated)
_LOCAL: Dict[str, Dict[str, float]] = {
    "DIBRUGARH": {"BJP":+0.04,"INC":-0.02}, "TINSUKIA":{"BJP":+0.04,"INC":-0.02},
    "MAJULI":    {"BJP":+0.05,"INC":-0.03}, "JORHAT":  {"BJP":+0.02},
    "GOLAGHAT":  {"BJP":+0.02,"AGP":+0.01}, "SILCHAR": {"BJP":+0.03,"INC":+0.01},
    "SONAI":     {"INC":+0.02,"AIUDF":+0.01},"DHING":  {"AIUDF":+0.04,"BJP":-0.03},
    "CHENGA":    {"AIUDF":+0.04,"BJP":-0.02},"BHABANIPUR":{"AIUDF":+0.03,"INC":+0.01},
    "KOKRAJHAR EAST":{"UPPL":+0.05,"BJP":-0.02},"KOKRAJHAR WEST":{"UPPL":+0.05,"INC":-0.02},
    "GOSSAIGAON":{"UPPL":+0.04},"SIDLI":{"UPPL":+0.05},
    "BARHAMPUR": {"BJP":+0.01,"INC":+0.01},"NAZIRA":{"INC":+0.01,"BJP":+0.01},
    "DHUBRI":    {"AIUDF":+0.04,"BJP":-0.02},"JANIA":{"AIUDF":+0.04},
    "MANKACHAR": {"AIUDF":+0.03},
}


def _agg(df: pd.DataFrame, score_col: str = "sentiment_score",
          weight_col: str = None) -> Dict[str, float]:
    result = {}
    for p in PARTIES:
        mask = df.get("party_tag", pd.Series(dtype=str)) == p
        sub  = df[mask] if mask.any() else df.head(0)
        if len(sub) == 0:
            result[p] = 0.0; continue
        scores = sub[score_col].fillna(0.0).values.astype(float)
        if weight_col and weight_col in sub.columns:
            w = sub[weight_col].fillna(0.5).values.clip(0.01, 1.0)
            w /= w.sum()
            val = float(np.dot(scores, w))
        else:
            val = float(scores.mean())
        result[p] = round(float(np.clip(val, -1.0, 1.0)), 4)
    return result


def _trend_score(df: pd.DataFrame, party: str) -> float:
    mask = df.get("party_tag", pd.Series(dtype=str)) == party
    sub  = df[mask].copy()
    if len(sub) < 3: return 0.0
    try:
        sub["dt"] = pd.to_datetime(sub["date"], errors="coerce")
        sub = sub.dropna(subset=["dt"]).sort_values("dt")
        if len(sub) < 3: return 0.0
        corr = float(np.corrcoef(np.arange(len(sub), dtype=float),
                                  sub["sentiment_score"].fillna(0).values)[0, 1])
        return 0.0 if np.isnan(corr) else round(corr, 4)
    except:
        return 0.0


def _volatility(df: pd.DataFrame, party: str) -> float:
    mask = df.get("party_tag", pd.Series(dtype=str)) == party
    sub  = df[mask]
    if len(sub) < 2: return 0.0
    return round(float(sub["sentiment_score"].fillna(0).std()), 4)


def _momentum(df: pd.DataFrame, party: str) -> float:
    if "date" not in df.columns or df.empty: return 0.0
    mask = df.get("party_tag", pd.Series(dtype=str)) == party
    sub  = df[mask].copy()
    if len(sub) < 4: return 0.0
    try:
        sub["dt"] = pd.to_datetime(sub["date"], errors="coerce")
        sub = sub.dropna(subset=["dt"]).sort_values("dt")
        mid = len(sub) // 2
        recent = float(sub.iloc[mid:]["sentiment_score"].fillna(0).mean())
        older  = float(sub.iloc[:mid]["sentiment_score"].fillna(0).mean())
        return round(recent - older, 4)
    except:
        return 0.0


def fuse(meta_df: pd.DataFrame, twitter_df: pd.DataFrame,
         news_df: pd.DataFrame, gtrends_df: pd.DataFrame,
         out_dir: Path, data_dir: Path) -> pd.DataFrame:
    """
    Produce per-party fused sentiment DataFrame.
    Saves: outputs/sentiment.csv + data/fused_sentiment.csv
    """
    from src.config.settings import CFG

    meta_agg   = _agg(meta_df,    "sentiment_score", "engagement_score")
    twitter_agg= _agg(twitter_df, "sentiment_score")
    news_agg   = _agg(news_df,    "sentiment_score")

    # GTrends → sentiment-scale signal
    gt_signal: Dict[str, float] = {}
    if not gtrends_df.empty:
        median_int = gtrends_df["normalised_interest"].median()
        for _, row in gtrends_df.iterrows():
            rel = float(np.clip(
                (row["normalised_interest"] - median_int) / max(median_int, 0.01),
                -1.0, 1.0))
            mom = float(row["momentum"])
            gt_signal[row["party"]] = round(0.70 * rel + 0.30 * mom, 4)

    rows = []
    for p in PARTIES:
        m_s  = meta_agg.get(p, 0.0)
        tw_s = twitter_agg.get(p, 0.0)
        n_s  = news_agg.get(p, 0.0)
        gt_s = gt_signal.get(p, 0.0)

        fused = float(np.clip(
            CFG.WEIGHT_META * m_s + CFG.WEIGHT_TWITTER * tw_s +
            CFG.WEIGHT_NEWS * n_s + CFG.WEIGHT_GTRENDS * gt_s,
            -1.0, 1.0))

        trend = _trend_score(twitter_df, p)
        vol   = _volatility(twitter_df, p)
        mom   = _momentum(twitter_df, p)

        gt_row = gtrends_df[gtrends_df["party"] == p] if not gtrends_df.empty else pd.DataFrame()
        s_mom = float(gt_row["momentum"].iloc[0]) if len(gt_row) else 0.0
        s_int = float(gt_row["normalised_interest"].iloc[0]) if len(gt_row) else 0.5

        final = float(np.clip(0.65*fused + 0.15*trend + 0.12*mom + 0.08*s_mom, -1.0, 1.0))

        n_meta = int((meta_df.get("party_tag", pd.Series(dtype=str)) == p).sum())
        n_tw   = int((twitter_df.get("party_tag", pd.Series(dtype=str)) == p).sum())
        n_news = int((news_df.get("party_tag", pd.Series(dtype=str)) == p).sum())

        rows.append({
            "party":           p,
            "meta_sentiment":  round(m_s,   4),
            "twitter_sentiment":round(tw_s, 4),
            "news_sentiment":  round(n_s,   4),
            "gtrends_signal":  round(gt_s,  4),
            "fused_sentiment": round(fused, 4),
            "trend_score":     round(trend, 4),
            "volatility":      round(vol,   4),
            "momentum":        round(mom,   4),
            "search_momentum": round(s_mom, 4),
            "search_interest": round(s_int, 4),
            "final_sentiment": round(final, 4),
            "sent_adj":        round(final, 4),   # alias for feature builder
            "n_meta":   n_meta, "n_twitter": n_tw, "n_news": n_news,
            "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        })

    fused_df = pd.DataFrame(rows)
    fused_df.to_csv(out_dir  / "sentiment.csv",      index=False)
    fused_df.to_csv(data_dir / "fused_sentiment.csv", index=False)

    # constituency-level adjustments
    _build_const_sentiment(fused_df, data_dir, out_dir)

    print("\n  Fusion: Meta30%+Twitter25%+News20%+GTrends25%")
    print(f"  {'Party':<8} {'Meta':>8} {'Twitter':>8} {'News':>8} {'GTrends':>8} {'Final':>8}")
    print(f"  {'─'*60}")
    for _, r in fused_df.iterrows():
        print(f"  {r.party:<8} {r.meta_sentiment:>+8.3f} "
              f"{r.twitter_sentiment:>+8.3f} {r.news_sentiment:>+8.3f} "
              f"{r.gtrends_signal:>+8.3f} {r.final_sentiment:>+8.3f}")
    return fused_df


def _build_const_sentiment(fused_df: pd.DataFrame,
                            data_dir: Path, out_dir: Path) -> None:
    sent_map = fused_df.set_index("party")["final_sentiment"].to_dict()
    cp = data_dir / "constituencies.csv"
    if not cp.exists(): return
    consts = (pd.read_csv(cp, header=None).iloc[:, 0]
              .str.strip().str.upper().tolist())
    rows = []
    for c in consts:
        adj = _LOCAL.get(c, {})
        for p in PARTIES:
            nat = sent_map.get(p, 0.0); loc = adj.get(p, 0.0)
            rows.append({"constituency": c, "party": p,
                         "national_sentiment": round(nat, 4),
                         "local_adj": round(loc, 4),
                         "constituency_sentiment": round(float(np.clip(nat+loc,-1,1)),4)})
    pd.DataFrame(rows).to_csv(out_dir / "constituency_sentiment.csv", index=False)


def load_sentiment(data_dir: str) -> pd.DataFrame:
    """Load best available fused sentiment."""
    dp = Path(data_dir)
    for fname in ["fused_sentiment.csv", "fused_sentiment_v14.csv",
                  "fused_sentiment_v13.csv", "fused_sentiment_v7.csv"]:
        p = dp / fname
        if p.exists():
            s = pd.read_csv(p)
            defaults = {"sent_adj": 0.0, "trend_score": 0.0, "final_sentiment": 0.0,
                        "sentiment_score": 0.0, "volatility": 0.0, "momentum": 0.0,
                        "search_momentum": 0.0, "search_interest": 0.5,
                        "gtrends_signal": 0.0, "media_bias": 0.0}
            for col, default in defaults.items():
                if col not in s.columns:
                    s[col] = (s.get("final_sentiment", pd.Series(0.0, index=s.index))
                               if col == "sentiment_score" else default)
            log.info(f"  Loaded sentiment: {fname}")
            return s
    log.warning("  No sentiment file — using zeros")
    return pd.DataFrame({
        "party": PARTIES, **{k: v for k, v in {
            "sent_adj": 0.0, "trend_score": 0.0, "final_sentiment": 0.0,
            "sentiment_score": 0.0, "volatility": 0.0, "momentum": 0.0,
            "search_momentum": 0.0, "search_interest": 0.5, "gtrends_signal": 0.0,
        }.items()}
    })
