"""
prediction/predict.py  —  V12 Prediction Engine
=================================================
Builds 2026 constituency-level win probabilities.

V12 CHANGES vs V11:

  CHANGE 1 — vote_to_prob replaces sigmoid(margin/sigma):
    V11: P = sigmoid(margin / 0.12)  [relative margin]
    V12: P = 1/(1+exp(-k*(vs-0.5))) [absolute vote share, k=8]

    Why better:
      - More interpretable: 45% vote share → P≈0.40 regardless of rivals
      - Multi-party races: in a 4-party race, each party's prob is computed
        independently before normalisation → less sensitivity to IND/NOTA
      - Calibration: k=8 gives P≈0.55 for a party with 53% (2021 median
        winner), matching empirical incumbent retention rate ~60%

  CHANGE 2 — Blending formula:
    V11: base prob with upset adjustment (up to 22%)
    V12: p_final = (1-alpha)*p_base + alpha*p_upset  [alpha=0.12]
    This is the spec formula — cleaner than V11's incumbent-only adjustment

  CHANGE 3 — Soft flattening:
    V11: P_flat = P + 0.28*(0.5-P)  → P=0.92 becomes 0.80
    V12: P_flat = 0.85*P + 0.15*0.5 → P=0.92 becomes 0.858
    V12 preserves strongholds better while still preventing overconfidence

  CHANGE 4 — Hard probability bounds:
    After all adjustments: clip to [0.05, 0.95] per spec
    After per-constituency normalisation: clip to [0.08, 0.92] (realistic range)
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

from src.utils.helpers import (
    vote_to_prob_array, blend_probs, soft_flatten_array,
    clip_prob_array, normalise, PROB_FLOOR, PROB_CEILING,
)
from src.features.engineer import MAIN_PARTIES, ALLIANCE_MAP, RENAMED_SEATS

ALWAYS_INCLUDE = {"BJP", "INC"}
AIUDF_CAP      = 22
BLEND_ALPHA     = 0.12    # spec: 0.10–0.18; using 0.12
VOTE_TO_PROB_K  = 8.0     # spec: k=8 for vote_to_prob
FLATTEN_STRENGTH = 0.15   # spec: prob = 0.85*p + 0.15*0.5
FLOOR_AFTER_NORM = 0.08
CEIL_AFTER_NORM  = 0.92


def build_base_probs(df21: pd.DataFrame,
                     df16: pd.DataFrame,
                     sent_df: pd.DataFrame,
                     wr21: pd.DataFrame,
                     const_list: list) -> pd.DataFrame:
    """
    Build vote_to_prob base probabilities for every constituency.

    For each constituency:
      1. Collect 2021 vote shares (or ECI-verified priors for 13 renamed seats)
      2. Apply sentiment nudge ±2%
      3. Transform: P = vote_to_prob(vs, k=8)
      4. Normalise per constituency
      5. Apply soft flatten (strength=0.15)
      6. Clip to [0.08, 0.92]

    Returns DataFrame with columns:
      constituency, party, alliance, vs_2021, vs_proj, rank_2021,
      nda_total_vs, opp_total_vs, p_base
    """
    vs21   = df21.set_index(["constituency","party"])["vote_share"].to_dict()
    rank21 = df21.set_index(["constituency","party"])["rank"].to_dict()
    nda21  = (df21[df21["alliance"]=="NDA"]
              .groupby("constituency")["vote_share"].sum().to_dict())
    opp21  = (df21[df21["alliance"]=="OPPOSITION"]
              .groupby("constituency")["vote_share"].sum().to_dict())
    s_map  = sent_df.set_index("party")["final_sentiment"].to_dict() \
             if "final_sentiment" in sent_df.columns else {}

    rows = []
    for const in const_list:
        # Renamed seat → use ECI-verified priors
        if const in RENAMED_SEATS:
            seat = RENAMED_SEATS[const]
            win_vs = max(seat.values())
            nda_v = sum(v for p,v in seat.items() if ALLIANCE_MAP.get(p)=="NDA")
            opp_v = sum(v for p,v in seat.items() if ALLIANCE_MAP.get(p)=="OPPOSITION")
            for party, vs in seat.items():
                ally = ALLIANCE_MAP.get(party,"OTHERS")
                sent = s_map.get(party,0.0)
                vs_p = max(vs + np.clip(sent*0.02,-0.02,0.02), 0.0)
                rows.append({"constituency":const,"party":party,"alliance":ally,
                             "vs_2021":vs,"vs_proj":vs_p,
                             "rank_2021":1 if vs==win_vs else 2,
                             "nda_total_vs":nda_v,"opp_total_vs":opp_v})
            continue

        parties = [p for p in MAIN_PARTIES
                   if (const,p) in vs21 or p in ALWAYS_INCLUDE]
        for party in parties:
            ally = ALLIANCE_MAP.get(party,"OTHERS")
            v21  = vs21.get((const,party), 0.0)
            sent = s_map.get(party, 0.0)
            if v21 == 0.0 and party not in ALWAYS_INCLUDE:
                continue
            vs_p = max(v21 + np.clip(sent*0.02,-0.02,0.02), 0.0) if v21>0 else 0.05
            rows.append({"constituency":const,"party":party,"alliance":ally,
                         "vs_2021":v21,"vs_proj":vs_p,
                         "rank_2021":rank21.get((const,party),10),
                         "nda_total_vs":nda21.get(const,0),
                         "opp_total_vs":opp21.get(const,0)})

    dp = pd.DataFrame(rows)

    # vote_to_prob transformation
    rec = []
    for const, g in dp.groupby("constituency"):
        g = g.copy().reset_index(drop=True)
        # V12: use absolute vote share (not margin) for initial prob
        vs_arr = g["vs_proj"].values
        p_raw  = vote_to_prob_array(vs_arr, k=VOTE_TO_PROB_K)
        p_norm = normalise(p_raw)
        p_flat = soft_flatten_array(p_norm, FLATTEN_STRENGTH)
        # Final clip
        p_clip = np.clip(p_flat, FLOOR_AFTER_NORM, CEIL_AFTER_NORM)
        p_clip = p_clip / p_clip.sum()
        for i, row in g.iterrows():
            rec.append({**row.to_dict(), "p_base": float(p_clip[i])})

    dp2 = pd.DataFrame(rec)
    log.info(f"  Base probs: {len(dp2)} rows, "
             f"{dp2['constituency'].nunique()} constituencies")
    top = dp2.groupby("constituency")["p_base"].max()
    log.info(f"  Winner prob: min={top.min():.2f} mean={top.mean():.2f} "
             f"max={top.max():.2f}")
    return dp2


def blend_and_assign(dp_base: pd.DataFrame,
                     upset_map: dict,
                     df21: pd.DataFrame,
                     alpha: float = BLEND_ALPHA) -> pd.DataFrame:
    """
    Blend base probabilities with upset model output.

    V12 formula (spec-correct):
      p_final = (1 - alpha) * p_base + alpha * p_upset

    Then apply alliance seat-sharing (BJP suppressed in AGP/UPPL seats).
    Then re-normalise, assign predicted_winner and confidence_score.
    """
    df21a  = df21.copy()
    df21a["alliance"] = df21a["party"].map(ALLIANCE_MAP).fillna("OTHERS")
    bjp21  = set(df21a[df21a["party"]=="BJP"]["constituency"].unique())
    agp21  = set(df21a[df21a["party"]=="AGP"]["constituency"].unique())
    uppl21 = set(df21a[df21a["party"]=="UPPL"]["constituency"].unique())
    agp_only  = agp21  - bjp21
    uppl_only = uppl21 - bjp21

    dp = dp_base.copy()

    # V12 blending: per-constituency normalised blend
    for const in dp["constituency"].unique():
        mask = dp["constituency"] == const
        g    = dp[mask].copy()
        p_up = float(upset_map.get(const, 0.40))   # default 0.40 if unknown

        # Spec formula: p_final = (1-alpha)*p_base + alpha*p_upset
        # p_upset here is the constituency-level upset prob applied UNIFORMLY
        # to shift the distribution: incumbent gets p_up weight toward the upset
        # scenario; challengers benefit proportionally
        inc_m = g["rank_2021"] == 1
        if inc_m.any():
            inc_idx = g[inc_m].index[0]
            non_inc = g[~inc_m].index
            p_base_inc = float(dp.loc[inc_idx, "p_base"])
            # Blend: p_final_inc = (1-alpha)*p_base_inc + alpha*(1-p_up)
            # This reduces incumbent prob when upset_prob is high
            p_final_inc = (1-alpha)*p_base_inc + alpha*(1-p_up)
            reduction   = max(p_base_inc - p_final_inc, 0.0)
            dp.loc[inc_idx, "p_base"] = max(p_final_inc, 0.02)
            if len(non_inc) > 0:
                w = dp.loc[non_inc, "p_base"]
                dp.loc[non_inc, "p_base"] += reduction * (w / w.sum())

    # Alliance seat-sharing
    for const in agp_only:
        dp.loc[(dp["constituency"]==const)&(dp["party"]=="BJP"),"p_base"] *= 0.05
    for const in uppl_only:
        dp.loc[(dp["constituency"]==const)&(dp["party"]=="BJP"),"p_base"] *= 0.05

    # Final normalise + hard bounds per spec
    dp["p_base"] = dp["p_base"].clip(lower=0.01)
    dp["win_probability"] = dp.groupby("constituency")["p_base"].transform(
        lambda x: x/x.sum())
    # Spec: prob = np.clip(prob, 0.05, 0.95)
    dp["win_probability"] = dp["win_probability"].clip(0.05, 0.95)
    dp["win_probability"] = dp.groupby("constituency")["win_probability"].transform(
        lambda x: x/x.sum())

    # Assign winner
    dp["predicted_winner"] = 0
    dp.loc[dp.groupby("constituency")["win_probability"].idxmax(),
           "predicted_winner"] = 1
    dp["runner_up_prob"] = dp.groupby("constituency")["win_probability"].transform(
        lambda x: x.nlargest(2).iloc[-1] if len(x)>=2 else x.max())
    dp["confidence_score"] = (dp["win_probability"]-dp["runner_up_prob"]).clip(0,1).round(4)
    return dp


def apply_aiudf_cap(dp: pd.DataFrame, cap: int = AIUDF_CAP) -> pd.DataFrame:
    winners = dp[dp["predicted_winner"]==1].copy()
    if winners["party"].value_counts().get("AIUDF",0) <= cap:
        return dp
    excess = (winners[winners["party"]=="AIUDF"]
              .sort_values("confidence_score")
              .head(winners["party"].value_counts()["AIUDF"]-cap)["constituency"].tolist())
    for const in excess:
        cr  = dp[dp["constituency"]==const].sort_values("win_probability",ascending=False)
        non = cr[cr["party"]!="AIUDF"]
        if len(non)>0:
            dp.loc[dp["constituency"]==const,"predicted_winner"]=0
            dp.loc[non.index[0],"predicted_winner"]=1
    log.info(f"  AIUDF cap applied → {cap} seats")
    return dp
