"""
simulation/monte_carlo.py  —  V12 Monte Carlo Engine
======================================================
Implements all V12 spec corrections over V11:

  CORRECTION 1 — Individual noise: 0.07 → 0.06
  CORRECTION 2 — Regional noise:   0.11 → 0.08
  CORRECTION 3 — Global swing:     28% (σ=0.042) → 22% (σ=0.035)
  CORRECTION 4 — Anti-incumbency:  20% (-7%) → 15% (-0.03 to -0.06)
  CORRECTION 5 — Turnout shock:    20% (σ=0.04) → 15% (σ=0.03)
  CORRECTION 6 — Opp surge:        8% (+4%) → 7% (+4%)  [unchanged ≈ spec]
  CORRECTION 7 — Prob bounds:      clip to [0.05, 0.95] per sim  ← NEW
  CORRECTION 8 — Soft flatten:     prob = 0.85*p + 0.15*0.5  ← spec formula

WHY LOWER NOISE?
  V11 had regional σ=0.11 + global σ=0.042 → combined effective std ≈ 0.12
  This was too noisy: BJP std=6.0, NDA range=22 seats.
  V12 with regional σ=0.08 + global σ=0.035 → effective std ≈ 0.09
  Expected: BJP std ≈ 4–5, NDA range ≈ 15–20 seats.
  This is more consistent with real election poll spread.

PROBABILITY BOUNDS (CRITICAL V12 FIX):
  V11 never clipped DURING simulation — only at the base probability stage.
  When noise is added, noisy_prob can exceed 0.95 or go below 0.05 momentarily.
  V12: after adding all noise, clip to [0.05, 0.95] before sampling.
  This prevents artificial 100% or 0% wins in individual simulations.
"""

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

from src.utils.helpers import soft_flatten_array, RANDOM_SEED

# ── V12 noise parameters (per spec) ──────────────────────────────────────────
IND_NOISE       = 0.06   # individual noise σ
REG_NOISE       = 0.08   # regional correlated noise σ
GLOBAL_PROB     = 0.22   # global swing probability
GLOBAL_SIGMA    = 0.035  # global swing magnitude σ
ANTI_INC_PROB   = 0.15   # anti-incumbency shock probability
ANTI_INC_MIN    = -0.06  # anti-incumbency range floor
ANTI_INC_MAX    = -0.03  # anti-incumbency range ceiling
TURNOUT_PROB    = 0.15   # turnout shock probability
TURNOUT_SIGMA   = 0.03   # turnout shock magnitude
OPP_SURGE_PROB  = 0.07   # opposition surge probability
OPP_SURGE_BOOST = 0.04   # opposition surge magnitude
SWING_REV_PROB  = 0.10   # swing reversal probability
TACTICAL_BONUS  = 0.020  # tactical voting bonus in close seats
CLOSE_GAP       = 0.15   # definition of "close seat"
UPSET_PROB      = 0.08   # per-seat random upset probability
UPSET_BOOST     = 0.28   # runner-up boost in upset simulations
PROB_SIM_MIN    = 0.05   # hard floor DURING simulation
PROB_SIM_MAX    = 0.95   # hard ceiling DURING simulation
MAJORITY        = 64

MAIN_PARTIES = ["BJP","INC","AGP","AIUDF","BPF","UPPL"]
ALLIANCE_MAP = {"BJP":"NDA","AGP":"NDA","UPPL":"NDA",
                "INC":"OPPOSITION","AIUDF":"OPPOSITION","BPF":"OPPOSITION"}

REGION_MAP = {
    "UPPER_ASSAM": [
        "DIBRUGARH","TINSUKIA","DIGBOI","MARGHERITA","CHABUA","DULIAJAN","MORAN",
        "DOOM DOOMA","NAHARKATIA","SONARI","LAHOWAL","MAHMARA","SADIYA",
        "DHAKUAKHANA","JONAI","DHEMAJI","MAJULI","JORHAT","TITABAR","MARIANI",
        "SARUPATHAR","HOWRAGHAT","GOLAGHAT","DERGAON","BOKAKHAT","KALIABOR",
        "TEOK","NAZIRA","THOWRA",
    ],
    "CENTRAL_ASSAM": [
        "NOWGONG","RAHA","SAMAGURI","DHING","BATADROBA","HOJAI","LUMDING",
        "JAGIROAD","MARIGAON","LAHARIGHAT","MANGALDOI","SIPAJHAR","BIHPURIA",
        "DHEKIAJULI","GOHPUR","SOOTEA","BISWANATH","BEHALI","BARCHALLA",
        "RANGAPARA","TEZPUR","BARHAMPUR","HAJO","LAKHIMPUR","NORTH LAKHIMPUR",
        "MORIGAON",
    ],
    "LOWER_ASSAM": [
        "NALBARI","BARKHETRI","PATACHARKUCHI","KAMALPUR","RANGIYA","JALUKBARI",
        "GAUHATI EAST","GAUHATI WEST","DISPUR","BOKO","CHAYGAON","PALASBARI",
        "BONGAIGAON","BIJNI","ABHAYAPURI NORTH","ABHAYAPURI SOUTH","BARPETA",
        "BAGHBAR","SARUKHETRI","CHENGA","BHABANIPUR","CHAPAGURI","GOLAKGANJ",
        "GAURIPUR","GOALPARA EAST","GOALPARA WEST","DALGAON","DUDHNOI",
    ],
    "BARAK_VALLEY": [
        "SILCHAR","SONAI","DHOLAI","BARKHOLA","KATIGORAH","LAKHIPUR","KATLICHERRA",
        "HAILAKANDI","ALGAPUR","BADARPUR","PATHERKANDI","RATABARI",
        "KARIMGANJ NORTH","KARIMGANJ SOUTH","JALESWAR","SOUTH SALMARA",
    ],
    "BODOLAND": [
        "KOKRAJHAR EAST","KOKRAJHAR WEST","GOSSAIGAON","SIDLI","BARAMA",
        "TAMULPUR","UDALGURI","MAJBAT","PANEERY","KALAIGAON","CHAPAGURI",
    ],
    "HILLS":      ["HAFLONG","DIPHU","BOKAJAN","BAITHALANGSO"],
    "AIUDF_BELT": ["DHUBRI","BILASIPARA EAST","BILASIPARA WEST","MANKACHAR",
                   "JANIA","JAMUNAMUKH","BORKHETRY","AMGOURI"],
}
_C2R = {c:r for r,cs in REGION_MAP.items() for c in cs}
_NATIONAL  = {"BJP","INC"}
_NDA_ALLY  = {"AGP","UPPL","BPF"}


def build_const_table(pred_df: pd.DataFrame) -> dict:
    """
    Build per-constituency lookup with V12 spec flattening.
    V12 flatten: prob = 0.85*p + 0.15*0.5
    """
    table = {}
    for const, g in pred_df.groupby("constituency"):
        probs = np.maximum(g["win_probability"].values.astype(float), 1e-4)
        probs /= probs.sum()
        flat  = soft_flatten_array(probs, 0.15)   # V12 spec strength

        sp  = np.sort(flat)[::-1]
        gap = float(sp[0]-sp[1]) if len(sp)>=2 else 1.0
        inc_m = g["rank_2021"] == 1
        inc   = g[inc_m]["party"].iloc[0] if inc_m.any() else None
        ally_map = g.set_index("party")["alliance"].to_dict()

        table[const] = {
            "parties":    g["party"].tolist(),
            "probs":      probs,
            "flat_probs": flat,
            "region":     _C2R.get(const,"CENTRAL_ASSAM"),
            "is_close":   gap < CLOSE_GAP,
            "gap":        gap,
            "incumbent":  inc,
            "ally_map":   ally_map,
        }

    close_n = sum(1 for cd in table.values() if cd["is_close"])
    raw_max = max(cd["probs"].max() for cd in table.values())
    flat_max= max(cd["flat_probs"].max() for cd in table.values())
    log.info(f"  Table: {len(table)} seats, {close_n} close (gap<{CLOSE_GAP:.0%})")
    log.info(f"  V12 flatten (0.85*P+0.15*0.5): max {raw_max:.2f}→{flat_max:.2f}")
    return table


def run_simulation(const_table: dict,
                   n_sims: int = 20_000,
                   seed:   int = RANDOM_SEED) -> pd.DataFrame:
    """
    Run V12 Monte Carlo elections with corrected noise levels.

    Key V12 differences:
    1. noise levels per spec (lower than V11)
    2. prob clipped to [0.05, 0.95] AFTER noise addition (before sampling)
    3. anti-incumbency uses uniform random draw from [-0.06, -0.03]
    """
    rng = np.random.default_rng(seed)

    reg_list = list(REGION_MAP.keys()) + ["CENTRAL_ASSAM"]
    reg_idx  = {r:i for i,r in enumerate(reg_list)}
    n_reg    = len(reg_list)
    consts   = list(const_table.keys())

    # Pre-draw all sim-level random variables
    reg_sw     = rng.normal(0.0, REG_NOISE,    (n_sims, n_reg))
    g_fires    = rng.random(n_sims) < GLOBAL_PROB
    g_mags     = rng.normal(0.0, GLOBAL_SIGMA,  n_sims)
    ai_fires   = rng.random(n_sims) < ANTI_INC_PROB
    ai_shocks  = rng.uniform(ANTI_INC_MIN, ANTI_INC_MAX, n_sims)  # V12: random in range
    ts_fires   = rng.random(n_sims) < TURNOUT_PROB
    ts_mags    = rng.normal(0.0, TURNOUT_SIGMA, n_sims)
    os_fires   = rng.random(n_sims) < OPP_SURGE_PROB
    sw_rev     = rng.random(n_sims) < SWING_REV_PROB

    log.info(f"  V12 simulation: {n_sims:,} × {len(consts)} consts")
    log.info(f"  σ_ind={IND_NOISE:.0%}  σ_reg={REG_NOISE:.0%}  "
             f"global={GLOBAL_PROB:.0%}(σ={GLOBAL_SIGMA:.1%})")

    all_counts = []

    for si in range(n_sims):
        counts = {p:0 for p in MAIN_PARTIES}; counts["OTHER"]=0

        rsw   = reg_sw[si]
        gf    = g_fires[si];   gm = g_mags[si]
        aif   = ai_fires[si];  ais = ai_shocks[si]
        tsf   = ts_fires[si];  tsm = ts_mags[si]
        osf   = os_fires[si]
        sr    = sw_rev[si]

        if sr:
            gm = -gm; rsw = -rsw

        for const in consts:
            cd    = const_table[const]
            probs = cd["flat_probs"].copy()
            n_p   = len(probs)
            ri    = reg_idx.get(cd["region"], 0)
            rsw_c = rsw[ri]
            amap  = cd["ally_map"]

            ind = rng.normal(0.0, IND_NOISE, n_p)

            for j, p in enumerate(cd["parties"]):
                ally = amap.get(p,"OTHERS")

                # Regional swing
                if   p in _NATIONAL:  ind[j] += rsw_c
                elif p in _NDA_ALLY:  ind[j] += rsw_c*0.30

                # Global swing
                if gf:
                    geff = gm if ally=="NDA" else (-gm if ally=="OPPOSITION" else gm*0.2)
                    if cd["is_close"]: geff *= 1.5   # reduced from 1.6
                    ind[j] += geff

                # Anti-incumbency (random severity in range)
                if aif and p==cd["incumbent"]:
                    ind[j] += ais   # uniform draw from [-0.06, -0.03]

                # Turnout shock (asymmetric: high turnout ↑OPP, ↓NDA)
                if tsf:
                    if ally=="OPPOSITION": ind[j] += tsm*0.5
                    elif ally=="NDA":      ind[j] -= tsm*0.25

                # Opposition surge
                if osf and ally=="OPPOSITION":
                    ind[j] += OPP_SURGE_BOOST

            # Tactical voting
            if cd["is_close"]:
                ind[int(np.argmax(probs))] += TACTICAL_BONUS

            # Upset injection
            if n_p>=2 and rng.random()<UPSET_PROB:
                ind[int(np.argsort(probs)[-2])] += UPSET_BOOST

            # V12 CRITICAL FIX: clip BEFORE sampling
            noisy = probs + ind
            noisy = np.clip(noisy, PROB_SIM_MIN, PROB_SIM_MAX)   # spec: clip(0.05, 0.95)
            noisy /= noisy.sum()

            winner = cd["parties"][rng.choice(n_p, p=noisy)]
            counts[winner if winner in counts else "OTHER"] += 1

        counts["NDA"]        = counts["BJP"]+counts["AGP"]+counts["UPPL"]
        counts["OPPOSITION"] = counts["INC"]+counts["AIUDF"]+counts["BPF"]
        all_counts.append(counts)

    return pd.DataFrame(all_counts)


def compute_summary(df_sims: pd.DataFrame) -> dict:
    S = {}
    for party in MAIN_PARTIES+["NDA","OPPOSITION"]:
        if party not in df_sims.columns: continue
        c = df_sims[party]
        p5,p25,p50,p75,p95 = map(int, np.percentile(c,[5,25,50,75,95]))
        S[party] = {
            "mean":round(float(c.mean()),1),"median":p50,
            "std":round(float(c.std()),1),
            "p5":p5,"p25":p25,"p75":p75,"p95":p95,
            "range_90":f"{p5}–{p95}","range_50":f"{p25}–{p75}",
            "majority_prob":f"{(c>=MAJORITY).mean():.1%}",
        }
    S["_meta"] = {
        "model_version":"V12",
        "n_simulations":len(df_sims),
        "ind_noise":IND_NOISE,"reg_noise":REG_NOISE,
        "global_prob":GLOBAL_PROB,"global_sigma":GLOBAL_SIGMA,
        "anti_inc_prob":ANTI_INC_PROB,
        "turnout_prob":TURNOUT_PROB,"opp_surge_prob":OPP_SURGE_PROB,
        "flatten_strength":0.15,"prob_sim_clip":f"[{PROB_SIM_MIN},{PROB_SIM_MAX}]",
        "nda_majority_prob":    f"{(df_sims['NDA']>=MAJORITY).mean():.1%}",
        "bjp_majority_prob":    f"{(df_sims['BJP']>=MAJORITY).mean():.1%}",
        "bjp_largest_prob":     f"{(df_sims['BJP']==df_sims[MAIN_PARTIES].max(axis=1)).mean():.1%}",
        "nda_upset_prob":       f"{(df_sims['NDA']<MAJORITY).mean():.1%}",
        "generated":            datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    return S
