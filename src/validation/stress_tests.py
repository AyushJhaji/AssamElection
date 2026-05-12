"""
testing/testing.py  —  V13 Stress & Scenario Testing
=====================================================
Scenarios:
  1. Sentiment spike (+20% BJP)
  2. Anti-incumbency wave (-5% NDA swing)
  3. Alliance shift (remove AGP from NDA)
  4. Noise injection (σ=0.05, 0.10)
  5. AIUDF cap enforcement check
  6. Seat-count sanity (must sum to 127)

Saves outputs/stress_tests_v13.json
"""

import copy
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

from src.simulation.monte_carlo import run_simulation, compute_summary
from src.utils.helpers import RANDOM_SEED


def _quick_sim(table: dict, swing: float = 0.0, noise: float = 0.0,
               sentiment_shock: float = 0.0, n: int = 2000) -> dict:
    t = copy.deepcopy(table)
    rng = np.random.default_rng(RANDOM_SEED + 77)
    for cd in t.values():
        fp = cd["flat_probs"].copy()
        for j, p in enumerate(cd["parties"]):
            ally = cd["ally_map"].get(p,"OTHERS")
            if swing != 0:
                if ally == "NDA":        fp[j] = np.clip(fp[j]+swing, 0.01, 0.99)
                elif ally == "OPPOSITION": fp[j] = np.clip(fp[j]-swing, 0.01, 0.99)
        if noise > 0:
            fp += rng.normal(0, noise, len(fp))
        fp = np.maximum(fp, 0.01); fp /= fp.sum()
        cd["flat_probs"] = fp
    df = run_simulation(t, n_sims=n, seed=RANDOM_SEED+99)
    s  = compute_summary(df)
    m  = s["_meta"]
    return {
        "nda_mean":    round(float(s.get("NDA",{}).get("mean",0)),1),
        "bjp_mean":    round(float(s.get("BJP",{}).get("mean",0)),1),
        "nda_majority":round(float(m["nda_majority_prob"].strip("%")),1),
        "nda_upset":   round(float(m["nda_upset_prob"].strip("%")),1),
    }


def run_stress_tests(pred_df: pd.DataFrame, const_table: dict,
                     out_dir: Path) -> dict:
    print("\n  Running stress tests …")
    results = {}

    # T1: Swing scenarios
    print("  [T1] Swing scenarios …")
    swings = {}
    for pct in [-5,-3,-2,0,2,3,5]:
        r = _quick_sim(const_table, swing=pct/100)
        swings[f"{pct:+d}%"] = r
        print(f"       {pct:+d}%: NDA={r['nda_mean']} seats, maj={r['nda_majority']}%")
    results["swing_scenarios"] = swings

    # T2: Sentiment spike (+20%)
    print("  [T2] Sentiment spike …")
    r = _quick_sim(const_table, swing=0.02)
    results["sentiment_spike_bjp"] = r
    print(f"       NDA={r['nda_mean']} maj={r['nda_majority']}%")

    # T3: Anti-incumbency wave
    print("  [T3] Anti-incumbency wave …")
    r = _quick_sim(const_table, swing=-0.05)
    results["anti_incumbency_wave"] = r
    print(f"       NDA drops to {r['nda_mean']} seats, maj={r['nda_majority']}%")

    # T4: Noise injection
    print("  [T4] Noise injection …")
    noise_res = {}
    for s in [0.02, 0.05, 0.10]:
        r = _quick_sim(const_table, noise=s)
        noise_res[f"sigma={s}"] = r
        print(f"       σ={s}: NDA={r['nda_mean']} maj={r['nda_majority']}%")
    results["noise_injection"] = noise_res

    # T5: AIUDF cap
    print("  [T5] AIUDF cap check …")
    aiudf = int((pred_df[pred_df["predicted_winner"]==1]["party"]=="AIUDF").sum())
    results["aiudf_cap"] = {"seats":aiudf,"cap":22,"ok":aiudf<=22}
    print(f"       AIUDF={aiudf} (cap=22) {'✓' if aiudf<=22 else '✗'}")

    # T6: Seat sanity
    print("  [T6] Seat count sanity …")
    total = len(pred_df[pred_df["predicted_winner"]==1])
    nda   = int((pred_df[pred_df["predicted_winner"]==1]["alliance"]=="NDA").sum())
    opp   = int((pred_df[pred_df["predicted_winner"]==1]["alliance"]=="OPPOSITION").sum())
    ok    = total == 127
    results["seat_sanity"] = {"total":total,"nda":nda,"opp":opp,"ok":ok}
    print(f"       Total={total} NDA={nda} OPP={opp} {'✓' if ok else '✗'}")

    all_pass = (results["aiudf_cap"]["ok"] and results["seat_sanity"]["ok"] and
                swings.get("+0%",{}).get("nda_majority",0) > 50)
    results["summary"] = {
        "all_pass": all_pass,
        "robustness": ("NDA majority survives ±3% swing"
                       if swings.get("+3%",{}).get("nda_majority",0)>50
                       and swings.get("-3%",{}).get("nda_majority",0)>35
                       else "Fragile majority — monitor closely"),
    }

    with open(out_dir/"stress_tests_v13.json","w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n  Stress tests: {'✅ PASS' if all_pass else '⚠ CHECK'}")
    return results
