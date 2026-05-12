"""
validation/checks.py  —  V12 Strict Validation
================================================
Per spec: model must pass ALL checks or is considered INVALID.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path

log = logging.getLogger(__name__)

from src.utils.helpers import expected_calibration_error

# V12 validation thresholds
VALID = {
    "close_seats_min":    15,
    "close_seats_max":    40,
    "prob_min":           0.40,
    "prob_max":           0.93,
    "upset_mean_min":     0.30,
    "upset_mean_max":     0.50,
    "std_min":            4.0,
    "nda_maj_min":        0.55,
    "nda_maj_max":        0.95,
    "nda_upset_min":      0.05,
    "spread_min":         0.08,
    "close_gap":          0.15,
}


def run_all_checks(pred_df: pd.DataFrame,
                   df_sims: pd.DataFrame,
                   summary: dict,
                   upset_probs: dict) -> tuple:
    """
    Run all V12 validation checks.
    Returns (all_pass: bool, results: list of (name, ok, value)).
    """
    m  = summary["_meta"]
    wp = pred_df[pred_df["predicted_winner"]==1]["win_probability"]
    gaps = [
        g["win_probability"].nlargest(2).diff().abs().iloc[-1]
        if len(g)>=2 else 1.0
        for _,g in pred_df.groupby("constituency")
    ]
    close_cnt = sum(1 for g in gaps if g < VALID["close_gap"])
    nda_maj   = float(m["nda_majority_prob"].strip("%"))/100
    nda_upset = float(m["nda_upset_prob"].strip("%"))/100
    bjp_std   = df_sims["BJP"].std()
    nda_range = int(df_sims["NDA"].max()-df_sims["NDA"].min())
    up_mean   = float(np.mean(list(upset_probs.values())))

    checks = [
        (f"Close seats (gap<{VALID['close_gap']:.0%}): {VALID['close_seats_min']}–{VALID['close_seats_max']}",
         VALID["close_seats_min"]<=close_cnt<=VALID["close_seats_max"], str(close_cnt)),
        (f"Min winner prob ≥ {VALID['prob_min']:.0%}",
         wp.min()>=VALID["prob_min"], f"{wp.min():.1%}"),
        (f"Max winner prob ≤ {VALID['prob_max']:.0%}",
         wp.max()<=VALID["prob_max"], f"{wp.max():.1%}"),
        (f"Upset prob mean {VALID['upset_mean_min']:.2f}–{VALID['upset_mean_max']:.2f}",
         VALID["upset_mean_min"]<=up_mean<=VALID["upset_mean_max"], f"{up_mean:.3f}"),
        (f"BJP std dev ≥ {VALID['std_min']}",
         bjp_std>=VALID["std_min"], f"{bjp_std:.1f}"),
        (f"NDA majority {VALID['nda_maj_min']:.0%}–{VALID['nda_maj_max']:.0%}",
         VALID["nda_maj_min"]<=nda_maj<=VALID["nda_maj_max"], m["nda_majority_prob"]),
        (f"NDA upset ≥ {VALID['nda_upset_min']:.0%}",
         nda_upset>=VALID["nda_upset_min"], m["nda_upset_prob"]),
        ("Winner prob spread ≥ 0.08",
         wp.std()>=VALID["spread_min"], f"{wp.std():.3f}"),
        ("NDA seat range > 15",
         nda_range>15, f"{nda_range} seats"),
        ("No party > 95% majority",
         all(float(summary.get(p,{}).get("majority_prob","0%").strip("%"))<95
             for p in ["BJP","INC","AGP","AIUDF","UPPL"]), "checked"),
        ("No deterministic outcomes",
         nda_maj<0.99 and float(m["bjp_majority_prob"].strip("%"))/100<0.99, "checked"),
    ]

    all_pass = all(ok for _,ok,_ in checks)
    return all_pass, checks


def print_checks(checks: list, all_pass: bool) -> None:
    print("\n" + "="*65)
    print("  VALIDATION — V12")
    print("="*65)
    for name, ok, val in checks:
        print(f"  [{'✓ PASS' if ok else '✗ FAIL'}]  {name:<48} → {val}")
    print()
    if all_pass:
        print("  ✅ ALL CHECKS PASSED — V12 is forecast-grade")
    else:
        print("  ⚠️  SOME CHECKS FAILED — tune parameters in monte_carlo.py")


def print_calibration(y_true: np.ndarray, y_pred: np.ndarray, label: str="") -> dict:
    cal = expected_calibration_error(y_true, y_pred, n_bins=7)
    if label: print(f"\n  Calibration — {label}")
    print(f"  ECE = {cal['ece']:.4f}")
    print(f"  {'Pred':>8} {'Actual':>8} {'Error':>8} {'N':>6}  Status")
    print(f"  {'─'*40}")
    for b in cal["bins"]:
        s = "✓" if b["error"]<0.10 else "⚠" if b["error"]<0.18 else "✗"
        print(f"  {b['mean_pred']:>7.2f}  {b['frac_pos']:>7.2f}  "
              f"{b['error']:>7.3f}  {b['n']:>5}  {s}")
    return cal
