"""
validation/validation.py  —  V14 Validation
============================================
Fixes V13 bug: json.dump fails on numpy.bool_ → use default=_json_safe.
"""

import json, logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss

log = logging.getLogger(__name__)

from src.utils.helpers import vote_to_prob_array, normalise, expected_calibration_error
from src.validation.checks import run_all_checks, print_checks, print_calibration


def _json_safe(obj):
    """Handle numpy types that json.dump can't serialize."""
    if isinstance(obj, (np.bool_,)):          return bool(obj)
    if isinstance(obj, (np.integer,)):         return int(obj)
    if isinstance(obj, (np.floating,)):        return float(obj)
    if isinstance(obj, np.ndarray):            return obj.tolist()
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def run_backtest(df21: pd.DataFrame, df16: pd.DataFrame) -> dict:
    vs16 = df16.set_index(["constituency","party"])["vote_share"].to_dict()
    rows = []
    for const, g in df21.groupby("constituency"):
        for _, r in g.iterrows():
            rows.append({"const":const,"party":r["party"],"vs16":vs16.get((const,r["party"]),0.0),"actual":r["real_winner"]})
    bt = pd.DataFrame(rows)
    for const, g in bt.groupby("const"):
        p = normalise(vote_to_prob_array(g["vs16"].values, k=8))
        bt.loc[g.index, "pred_prob"] = p
    y, yh = bt["actual"].values, bt["pred_prob"].values.astype(float)
    correct = total = 0
    for const, g in bt.groupby("const"):
        pw = g.loc[g["pred_prob"].idxmax(),"party"]
        aw = g[g["actual"]==1]["party"]
        if len(aw) > 0: correct += int(pw==aw.iloc[0]); total += 1
    cal = print_calibration(y, yh, "Backtest 2016→2021")
    return {"auc":round(roc_auc_score(y,yh),3),"brier":round(brier_score_loss(y,yh),4),"seat_acc":round(correct/total,3) if total else 0,"bt_df":bt,"cal":cal}


def run_validation(pred_df, df_sims, summary, upset_map, out_dir, bt=None):
    all_pass, checks = run_all_checks(pred_df, df_sims, summary, upset_map)
    print_checks(checks, all_pass)
    if bt and "cal" in bt:
        pd.DataFrame(bt["cal"]["bins"]).to_csv(out_dir/"calibration_v14.csv", index=False)
    report = {
        "version":           "V14",
        "all_pass":          bool(all_pass),   # ensure plain Python bool
        "n_checks":          len(checks),
        "n_passed":          int(sum(1 for _,ok,_ in checks if ok)),
        "checks":            [{"name":n,"passed":bool(ok),"value":str(v)} for n,ok,v in checks],
        "backtest_auc":      float(bt["auc"])      if bt else None,
        "backtest_brier":    float(bt["brier"])    if bt else None,
        "backtest_seat_acc": float(bt["seat_acc"]) if bt else None,
    }
    with open(out_dir/"validation_report_v14.json","w") as fh:
        json.dump(report, fh, indent=2, default=_json_safe)
    log.info(f"  Validation: {'PASS' if all_pass else 'FAIL'} ({report['n_passed']}/{report['n_checks']})")
    return all_pass, checks
