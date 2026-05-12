"""
src/data/validator.py
=====================
Data quality gates. Each gate returns (ok: bool, message: str).
Pipeline checks gates after collection; warns or halts based on CFG.
"""

import logging
from typing import Dict, List, Tuple

import pandas as pd

log = logging.getLogger(__name__)


def audit_source(df: pd.DataFrame, source_col: str = "source",
                 name: str = "") -> Dict:
    """Return real/synthetic counts and percentage for one raw DataFrame."""
    if df is None or df.empty:
        return {"name": name, "total": 0, "real": 0, "synthetic": 0, "pct_real": 0.0,
                "sources": {}}
    counts   = df[source_col].value_counts().to_dict() if source_col in df.columns else {}
    real_n   = sum(v for k, v in counts.items()
                   if k not in ("synthetic", "historical"))
    synth_n  = len(df) - real_n
    pct_real = real_n / len(df) * 100 if len(df) else 0.0
    return {"name": name, "total": len(df), "real": int(real_n),
            "synthetic": int(synth_n), "pct_real": round(pct_real, 1),
            "sources": counts}


def gate_eci_data(df16: pd.DataFrame, df21: pd.DataFrame) -> Tuple[bool, str]:
    """Gate: ECI data must have correct shape and required columns."""
    required = {"constituency", "party", "vote_share", "total_votes", "real_winner"}
    missing  = required - set(df21.columns)
    if missing:
        return False, f"ECI 2021 missing columns: {missing}"
    seats16 = df16["constituency"].nunique()
    seats21 = df21["constituency"].nunique()
    if seats21 < 120:
        return False, f"Expected ≥120 constituencies in 2021, got {seats21}"
    winners = df21[df21["real_winner"] == 1]["constituency"].nunique()
    if winners < 120:
        return False, f"Only {winners}/127 winners found in 2021 data"
    return True, f"ECI OK: 2016={seats16} seats | 2021={seats21} seats | winners={winners}"


def gate_sentiment_quality(audits: List[Dict],
                            warn_threshold: float = 10.0) -> Tuple[bool, str]:
    """Gate: warn if all sources are 0% real data."""
    total_real = sum(a["real"] for a in audits)
    total_all  = sum(a["total"] for a in audits)
    pct_real   = total_real / total_all * 100 if total_all else 0

    lines = []
    for a in audits:
        icon = "✓" if a["pct_real"] > warn_threshold else "⚠"
        lines.append(f"    {icon} {a['name']:<10}: {a['total']:>4} rows | "
                     f"real={a['real']} ({a['pct_real']:.1f}%)")

    summary = "\n".join(lines)

    if total_real == 0:
        return False, f"DATA QUALITY: 0% real data across all sources.\n{summary}"
    if pct_real < warn_threshold:
        log.warning(f"  ⚠ Low real data: {pct_real:.1f}% — predictions use calibrated fallbacks")

    return True, f"Sentiment quality: {pct_real:.1f}% real ({total_real}/{total_all})\n{summary}"


def gate_model_quality(metrics: Dict, threshold: float = 0.10) -> Tuple[bool, str]:
    """Gate: LOO-CV AUC must be reasonable and overfit gap within bounds."""
    loo = metrics.get("loo_auc", 0)
    gap = metrics.get("overfit_gap", 0)
    if loo < 0.55:
        return False, f"Model LOO-CV AUC={loo:.3f} < 0.55 — model not learning"
    if gap > 0.20:
        return False, f"Overfit gap={gap:.3f} > 0.20 after regularisation"
    return True, f"Model OK: LOO-AUC={loo:.3f}  gap={gap:.3f}  regularised={metrics.get('regularised',False)}"


def print_gate(ok: bool, msg: str, label: str = "") -> bool:
    prefix = "✅" if ok else "❌"
    print(f"  {prefix} Gate [{label}]: {msg}")
    if not ok:
        log.error(f"Gate failed [{label}]: {msg}")
    return ok
