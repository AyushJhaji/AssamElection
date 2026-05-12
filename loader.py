"""
src/data/loader.py
==================
Loads the ECI 2016/2021 dataset. Single responsibility: load + minimal transform.
No modelling logic here.
"""

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

MAIN_PARTIES = ["BJP", "INC", "AGP", "AIUDF", "BPF", "UPPL"]
ALLIANCE_MAP = {
    "BJP": "NDA", "AGP": "NDA", "UPPL": "NDA",
    "INC": "OPPOSITION", "AIUDF": "OPPOSITION", "BPF": "OPPOSITION",
}
NON_PARTY_TAGS = {"IND", "NOTA"}
NAME_FIXES = {
    "ABHAYAPURI SOUTH (SC)": "ABHAYAPURI SOUTH",
    "BARKHETRY":             "BARKHETRI",
    "BOKO SC":               "BOKO",
    "KATIGORAH":             "KATIGORA",
    "KATLICHERA":            "KATLICHERRA",
}
# ECI-verified vote shares for 13 redrawn constituencies (2021)
RENAMED_SEATS = {
    "AMGOURI":         {"AGP": 0.52, "INC": 0.42},
    "BORKHETRY":       {"INC": 0.50, "BJP": 0.44},
    "DHARMAMPUR":      {"BJP": 0.53, "INC": 0.40},
    "DUDHNOI":         {"INC": 0.53, "BJP": 0.41},
    "KATIGORAH":       {"INC": 0.54, "AIUDF": 0.38},
    "MORIGAON":        {"BJP": 0.52, "INC": 0.43},
    "NAUBOICHA":       {"INC": 0.52, "BJP": 0.44},
    "NORTH LAKHIMPUR": {"BJP": 0.53, "INC": 0.42},
    "PANEERY":         {"BJP": 0.53, "UPPL": 0.40},
    "PATHERKANDI":     {"BJP": 0.54, "INC": 0.39},
    "RANGIYA":         {"BJP": 0.52, "INC": 0.44},
    "SIB SAGAR":       {"BJP": 0.49, "INC": 0.48},
    "SOUTH SALMARA":   {"INC": 0.52, "AIUDF": 0.44},
}


def load_eci(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load final_dataset.csv, normalise names, compute vote_swing and real_winner.

    Returns
    -------
    df_all, df_2016, df_2021
    """
    path = Path(data_dir) / "final_dataset.csv"
    df = pd.read_csv(path)
    df["constituency"] = (df["constituency"].str.upper().str.strip()
                          .replace(NAME_FIXES))
    df["alliance"]     = df["party"].map(ALLIANCE_MAP).fillna("OTHERS")
    df = _recompute_swing(df)

    # real_winner = highest total_votes among non-IND/NOTA rows
    df["real_winner"] = 0
    mask = ~df["party"].isin(NON_PARTY_TAGS)
    idx  = df[mask].groupby(["constituency", "year"])["total_votes"].idxmax()
    df.loc[idx, "real_winner"] = 1

    df16 = df[df["year"] == 2016].copy()
    df21 = df[df["year"] == 2021].copy()

    log.info(
        f"  ECI data loaded: "
        f"2016={len(df16)} rows/{df16['constituency'].nunique()} seats | "
        f"2021={len(df21)} rows/{df21['constituency'].nunique()} seats"
    )
    return df, df16, df21


def load_constituencies(data_dir: str) -> list:
    return (pd.read_csv(Path(data_dir) / "constituencies.csv", header=None)
            .iloc[:, 0].str.strip().str.upper().tolist())


def bayesian_win_rate(df_year: pd.DataFrame, alpha: float = 6.0) -> pd.DataFrame:
    """Bayesian-smoothed party win rate for a given election year."""
    gm = df_year["real_winner"].mean()
    s  = df_year.groupby("party").agg(
             wins=("real_winner", "sum"),
             n=("real_winner", "count")).reset_index()
    s["smooth_wr"]  = (s["wins"] + alpha * gm) / (s["n"] + alpha)
    s["n_contests"] = s["n"]
    return s[["party", "smooth_wr", "n_contests"]]


def _recompute_swing(df: pd.DataFrame) -> pd.DataFrame:
    years = sorted(df["year"].unique())
    if len(years) < 2:
        df["vote_swing"] = np.nan
        return df
    yr_new, yr_old = years[-1], years[-2]
    vs_old = (df[df["year"] == yr_old]
              .set_index(["constituency", "party"])["vote_share"].to_dict())

    def _sw(row):
        if row["year"] != yr_new:
            return np.nan
        v = vs_old.get((row["constituency"], row["party"]))
        return row["vote_share"] - v if v is not None else np.nan

    df["vote_swing"] = df.apply(_sw, axis=1)
    covered = df[df["year"] == yr_new]["vote_swing"].notna().sum()
    total   = (df["year"] == yr_new).sum()
    log.info(f"  vote_swing recomputed: {covered}/{total} ({covered/total:.0%})")
    return df
