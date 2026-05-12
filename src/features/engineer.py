"""
features/build_features.py  —  V14 Feature Engineering
========================================================
V14 additions over V13:
  NEW 1 — search_momentum from Google Trends
  NEW 2 — search_interest (normalised GTrends interest score)
  NEW 3 — gtrends_signal (combined search behaviour signal)
  NEW 4 — data_quality_score (fraction of real vs synthetic data per party)
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Dict

log = logging.getLogger(__name__)

MAIN_PARTIES = ["BJP","INC","AGP","AIUDF","BPF","UPPL"]
ALLIANCE_MAP = {"BJP":"NDA","AGP":"NDA","UPPL":"NDA","INC":"OPPOSITION","AIUDF":"OPPOSITION","BPF":"OPPOSITION"}
NON_PARTY_TAGS = {"IND","NOTA"}

NAME_FIXES = {"ABHAYAPURI SOUTH (SC)":"ABHAYAPURI SOUTH","BARKHETRY":"BARKHETRI","BOKO SC":"BOKO","KATIGORAH":"KATIGORA","KATLICHERA":"KATLICHERRA"}

RENAMED_SEATS = {
    "AMGOURI":{"AGP":0.52,"INC":0.42},"BORKHETRY":{"INC":0.50,"BJP":0.44},"DHARMAMPUR":{"BJP":0.53,"INC":0.40},
    "DUDHNOI":{"INC":0.53,"BJP":0.41},"KATIGORAH":{"INC":0.54,"AIUDF":0.38},"MORIGAON":{"BJP":0.52,"INC":0.43},
    "NAUBOICHA":{"INC":0.52,"BJP":0.44},"NORTH LAKHIMPUR":{"BJP":0.53,"INC":0.42},"PANEERY":{"BJP":0.53,"UPPL":0.40},
    "PATHERKANDI":{"BJP":0.54,"INC":0.39},"RANGIYA":{"BJP":0.52,"INC":0.44},"SIB SAGAR":{"BJP":0.49,"INC":0.48},
    "SOUTH SALMARA":{"INC":0.52,"AIUDF":0.44},
}

BAYES_ALPHA = 6.0

UPSET_FEATURE_NAMES = [
    # V12 core
    "nda_adv","nda_total_vs","opp_total_vs","effective_n_parties","top2_margin",
    "constituency_vol","vote_swing","vote_swing_sq","sent_adj","alliance_is_nda",
    "incumbent_win_rate","prev_winner_vs","log_total_votes","n_candidates",
    # V13 sentiment
    "sentiment_score","trend_score","volatility","momentum","media_bias",
    # V14 new
    "search_momentum","search_interest","gtrends_signal","data_quality_score",
]


def load_data(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    path = Path(data_dir)/"final_dataset.csv"
    df   = pd.read_csv(path)
    df["constituency"] = df["constituency"].str.upper().str.strip().replace(NAME_FIXES)
    df["alliance"]     = df["party"].map(ALLIANCE_MAP).fillna("OTHERS")
    df = _recompute_swing(df)
    df["real_winner"] = 0
    mask = ~df["party"].isin(NON_PARTY_TAGS)
    idx  = df[mask].groupby(["constituency","year"])["total_votes"].idxmax()
    df.loc[idx, "real_winner"] = 1
    df16 = df[df["year"]==2016].copy()
    df21 = df[df["year"]==2021].copy()
    log.info(f"  Loaded: 2016={len(df16)}/{df16['constituency'].nunique()} | 2021={len(df21)}/{df21['constituency'].nunique()}")
    return df, df16, df21


def _recompute_swing(df):
    years = sorted(df["year"].unique())
    if len(years) < 2: df["vote_swing"]=np.nan; return df
    yr_new,yr_old = years[-1],years[-2]
    vs_old = df[df["year"]==yr_old].set_index(["constituency","party"])["vote_share"].to_dict()
    def _sw(row):
        if row["year"]!=yr_new: return np.nan
        v = vs_old.get((row["constituency"],row["party"]))
        return row["vote_share"]-v if v is not None else np.nan
    df["vote_swing"] = df.apply(_sw, axis=1)
    covered = df[df["year"]==yr_new]["vote_swing"].notna().sum()
    total   = (df["year"]==yr_new).sum()
    log.info(f"  vote_swing: {covered}/{total} ({covered/total:.0%})")
    return df


def load_constituencies(data_dir: str) -> list:
    return pd.read_csv(Path(data_dir)/"constituencies.csv",header=None).iloc[:,0].str.strip().str.upper().tolist()


def load_sentiment(data_dir: str) -> pd.DataFrame:
    dp = Path(data_dir)
    for fname in ["fused_sentiment_v14.csv","fused_sentiment_v13.csv","fused_sentiment_v7.csv"]:
        p = dp/fname
        if p.exists():
            s = pd.read_csv(p)
            for c in ["sent_adj","trend_score","final_sentiment","sentiment_score","volatility","momentum","media_bias","search_momentum","search_interest","gtrends_signal"]:
                if c not in s.columns:
                    s[c] = s.get("final_sentiment", pd.Series(0.0,index=s.index)) if c=="sentiment_score" else 0.5 if c=="search_interest" else 0.0
            log.info(f"  Loaded sentiment: {fname} ({len(s)} rows)")
            return s
    log.warning("  No sentiment file — using zeros")
    return pd.DataFrame({"party":MAIN_PARTIES,"sent_adj":0.0,"trend_score":0.0,"final_sentiment":0.0,"sentiment_score":0.0,"volatility":0.0,"momentum":0.0,"media_bias":0.0,"search_momentum":0.0,"search_interest":0.5,"gtrends_signal":0.0})


def bayesian_win_rate(df_year, alpha=BAYES_ALPHA):
    gm = df_year["real_winner"].mean()
    s  = df_year.groupby("party").agg(wins=("real_winner","sum"),n=("real_winner","count")).reset_index()
    s["smooth_wr"] = (s["wins"]+alpha*gm)/(s["n"]+alpha)
    s["n_contests"] = s["n"]
    return s[["party","smooth_wr","n_contests"]]


def constituency_stats(df):
    records = []
    for const, g in df.groupby("constituency"):
        vs=g["vote_share"].values; nda_vs=float(g[g["alliance"]=="NDA"]["vote_share"].sum()); opp_vs=float(g[g["alliance"]=="OPPOSITION"]["vote_share"].sum())
        maj_vs=vs[vs>0.02]; enp=1.0/float(np.sum(maj_vs**2)) if len(maj_vs)>0 else 2.0
        sv=np.sort(vs)[::-1]; t2m=float(sv[0]-sv[1]) if len(sv)>=2 else float(sv[0])
        records.append({"constituency":const,"nda_total_vs":nda_vs,"opp_total_vs":opp_vs,"nda_adv":nda_vs-opp_vs,"effective_n_parties":enp,"top2_margin":t2m,"constituency_vol":float(vs.std()),"n_candidates":int(len(g)),"total_valid_votes":float(g["total_votes"].sum()),"log_total_votes":float(np.log1p(g["total_votes"].sum()))})
    return pd.DataFrame(records)


def _sent_feats(party: str, sent_df: pd.DataFrame,
                raw_dfs: Optional[Dict[str, pd.DataFrame]] = None) -> dict:
    """Extract all V14 sentiment + GTrends features for one party."""
    def _col(name, default=0.0):
        if name in sent_df.columns:
            row = sent_df[sent_df["party"]==party]
            return float(row[name].iloc[0]) if len(row)>0 else default
        return default

    sent    = _col("final_sentiment", _col("sent_adj", 0.0))
    trend   = _col("trend_score",   0.0)
    vol     = _col("volatility",    0.0)
    mom     = _col("momentum",      0.0)
    news    = _col("news_sentiment", sent)
    bias    = float(np.clip(0.6*news - 0.4*sent, -1.0, 1.0))
    s_mom   = _col("search_momentum",  0.0)
    s_int   = _col("search_interest",  0.5)
    gt_sig  = _col("gtrends_signal",   0.0)

    # Data quality score: fraction of real data for this party across all sources
    dq = 0.0
    if raw_dfs:
        total_real = total_all = 0
        for df in raw_dfs.values():
            if df is None or df.empty: continue
            ptag = "party_tag" if "party_tag" in df.columns else None
            if ptag:
                sub = df[df[ptag]==party]
                src = sub.get("source", pd.Series(dtype=str))
                total_all  += len(sub)
                total_real += (src != "synthetic").sum()
        dq = round(float(total_real / max(total_all, 1)), 4)

    return {
        "sentiment_score":    round(sent,  4),
        "trend_score":        round(trend, 4),
        "volatility":         round(vol,   4),
        "momentum":           round(mom,   4),
        "media_bias":         round(bias,  4),
        "search_momentum":    round(s_mom, 4),
        "search_interest":    round(s_int, 4),
        "gtrends_signal":     round(gt_sig,4),
        "data_quality_score": dq,
    }


def build_upset_rows(df_target, df_prev, wr_prev, sent_df,
                     raw_dfs: Optional[Dict] = None):
    cs      = constituency_stats(df_prev)
    wr_map  = wr_prev.set_index("party")["smooth_wr"].to_dict()
    s_map   = sent_df.set_index("party")["final_sentiment"].to_dict() if "final_sentiment" in sent_df.columns else {}
    prev_w  = df_prev[df_prev["winner"]==1].groupby("constituency")["party"].first().to_dict()
    curr_w  = df_target[df_target["real_winner"]==1].groupby("constituency")["party"].first().to_dict()
    prev_vs = {(r["constituency"],r["party"]):r["vote_share"] for _,r in df_prev[df_prev["winner"]==1].iterrows()}
    prev_sw = {(r["constituency"],r["party"]):r.get("vote_swing",np.nan) for _,r in df_prev[df_prev["winner"]==1].iterrows()}
    common  = set(prev_w)&set(curr_w)
    log.info(f"  Upset training: {len(common)} constituencies")
    rows = []
    for const in sorted(common):
        pw=prev_w[const]; cw=curr_w[const]; upset=int(pw!=cw)
        pr=df_prev[(df_prev["constituency"]==const)&(df_prev["party"]==pw)]
        inc_ally=pr["alliance"].iloc[0] if len(pr)>0 else "OTHERS"
        vsw=float(prev_sw.get((const,pw),0.0) or 0.0)
        vs_p=float(prev_vs.get((const,pw),0.15))
        r=cs[cs["constituency"]==const]
        if len(r)==0: continue
        r=r.iloc[0]
        sf=_sent_feats(pw,sent_df,raw_dfs)
        rows.append({"constituency":const,"upset":upset,"nda_adv":r["nda_adv"],"nda_total_vs":r["nda_total_vs"],"opp_total_vs":r["opp_total_vs"],"effective_n_parties":r["effective_n_parties"],"top2_margin":r["top2_margin"],"constituency_vol":r["constituency_vol"],"n_candidates":r["n_candidates"],"log_total_votes":r["log_total_votes"],"vote_swing":vsw,"vote_swing_sq":vsw**2,"sent_adj":s_map.get(pw,0.0),"alliance_is_nda":float(inc_ally=="NDA"),"incumbent_win_rate":wr_map.get(pw,0.15),"prev_winner_vs":vs_p,**sf})
    df_out = pd.DataFrame(rows)
    log.info(f"  Upsets: {df_out['upset'].sum()}/{len(df_out)} = {df_out['upset'].mean():.1%}")
    return df_out


def build_2026_features(df21, wr21, sent_df, const_list,
                        raw_dfs: Optional[Dict] = None):
    cs    = constituency_stats(df21)
    wr_map= wr21.set_index("party")["smooth_wr"].to_dict()
    s_map = sent_df.set_index("party")["final_sentiment"].to_dict() if "final_sentiment" in sent_df.columns else {}
    prev_w = df21[df21["real_winner"]==1].groupby("constituency")["party"].first().to_dict()
    prev_vs= {(r["constituency"],r["party"]):r["vote_share"] for _,r in df21[df21["real_winner"]==1].iterrows()}
    prev_sw= {(r["constituency"],r["party"]):r.get("vote_swing",0.0) or 0.0 for _,r in df21[df21["real_winner"]==1].iterrows()}
    rows   = []
    for const in const_list:
        pw=prev_w.get(const); pr=df21[(df21["constituency"]==const)&(df21["party"]==pw)] if pw else pd.DataFrame()
        inc_ally=pr["alliance"].iloc[0] if len(pr)>0 else "OTHERS"
        vsw=float(prev_sw.get((const,pw),0.0)) if pw else 0.0
        vs_p=float(prev_vs.get((const,pw),0.15)) if pw else 0.15
        r=cs[cs["constituency"]==const]
        if len(r)>0:
            r=r.iloc[0]; nda_v,opp_v,enp,t2m,cvol,nc,lv=(r["nda_total_vs"],r["opp_total_vs"],r["effective_n_parties"],r["top2_margin"],r["constituency_vol"],r["n_candidates"],r["log_total_votes"])
        else:
            nda_v=opp_v=0.0; enp=2.5; t2m=0.15; cvol=0.15; nc=6; lv=11.5
        sf=_sent_feats(pw or "BJP",sent_df,raw_dfs)
        rows.append({"constituency":const,"prev_winner":pw or "","nda_adv":nda_v-opp_v,"nda_total_vs":nda_v,"opp_total_vs":opp_v,"effective_n_parties":enp,"top2_margin":t2m,"constituency_vol":cvol,"n_candidates":nc,"log_total_votes":lv,"vote_swing":vsw,"vote_swing_sq":vsw**2,"sent_adj":s_map.get(pw,0.0) if pw else 0.0,"alliance_is_nda":float(inc_ally=="NDA"),"incumbent_win_rate":wr_map.get(pw,0.15) if pw else 0.15,"prev_winner_vs":vs_p,**sf})
    return pd.DataFrame(rows)
