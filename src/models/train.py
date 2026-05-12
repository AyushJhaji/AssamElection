"""
training/training.py  —  V14 Model Training
============================================
V14 additions over V13:
  1. Overfitting guard: if train_auc - loo_auc > threshold → add regularisation
  2. Feature selection: drop features with importance < 0.005
  3. 4 new features: search_momentum, search_interest, gtrends_signal, data_quality_score
  4. More conservative GBM params (max_depth=2, lower lr)
"""

import logging, warnings
from pathlib import Path
import sys

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

import joblib
import numpy as np
import pandas as pd
from sklearn.base         import clone
from sklearn.ensemble     import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics      import roc_auc_score, brier_score_loss
from sklearn.model_selection import LeaveOneOut

from src.utils.helpers import standardise_features, temperature_scale_array, expected_calibration_error, log_loss_safe, RANDOM_SEED

# V14 expanded feature set
V14_FEATURES = [
    # V12 core
    "nda_adv","nda_total_vs","opp_total_vs","effective_n_parties","top2_margin",
    "constituency_vol","vote_swing","vote_swing_sq","sent_adj","alliance_is_nda",
    "incumbent_win_rate","prev_winner_vs","log_total_votes","n_candidates",
    # V13 sentiment
    "sentiment_score","trend_score","volatility","momentum","media_bias",
    # V14 new: Google Trends
    "search_momentum","search_interest","gtrends_signal",
    # V14 new: data quality
    "data_quality_score",
]

# V14: more conservative to reduce overfitting (V13 had train_auc 0.948 vs LOO 0.859)
_GB_NORMAL = dict(n_estimators=80, max_depth=2, learning_rate=0.04, subsample=0.70, min_samples_leaf=10, random_state=RANDOM_SEED)
_GB_STRICT = dict(n_estimators=60, max_depth=2, learning_rate=0.03, subsample=0.65, min_samples_leaf=12, random_state=RANDOM_SEED)
_RF_NORMAL = dict(n_estimators=100, max_depth=3, min_samples_leaf=10, class_weight="balanced", max_features="sqrt", random_state=RANDOM_SEED, n_jobs=-1)
_RF_STRICT = dict(n_estimators=80,  max_depth=2, min_samples_leaf=12, class_weight="balanced", max_features="sqrt", random_state=RANDOM_SEED, n_jobs=-1)
_LR        = dict(C=0.08, max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED, solver="lbfgs")

TEMPERATURE = 2.0
UPSET_MIN, UPSET_MAX = 0.20, 0.50
FI_DROP_THRESHOLD = 0.005   # drop features below this importance


class UpsetModelV14:
    """
    V14 ensemble with overfitting guard and feature selection.
    If train_auc - loo_auc > OVERFIT_THRESHOLD, applies stricter regularisation
    and drops low-importance features.
    """

    def __init__(self, overfit_threshold: float = 0.10):
        self.overfit_threshold = overfit_threshold
        self.gb = GradientBoostingClassifier(**_GB_NORMAL)
        self.rf = RandomForestClassifier(**_RF_NORMAL)
        self.lr = LogisticRegression(**_LR)
        self.feat_mean = self.feat_std = None
        self.features: list = []
        self.selected_features: list = []
        self.regularised = False
        self.fitted = False

    def _ensemble(self, Xs):
        return (0.50*self.gb.predict_proba(Xs)[:,1] +
                0.30*self.rf.predict_proba(Xs)[:,1] +
                0.20*self.lr.predict_proba(Xs)[:,1])

    def _calibrate(self, p):
        return np.clip(temperature_scale_array(p, TEMPERATURE), UPSET_MIN, UPSET_MAX)

    def fit(self, df: pd.DataFrame) -> dict:
        # Resolve available features
        feat = [f for f in V14_FEATURES if f in df.columns]
        if not feat:
            feat = [c for c in df.select_dtypes("number").columns if c != "upset"]
        self.features = feat

        X_raw = df[feat].fillna(0).values
        y     = df["upset"].values
        Xs, self.feat_mean, self.feat_std = standardise_features(X_raw)

        # ── Phase 1: LOO-CV ────────────────────────────────────────────────────
        oof = np.zeros(len(y))
        for tr, va in LeaveOneOut().split(Xs):
            gb_ = clone(self.gb); rf_ = clone(self.rf); lr_ = clone(self.lr)
            gb_.fit(Xs[tr],y[tr]); rf_.fit(Xs[tr],y[tr]); lr_.fit(Xs[tr],y[tr])
            oof[va] = (0.50*gb_.predict_proba(Xs[va])[:,1] +
                       0.30*rf_.predict_proba(Xs[va])[:,1] +
                       0.20*lr_.predict_proba(Xs[va])[:,1])

        loo_auc = roc_auc_score(y, oof)

        # ── Phase 2: Full fit to get feature importances ───────────────────────
        self.gb.fit(Xs,y); self.rf.fit(Xs,y); self.lr.fit(Xs,y)
        p_raw_phase1 = self._ensemble(Xs)
        train_auc_phase1 = roc_auc_score(y, p_raw_phase1)
        gap = train_auc_phase1 - loo_auc

        fi = pd.Series(self.gb.feature_importances_, index=feat).sort_values(ascending=False)

        # ── Phase 3: Overfitting guard ────────────────────────────────────────
        if gap > self.overfit_threshold:
            log.warning(f"  ⚠ Overfit detected: train_auc={train_auc_phase1:.3f} "
                        f"loo_auc={loo_auc:.3f} gap={gap:.3f} > {self.overfit_threshold}")
            log.warning(f"  Applying stricter regularisation + feature selection")
            self.regularised = True

            # Drop low-importance features
            keep  = fi[fi >= FI_DROP_THRESHOLD].index.tolist()
            drop  = fi[fi < FI_DROP_THRESHOLD].index.tolist()
            if drop:
                log.info(f"  Feature selection: dropping {len(drop)} low-importance: {drop}")
            self.selected_features = keep if keep else feat

            # Re-standardise on selected features
            X_sel = df[self.selected_features].fillna(0).values
            Xs2, self.feat_mean, self.feat_std = standardise_features(X_sel)

            # Re-fit with stricter params
            self.gb = GradientBoostingClassifier(**_GB_STRICT)
            self.rf = RandomForestClassifier(**_RF_STRICT)
            self.gb.fit(Xs2,y); self.rf.fit(Xs2,y); self.lr.fit(Xs2,y)

            # Re-run LOO on selected features
            oof2 = np.zeros(len(y))
            for tr, va in LeaveOneOut().split(Xs2):
                gb_=clone(self.gb); rf_=clone(self.rf); lr_=clone(self.lr)
                gb_.fit(Xs2[tr],y[tr]); rf_.fit(Xs2[tr],y[tr]); lr_.fit(Xs2[tr],y[tr])
                oof2[va] = (0.50*gb_.predict_proba(Xs2[va])[:,1] +
                            0.30*rf_.predict_proba(Xs2[va])[:,1] +
                            0.20*lr_.predict_proba(Xs2[va])[:,1])

            loo_auc = roc_auc_score(y, oof2)
            p_raw   = self._ensemble(Xs2)
            fi      = pd.Series(self.gb.feature_importances_,
                                index=self.selected_features).sort_values(ascending=False)
            # Use selected features going forward
            self.features = self.selected_features
            Xs = Xs2
        else:
            log.info(f"  ✓ No overfit: gap={gap:.3f} ≤ {self.overfit_threshold}")
            self.selected_features = feat
            p_raw = p_raw_phase1

        self.fitted = True
        p_cal  = self._calibrate(p_raw)
        cal    = expected_calibration_error(y, self._calibrate(oof))
        sent_fi = {k:v for k,v in fi.items() if any(s in k for s in ["sent","trend","vol","mom","media","search","gtrend","quality"])}

        return {
            "loo_auc":            round(loo_auc, 3),
            "train_auc":          round(roc_auc_score(y,p_raw), 3),
            "brier":              round(brier_score_loss(y,p_cal), 4),
            "log_loss":           round(log_loss_safe(y,p_cal), 4),
            "ece":                cal["ece"],
            "n_upsets":           int(y.sum()), "n_total": int(len(y)),
            "upset_rate":         round(float(y.mean()), 3),
            "mean_upset_prob_cal":round(float(p_cal.mean()), 3),
            "mean_upset_prob_raw":round(float(p_raw.mean()), 3),
            "max_fi":             round(float(fi.max()), 3),
            "overfit_gap":        round(gap, 4),
            "regularised":        self.regularised,
            "feature_importances":fi.round(4).to_dict(),
            "sentiment_fi_total": round(sum(sent_fi.values()), 4),
            "n_features":         len(self.features),
            "n_features_dropped": len(self.selected_features) - len(self.features) if self.regularised else 0,
        }

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if not self.fitted: raise RuntimeError("Call fit() first")
        feat  = [f for f in self.features if f in df.columns]
        X_raw = df[feat].fillna(0).values
        if len(feat) < len(self.features):
            full = np.zeros((len(df), len(self.features)))
            for i, f in enumerate(self.features):
                if f in feat: full[:,i] = X_raw[:,feat.index(f)]
            X_raw = full
        Xs,_,_ = standardise_features(X_raw, self.feat_mean, self.feat_std)
        return self._calibrate(self._ensemble(Xs))

    def print_importances(self):
        if not self.features: return
        fi = pd.Series(self.gb.feature_importances_, index=self.features).sort_values(ascending=False)
        tag = "(REGULARISED) " if self.regularised else ""
        print(f"  Feature importances (GBM) — V14 {tag}:")
        for f, v in fi.items():
            if v > 0.005:
                bar  = "█" * int(v*50)
                src  = " [GT]" if "search" in f or "gtrend" in f else " [SENT]" if any(s in f for s in ["sent","trend","vol","mom","media"]) else ""
                flag = " ⚠ HIGH" if v > 0.35 else ""
                print(f"    {f:<26} {v:.4f}  {bar}{src}{flag}")


def train_model(upset_rows: pd.DataFrame, model_dir: Path) -> tuple:
    from src.config.settings import CFG
    model   = UpsetModelV14(overfit_threshold=CFG.OVERFIT_THRESHOLD)
    metrics = model.fit(upset_rows)
    model.print_importances()

    if metrics["regularised"]:
        print(f"\n  ⚠ Regularisation applied (gap={metrics['overfit_gap']:.3f})")
    else:
        print(f"\n  ✓ No overfit (gap={metrics['overfit_gap']:.3f})")

    print(f"  LOO-CV AUC: {metrics['loo_auc']:.3f}  Train AUC: {metrics['train_auc']:.3f}")
    print(f"  Gap: {metrics['overfit_gap']:.3f}  Threshold: {CFG.OVERFIT_THRESHOLD}")

    joblib.dump(model, model_dir/"upset_model_v14.pkl")
    log.info(f"  Saved upset_model_v14.pkl | AUC={metrics['loo_auc']} sent_fi={metrics['sentiment_fi_total']}")
    return model, metrics
