"""
src/pipeline.py
===============
Pipeline orchestrator. Encapsulates all steps so main.py stays thin.
Each step returns self for optional chaining; state is stored on self.
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


class Pipeline:
    """
    Full forecasting pipeline for Assam 2026.
    Call run() or individual step methods.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.cfg.OUT_DIR.mkdir(exist_ok=True)
        self.cfg.MDL_DIR.mkdir(exist_ok=True)
        self.cfg.CACHE_DIR.mkdir(exist_ok=True)

        # State populated as pipeline runs
        self.df = self.df16 = self.df21 = None
        self.const_list   = []
        self.wr21 = self.wr16 = None
        self.sent_df      = None
        self.raw_data     = None
        self.upset_rows   = None
        self.feats_2026   = None
        self.model        = None
        self.train_metrics= {}
        self.backtest     = {}
        self.pred_df      = None
        self.winners      = None
        self.upset_map    = {}
        self.df_sims      = None
        self.summary      = {}
        self.const_table  = {}
        self.all_pass     = False

    # ── Step 1: Load ECI data ─────────────────────────────────────────────────
    def load_data(self) -> "Pipeline":
        from src.data.loader import load_eci, load_constituencies, bayesian_win_rate
        from src.data.validator import gate_eci_data, print_gate

        self.df, self.df16, self.df21 = load_eci(str(self.cfg.DATA_DIR))
        self.const_list = load_constituencies(str(self.cfg.DATA_DIR))
        self.wr21 = bayesian_win_rate(self.df21)
        self.wr16 = bayesian_win_rate(self.df16)

        ok, msg = gate_eci_data(self.df16, self.df21)
        print_gate(ok, msg, "ECI Data")
        if not ok:
            raise RuntimeError(f"ECI data gate failed: {msg}")
        return self

    # ── Step 2: Collect sentiment ─────────────────────────────────────────────
    def collect_sentiment(self, skip: bool = False) -> "Pipeline":
        from src.data.validator import audit_source, gate_sentiment_quality, print_gate

        if skip:
            from src.sentiment.fusion import load_sentiment
            self.sent_df  = load_sentiment(str(self.cfg.DATA_DIR))
            self.raw_data = None
            log.info("  Sentiment: loaded from cache")
            return self

        from src.sentiment.collector import collect_all
        from src.data.cleaner       import clean
        from src.sentiment.fusion   import fuse

        collected = collect_all(seed=self.cfg.RANDOM_SEED)

        # Clean text sources
        collected.meta    = clean(collected.meta,    "post_text",   "source")
        collected.twitter = clean(collected.twitter, "tweet_text",  "source")
        collected.news    = clean(collected.news,    "title",       "source",
                                   near_dedup=False)

        # Audit
        audits = [
            audit_source(collected.meta,    name="meta"),
            audit_source(collected.twitter, name="twitter"),
            audit_source(collected.news,    name="news"),
        ]
        ok, msg = gate_sentiment_quality(audits, warn_threshold=self.cfg.MIN_REAL_DATA_WARN)
        print_gate(ok, msg, "Sentiment Quality")
        if not ok and self.cfg.HALT_ON_ZERO_REAL:
            raise RuntimeError(f"Sentiment quality gate failed: {msg}")

        # Save raw files
        collected.meta.to_csv(   self.cfg.OUT_DIR / "raw_meta.csv",    index=False)
        collected.twitter.to_csv(self.cfg.OUT_DIR / "raw_twitter.csv", index=False)
        collected.news.to_csv(   self.cfg.OUT_DIR / "raw_news.csv",    index=False)
        collected.gtrends.to_csv(self.cfg.OUT_DIR / "raw_gtrends.csv", index=False)

        # Fuse
        self.sent_df = fuse(
            collected.meta, collected.twitter, collected.news, collected.gtrends,
            self.cfg.OUT_DIR, self.cfg.DATA_DIR,
        )
        self.raw_data = {"meta": collected.meta, "twitter": collected.twitter,
                         "news": collected.news}
        return self

    # ── Step 3: Feature engineering ───────────────────────────────────────────
    def build_features(self) -> "Pipeline":
        from src.features.engineer import build_upset_rows, build_2026_features
        self.upset_rows = build_upset_rows(
            self.df21, self.df16, self.wr16, self.sent_df, self.raw_data)
        self.feats_2026 = build_2026_features(
            self.df21, self.wr21, self.sent_df, self.const_list, self.raw_data)
        new_cols = [c for c in self.feats_2026.columns
                    if any(s in c for s in ["sentiment","trend","vol","mom","media",
                                             "search","gtrend","quality"])]
        log.info(f"  Features: {len(self.upset_rows)} training | "
                 f"{len(self.feats_2026)} prediction | V14 cols={new_cols}")
        return self

    # ── Step 4: Train model ───────────────────────────────────────────────────
    def train(self) -> "Pipeline":
        from src.models.train import train_model
        from src.data.validator import gate_model_quality, print_gate
        self.model, self.train_metrics = train_model(self.upset_rows, self.cfg.MDL_DIR)
        ok, msg = gate_model_quality(self.train_metrics, self.cfg.OVERFIT_THRESHOLD)
        print_gate(ok, msg, "Model Quality")
        return self

    # ── Step 5: Backtest + calibration ───────────────────────────────────────
    def validate(self) -> "Pipeline":
        from src.validation.backtest import run_backtest
        self.backtest = run_backtest(self.df21, self.df16)
        self.backtest["bt_df"].to_csv(self.cfg.OUT_DIR / "backtest.csv", index=False)
        log.info(f"  Backtest: seat_acc={self.backtest['seat_acc']:.1%} "
                 f"AUC={self.backtest['auc']:.3f} Brier={self.backtest['brier']:.4f}")
        return self

    # ── Step 6: Predict ───────────────────────────────────────────────────────
    def predict(self) -> "Pipeline":
        from src.models.predict import build_base_probs, blend_and_assign, apply_aiudf_cap

        upset_probs    = self.model.predict(self.feats_2026)
        self.upset_map = dict(zip(self.feats_2026["constituency"], upset_probs))

        dp_base      = build_base_probs(self.df21, self.df16, self.sent_df,
                                         self.wr21, self.const_list)
        dp           = blend_and_assign(dp_base, self.upset_map, self.df21)
        dp           = apply_aiudf_cap(dp)
        self.pred_df = dp
        self.winners = dp[dp["predicted_winner"] == 1].copy()

        self.winners.to_csv(self.cfg.OUT_DIR / "winners.csv",     index=False, encoding="utf-8-sig")
        self.pred_df.to_csv(self.cfg.OUT_DIR / "predictions.csv", index=False, encoding="utf-8-sig")
        return self

    # ── Step 7: Monte Carlo ───────────────────────────────────────────────────
    def simulate(self, n_sims: Optional[int] = None) -> "Pipeline":
        from src.simulation.monte_carlo import build_const_table, run_simulation, compute_summary
        from src.utils.helpers import save_json

        n = n_sims or self.cfg.N_SIMULATIONS
        self.const_table = build_const_table(self.pred_df)
        self.df_sims     = run_simulation(self.const_table, n_sims=n,
                                           seed=self.cfg.RANDOM_SEED)
        self.summary     = compute_summary(self.df_sims)
        self.df_sims.to_csv(self.cfg.OUT_DIR / "seat_distribution.csv", index=False)
        save_json(self.cfg.OUT_DIR / "summary.json", self.summary)
        return self

    # ── Step 8: Full validation checks ───────────────────────────────────────
    def check(self) -> "Pipeline":
        from src.validation.backtest import run_validation
        self.all_pass, _ = run_validation(
            self.pred_df, self.df_sims, self.summary,
            self.upset_map, self.cfg.OUT_DIR, self.backtest)
        return self

    # ── Step 9: Stress tests ──────────────────────────────────────────────────
    def stress_test(self) -> "Pipeline":
        from src.validation.stress_tests import run_stress_tests
        run_stress_tests(self.pred_df, self.const_table, self.cfg.OUT_DIR)
        return self

    # ── Step 10: Save metadata ────────────────────────────────────────────────
    def save_meta(self) -> "Pipeline":
        joblib.dump({
            "wr21": self.wr21, "wr16": self.wr16,
            "sent_df": self.sent_df, "df21": self.df21, "df16": self.df16,
            "upset_map": self.upset_map, "train_metrics": self.train_metrics,
            "backtest": self.backtest,
        }, self.cfg.MDL_DIR / "pipeline_meta.pkl")
        return self

    # ── Validate outputs exist ────────────────────────────────────────────────
    def outputs_ready(self) -> bool:
        required = ["summary.json", "winners.csv", "predictions.csv",
                    "seat_distribution.csv"]
        missing  = [f for f in required if not (self.cfg.OUT_DIR / f).exists()]
        if missing:
            # Fallback: check V14 file names
            v14_missing = [f for f in [f.replace(".","_v14.") for f in missing]
                           if not (self.cfg.OUT_DIR / f).exists()]
            if v14_missing:
                log.error(f"  Missing outputs: {missing}")
                return False
        return True
