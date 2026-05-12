"""
main.py — Assam Election ML
============================
Single entry point for the entire forecasting pipeline.

Usage:
    python main.py                    # full pipeline + auto-launch browser
    python main.py --no-ui            # pipeline only, no browser
    python main.py --skip-sentiment   # use cached sentiment (faster)
    python main.py --quick            # 5k simulations, skip stress tests
    python main.py --port 9000        # custom server port
    python main.py --serve            # serve existing outputs only

Pipeline (10 stages):
    1.  Load & validate ECI data (2016/2021)
    2.  Collect sentiment    Meta · Twitter · News · Google Trends
    3.  Clean collected data  dedup · spam · near-duplicate
    4.  Fuse sentiment        weighted combination + trend/momentum
    5.  Feature engineering   23 pseph + sentiment + GTrends features
    6.  Train model           GBM + RF + LR ensemble, overfit guard
    7.  Backtest              2016 → 2021, AUC / Brier / seat accuracy
    8.  Predict               2026 constituency win probabilities
    9.  Monte Carlo           20,000 simulations, correlated noise
   10.  Validate & serve      11 checks, Flask server, browser
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Bootstrap: add repo root to path ─────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from src.utils.logger import setup as setup_logging
setup_logging("INFO", _ROOT / "outputs" / "pipeline.log")

log = logging.getLogger("main")

from src.config.settings import CFG
from src.pipeline        import Pipeline

DIV = "═" * 65


def _step(n: int | str, title: str) -> None:
    print(f"\n[{str(n):>2}] {title}")
    log.info(f"Step {n}: {title}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Assam Election ML — Production Forecasting Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--no-ui",          action="store_true",
                        help="Skip browser launch")
    parser.add_argument("--serve",          action="store_true",
                        help="Serve existing outputs without rerunning pipeline")
    parser.add_argument("--skip-sentiment", action="store_true",
                        help="Use cached sentiment file (faster iteration)")
    parser.add_argument("--quick",          action="store_true",
                        help="5k simulations, skip stress tests")
    parser.add_argument("--port",           type=int, default=CFG.SERVER_PORT,
                        help=f"HTTP port (default {CFG.SERVER_PORT})")
    args = parser.parse_args()

    t_start = time.time()
    print(DIV)
    print("  Assam 2026 Election Forecast  —  Production ML Pipeline")
    print(DIV)
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Mode   : {'SERVE ONLY' if args.serve else 'FULL PIPELINE'}")
    print(CFG.summary())

    pipeline = Pipeline(CFG)

    # ── Full pipeline ──────────────────────────────────────────────────────────
    if not args.serve:
        try:
            _step(1, "Load & validate ECI data")
            pipeline.load_data()
            print(f"  ✓ 2016: {pipeline.df16['constituency'].nunique()} seats  "
                  f"| 2021: {pipeline.df21['constituency'].nunique()} seats  "
                  f"| {len(pipeline.const_list)} constituencies")

            _step(2, "Collect sentiment  (Meta · Twitter · News · Google Trends)")
            pipeline.collect_sentiment(skip=args.skip_sentiment)

            _step(3, "Feature engineering  (23 features)")
            pipeline.build_features()
            new_cols = [c for c in pipeline.feats_2026.columns
                        if any(s in c for s in ["sentiment","trend","vol","mom",
                                                  "media","search","gtrend","quality"])]
            print(f"  ✓ Training rows: {len(pipeline.upset_rows)}")
            print(f"  ✓ V14 features : {new_cols}")

            _step(4, "Train model  (overfit guard: threshold=0.10)")
            pipeline.train()
            m = pipeline.train_metrics
            print(f"  LOO-CV AUC : {m['loo_auc']:.3f}")
            print(f"  Train AUC  : {m['train_auc']:.3f}")
            print(f"  Overfit gap: {m['overfit_gap']:.3f}  "
                  f"{'⚠ REGULARISED' if m['regularised'] else '✓ OK'}")
            print(f"  Upset prob : {m['mean_upset_prob_cal']:.3f}  "
                  f"Sentiment FI: {m.get('sentiment_fi_total',0):.4f}")

            _step(5, "Backtest  (2016 → 2021)")
            pipeline.validate()
            bt = pipeline.backtest
            print(f"  Seat accuracy: {bt['seat_acc']:.1%}  "
                  f"AUC={bt['auc']:.3f}  Brier={bt['brier']:.4f}")

            _step(6, "Generate predictions  (2026 constituencies)")
            pipeline.predict()
            winners = pipeline.winners
            print(f"\n  {'─'*55}")
            print(f"  PREDICTED TALLY — ASSAM 2026")
            print(f"  {'─'*55}")
            for p_, s in winners["party"].value_counts().items():
                print(f"    {p_:<10}: {s:>3}  {'█' * s}")
            print()
            for a_, s in winners["alliance"].value_counts().items():
                print(f"    {a_:<16}: {s:>3}")

            _step(7, f"Monte Carlo  ({CFG.N_SIMULATIONS if not args.quick else 5000:,} sims)")
            pipeline.simulate(n_sims=5_000 if args.quick else None)
            m2 = pipeline.summary.get("_meta", {})
            print(f"  NDA majority: {m2.get('nda_majority_prob','–')}")
            print(f"  NDA upset   : {m2.get('nda_upset_prob','–')}")
            print(f"  BJP mean    : {pipeline.summary.get('BJP',{}).get('mean','–')} seats")

            _step(8, "Validation checks")
            pipeline.check()

            if not args.quick:
                _step(9, "Stress tests  (7 scenarios)")
                pipeline.stress_test()

            _step(10, "Save pipeline metadata")
            pipeline.save_meta()

            elapsed = time.time() - t_start
            status  = "✅ ALL CHECKS PASSED" if pipeline.all_pass else "⚠ SOME CHECKS FAILED"
            print(f"\n{DIV}")
            print(f"  COMPLETE  |  {status}")
            print(f"  Pipeline time: {elapsed:.0f}s  ({elapsed/60:.1f} min)")
            print(DIV)

        except Exception as exc:
            log.error(f"Pipeline error: {exc}", exc_info=True)
            print(f"\n  ⚠ Pipeline failed: {exc}")
            print("  Attempting to serve any existing outputs …")

    # ── Deploy ────────────────────────────────────────────────────────────────
    if not pipeline.outputs_ready():
        print("  No outputs found. Run without --serve first.")
        print("  Quick run: python main.py --skip-sentiment --quick")
        sys.exit(1)

    if args.no_ui:
        print(f"\n  --no-ui: outputs at {CFG.OUT_DIR}")
        return

    # Start server + browser
    from server.launch import ServerManager
    mgr = ServerManager(port=args.port, no_ui=False,
                        outputs_dir=CFG.OUT_DIR, web_dir=CFG.WEB_DIR)
    mgr.start()
    mgr.open_browser(delay=0.5)

    print(); print(DIV)
    print("  🗳  Assam 2026 Election Forecast")
    print(DIV)
    print(f"  Dashboard    : http://localhost:{mgr.port}")
    print(f"  API summary  : http://localhost:{mgr.port}/api/summary")
    print(f"  Data quality : http://localhost:{mgr.port}/api/data-quality")
    print(f"  Trends       : http://localhost:{mgr.port}/api/trends")
    print()
    print(f"  Auto-refreshes every {CFG.AUTO_REFRESH_SECS}s  |  Ctrl-C to stop")
    print(DIV)
    mgr.wait()


if __name__ == "__main__":
    main()
