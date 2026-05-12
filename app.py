"""
server/app.py — Flask API
=========================
GET /                     → web/index.html
GET /api/health           → {"status":"ok","model":"assam-election-ml"}
GET /api/status           → pipeline ready + timestamps
GET /api/summary          → MC summary JSON
GET /api/winners          → 127 constituency results
GET /api/seat-distribution → histogram data
GET /api/calibration      → calibration curve
GET /api/sentiment        → fused sentiment
GET /api/trends           → Google Trends data
GET /api/data-quality     → real vs synthetic audit
GET /api/stress-tests     → stress scenarios
GET /api/validation       → validation report
GET /outputs/<file>       → raw file passthrough
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, abort, jsonify, request, send_file

log = logging.getLogger(__name__)
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent

_REGION = {
    **{c: "Upper Assam"  for c in ["DIBRUGARH","TINSUKIA","DIGBOI","MARGHERITA","CHABUA","DULIAJAN","MORAN","DOOM DOOMA","NAHARKATIA","SONARI","LAHOWAL","MAHMARA","SADIYA","DHAKUAKHANA","JONAI","DHEMAJI","MAJULI","JORHAT","TITABAR","MARIANI","SARUPATHAR","HOWRAGHAT","GOLAGHAT","DERGAON","BOKAKHAT","KALIABOR","TEOK","NAZIRA","THOWRA"]},
    **{c: "Central Assam" for c in ["NOWGONG","RAHA","SAMAGURI","DHING","BATADROBA","HOJAI","LUMDING","JAGIROAD","MARIGAON","LAHARIGHAT","MANGALDOI","SIPAJHAR","BIHPURIA","DHEKIAJULI","GOHPUR","SOOTEA","BISWANATH","BEHALI","BARCHALLA","RANGAPARA","TEZPUR","BARHAMPUR","HAJO","LAKHIMPUR","NORTH LAKHIMPUR","MORIGAON","SIB SAGAR","RUPOHIHAT","SORBHOG","DHARMAMPUR","PANEERY","NAUBOICHA"]},
    **{c: "Lower Assam"  for c in ["NALBARI","BARKHETRI","PATACHARKUCHI","KAMALPUR","RANGIYA","JALUKBARI","GAUHATI EAST","GAUHATI WEST","DISPUR","BOKO","CHAYGAON","PALASBARI","BONGAIGAON","BIJNI","ABHAYAPURI NORTH","ABHAYAPURI SOUTH","BARPETA","BAGHBAR","SARUKHETRI","CHENGA","BHABANIPUR","CHAPAGURI","GOLAKGANJ","GAURIPUR","GOALPARA EAST","GOALPARA WEST","DALGAON","DUDHNOI"]},
    **{c: "Barak Valley" for c in ["SILCHAR","SONAI","DHOLAI","BARKHOLA","KATIGORAH","LAKHIPUR","KATLICHERRA","HAILAKANDI","ALGAPUR","BADARPUR","PATHERKANDI","RATABARI","KARIMGANJ NORTH","KARIMGANJ SOUTH","JALESWAR","SOUTH SALMARA"]},
    **{c: "Bodoland"     for c in ["KOKRAJHAR EAST","KOKRAJHAR WEST","GOSSAIGAON","SIDLI","BARAMA","TAMULPUR","UDALGURI","MAJBAT","KALAIGAON"]},
    **{c: "Hills"        for c in ["HAFLONG","DIPHU","BOKAJAN","BAITHALANGSO"]},
    **{c: "AIUDF Belt"   for c in ["DHUBRI","BILASIPARA EAST","BILASIPARA WEST","MANKACHAR","JANIA","JAMUNAMUKH","BORKHETRY","AMGOURI"]},
}

# Files to try in preference order (canonical name first, then legacy V14/V13/V12)
_CANDIDATES = {
    "summary":      ["summary.json",     "summary_v14.json",     "summary_v13.json",     "summary_v12.json"],
    "winners":      ["winners.csv",      "winners_v14.csv",      "winners_v13.csv",      "winners_v12.csv"],
    "predictions":  ["predictions.csv",  "predictions_v14.csv",  "predictions_v13.csv",  "predictions_v12.csv"],
    "seat_dist":    ["seat_distribution.csv","seat_distribution_v14.csv","seat_distribution_v13.csv","seat_distribution_v12.csv"],
    "calibration":  ["calibration.csv",  "calibration_v14.csv",  "calibration_v13.csv",  "calibration_v12.csv"],
    "sentiment":    ["sentiment.csv",    "sentiment_v14.csv",    "sentiment_v13.csv",    "fused_sentiment_v7.csv"],
    "gtrends":      ["raw_gtrends.csv",  "raw_gtrends_v14.csv"],
    "stress":       ["stress_tests.json","stress_tests_v14.json","stress_tests_v13.json"],
    "validation":   ["validation_report.json","validation_report_v14.json","validation_report_v13.json"],
}


def create_app(outputs_dir: Path = None, web_dir: Path = None) -> Flask:
    out = Path(outputs_dir) if outputs_dir else _ROOT / "outputs"
    web = Path(web_dir)     if web_dir     else _ROOT / "web"
    app = Flask(__name__, static_folder=None)

    @app.after_request
    def _cors(r):
        o = request.headers.get("Origin", "")
        if "localhost" in o or "127.0.0.1" in o:
            r.headers["Access-Control-Allow-Origin"] = o
        return r

    def _nc(r):
        r.headers.update({"Cache-Control": "no-cache,no-store,must-revalidate",
                           "Pragma": "no-cache", "Expires": "0"})
        return r

    def _mt(p: Path) -> str:
        try: return datetime.fromtimestamp(os.path.getmtime(p)).strftime("%Y-%m-%d %H:%M:%S")
        except: return "unknown"

    def _best(key: str) -> Path | None:
        for fname in _CANDIDATES.get(key, []):
            p = out / fname
            if p.exists(): return p
        return None

    @app.route("/")
    def index():
        f = web / "index.html"
        if not f.exists(): abort(404)
        return send_file(str(f))

    @app.route("/outputs/<path:filename>")
    def serve_output(filename):
        f = out / filename
        if not f.exists(): abort(404)
        return _nc(send_file(str(f)))

    @app.route("/api/health")
    def health():
        return _nc(jsonify({"status": "ok", "model": "assam-election-ml"}))

    @app.route("/api/status")
    def status():
        files, latest = {}, 0.0
        for key in ["summary", "winners", "predictions", "seat_dist"]:
            p = _best(key)
            if p:
                mt = os.path.getmtime(p)
                files[key] = {"exists": True, "file": p.name,
                               "last_modified": _mt(p),
                               "size_kb": round(p.stat().st_size / 1024, 1)}
                latest = max(latest, mt)
            else:
                files[key] = {"exists": False}
        return _nc(jsonify({
            "ready": all(v["exists"] for v in files.values()),
            "files": files,
            "last_run": datetime.fromtimestamp(latest).strftime("%Y-%m-%d %H:%M:%S") if latest else None,
            "server_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }))

    @app.route("/api/summary")
    def api_summary():
        p = _best("summary")
        if not p: return _nc(jsonify({"error": "Run main.py first"})), 404
        with open(p, encoding="utf-8") as fh: data = json.load(fh)
        data["_server"] = {"file": p.name, "served_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        return _nc(jsonify(data))

    @app.route("/api/winners")
    def api_winners():
        p = _best("winners")
        if not p: return _nc(jsonify({"error": "No winners file"})), 404
        df = pd.read_csv(p)
        df["region"] = df["constituency"].map(_REGION).fillna("Central Assam")
        for col in ["win_probability", "confidence_score"]:
            if col in df.columns: df[col] = df[col].round(4)
        return _nc(jsonify({"data": json.loads(df.to_json(orient="records")),
                             "total": len(df), "source": p.name}))

    @app.route("/api/seat-distribution")
    def api_seat_dist():
        p = _best("seat_dist")
        if not p: return _nc(jsonify({"error": "No seat distribution"})), 404
        df = pd.read_csv(p)
        result = {}
        for party in ["BJP","INC","AGP","AIUDF","UPPL","NDA","OPPOSITION"]:
            if party not in df.columns: continue
            col = df[party]
            p5,p25,p50,p75,p95 = [int(v) for v in np.percentile(col,[5,25,50,75,95])]
            lo, hi = max(0,int(col.min())-2), min(130,int(col.max())+7)
            counts, edges = np.histogram(col.values, bins=list(range(lo,hi+1,5)))
            result[party] = {
                "mean": round(float(col.mean()),1), "std": round(float(col.std()),1),
                "median": p50, "p5": p5, "p25": p25, "p75": p75, "p95": p95,
                "range_90": f"{p5}–{p95}", "majority_prob": round(float((col>=64).mean())*100,1),
                "histogram": [{"lo":int(edges[i]),"hi":int(edges[i+1])-1,
                               "count":int(counts[i]),"pct":round(float(counts[i])/len(col)*100,2)}
                               for i in range(len(counts)) if counts[i]>0],
            }
        return _nc(jsonify({"parties": result, "n_simulations": len(df), "source": p.name}))

    @app.route("/api/calibration")
    def api_calibration():
        p = _best("calibration")
        if not p: return _nc(jsonify([])), 200
        return _nc(jsonify(json.loads(pd.read_csv(p).to_json(orient="records"))))

    @app.route("/api/sentiment")
    def api_sentiment():
        p = _best("sentiment")
        if not p:
            # fallback: data dir
            dp = _ROOT / "data" / "fused_sentiment.csv"
            p  = dp if dp.exists() else None
        if not p: return _nc(jsonify([])), 200
        return _nc(jsonify(json.loads(pd.read_csv(p).to_json(orient="records"))))

    @app.route("/api/trends")
    def api_trends():
        p = _best("gtrends")
        if not p: return _nc(jsonify({"error": "Run pipeline first"})), 404
        df = pd.read_csv(p)
        return _nc(jsonify({"data": json.loads(df.to_json(orient="records")),
                             "source": p.name}))

    @app.route("/api/data-quality")
    def api_data_quality():
        q = {}
        for key, fname in [("meta","raw_meta.csv"),("twitter","raw_twitter.csv"),
                            ("news","raw_news.csv"),("gtrends","raw_gtrends.csv")]:
            p = out/fname
            if not p.exists():
                p = out/(fname.replace(".csv","_v14.csv"))
            if not p.exists():
                q[key] = {"available": False}; continue
            df     = pd.read_csv(p)
            counts = df.get("source",pd.Series()).value_counts().to_dict()
            real_n = sum(v for k,v in counts.items() if k not in ("synthetic","historical"))
            q[key] = {"available": True, "total": len(df), "real": int(real_n),
                      "synthetic": len(df)-int(real_n),
                      "pct_real": round(real_n/len(df)*100,1) if len(df) else 0,
                      "sources": counts}
        return _nc(jsonify({"quality": q,
                             "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}))

    @app.route("/api/stress-tests")
    def api_stress():
        p = _best("stress")
        if not p: return _nc(jsonify({"error": "Not yet generated"})), 404
        with open(p) as fh: return _nc(jsonify(json.load(fh)))

    @app.route("/api/validation")
    def api_validation():
        p = _best("validation")
        if not p: return _nc(jsonify({"error": "Not yet generated"})), 404
        with open(p) as fh: return _nc(jsonify(json.load(fh)))

    return app
