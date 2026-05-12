<<<<<<< HEAD
# assam-election-ml

> **Production-grade real-time election forecasting system for Assam Legislative Assembly 2026.**  
> 20,000 Monte Carlo simulations · 4-source live sentiment · 23 ML features · Auto-launching dashboard

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-lightgrey.svg)](https://flask.palletsprojects.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Dashboard

### Overview — Seat Projection & Majority Probability
![Overview tab showing NDA majority probability at 88.8%, seat projection bars for all parties, and probability rings](https://github.com/AyushJhaji/AssamElection/blob/c71118f6375451f7e72c87bf48d393359493fe5f/screenshots/s1.png)

### Simulation — 20,000 Monte Carlo Runs
![Simulation tab with BJP mean 53.9 seats and NDA mean 70.5 seats distribution histograms and all-party confidence ranges](https://github.com/AyushJhaji/AssamElection/blob/212e40e175c8f677f9cf6c479383a2af98cb11db/screenshots/s2.png))

### Constituencies — Full 127-Seat Searchable Table
![Constituencies tab with filterable table of all 127 seats sorted by win probability with party colours and gap bars](https://github.com/AyushJhaji/AssamElection/blob/c71118f6375451f7e72c87bf48d393359493fe5f/screenshots/s3.png)

### Battleground — Top 20 Toss-up Seats
![Battleground tab showing 20 closest seats including NAZIRA 0.8%, TEOK 1.6%, BARHAMPUR 1.9% with red BATTLEFIELD tags](https://github.com/AyushJhaji/AssamElection/blob/c71118f6375451f7e72c87bf48d393359493fe5f/screenshots/s4.png)

### Regions — 7-Region Breakdown
![Regions tab with NDA vs OPP breakdown across Upper Assam, Central, Lower, Barak Valley, Bodoland, Hills and AIUDF Belt](https://github.com/AyushJhaji/AssamElection/blob/c71118f6375451f7e72c87bf48d393359493fe5f/screenshots/s5.png)

### Sentiment — Live Multi-Source Fusion
![Sentiment tab showing BJP +0.433, INC +0.354 fused scores with Meta/Twitter/News source comparison and trend-volatility-momentum table](https://github.com/AyushJhaji/AssamElection/blob/c71118f6375451f7e72c87bf48d393359493fe5f/screenshots/s6.png)

### Trends — Google Trends Search Interest
![Trends tab showing BJP 83/100, INC 35/100 Google search interest bars with trend direction and momentum](https://github.com/AyushJhaji/AssamElection/blob/c71118f6375451f7e72c87bf48d393359493fe5f/screenshots/s7.png)

### Scenarios — Swing Simulator & Stress Tests
![Scenarios tab with interactive NDA swing slider and 7-scenario stress test grid showing seat outcomes](https://github.com/AyushJhaji/AssamElection/blob/c71118f6375451f7e72c87bf48d393359493fe5f/screenshots/s8.png)

### Model — Validation Checks & Version History
![Model tab showing all 11 validation checks passing, version comparison table V10–V13, and calibration curve](https://github.com/AyushJhaji/AssamElection/blob/c71118f6375451f7e72c87bf48d393359493fe5f/screenshots/s9.png)

---

## Quick start

```bash
# 1. Clone & install
git clone https://github.com/your-username/assam-election-ml.git
cd assam-election-ml
pip install -r requirements.txt

# 2. Configure API keys (optional — pipeline runs without them via fallbacks)
cp .env.example .env
# Edit .env with your NewsAPI key and Meta access token

# 3. Run
python main.py
```

Browser opens automatically at `http://localhost:8000`.

---

## CLI options

```bash
python main.py                    # full pipeline + auto-launch browser
python main.py --no-ui            # pipeline only, no browser
python main.py --skip-sentiment   # use cached sentiment (faster iteration)
python main.py --quick            # 5k simulations, skip stress tests (~3 min)
python main.py --port 9000        # custom server port
python main.py --serve            # serve existing outputs without rerunning
```

---

## Architecture

```
assam-election-ml/
│
├── main.py                    ← SINGLE ENTRY POINT
├── requirements.txt
├── .env.example
│
├── src/
│   ├── config/
│   │   └── settings.py        ← all config loaded from .env
│   ├── pipeline.py            ← Pipeline class (10 step methods)
│   ├── data/
│   │   ├── loader.py          ← ECI 2016/2021 loading + swing computation
│   │   ├── cleaner.py         ← dedup, spam, near-duplicate removal
│   │   └── validator.py       ← quality gates + audit reporting
│   ├── sentiment/
│   │   ├── collector.py       ← Meta + Twitter + News + GTrends (unified)
│   │   ├── fusion.py          ← 4-source weighted fusion + trend/momentum
│   │   └── nlp.py             ← VADER-inspired pure-Python NLP scorer
│   ├── features/
│   │   └── engineer.py        ← 23-feature matrix
│   ├── models/
│   │   ├── train.py           ← GBM+RF+LR ensemble + overfit guard
│   │   └── predict.py         ← 2026 constituency probabilities
│   ├── simulation/
│   │   └── monte_carlo.py     ← 20k correlated-noise MC simulations
│   ├── validation/
│   │   ├── backtest.py        ← 2016→2021 backtest + JSON fix
│   │   ├── checks.py          ← 11 validation gates
│   │   └── stress_tests.py    ← 7 scenario stress tests
│   └── utils/
│       ├── logger.py
│       └── helpers.py         ← shared math + JSON helpers
│
├── server/
│   ├── app.py                 ← Flask API (12 endpoints)
│   └── launch.py              ← port scan + daemon thread + browser open
├── web/
│   └── index.html             ← dynamic dashboard (9 tabs, auto-refresh 10s)
├── data/                      ← ECI source files (never modified)
└── docs/screenshots/          ← dashboard screenshots
```

---

## Data sources & fusion weights

| Source | Weight | Collection tiers |
|--------|:------:|-----------------|
| Meta (Facebook / Instagram / Threads) | **30%** | Graph API → page scrape → synthetic |
| Twitter / X | **25%** | snscrape → Nitter RSS → synthetic |
| NewsAPI / RSS feeds | **20%** | NewsAPI → Sentinel Assam / NDTV / ToI → synthetic |
| Google Trends | **25%** | pytrends → historical calibration |

Every row is labelled with its true source (`graph_api`, `live`, `newsapi`, `rss`, `pytrends`, `synthetic`, `historical`).  
Check `/api/data-quality` for a live breakdown of real vs synthetic percentages.

---

## ML features (23 total)

**Psephological (14)** — from ECI 2016/2021 data:
`nda_adv`, `nda_total_vs`, `opp_total_vs`, `effective_n_parties`, `top2_margin`,
`constituency_vol`, `vote_swing`, `vote_swing_sq`, `sent_adj`, `alliance_is_nda`,
`incumbent_win_rate`, `prev_winner_vs`, `log_total_votes`, `n_candidates`

**Sentiment (5)** — from live fusion pipeline:
`sentiment_score`, `trend_score`, `volatility`, `momentum`, `media_bias`

**Google Trends (3)** — search behaviour signals:
`search_momentum`, `search_interest`, `gtrends_signal`

**Data quality (1)**:
`data_quality_score` — fraction of real (non-synthetic) data for each party

---

## Model

| Property | Value |
|----------|-------|
| Architecture | GBM + Random Forest + Logistic Regression (50% / 30% / 20%) |
| Calibration | Temperature scaling T=2.0, clipped to [0.20, 0.50] |
| Cross-validation | Leave-One-Out (n=126 constituencies) |
| Overfit guard | If `train_auc − loo_auc > 0.10` → stricter params + feature selection |
| LOO-CV AUC | ~0.85 |
| Backtest seat accuracy | ~60% (2016→2021) |

---

## Validation (11 checks — all must pass)

| Check | Threshold | Result |
|-------|-----------|--------|
| Close seats | 15–40 | 30 ✅ |
| Min winner probability | ≥ 40% | 44.7% ✅ |
| Max winner probability | ≤ 93% | 87.2% ✅ |
| Upset probability mean | 0.30–0.50 | 0.441 ✅ |
| BJP simulation std dev | ≥ 4.0 | 5.4 ✅ |
| NDA majority probability | 55%–95% | 87.6% ✅ |
| NDA upset probability | ≥ 5% | 12.4% ✅ |
| Win prob spread | ≥ 0.08 | 0.105 ✅ |
| NDA seat range | > 15 seats | 51 seats ✅ |
| No party > 95% solo majority | checked | ✅ |
| No deterministic outcomes | checked | ✅ |

---

## API (12 endpoints)

| Endpoint | Description |
|----------|-------------|
| `GET /` | Live 9-tab dashboard |
| `GET /api/health` | `{"status":"ok","model":"assam-election-ml"}` |
| `GET /api/status` | Pipeline ready state + file timestamps |
| `GET /api/summary` | MC summary — NDA majority %, seat ranges |
| `GET /api/winners` | All 127 constituency predictions |
| `GET /api/seat-distribution` | Party seat histograms |
| `GET /api/calibration` | Calibration curve (2016→2021) |
| `GET /api/sentiment` | Fused sentiment signals per party |
| `GET /api/trends` | Google Trends search interest data |
| `GET /api/data-quality` | Real vs synthetic audit per source |
| `GET /api/stress-tests` | 7-scenario stress test results |
| `GET /api/validation` | Full 11-check validation report |

---

## Important: probabilistic model

Results are **probability distributions, not point forecasts**.  
"NDA majority probability 88.8%" means 88.8% of 20,000 simulated elections gave NDA ≥ 64 seats — it is not a guarantee.  
The model quantifies known uncertainty: swing volatility, regional noise, sentiment shifts, anti-incumbency waves.
=======
# assam-election-forecast-ml
Production-grade machine learning pipeline for real-time Assam election forecasting using ensemble models, sentiment analysis, Google Trends, and Monte Carlo simulation.
>>>>>>> 27a66d87d6ed8c39a5271684d6757f13bce480bf
