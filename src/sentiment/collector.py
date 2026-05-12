"""
src/sentiment/collector.py
==========================
Unified entry point for all 4 sentiment/trend data sources.
Each source tries real data first, falls back gracefully, labels source honestly.

collect_all() → CollectedData(meta, twitter, news, gtrends)
"""

import logging
import random
import re
import time
import json
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

PARTIES = ["BJP", "INC", "AIUDF", "AGP", "UPPL"]

# ── Corpora (realistic Assam 2026 political content) ─────────────────────────
_CORPUS: Dict[str, Dict[str, List]] = {
    "meta": {
        "BJP": [
            "BJP development record praised across Assam — roads, bridges, schools improved",
            "Himanta Biswa Sarma government infrastructure work visible in all districts",
            "NDA confident of strong performance in 2026 Assam assembly polls",
            "BJP 2026 campaign gaining momentum in Upper Assam constituencies",
            "Double engine government delivering for Assam — NDA alliance strong",
        ],
        "INC": [
            "Congress Assam promises employment and welfare for youth in 2026 manifesto",
            "INC Bharat Jodo gathering support in Brahmaputra valley districts",
            "Rahul Gandhi Assam visit energises Congress workers for 2026 polls",
            "Congress targeting 35+ seats in upcoming Assam assembly elections",
            "INC Assam working on grassroots level across all 127 constituencies",
        ],
        "AIUDF": [
            "AIUDF consolidating community support in Lower Assam and Barak Valley",
            "Badruddin Ajmal confident AIUDF will win 20+ seats in Assam 2026",
            "All India United Democratic Front active campaign ahead of 2026 elections",
            "AIUDF community welfare schemes drawing voter support across constituencies",
        ],
        "AGP": [
            "Asom Gana Parishad protecting Assamese cultural identity ahead of 2026",
            "AGP NDA alliance confident in traditional strongholds Upper Assam",
            "Asom Gana Parishad candidates active across Central Assam constituencies",
        ],
        "UPPL": [
            "UPPL Bodoland development record praised ahead of 2026 elections",
            "United People's Party Liberal expected to dominate BTR area seats",
            "UPPL Pramod Boro governance agenda bringing peace and progress Bodoland",
        ],
    },
    "twitter": {
        "BJP": [
            "BJP winning Assam 2026 — strong governance and development momentum!",
            "NDA Assam rally massive crowd shows strong BJP support base",
            "Himanta government delivering on promises — BJP confident for 2026",
            "BJP development agenda Assam 2026 winning voter hearts across regions",
        ],
        "INC": [
            "Congress Assam fighting for youth jobs and employment in 2026",
            "INC Bharat Jodo Assam gathering strong support at grassroots",
            "Congress promises welfare governance if voted to power in Assam 2026",
            "INC Assam targeting strong seat count with ground-level work",
        ],
        "AIUDF": [
            "AIUDF targeting 20+ seats in Lower Assam and Barak Valley 2026",
            "Badruddin Ajmal confident AIUDF strong ground support persists",
            "AIUDF community mobilisation solid ahead of Assam elections",
        ],
        "AGP": [
            "AGP NDA ally confident winning Assam seats 2026 polls",
            "Asom Gana Parishad cultural identity message resonating voters",
        ],
        "UPPL": [
            "UPPL Bodoland sweep expected 2026 Assam elections",
            "Pramod Boro UPPL development agenda praised BTR voters",
        ],
    },
    "news": {
        "BJP": [
            ("BJP projects comfortable majority in Assam 2026",
             "Internal surveys project BJP winning 60+ seats. Himanta Biswa Sarma expressed confidence."),
            ("NDA alliance Assam 2026 seat sharing finalised",
             "BJP AGP UPPL seat sharing agreement reached for upcoming polls."),
            ("BJP faces questions on price rise Assam",
             "Opposition challenges BJP record on inflation and employment."),
        ],
        "INC": [
            ("Congress Assam 2026 manifesto promises 10 lakh jobs",
             "INC releases manifesto with employment, farm loan waiver promises."),
            ("Rahul Gandhi Guwahati rally targets BJP policies",
             "Congress leader addresses large rally on price rise and jobs."),
        ],
        "AIUDF": [
            ("AIUDF targets 20 seats Assam 2026 elections",
             "Badruddin Ajmal says party will contest 25 and win 20 seats."),
            ("AIUDF consolidates minority vote Lower Assam",
             "Party focusing campaign on Lower Assam and Barak Valley."),
        ],
        "AGP": [
            ("AGP confident 10-12 seats NDA ally Assam",
             "Party contesting over 20 seats expecting solid performance."),
        ],
        "UPPL": [
            ("UPPL expected dominate Bodoland seats Assam 2026",
             "UPPL projects winning all BTR area constituencies."),
        ],
    },
}

_GTRENDS_HIST: Dict[str, Dict] = {
    "BJP":   {"mean": 72, "std": 18, "trend": +0.15},
    "INC":   {"mean": 48, "std": 15, "trend": +0.08},
    "AIUDF": {"mean": 35, "std": 14, "trend": +0.05},
    "AGP":   {"mean": 28, "std": 11, "trend": +0.03},
    "UPPL":  {"mean": 22, "std":  9, "trend": +0.02},
}

_NITTER = ["https://nitter.poast.org", "https://nitter.privacydev.net"]
_RSS_FEEDS = [
    "https://www.sentinelassam.com/rss.xml",
    "https://nenow.in/feed",
    "https://www.ndtv.com/rss/2012",
]
_GRAPH = "https://graph.facebook.com/v18.0"
_NEWS_BASE = "https://newsapi.org/v2"
_PARTY_KW = {
    "BJP":   ["bjp", "himanta", "nda assam"],
    "INC":   ["congress assam", "inc assam", "rahul assam"],
    "AIUDF": ["aiudf", "badruddin"],
    "AGP":   ["agp", "asom gana"],
    "UPPL":  ["uppl", "bodoland", "pramod boro"],
}


# ── Output container ──────────────────────────────────────────────────────────
@dataclass
class CollectedData:
    meta:    pd.DataFrame = field(default_factory=pd.DataFrame)
    twitter: pd.DataFrame = field(default_factory=pd.DataFrame)
    news:    pd.DataFrame = field(default_factory=pd.DataFrame)
    gtrends: pd.DataFrame = field(default_factory=pd.DataFrame)

    def audit(self) -> Dict[str, Dict]:
        """Real vs synthetic breakdown per source."""
        result = {}
        for name, df in [("meta", self.meta), ("twitter", self.twitter),
                          ("news", self.news), ("gtrends", self.gtrends)]:
            if df.empty:
                result[name] = {"total": 0, "real": 0, "synthetic": 0, "pct_real": 0.0}
                continue
            src_col = "source" if "source" in df.columns else None
            counts  = df[src_col].value_counts().to_dict() if src_col else {}
            real_n  = sum(v for k, v in counts.items()
                          if k not in ("synthetic", "historical"))
            result[name] = {"total": len(df), "real": int(real_n),
                            "synthetic": len(df) - int(real_n),
                            "pct_real": round(real_n / len(df) * 100, 1),
                            "sources": counts}
        return result


# ── Internal helpers ──────────────────────────────────────────────────────────

def _score(text: str) -> float:
    from src.sentiment.nlp import score_text
    return score_text(text)

def _score_hl(title: str, desc: str = "") -> float:
    from src.sentiment.nlp import score_headline
    return score_headline(title, desc)


def _synth_meta(party: str, n: int, seed: int) -> List[Dict]:
    rng_py = random.Random(seed); rng_np = np.random.default_rng(seed)
    templates = _CORPUS["meta"].get(party, [f"{party} Assam 2026"])
    eng_prof  = {"BJP":(1200,700,0.14),"INC":(850,550,0.11),"AIUDF":(700,450,0.16),
                 "AGP":(400,280,0.08),"UPPL":(350,240,0.09)}.get(party,(500,300,0.10))
    base_dt   = datetime.now() - timedelta(days=45)
    rows = []
    for _ in range(n):
        txt   = rng_py.choice(templates)
        likes = max(0, int(rng_np.normal(eng_prof[0], eng_prof[1])))
        shar  = int(likes * eng_prof[2] * rng_np.uniform(0.5, 1.5))
        raw   = _score(txt)
        rows.append({
            "post_text": txt, "platform": "facebook",
            "likes": likes, "shares": shar,
            "engagement_score": round(min(1.0, (likes + shar*5)/15000), 4),
            "raw_score": round(raw, 4),
            "sentiment_score": round(float(np.clip(raw + rng_np.normal(0, 0.12), -1, 1)), 4),
            "party_tag": party,
            "date": (base_dt + timedelta(days=rng_py.randint(0,44))).strftime("%Y-%m-%d"),
            "source": "synthetic",
        })
    return rows


def _synth_twitter(party: str, n: int, seed: int) -> List[Dict]:
    rng_py = random.Random(seed); rng_np = np.random.default_rng(seed)
    templates = _CORPUS["twitter"].get(party, [f"{party} Assam 2026"])
    base_dt   = datetime.now() - timedelta(days=30)
    rows = []
    for _ in range(n):
        txt = rng_py.choice(templates)
        raw = _score(txt)
        rows.append({
            "tweet_text": txt,
            "date": (base_dt + timedelta(days=rng_py.randint(0,29))).strftime("%Y-%m-%d"),
            "party_tag": party,
            "raw_score": round(raw, 4),
            "sentiment_score": round(float(np.clip(raw + rng_np.normal(0, 0.10), -1, 1)), 4),
            "source": "synthetic",
        })
    return rows


def _synth_news(party: str, n: int, seed: int) -> List[Dict]:
    rng = random.Random(seed)
    templates = _CORPUS["news"].get(party, [(f"{party} Assam 2026", "Party campaigns.")])
    base_dt   = datetime.now() - timedelta(days=60)
    rows = []
    for i in range(n):
        title, desc = templates[i % len(templates)]
        rows.append({
            "title": title, "description": desc, "url": "",
            "source": "synthetic", "party_tag": party,
            "sentiment_score": round(float(np.clip(_score_hl(title, desc) + rng.gauss(0,0.07), -1, 1)), 4),
            "date": (base_dt + timedelta(days=rng.randint(0,59))).strftime("%Y-%m-%d"),
        })
    return rows


# ── Real collectors ───────────────────────────────────────────────────────────

def _meta_api(query: str, token: str, n: int = 10) -> List[Dict]:
    params = urllib.parse.urlencode({
        "q": query, "type": "post",
        "fields": "message,created_time,likes.summary(true),shares",
        "limit": n, "access_token": token,
    })
    try:
        req = urllib.request.Request(f"{_GRAPH}/search?{params}",
                                      headers={"User-Agent": "election-research/1.0"})
        with urllib.request.urlopen(req, timeout=8) as r:
            data = json.loads(r.read().decode())
        rows = []
        for item in data.get("data", []):
            msg = item.get("message", "")
            if len(msg) < 15: continue
            likes = item.get("likes",{}).get("summary",{}).get("total_count",0)
            shar  = item.get("shares",{}).get("count",0) if "shares" in item else 0
            raw   = _score(msg)
            rows.append({"post_text": msg[:400], "platform": "facebook",
                         "likes": likes, "shares": shar,
                         "engagement_score": round(min(1.0,(likes+shar*5)/15000),4),
                         "raw_score": round(raw,4), "sentiment_score": round(raw,4),
                         "party_tag": "", "date": (item.get("created_time","") or "")[:10],
                         "source": "graph_api"})
        return rows
    except Exception as exc:
        log.debug(f"  Meta API '{query}': {type(exc).__name__}")
        return []


def _nitter_rss(query: str, n: int = 15) -> List[str]:
    enc = urllib.parse.quote(query)
    for inst in _NITTER:
        try:
            req = urllib.request.Request(f"{inst}/search/rss?q={enc}",
                                          headers={"User-Agent": "election-research/1.0"})
            with urllib.request.urlopen(req, timeout=5) as r:
                root = ET.fromstring(r.read())
            texts = []
            for item in root.iter("item"):
                title = item.findtext("title", "").strip()
                desc  = re.sub(r"<[^>]+>", "",
                               item.findtext("description", "")).strip()
                combined = (title + " " + desc).strip()
                if len(combined) > 20:
                    texts.append(combined[:280])
                if len(texts) >= n: break
            if texts: return texts
        except: continue
    return []


def _rss_news(max_articles: int = 40) -> List[Dict]:
    arts = []
    for feed in _RSS_FEEDS:
        try:
            req = urllib.request.Request(feed, headers={"User-Agent": "election-research/1.0"})
            with urllib.request.urlopen(req, timeout=5) as r:
                root = ET.fromstring(r.read())
            for item in root.iter("item"):
                t = item.findtext("title", "").strip()
                d = re.sub(r"<[^>]+>", "", item.findtext("description", "")).strip()
                if not t: continue
                if "assam" in (t+d).lower() or "election" in (t+d).lower():
                    arts.append({"title": t, "description": d[:300],
                                 "url": item.findtext("link",""),
                                 "source": feed.split("/")[2],
                                 "date": datetime.now().strftime("%Y-%m-%d")})
            if len(arts) >= max_articles: break
        except: continue
    return arts[:max_articles]


def _newsapi_fetch(query: str, key: str, n: int = 12) -> List[Dict]:
    params = urllib.parse.urlencode({
        "q": query, "language": "en", "sortBy": "publishedAt",
        "pageSize": n, "apiKey": key,
    })
    try:
        req = urllib.request.Request(f"{_NEWS_BASE}/everything?{params}",
                                      headers={"User-Agent": "election-research/1.0"})
        with urllib.request.urlopen(req, timeout=8) as r:
            data = json.loads(r.read().decode())
        arts = []
        for a in data.get("articles", [])[:n]:
            t = a.get("title", "") or ""; d = a.get("description", "") or ""
            if len(t) < 10: continue
            arts.append({"title": t[:200], "description": d[:300],
                         "url": a.get("url", ""),
                         "source": a.get("source", {}).get("name", "newsapi"),
                         "date": (a.get("publishedAt", "") or "")[:10]})
        return arts
    except Exception as exc:
        log.debug(f"  NewsAPI '{query}': {type(exc).__name__}")
        return []


def _gtrends_pytrends(party: str, geo: str, timeframe: str) -> Optional[Dict]:
    try:
        from pytrends.request import TrendReq
        queries = {"BJP":["BJP Assam","Himanta Biswa Sarma"],
                   "INC":["Congress Assam","INC Assam election"],
                   "AIUDF":["AIUDF Assam","Badruddin Ajmal"],
                   "AGP":["AGP Assam"],"UPPL":["UPPL Bodoland"]}.get(party, [party+" Assam"])[:2]
        pt = TrendReq(hl="en-US", tz=330, timeout=(5,8), retries=1, backoff_factor=0.5)
        pt.build_payload(queries, cat=396, timeframe=timeframe, geo=geo)
        df = pt.interest_over_time()
        if df.empty: return None
        scores = df[queries].mean(axis=1)
        current = float(scores.iloc[-1])
        recent  = float(scores.tail(4).mean())
        older   = float(scores.tail(12).head(8).mean())
        mom     = (recent - older) / max(older, 1.0)
        return {
            "interest": round(current, 1),
            "normalised_interest": round(current / 100.0, 4),
            "trend_direction": "rising" if mom > 0.05 else "falling" if mom < -0.05 else "stable",
            "momentum": round(float(np.clip(mom, -1.0, 1.0)), 4),
            "source": "pytrends",
        }
    except ImportError:
        return None
    except Exception as exc:
        log.debug(f"  pytrends {party}: {type(exc).__name__}")
        return None


def _gtrends_hist(party: str, seed: int) -> Dict:
    rng  = np.random.default_rng(seed + abs(hash(party)) % 1000)
    hist = _GTRENDS_HIST.get(party, {"mean": 30, "std": 10, "trend": 0.0})
    val  = float(np.clip(rng.normal(hist["mean"], hist["std"]), 5, 100))
    mom  = float(np.clip(rng.normal(hist["trend"], 0.04), -0.5, 0.5))
    return {
        "interest": round(val, 1),
        "normalised_interest": round(val / 100.0, 4),
        "trend_direction": "rising" if mom > 0.05 else "falling" if mom < -0.05 else "stable",
        "momentum": round(mom, 4),
        "source": "historical",
    }


# ── Main entry point ──────────────────────────────────────────────────────────

def collect_all(n_meta: int = 20, n_twitter: int = 25, n_news: int = 15,
                seed: int = 42) -> CollectedData:
    """
    Collect from all 4 sources. Each source: real → fallback → synthetic.
    All rows labelled with true source. Returns CollectedData.
    """
    from src.config.settings import CFG

    # ── Meta ──────────────────────────────────────────────────────────────────
    meta_rows: List[Dict] = []
    for party in PARTIES:
        real = []
        if CFG.has_meta_api():
            for q in [f"{party} Assam", f"{party} Assam 2026"][:2]:
                for row in _meta_api(q, CFG.META_ACCESS_TOKEN, n=8):
                    row["party_tag"] = party
                    real.append(row)
                time.sleep(0.3)
        n_real = len(real)
        synth  = _synth_meta(party, max(0, n_meta - n_real), seed + abs(hash(party)) % 500)
        meta_rows.extend(real + synth)
    meta_df = pd.DataFrame(meta_rows)
    _log_src("Meta", meta_df)

    # ── Twitter ───────────────────────────────────────────────────────────────
    twitter_rows: List[Dict] = []
    for party in PARTIES:
        real_texts: List[str] = []
        for q in [f"{party} Assam 2026", f"#{party}Assam"][:2]:
            try:
                import snscrape.modules.twitter as sn
                for i, t in enumerate(sn.TwitterSearchScraper(
                        f"{q} lang:en since:2026-01-01").get_items()):
                    if i >= n_twitter: break
                    txt = t.rawContent or t.content
                    if txt: real_texts.append(txt)
            except: pass
            if len(real_texts) < n_twitter // 2:
                real_texts.extend(_nitter_rss(q, n=12))
            if len(real_texts) >= n_twitter: break
        real_rows = [{"tweet_text": re.sub(r"\s+"," ",t).strip()[:280],
                      "date": datetime.now().strftime("%Y-%m-%d"),
                      "party_tag": party,
                      "raw_score": round(_score(t), 4),
                      "sentiment_score": round(_score(t), 4),
                      "source": "live"} for t in real_texts[:n_twitter]]
        needed = max(0, n_twitter - len(real_rows))
        synth  = _synth_twitter(party, needed, seed + abs(hash(party)) % 700)
        twitter_rows.extend(real_rows + synth)
    twitter_df = pd.DataFrame(twitter_rows)
    _log_src("Twitter", twitter_df)

    # ── News ──────────────────────────────────────────────────────────────────
    raw_arts: List[Dict] = []
    if CFG.has_news_api():
        for q in ["Assam election 2026", "BJP INC Assam poll"]:
            raw_arts.extend(_newsapi_fetch(q, CFG.NEWS_API_KEY, n=12))
            if len(raw_arts) >= 50: break
            time.sleep(0.3)
    if len(raw_arts) < 20:
        raw_arts.extend(_rss_news(max_articles=40))

    tagged: Dict[str, List] = {p: [] for p in PARTIES}
    for art in raw_arts:
        combined = (art["title"] + " " + art.get("description", "")).lower()
        for p, kws in _PARTY_KW.items():
            if any(kw in combined for kw in kws):
                sc = _score_hl(art["title"], art.get("description",""))
                tagged[p].append({**art, "party_tag": p, "sentiment_score": round(sc,4)})
                break
    news_rows = []
    for party in PARTIES:
        real = tagged[party]
        synth = _synth_news(party, max(0, n_news - len(real)), seed + abs(hash(party)) % 300)
        news_rows.extend(real + synth)
    news_df = pd.DataFrame(news_rows)
    _log_src("News", news_df)

    # ── Google Trends ─────────────────────────────────────────────────────────
    gt_rows = []
    for party in PARTIES:
        result = _gtrends_pytrends(party, CFG.GTRENDS_GEO, CFG.GTRENDS_TIMEFRAME)
        if result:
            log.info(f"  GTrends pytrends: {party} interest={result['interest']:.0f}")
            time.sleep(2.5)
        else:
            result = _gtrends_hist(party, seed)
            log.warning(f"  GTrends FALLBACK (historical) {party} "
                        f"→ interest={result['interest']:.0f} "
                        f"[install pytrends for real data]")
        gt_rows.append({"party": party,
                        "interest":           result["interest"],
                        "normalised_interest": result["normalised_interest"],
                        "trend_direction":    result["trend_direction"],
                        "momentum":           result["momentum"],
                        "source":             result["source"],
                        "collected_at":       datetime.now().strftime("%Y-%m-%d %H:%M")})
    gtrends_df = pd.DataFrame(gt_rows)
    real_gt = (gtrends_df["source"] != "historical").sum()
    log.info(f"  GTrends: {len(gtrends_df)} parties | real={real_gt}")

    return CollectedData(meta=meta_df, twitter=twitter_df,
                         news=news_df, gtrends=gtrends_df)


def _log_src(name: str, df: pd.DataFrame) -> None:
    if df.empty:
        log.info(f"  {name}: empty")
        return
    counts  = df.get("source", pd.Series()).value_counts().to_dict()
    real_n  = sum(v for k, v in counts.items() if k not in ("synthetic","historical"))
    pct     = real_n / len(df) * 100
    log.info(f"  {name}: {len(df)} rows | real={real_n} ({pct:.0f}%) | {counts}")
