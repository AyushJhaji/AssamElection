"""
src/config/settings.py
======================
Central configuration. All values load from .env.
Never hardcode secrets. Import via: from src.config.settings import CFG
"""

import os
import logging
from pathlib import Path

log = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent.parent
_ENV  = _ROOT / ".env"

# ── Load .env ─────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    if _ENV.exists():
        load_dotenv(_ENV)
except ImportError:
    if _ENV.exists():
        with open(_ENV) as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    os.environ.setdefault(k.strip(), v.strip())


class Settings:
    """Immutable settings object. Access via CFG.FIELD."""

    # ── API credentials ───────────────────────────────────────────────────────
    NEWS_API_KEY:      str = os.getenv("NEWS_API_KEY", "")
    META_ACCESS_TOKEN: str = os.getenv("META_ACCESS_TOKEN", "")
    META_APP_ID:       str = os.getenv("META_APP_ID", "")
    META_APP_SECRET:   str = os.getenv("META_APP_SECRET", "")

    # ── Server ────────────────────────────────────────────────────────────────
    SERVER_PORT:       int = int(os.getenv("SERVER_PORT", "8000"))
    SERVER_HOST:       str = os.getenv("SERVER_HOST", "127.0.0.1")
    AUTO_REFRESH_SECS: int = int(os.getenv("AUTO_REFRESH_SECONDS", "10"))

    # ── Pipeline ──────────────────────────────────────────────────────────────
    N_SIMULATIONS:     int   = int(os.getenv("N_SIMULATIONS", "20000"))
    RANDOM_SEED:       int   = 42
    BLEND_ALPHA:       float = 0.12

    # ── Overfitting guard ─────────────────────────────────────────────────────
    OVERFIT_THRESHOLD: float = 0.10   # regularise if train_auc - loo_auc > this

    # ── Data quality gate ─────────────────────────────────────────────────────
    # Pipeline warns (but continues) if real data fraction falls below this
    MIN_REAL_DATA_WARN:  float = 0.10
    # Pipeline halts if ALL sources are 0% real
    HALT_ON_ZERO_REAL:   bool  = False   # set True for strict production mode

    # ── Fusion weights (must sum to 1.0) ──────────────────────────────────────
    WEIGHT_META:    float = 0.30
    WEIGHT_TWITTER: float = 0.25
    WEIGHT_NEWS:    float = 0.20
    WEIGHT_GTRENDS: float = 0.25

    # ── Google Trends ─────────────────────────────────────────────────────────
    GTRENDS_GEO:       str = "IN-AS"       # Assam geo code
    GTRENDS_TIMEFRAME: str = "today 3-m"   # last 3 months
    GTRENDS_TIMEOUT:   int = 8

    # ── Paths (all relative to repo root) ─────────────────────────────────────
    ROOT:     Path = _ROOT
    DATA_DIR: Path = _ROOT / "data"
    OUT_DIR:  Path = _ROOT / "outputs"
    MDL_DIR:  Path = _ROOT / "models"
    WEB_DIR:  Path = _ROOT / "web"
    CACHE_DIR:Path = _ROOT / "cache"

    # ── Validation helpers ────────────────────────────────────────────────────
    def has_news_api(self) -> bool:
        return bool(self.NEWS_API_KEY and len(self.NEWS_API_KEY) > 10)

    def has_meta_api(self) -> bool:
        return bool(self.META_ACCESS_TOKEN and len(self.META_ACCESS_TOKEN) > 20)

    def weights_ok(self) -> bool:
        total = self.WEIGHT_META + self.WEIGHT_TWITTER + self.WEIGHT_NEWS + self.WEIGHT_GTRENDS
        return abs(total - 1.0) < 1e-6

    def __post_init__(self):
        assert self.weights_ok(), "Fusion weights must sum to 1.0"

    def summary(self) -> str:
        return (
            f"  NewsAPI    : {'✓ set' if self.has_news_api() else '✗ → RSS/synthetic'}\n"
            f"  Meta API   : {'✓ set' if self.has_meta_api() else '✗ → scrape/synthetic'}\n"
            f"  GTrends    : ✓ (geo={self.GTRENDS_GEO}, pytrends → historical fallback)\n"
            f"  Fusion     : Meta {self.WEIGHT_META:.0%} · Twitter {self.WEIGHT_TWITTER:.0%}"
            f" · News {self.WEIGHT_NEWS:.0%} · GTrends {self.WEIGHT_GTRENDS:.0%}\n"
            f"  Simulations: {self.N_SIMULATIONS:,}  |  Port: {self.SERVER_PORT}\n"
        )


CFG = Settings()
