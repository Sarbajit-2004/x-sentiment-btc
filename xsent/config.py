"""Central configuration. Everything can be overridden from the .env file."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # dotenv is optional
    load_dotenv = None

ROOT = Path(__file__).resolve().parents[1]
if load_dotenv is not None:
    load_dotenv(ROOT / ".env")


def _env(name: str, default, cast=str):
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    if cast is bool:
        return raw.strip().lower() in {"1", "true", "yes", "y"}
    return cast(raw.strip())


# Default Apify actors. Any actor that accepts a search query and returns tweets
# in the common X JSON shape works; set APIFY_ACTOR_ID in .env to switch.
KAITO_ACTOR = "kaitoeasyapi/twitter-x-data-tweet-scraper-pay-per-result-cheapest"
APIDOJO_ACTOR = "apidojo/tweet-scraper"
SCWEET_ACTOR = "altimis/scweet"


@dataclass
class Config:
    # ---- asset -----------------------------------------------------------
    asset_name: str = _env("ASSET_NAME", "Bitcoin")
    price_ticker: str = _env("PRICE_TICKER", "BTC-USD")          # yfinance symbol
    coinbase_product: str = _env("COINBASE_PRODUCT", "BTC-USD")  # price fallback
    # ticker, cashtag and keyword searches in one X advanced-search query
    x_query: str = _env("X_QUERY", "(bitcoin OR BTC OR $BTC OR #bitcoin OR #BTC)")
    lookback_days: int = _env("LOOKBACK_DAYS", 180, int)

    # ---- Apify (X scraping) ---------------------------------------------
    apify_token: str = _env("APIFY_TOKEN", "")
    apify_actor_id: str = _env("APIFY_ACTOR_ID", KAITO_ACTOR)
    tweets_per_window: int = _env("TWEETS_PER_WINDOW", 80, int)
    window_hours: int = _env("WINDOW_HOURS", 24, int)
    apify_query_type: str = _env("APIFY_QUERY_TYPE", "Top")  # Top | Latest
    apify_lang: str = _env("APIFY_LANG", "en")
    apify_max_charge_usd: float = _env("APIFY_MAX_CHARGE_USD", 0.5, float)  # per run cap
    apify_concurrency: int = _env("APIFY_CONCURRENCY", 3, int)  # parallel actor runs
    apify_run_timeout_s: int = _env("APIFY_RUN_TIMEOUT_S", 600, int)
    # optional JSON template overriding the actor input (see README)
    apify_input_template: str = _env("APIFY_INPUT_TEMPLATE", "")

    # ---- Tavily (news) ---------------------------------------------------
    tavily_api_key: str = _env("TAVILY_API_KEY", "")
    news_queries: tuple = tuple(
        q.strip() for q in _env(
            "NEWS_QUERIES",
            "Bitcoin price news|Bitcoin ETF flows|Bitcoin regulation SEC|crypto market outlook Bitcoin",
        ).split("|") if q.strip()
    )
    news_time_range: str = _env("NEWS_TIME_RANGE", "week")  # day|week|month|year
    news_max_results: int = _env("NEWS_MAX_RESULTS", 10, int)

    # ---- sentiment --------------------------------------------------------
    sentiment_engine: str = _env("SENTIMENT_ENGINE", "vader")  # vader | finbert | cryptobert
    max_cashtags: int = _env("MAX_CASHTAGS", 5, int)  # drop shill spam

    # ---- modelling ------------------------------------------------------
    horizons: tuple = (0, 1, 2, 3, 4, 5)
    min_train_days: int = _env("MIN_TRAIN_DAYS", 60, int)
    refit_every: int = _env("REFIT_EVERY", 1, int)
    n_jobs: int = _env("N_JOBS", -1, int)  # parallel horizons (-1 = all cores, 1 = sequential)
    primary_model: str = _env("PRIMARY_MODEL", "ensemble")
    cost_bps: float = _env("COST_BPS", 10.0, float)          # per unit turnover
    signal_threshold: float = _env("SIGNAL_THRESHOLD", 0.0, float)
    allow_short: bool = _env("ALLOW_SHORT", False, bool)
    eval_days: int = _env("EVAL_DAYS", 180, int)  # report window (most recent N days)
    random_seed: int = 42

    # ---- paths (override to run a variant on its own database / output folder) ----
    data_dir: Path = field(default_factory=lambda: ROOT / _env("DATA_DIR", "data"))
    out_dir: Path = field(default_factory=lambda: ROOT / _env("OUT_DIR", "outputs"))

    @property
    def db_path(self) -> Path:
        return self.data_dir / "xsent.db"

    def demo(self) -> "Config":
        """Return a copy pointed at the offline demo database/output folder."""
        c = Config()
        c.data_dir = ROOT / "data" / "demo"
        c.out_dir = ROOT / "outputs" / "demo"
        return c

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "charts").mkdir(parents=True, exist_ok=True)
