"""Command-line entry point.

    python run_pipeline.py all                  # scrape X + news + prices, then analyse
    python run_pipeline.py scrape --max-windows 1   # test Apify with one day first
    python run_pipeline.py historical --csv path/to/bitcoin_tweets.csv [--start 2025-01-01]
    python run_pipeline.py prices | news | analyse
    python run_pipeline.py demo                 # offline end-to-end test on synthetic data
    python run_pipeline.py check                # verify keys + connectivity
"""
from __future__ import annotations

import argparse
import logging
import sys

from xsent.config import Config


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="X sentiment -> price prediction pipeline")
    ap.add_argument("step", choices=["all", "scrape", "news", "prices", "historical", "analyse",
                                     "demo", "check"])
    ap.add_argument("--max-windows", type=int, default=None, help="scrape only N windows (testing)")
    ap.add_argument("--csv", help="historical tweets CSV (for the 'historical' step)")
    ap.add_argument("--start", help="historical: keep tweets on/after YYYY-MM-DD")
    ap.add_argument("--end", help="historical: keep tweets before YYYY-MM-DD")
    ap.add_argument("--per-day", type=int, default=300, help="historical: max tweets kept per day")
    ap.add_argument("--no-pdf", action="store_true", help="skip PDF export")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for noisy in ["urllib3", "matplotlib", "yfinance", "peewee"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)
    cfg = Config()
    cfg.ensure_dirs()

    if a.step == "check":
        from xsent.check import check
        return check(cfg)
    if a.step == "demo":
        from xsent.ingest.demo import generate
        from xsent.pipeline import analyse
        cfg = cfg.demo()
        cfg.ensure_dirs()
        generate(cfg)
        analyse(cfg, demo=True, pdf=not a.no_pdf)
        return 0
    if a.step in ("scrape", "all"):
        from xsent.ingest.apify_x import scrape
        scrape(cfg, max_windows=a.max_windows)
    if a.step in ("news", "all"):
        from xsent.ingest.tavily_news import fetch_news
        try:
            fetch_news(cfg)
        except Exception as e:  # news only feeds the outlook; never block the model
            logging.getLogger("news").warning("news step failed: %s", e)
    if a.step == "historical":
        if not a.csv:
            ap.error("--csv is required for the historical step")
        from xsent.ingest.historical import load_csv
        load_csv(cfg, a.csv, start=a.start, end=a.end, per_day=a.per_day)
    if a.step in ("prices", "all", "historical"):
        from xsent.ingest.prices import fetch_prices
        fetch_prices(cfg)
    if a.step in ("analyse", "all"):
        from xsent.pipeline import analyse
        analyse(cfg, pdf=not a.no_pdf)
    return 0


if __name__ == "__main__":
    sys.exit(main())
