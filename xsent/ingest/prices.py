"""Daily OHLC price bars (UTC). yfinance first, Coinbase Exchange public API as fallback."""
from __future__ import annotations

import logging

import pandas as pd
import requests

from ..config import Config
from ..db import connect, read, upsert
from ..timeutil import today_utc

log = logging.getLogger(__name__)


def _yfinance(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    import yfinance as yf

    h = yf.Ticker(ticker).history(start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"),
                                  interval="1d", auto_adjust=False)
    if h is None or h.empty:
        return pd.DataFrame()
    idx = h.index.tz_convert("UTC") if h.index.tz is not None else h.index.tz_localize("UTC")
    return pd.DataFrame({
        "date_utc": idx.strftime("%Y-%m-%d"),
        "open": h["Open"].values, "high": h["High"].values, "low": h["Low"].values,
        "close": h["Close"].values, "volume": h["Volume"].values, "source": "yfinance",
    })


def _coinbase(product: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    rows, t = [], start
    while t < end:
        t2 = min(t + pd.Timedelta(days=299), end)
        r = requests.get(f"https://api.exchange.coinbase.com/products/{product}/candles",
                         params={"granularity": 86400, "start": t.isoformat(), "end": t2.isoformat()},
                         headers={"User-Agent": "xsent-pipeline"}, timeout=30)
        r.raise_for_status()
        rows.extend(r.json())  # [time, low, high, open, close, volume]
        t = t2
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["time", "low", "high", "open", "close", "volume"])
    df["date_utc"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.strftime("%Y-%m-%d")
    df["source"] = "coinbase"
    return df.drop(columns="time").drop_duplicates("date_utc")


def fetch_prices(cfg: Config, start: pd.Timestamp | None = None) -> int:
    con = connect(cfg.db_path)
    end = today_utc()  # exclusive -> today's still-forming candle is never stored
    if start is None:
        # cover the tweet history plus a 60-day warm-up for rolling price features
        first = read(con, "SELECT MIN(date_utc) AS d FROM tweets")["d"].iloc[0]
        start = pd.Timestamp(first, tz="UTC") if first else end - pd.Timedelta(days=cfg.lookback_days)
        start = start - pd.Timedelta(days=60)
    df = pd.DataFrame()
    try:
        df = _yfinance(cfg.price_ticker, start, end)
    except Exception as e:  # network / API change
        log.warning("yfinance failed (%s); falling back to Coinbase", e)
    if df.empty:
        df = _coinbase(cfg.coinbase_product, start, end)
    df = df[df["date_utc"] < end.strftime("%Y-%m-%d")].dropna(subset=["close"])
    n = upsert(con, "prices", df)
    log.info("stored %d daily bars %s .. %s (%s)", n, df["date_utc"].min(), df["date_utc"].max(),
             df["source"].iloc[0] if n else "-")
    con.close()
    return n
