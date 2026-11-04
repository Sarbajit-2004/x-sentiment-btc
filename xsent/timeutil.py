"""Timestamp parsing and UTC / US-Eastern alignment."""
from __future__ import annotations

import pandas as pd

EASTERN = "America/New_York"
X_FORMAT = "%a %b %d %H:%M:%S %z %Y"  # e.g. "Tue Oct 29 12:00:00 +0000 2024"


def parse_ts(values) -> pd.Series:
    """Parse X/Twitter or ISO timestamps into tz-aware UTC. Unparseable -> NaT."""
    s = pd.Series(values)
    out = pd.to_datetime(s, format=X_FORMAT, errors="coerce", utc=True)
    missing = out.isna() & s.notna()
    if missing.any():
        # ISO strings, '2021-02-10 23:59:04', epoch seconds, etc.
        alt = pd.to_datetime(s[missing], errors="coerce", utc=True, format="mixed")
        out.loc[missing] = alt
    return out


def add_time_columns(df: pd.DataFrame, ts: pd.Series) -> pd.DataFrame:
    """Attach created_at_utc / created_at_est / date_utc columns.

    The daily price bar for crypto (yfinance, Coinbase) runs 00:00-24:00 UTC,
    so each post is bucketed on its UTC calendar date. The Eastern timestamp is
    kept for reference / equity-market alignment.
    """
    ts = pd.to_datetime(ts, utc=True)
    df = df.copy()
    df["created_at_utc"] = ts.dt.strftime("%Y-%m-%dT%H:%M:%SZ").values
    df["created_at_est"] = ts.dt.tz_convert(EASTERN).dt.strftime("%Y-%m-%dT%H:%M:%S%z").values
    df["date_utc"] = ts.dt.strftime("%Y-%m-%d").values
    return df


def today_utc() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").normalize()
