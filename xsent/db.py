"""SQLite storage: raw tweets, prices, news, scrape-run log, and the joined daily view."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS tweets (
    tweet_id         TEXT PRIMARY KEY,
    source           TEXT,      -- 'apify:<actor>' | 'hist:<file>' | 'demo'
    query            TEXT,
    created_at_utc   TEXT,      -- ISO-8601, UTC
    created_at_est   TEXT,      -- ISO-8601, America/New_York (EST/EDT)
    date_utc         TEXT,      -- YYYY-MM-DD bucket that aligns with the daily price bar
    text             TEXT,
    lang             TEXT,
    likes            INTEGER,
    retweets         INTEGER,
    replies          INTEGER,
    quotes           INTEGER,
    views            INTEGER,
    author           TEXT,
    author_followers INTEGER,
    author_verified  INTEGER,
    sent_compound    REAL,
    sent_pos         REAL,
    sent_neg         REAL,
    is_spam          INTEGER
);
CREATE INDEX IF NOT EXISTS ix_tweets_date ON tweets(date_utc);

CREATE TABLE IF NOT EXISTS prices (
    date_utc TEXT PRIMARY KEY,   -- daily bar 00:00-24:00 UTC
    open REAL, high REAL, low REAL, close REAL, volume REAL,
    source TEXT
);

CREATE TABLE IF NOT EXISTS news (
    url            TEXT PRIMARY KEY,
    title          TEXT,
    content        TEXT,
    score          REAL,
    published_date TEXT,
    query          TEXT,
    fetched_at     TEXT,
    answer         TEXT,
    sent_compound  REAL
);

CREATE TABLE IF NOT EXISTS scrape_runs (
    window_start  TEXT,
    window_end    TEXT,
    actor_id      TEXT,
    run_id        TEXT,
    dataset_id    TEXT,
    status        TEXT,
    n_items       INTEGER,
    run_input     TEXT,
    finished_at   TEXT,
    PRIMARY KEY (window_start, window_end, actor_id)
);

-- Daily join of X sentiment aggregates with the price bar (built by features.py)
CREATE TABLE IF NOT EXISTS daily (
    date_utc TEXT PRIMARY KEY
);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    con.executescript(SCHEMA)
    return con


def upsert(con: sqlite3.Connection, table: str, df: pd.DataFrame) -> int:
    """INSERT OR REPLACE rows of df into table (columns must exist)."""
    if df is None or df.empty:
        return 0
    cols = list(df.columns)
    placeholders = ",".join("?" * len(cols))
    sql = f"INSERT OR REPLACE INTO {table} ({','.join(cols)}) VALUES ({placeholders})"
    rows = [tuple(None if pd.isna(v) else (v.item() if hasattr(v, "item") else v) for v in r)
            for r in df.itertuples(index=False, name=None)]
    con.executemany(sql, rows)
    con.commit()
    return len(rows)


def insert_ignore(con: sqlite3.Connection, table: str, df: pd.DataFrame) -> int:
    """INSERT OR IGNORE (keeps already-scored rows untouched). Returns rows inserted."""
    if df is None or df.empty:
        return 0
    cols = list(df.columns)
    sql = f"INSERT OR IGNORE INTO {table} ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})"
    before = con.total_changes
    rows = [tuple(None if pd.isna(v) else (v.item() if hasattr(v, "item") else v) for v in r)
            for r in df.itertuples(index=False, name=None)]
    con.executemany(sql, rows)
    con.commit()
    return con.total_changes - before


def read(con: sqlite3.Connection, sql: str, params=()) -> pd.DataFrame:
    return pd.read_sql_query(sql, con, params=params)


def replace_table(con: sqlite3.Connection, table: str, df: pd.DataFrame) -> None:
    df.to_sql(table, con, if_exists="replace", index=False)
    con.commit()
