"""Load historical tweet CSVs (e.g. the Kaggle Bitcoin datasets in the brief) into the tweets table.

Column names differ between datasets, so a set of aliases is tried for each
field. Files can be many GB, so they are streamed in chunks and optionally
down-sampled to `per_day` tweets per UTC day (top by engagement), which keeps
the daily sentiment series comparable to the capped Apify sample.

Tested shapes:
  * kaushiksuresh147/bitcoin-tweets  (user_name, user_followers, user_verified, date, text, ...)
  * pokeash/bitcoin-tweets-dataset-20252026 (same family of columns)
  * alaix14/bitcoin-tweets-20160101-to-20190329 (';'-separated: id;user;fullname;url;timestamp;replies;likes;retweets;text)
  * generic: any CSV with a timestamp column and a text column
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pandas as pd

from ..config import Config
from ..db import connect, insert_ignore
from ..timeutil import add_time_columns, parse_ts

log = logging.getLogger(__name__)

ALIASES = {
    "tweet_id": ["id", "tweet_id", "id_str", "tweetid"],
    "created": ["date", "created_at", "createdat", "timestamp", "datetime", "time"],
    "text": ["text", "full_text", "tweet", "content", "body"],
    "likes": ["likes", "like_count", "likecount", "favorite_count", "favorites", "user_favourites_ignore"],
    "retweets": ["retweets", "retweet_count", "retweetcount"],
    "replies": ["replies", "reply_count", "replycount"],
    "quotes": ["quotes", "quote_count", "quotecount"],
    "views": ["views", "view_count", "viewcount", "impressions"],
    "author": ["user_name", "username", "user", "author", "screen_name", "user_screen_name"],
    "author_followers": ["user_followers", "followers", "followers_count", "user_followers_count",
                         "author_followers"],
    "author_verified": ["user_verified", "verified", "is_verified", "isblueverified"],
    "lang": ["lang", "language"],
}


def _detect_sep(path: Path) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        head = f.readline()
    return ";" if head.count(";") > head.count(",") else ","


def _map_columns(cols) -> dict:
    lower = {c.lower().strip(): c for c in cols}
    mapping = {}
    for target, names in ALIASES.items():
        for n in names:
            if n in lower:
                mapping[target] = lower[n]
                break
    missing = {"created", "text"} - set(mapping)
    if missing:
        raise ValueError(f"CSV is missing required columns {missing}; found {list(cols)}")
    return mapping


def _to_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0).astype("int64")


def load_csv(cfg: Config, path: str | Path, start: str | None = None, end: str | None = None,
             per_day: int | None = 300, chunksize: int = 200_000) -> int:
    path = Path(path)
    sep = _detect_sep(path)
    con = connect(cfg.db_path)
    source = f"hist:{path.name}"
    pool = pd.DataFrame()
    mapping = None

    def reduce(df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop_duplicates("tweet_id")
        if not per_day:
            return df
        eng = df["likes"] + 2 * df["retweets"] + df["replies"] + 1e-6 * df["author_followers"]
        return (df.assign(_e=eng).sort_values("_e", ascending=False)
                  .groupby("date_utc", group_keys=False).head(per_day).drop(columns="_e"))

    reader = pd.read_csv(path, sep=sep, chunksize=chunksize, dtype=str, on_bad_lines="skip",
                         engine="python" if sep == ";" else "c", encoding_errors="ignore")
    for k, chunk in enumerate(reader):
        if mapping is None:
            mapping = _map_columns(chunk.columns)
            log.info("column mapping: %s", mapping)
        ts = parse_ts(chunk[mapping["created"]].values)
        ok = ts.notna().values & chunk[mapping["text"]].notna().values
        chunk, ts = chunk[ok], ts[ok]
        if start:
            m = (ts >= pd.Timestamp(start, tz="UTC")).values
            chunk, ts = chunk[m], ts[m]
        if end:
            m = (ts < pd.Timestamp(end, tz="UTC")).values
            chunk, ts = chunk[m], ts[m]
        if chunk.empty:
            continue
        df = pd.DataFrame({"text": chunk[mapping["text"]].astype(str).values})
        for f in ["likes", "retweets", "replies", "quotes", "views", "author_followers"]:
            df[f] = _to_int(chunk[mapping[f]]).values if f in mapping else 0
        df["author"] = chunk[mapping["author"]].values if "author" in mapping else None
        df["author_verified"] = (chunk[mapping["author_verified"]].astype(str).str.lower()
                                 .isin(["true", "1", "yes"]).astype(int).values
                                 if "author_verified" in mapping else 0)
        df["lang"] = chunk[mapping["lang"]].values if "lang" in mapping else None
        if "tweet_id" in mapping:
            df["tweet_id"] = chunk[mapping["tweet_id"]].astype(str).values
        else:
            df["tweet_id"] = [hashlib.md5(f"{t}|{x}|{a}".encode()).hexdigest()
                              for t, x, a in zip(ts.astype(str), df["text"], df["author"])]
        df["source"] = source
        df["query"] = "historical"
        df = add_time_columns(df, ts.reset_index(drop=True))
        pool = reduce(pd.concat([pool, df], ignore_index=True))
        log.info("chunk %d: %d rows parsed, pool=%d", k, len(df), len(pool))
    if pool.empty:
        log.warning("no rows loaded from %s", path)
        return 0
    df = pool
    n = insert_ignore(con, "tweets", df)
    log.info("inserted %d historical tweets from %s (%s .. %s)", n, path.name,
             df["date_utc"].min(), df["date_utc"].max())
    con.close()
    return n
