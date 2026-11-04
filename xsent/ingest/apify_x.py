"""Scrape X (Twitter) posts through an Apify actor, one time window at a time.

Why windows: a single "last 180 days" query returns mostly the newest posts.
Querying each day (or N-hour window) separately with a fixed cap gives an
even sample across the whole period, which is what a daily sentiment series
needs. Windows already scraped are skipped, so the job is resumable.

Uses the Apify REST API directly (no SDK version drift):
  POST /v2/acts/{actor}/runs        -> start run
  GET  /v2/actor-runs/{id}          -> poll until finished
  GET  /v2/datasets/{id}/items      -> download tweets
"""
from __future__ import annotations

import hashlib
import json
import logging
import time

import pandas as pd
import requests

from ..config import APIDOJO_ACTOR, KAITO_ACTOR, SCWEET_ACTOR, Config
from ..db import connect, insert_ignore, read, upsert
from ..timeutil import add_time_columns, parse_ts, today_utc

log = logging.getLogger(__name__)
API = "https://api.apify.com/v2"

# Input templates per actor. Placeholders are filled by str.format-like replace.
# since_time / until_time (unix seconds) are X advanced-search operators.
TEMPLATES = {
    KAITO_ACTOR: {
        "twitterContent": "{query} lang:{lang} since_time:{since_ts} until_time:{until_ts}",
        "queryType": "{query_type}",
        "lang": "{lang}",
        "maxItems": "{max_items}",
    },
    APIDOJO_ACTOR: {
        "searchTerms": ["{query} since_time:{since_ts} until_time:{until_ts}"],
        "sort": "{query_type}",
        "tweetLanguage": "{lang}",
        "maxItems": "{max_items}",
    },
    SCWEET_ACTOR: {
        "search_query": "{query} lang:{lang}",
        "since": "{since_iso}",
        "until": "{until_iso}",
        "search_sort": "{query_type}",
        "max_items": "{max_items}",
    },
}

# approximate USD per 1,000 tweets (Apify Store, free tier, checked Nov 2026)
PRICE_PER_1K = {KAITO_ACTOR: 0.25, APIDOJO_ACTOR: 0.40, SCWEET_ACTOR: 0.30}


def _fill(obj, values: dict):
    if isinstance(obj, dict):
        return {k: _fill(v, values) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_fill(v, values) for v in obj]
    if isinstance(obj, str):
        out = obj
        for k, v in values.items():
            out = out.replace("{" + k + "}", str(v))
        if obj == "{max_items}":
            return int(out)
        return out
    return obj


def build_input(cfg: Config, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    if cfg.apify_input_template:
        template = json.loads(cfg.apify_input_template)
    else:
        template = TEMPLATES.get(cfg.apify_actor_id, TEMPLATES[KAITO_ACTOR])
    values = {
        "query": cfg.x_query,
        "lang": cfg.apify_lang,
        "query_type": cfg.apify_query_type,
        "max_items": cfg.tweets_per_window,
        "since_ts": int(start.timestamp()),
        "until_ts": int(end.timestamp()),
        "since": start.strftime("%Y-%m-%d"),
        "until": end.strftime("%Y-%m-%d"),
        "since_iso": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "until_iso": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    return _fill(template, values)


# --------------------------------------------------------------------------
# Normalisation (shared by Apify output and historical CSVs)
# --------------------------------------------------------------------------
def _get(item: dict, *paths, default=None):
    for p in paths:
        cur = item
        ok = True
        for part in p.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                ok = False
                break
        if ok and cur is not None:
            return cur
    return default


def _num(v) -> int:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return 0


def normalise_items(items: list[dict], source: str, query: str) -> pd.DataFrame:
    rows = []
    for it in items:
        text = _get(it, "text", "full_text", "fullText", "content")
        created = _get(it, "createdAt", "created_at", "date")
        if not text or not created:
            continue  # error/mock/no-result placeholder items
        tid = str(_get(it, "id", "id_str", "tweetId", "tweet_id", default=""))
        if not tid or tid in {"-1", "0"}:
            tid = hashlib.md5(f"{created}|{text}".encode()).hexdigest()
        rows.append({
            "tweet_id": tid,
            "source": source,
            "query": query,
            "_created": created,
            "text": str(text),
            "lang": _get(it, "lang", default=None),
            "likes": _num(_get(it, "likeCount", "favorite_count", "likes", default=0)),
            "retweets": _num(_get(it, "retweetCount", "retweet_count", "retweets", default=0)),
            "replies": _num(_get(it, "replyCount", "reply_count", "replies", default=0)),
            "quotes": _num(_get(it, "quoteCount", "quote_count", default=0)),
            "views": _num(_get(it, "viewCount", "views", "view_count", default=0)),
            "author": _get(it, "author.userName", "author.username", "user.screen_name",
                           "user.username", default=None),
            "author_followers": _num(_get(it, "author.followers", "author.followersCount",
                                          "user.followers_count", default=0)),
            "author_verified": int(bool(_get(it, "author.isBlueVerified", "author.isVerified",
                                             "user.verified", default=False))),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    ts = parse_ts(df.pop("_created"))
    df = df[ts.notna().values]
    ts = ts[ts.notna()]
    return add_time_columns(df, ts)


# --------------------------------------------------------------------------
# Apify REST calls
# --------------------------------------------------------------------------
class ApifyError(RuntimeError):
    pass


def _session(token: str) -> requests.Session:
    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {token}"
    return s


def run_actor(cfg: Config, run_input: dict) -> tuple[dict, list[dict]]:
    s = _session(cfg.apify_token)
    actor = cfg.apify_actor_id.replace("/", "~")
    params = {"timeout": cfg.apify_run_timeout_s}
    if cfg.apify_max_charge_usd > 0:
        params["maxTotalChargeUsd"] = cfg.apify_max_charge_usd
    r = s.post(f"{API}/acts/{actor}/runs", json=run_input, params=params, timeout=60)
    if r.status_code >= 400:
        raise ApifyError(f"start failed {r.status_code}: {r.text[:300]}")
    run = r.json()["data"]
    deadline = time.time() + cfg.apify_run_timeout_s + 120
    while run["status"] in {"READY", "RUNNING"}:
        if time.time() > deadline:
            raise ApifyError(f"run {run['id']} timed out")
        r = s.get(f"{API}/actor-runs/{run['id']}", params={"waitForFinish": 60}, timeout=90)
        r.raise_for_status()
        run = r.json()["data"]
    if run["status"] != "SUCCEEDED":
        raise ApifyError(f"run {run['id']} ended with status {run['status']}")
    items, offset = [], 0
    while True:
        r = s.get(f"{API}/datasets/{run['defaultDatasetId']}/items",
                  params={"clean": "true", "format": "json", "offset": offset, "limit": 1000},
                  timeout=120)
        r.raise_for_status()
        batch = r.json()
        items.extend(batch)
        if len(batch) < 1000:
            break
        offset += 1000
    return run, items


def windows(cfg: Config, end: pd.Timestamp | None = None) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Non-overlapping UTC windows covering the last `lookback_days` complete days."""
    end = end or today_utc()  # exclusive: today's bar is still forming
    start = end - pd.Timedelta(days=cfg.lookback_days)
    step = pd.Timedelta(hours=cfg.window_hours)
    out, t = [], start
    while t < end:
        out.append((t, min(t + step, end)))
        t += step
    return out


def _log_run(con, cfg, ws, we, run_input, status, run=None, n=0):
    upsert(con, "scrape_runs", pd.DataFrame([{
        "window_start": ws.isoformat(), "window_end": we.isoformat(),
        "actor_id": cfg.apify_actor_id, "run_id": run["id"] if run else None,
        "dataset_id": run["defaultDatasetId"] if run else None, "status": status,
        "n_items": int(n), "run_input": json.dumps(run_input),
        "finished_at": pd.Timestamp.now(tz="UTC").isoformat()}]))


def _one_window(cfg: Config, ws, we):
    run_input = build_input(cfg, ws, we)
    try:
        run, items = run_actor(cfg, run_input)
    except (ApifyError, requests.RequestException) as e:
        return ws, we, run_input, None, None, e
    df = normalise_items(items, f"apify:{cfg.apify_actor_id}", cfg.x_query)
    if not df.empty:
        # guard against actors that return mock / out-of-window items
        t = pd.to_datetime(df["created_at_utc"], utc=True)
        df = df[((t >= ws) & (t < we)).values]
    return ws, we, run_input, run, df, None


def scrape(cfg: Config, max_windows: int | None = None) -> int:
    """Scrape every missing window, newest first, `APIFY_CONCURRENCY` runs at a time.

    Windows that succeed are recorded in scrape_runs and skipped next time; failed or
    empty windows are retried on the next call. Stops early on auth/plan/credit errors
    or when the first windows all come back empty (typical of free-plan demo mode).
    """
    if not cfg.apify_token:
        raise SystemExit("APIFY_TOKEN is missing. Add it to .env (see .env.example).")
    from concurrent.futures import ThreadPoolExecutor, as_completed

    con = connect(cfg.db_path)
    done = read(con, "SELECT window_start FROM scrape_runs WHERE actor_id=? AND status='SUCCEEDED'",
                (cfg.apify_actor_id,))
    done_set = set(done["window_start"])
    todo = [w for w in windows(cfg) if w[0].isoformat() not in done_set][::-1]  # newest first
    if max_windows:
        todo = todo[:max_windows]
    est = len(todo) * cfg.tweets_per_window / 1000 * PRICE_PER_1K.get(cfg.apify_actor_id, 0.40)
    log.info("Apify actor %s: %d windows to scrape (%d already done), <= %d tweets, est. cost ~$%.2f",
             cfg.apify_actor_id, len(todo), len(done_set), len(todo) * cfg.tweets_per_window, est)
    total, finished, empties, fatal = 0, 0, 0, None
    with ThreadPoolExecutor(max_workers=max(1, cfg.apify_concurrency)) as pool:
        futures = [pool.submit(_one_window, cfg, ws, we) for ws, we in todo]
        for fut in as_completed(futures):
            ws, we, run_input, run, df, err = fut.result()
            finished += 1
            if err is not None:
                log.warning("window %s failed: %s", ws.date(), err)
                _log_run(con, cfg, ws, we, run_input, "FAILED")
                if "start failed 40" in str(err):   # 401/402/403: token, plan or credit problem
                    fatal = err
                    for f in futures:
                        f.cancel()
                    break
                continue
            if df.empty:
                empties += 1
                _log_run(con, cfg, ws, we, run_input, "EMPTY", run)
                log.warning("window %s returned no usable tweets (run %s)", ws.date(), run["id"])
                if empties >= 3 and empties == finished:
                    fatal = ApifyError(
                        "the first windows all returned no tweets. The actor may be in free-plan demo "
                        "mode or the query/input format is not accepted - check the run log in the Apify "
                        "console, or switch APIFY_ACTOR_ID (see README).")
                    for f in futures:
                        f.cancel()
                    break
                continue
            n = insert_ignore(con, "tweets", df)
            total += n
            _log_run(con, cfg, ws, we, run_input, "SUCCEEDED", run, len(df))
            log.info("[%d/%d] %s -> %d tweets (run %s)", finished, len(todo), ws.date(), len(df), run["id"])
    con.close()
    if fatal:
        raise SystemExit(f"Apify scraping stopped: {fatal}")
    log.info("scrape finished: %d new tweets", total)
    return total
