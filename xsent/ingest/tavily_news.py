"""Recent, relevant news for the asset via the Tavily Search API (topic='news')."""
from __future__ import annotations

import logging

import pandas as pd
import requests

from ..config import Config
from ..db import connect, upsert

log = logging.getLogger(__name__)
URL = "https://api.tavily.com/search"


def search(cfg: Config, query: str) -> dict:
    r = requests.post(URL, headers={"Authorization": f"Bearer {cfg.tavily_api_key}"}, json={
        "query": query,
        "topic": "news",
        "time_range": cfg.news_time_range,
        "search_depth": "advanced",
        "max_results": cfg.news_max_results,
        "include_answer": "basic",
    }, timeout=60)
    if r.status_code >= 400:
        raise RuntimeError(f"Tavily {r.status_code}: {r.text[:300]}")
    return r.json()


def fetch_news(cfg: Config) -> int:
    if not cfg.tavily_api_key:
        raise SystemExit("TAVILY_API_KEY is missing. Add it to .env (see .env.example).")
    from ..sentiment import score_texts

    con = connect(cfg.db_path)
    now = pd.Timestamp.now(tz="UTC").isoformat()
    frames = []
    for q in cfg.news_queries:
        res = search(cfg, q)
        rows = [{
            "url": it.get("url"), "title": it.get("title"), "content": it.get("content"),
            "score": it.get("score"), "published_date": it.get("published_date"),
            "query": q, "fetched_at": now, "answer": res.get("answer"),
        } for it in res.get("results", []) if it.get("url")]
        log.info("Tavily '%s': %d articles", q, len(rows))
        frames.append(pd.DataFrame(rows))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if df.empty:
        return 0
    df = df.drop_duplicates("url")
    df["sent_compound"] = score_texts((df["title"].fillna("") + ". " + df["content"].fillna("")).tolist(),
                                      engine="vader")["compound"].values
    n = upsert(con, "news", df)
    con.close()
    return n
