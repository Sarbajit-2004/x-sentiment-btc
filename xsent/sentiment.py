"""Post-level sentiment scoring + spam filtering.

Default engine: VADER (rule-based, built for social media text) extended with a
crypto-trading lexicon. It is fast (~10k posts/s), deterministic and needs no
GPU, so the full pipeline is reproducible on a laptop. Optional engine
'finbert' (ProsusAI/finbert via transformers) can be switched on in .env.
"""
from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd

from .config import Config
from .db import connect, read

log = logging.getLogger(__name__)

CRYPTO_LEXICON = {
    "bullish": 2.6, "bull": 1.5, "bulls": 1.5, "bullrun": 2.6, "moon": 2.2, "mooning": 2.5,
    "pump": 1.2, "pumping": 1.5, "rally": 2.0, "rallying": 2.0, "breakout": 1.8, "surge": 2.0,
    "soar": 2.2, "soaring": 2.2, "ath": 2.0, "hodl": 1.5, "hodling": 1.5, "accumulate": 1.5,
    "accumulating": 1.5, "btfd": 1.5, "inflows": 1.3, "inflow": 1.3, "uptrend": 1.8,
    "wagmi": 2.0, "undervalued": 1.5, "adoption": 1.2, "green": 0.8, "rip": 0.0,
    "bearish": -2.6, "bear": -1.5, "bears": -1.5, "dump": -2.0, "dumping": -2.2, "dumped": -2.0,
    "rekt": -2.6, "crash": -2.8, "crashing": -2.8, "plunge": -2.5, "plunging": -2.5,
    "tank": -2.0, "tanking": -2.2, "breakdown": -1.8, "capitulation": -2.2, "liquidated": -2.2,
    "liquidations": -1.8, "fud": -1.8, "rug": -2.8, "rugpull": -3.0, "scam": -2.5,
    "outflows": -1.3, "outflow": -1.3, "downtrend": -1.8, "ngmi": -2.0, "overvalued": -1.5,
    "correction": -1.2, "selloff": -2.0, "red": -0.8, "bubble": -1.5, "ponzi": -2.8,
}

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
CASHTAG_RE = re.compile(r"\$[A-Za-z]{2,10}\b")
WS_RE = re.compile(r"\s+")

_vader = None


def _get_vader():
    global _vader
    if _vader is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _vader = SentimentIntensityAnalyzer()
        _vader.lexicon.update(CRYPTO_LEXICON)
    return _vader


def clean(text: str) -> str:
    t = URL_RE.sub(" ", text or "")
    t = MENTION_RE.sub(" ", t)
    t = t.replace("#", " ").replace("$", " ")
    return WS_RE.sub(" ", t).strip()


# transformer engines: model id, positive label, negative label. compound = P(pos) - P(neg)
TRANSFORMER_ENGINES = {
    "finbert": ("ProsusAI/finbert", "positive", "negative"),
    "cryptobert": ("ElKulako/cryptobert", "bullish", "bearish"),  # trained on crypto social posts
}


def score_texts(texts: list[str], engine: str = "vader") -> pd.DataFrame:
    cleaned = [clean(t) for t in texts]
    if engine in TRANSFORMER_ENGINES:
        # no silent fallback to VADER: a comparison run must really use the engine it names
        return _transformer(cleaned, *TRANSFORMER_ENGINES[engine])
    v = _get_vader()
    out = [v.polarity_scores(t) for t in cleaned]
    return pd.DataFrame({"compound": [o["compound"] for o in out],
                         "pos": [o["pos"] for o in out], "neg": [o["neg"] for o in out]})


def _transformer(texts: list[str], model: str, pos: str, neg: str) -> pd.DataFrame:
    from transformers import pipeline  # optional heavy dependency (torch + transformers)

    clf = pipeline("text-classification", model=model, top_k=None, truncation=True, max_length=128)
    res = clf(texts, batch_size=64)
    rows = []
    for r in res:
        p = {d["label"].lower(): d["score"] for d in r}
        rows.append({"compound": p.get(pos, 0) - p.get(neg, 0), "pos": p.get(pos, 0), "neg": p.get(neg, 0)})
    return pd.DataFrame(rows)


# engagement-farming / giveaway posts: they carry no view on price but collect huge engagement
PROMO_RE = re.compile(
    r"giveaway|give ?away|airdrop|to enter\b|retweet to win|\brt to win|repost (this|and|&)|"
    r"like (this|and|&) (post|tweet|retweet|repost)|follow (me|us) (and|&)|"
    r"(reply|drop|comment) (with )?your (btc |bitcoin |sol |eth )?(address|wallet)|"
    r"will go to one of you|winners? (will|gets?)|free (btc|bitcoin|crypto)\b|"
    r"use (my )?(code|link)|referral|sign ?up bonus|promo code",
    re.IGNORECASE)


def spam_flags(df: pd.DataFrame, max_cashtags: int) -> np.ndarray:
    """Flag shill/bot posts: cashtag stuffing, giveaways/engagement farming,
    exact duplicates on the same day, near-empty text."""
    text = df["text"].fillna("")
    n_cash = text.str.count(CASHTAG_RE.pattern)
    promo = text.map(lambda x: PROMO_RE.search(x) is not None)
    norm = text.map(clean).str.lower().str.replace(r"[^a-z ]", "", regex=True)
    dup_day = pd.Series(list(zip(df["date_utc"], norm))).duplicated(keep="first").values & norm.str.len().gt(0).values
    short = norm.str.split().str.len().fillna(0) < 3
    return ((n_cash > max_cashtags).values | promo.values | dup_day | short.values).astype(int)


def score_db(cfg: Config, rescore: bool = False) -> int:
    """Score posts that have no sentiment yet, and refresh the spam flag of every post
    (cheap, and keeps flags consistent when the filter rules change)."""
    con = connect(cfg.db_path)
    alltxt = read(con, "SELECT tweet_id, date_utc, text, sent_compound FROM tweets ORDER BY created_at_utc")
    if alltxt.empty:
        con.close()
        return 0
    flags = spam_flags(alltxt, cfg.max_cashtags)
    con.executemany("UPDATE tweets SET is_spam=? WHERE tweet_id=?",
                    list(zip(flags.tolist(), alltxt["tweet_id"])))
    todo = alltxt if rescore else alltxt[alltxt["sent_compound"].isna()]
    if len(todo):
        s = score_texts(todo["text"].tolist(), engine=cfg.sentiment_engine)
        con.executemany("UPDATE tweets SET sent_compound=?, sent_pos=?, sent_neg=? WHERE tweet_id=?",
                        list(zip(s["compound"].astype(float), s["pos"].astype(float),
                                 s["neg"].astype(float), todo["tweet_id"])))
    con.commit()
    con.close()
    log.info("scored %d new posts (%s); %d of %d posts flagged as spam/promo", len(todo),
             cfg.sentiment_engine, int(flags.sum()), len(alltxt))
    return len(todo)
