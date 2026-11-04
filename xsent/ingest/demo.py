"""Synthetic offline data used ONLY to test the pipeline without API keys / internet.

A latent 'crowd mood' follows an AR(1) process. Next-day returns load weakly
on yesterday's mood, and posts are generated with a bullish/bearish lean that
depends on the mood and on the same day's return. Every number produced from
this data is fake; the report is stamped as a demo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config
from ..db import connect, insert_ignore, upsert
from ..timeutil import add_time_columns, today_utc

BULL = ["Bitcoin looks bullish, breakout above resistance", "$BTC to the moon, accumulating more",
        "Strong ETF inflows, #bitcoin rally continues", "HODL, this dip is a gift, buying BTC",
        "BTC surge incoming, bulls in control", "Great momentum on #BTC, new highs soon"]
BEAR = ["Bitcoin looks weak, breakdown below support", "$BTC dump continues, getting rekt",
        "ETF outflows again, #bitcoin crash risk", "Selling my BTC, this rally is fake",
        "BTC plunge, liquidations everywhere, bearish", "Bearish divergence on #BTC, expect correction"]
FILLER = ("today now chart market crypto traders week price watch level desk fund volume news "
          "asia europe us session close open candle trend range support zone signal timeframe daily "
          "weekly hourly spot futures options macro rates dollar gold stocks miners halving cycle "
          "wallet exchange whales retail funds flows desk analyst thread").split()
NEUTRAL = ["Bitcoin trading sideways today", "Watching $BTC price action closely",
           "#bitcoin weekly close coming up", "What is everyone's BTC plan this week?"]


def generate(cfg: Config, days: int = 240, posts_per_day: int = 100, seed: int = 7,
             signal: float = 0.25) -> None:
    rng = np.random.default_rng(seed)
    end = today_utc()
    dates = pd.date_range(end - pd.Timedelta(days=days), periods=days, freq="D", tz="UTC")
    mood = np.zeros(days)
    for t in range(1, days):
        mood[t] = 0.85 * mood[t - 1] + rng.normal(0, 0.5)
    sigma = 0.025
    r = np.zeros(days)
    for t in range(1, days):
        r[t] = 0.0005 + signal * sigma * np.tanh(mood[t - 1]) + rng.normal(0, sigma)
    close = 60000 * np.exp(np.cumsum(r))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 0.008, days)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.008, days)))
    prices = pd.DataFrame({"date_utc": dates.strftime("%Y-%m-%d"), "open": open_, "high": high,
                           "low": low, "close": close, "volume": rng.lognormal(23, 0.3, days),
                           "source": "demo"})

    rows, ts = [], []
    for t, d in enumerate(dates):
        n = rng.poisson(posts_per_day * (1 + 0.3 * abs(np.tanh(mood[t]))))
        p_bull = 1 / (1 + np.exp(-(1.2 * np.tanh(mood[t]) + 8 * r[t])))
        for j in range(n):
            u = rng.random()
            if u < 0.25:
                text = NEUTRAL[rng.integers(len(NEUTRAL))]
            elif rng.random() < p_bull:
                text = BULL[rng.integers(len(BULL))]
            else:
                text = BEAR[rng.integers(len(BEAR))]
            ts.append(d + pd.Timedelta(seconds=int(rng.integers(0, 86400))))
            rows.append({
                "tweet_id": f"demo-{t}-{j}", "source": "demo", "query": cfg.x_query,
                "text": f"{text} " + " ".join(rng.choice(FILLER, 3)), "lang": "en",
                "likes": int(rng.lognormal(2, 1.5)), "retweets": int(rng.lognormal(0.5, 1.3)),
                "replies": int(rng.lognormal(0.3, 1.0)), "quotes": int(rng.lognormal(0, 0.8)),
                "views": int(rng.lognormal(7, 1.5)), "author": f"user{rng.integers(5000)}",
                "author_followers": int(rng.lognormal(6.5, 2.0)), "author_verified": int(rng.random() < 0.1),
            })
    tweets = add_time_columns(pd.DataFrame(rows), pd.Series(ts))

    news = pd.DataFrame([{
        "url": f"https://example.com/demo-news-{i}", "title": title, "content": title,
        "score": 0.9 - 0.1 * i, "published_date": (end - pd.Timedelta(days=i)).strftime("%a, %d %b %Y"),
        "query": "demo", "fetched_at": end.isoformat(), "answer": "DEMO: synthetic news answer.",
    } for i, title in enumerate([
        "DEMO headline: Bitcoin ETF inflows rise for third week",
        "DEMO headline: Regulators weigh new crypto custody rules",
        "DEMO headline: Bitcoin slips as risk assets sell off",
        "DEMO headline: Miners report record hashrate after upgrade",
    ])])
    from ..sentiment import score_texts
    news["sent_compound"] = score_texts(news["title"].tolist(), engine="vader")["compound"].values

    con = connect(cfg.db_path)
    for t in ["tweets", "prices", "news", "scrape_runs"]:
        con.execute(f"DELETE FROM {t}")
    con.commit()
    upsert(con, "prices", prices)
    insert_ignore(con, "tweets", tweets)
    upsert(con, "news", news)
    con.close()
