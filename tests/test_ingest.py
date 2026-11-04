"""Ingestion with mocked HTTP: Apify actor runs, Tavily news, Coinbase prices, historical CSVs."""
import json
import re

import pandas as pd
import pytest

from xsent.config import APIDOJO_ACTOR, KAITO_ACTOR, SCWEET_ACTOR
from xsent.db import connect, read
from xsent.ingest import apify_x, historical, prices, tavily_news
from xsent.timeutil import parse_ts


# ------------------------------------------------------------------ parsing
def test_parse_x_and_iso_timestamps():
    ts = parse_ts(["Tue Oct 29 23:30:00 +0000 2024", "2024-10-29T23:30:00.000Z",
                   "2021-02-10 23:59:04", None, "garbage"])
    assert ts[0] == pd.Timestamp("2024-10-29 23:30", tz="UTC")
    assert ts[1] == ts[0]
    assert ts[2] == pd.Timestamp("2021-02-10 23:59:04", tz="UTC")
    assert ts[3] is pd.NaT and ts[4] is pd.NaT


def test_normalise_kaito_item_and_drop_mock():
    items = [
        {"type": "tweet", "id": "1850", "text": "Bitcoin breakout! $BTC", "createdAt":
         "Tue Oct 29 23:30:00 +0000 2024", "likeCount": 10, "retweetCount": 2, "replyCount": 1,
         "quoteCount": 0, "viewCount": 900, "lang": "en",
         "author": {"userName": "alice", "followers": 12000, "isBlueVerified": True}},
        {"type": "mock_tweet", "noResults": True},          # no text/createdAt -> dropped
    ]
    df = apify_x.normalise_items(items, "apify:x", "q")
    assert len(df) == 1
    r = df.iloc[0]
    assert r.tweet_id == "1850" and r.author == "alice" and r.author_followers == 12000
    assert r.date_utc == "2024-10-29"                      # UTC bucket
    assert r.created_at_est.startswith("2024-10-29T19:30")  # EDT = UTC-4 in October
    assert r.views == 900 and r.author_verified == 1


def test_normalise_scweet_shape():
    items = [{"tweet_id": "9", "text": "btc dump", "created_at": "2026-05-01T10:00:00Z",
              "favorite_count": 3, "retweet_count": 1, "view_count": 50,
              "user": {"username": "bob", "followers_count": 77}}]
    r = apify_x.normalise_items(items, "s", "q").iloc[0]
    assert (r.tweet_id, r.likes, r.author, r.author_followers) == ("9", 3, "bob", 77)


@pytest.mark.parametrize("actor", [KAITO_ACTOR, APIDOJO_ACTOR, SCWEET_ACTOR])
def test_build_input_templates(cfg, actor):
    cfg.apify_actor_id = actor
    s, e = pd.Timestamp("2026-05-01", tz="UTC"), pd.Timestamp("2026-05-02", tz="UTC")
    inp = apify_x.build_input(cfg, s, e)
    blob = json.dumps(inp)
    assert not re.search(r"\{[a-z_]+\}", blob)          # every placeholder filled
    assert "since_time:1777593600" in blob or "2026-05-01T00:00:00Z" in blob
    max_key = "max_items" if actor == SCWEET_ACTOR else "maxItems"
    assert inp[max_key] == cfg.tweets_per_window and isinstance(inp[max_key], int)


def test_windows_cover_lookback_without_today(cfg):
    cfg.lookback_days = 180
    end = pd.Timestamp("2026-11-03", tz="UTC")
    w = apify_x.windows(cfg, end)
    assert len(w) == 180 and w[-1][1] == end and w[0][0] == end - pd.Timedelta(days=180)


# ------------------------------------------------------------------ fake HTTP
class FakeResp:
    def __init__(self, data, status=200):
        self._d, self.status_code, self.text = data, status, json.dumps(data)[:300]

    @property
    def ok(self):
        return self.status_code < 400

    def json(self):
        return self._d

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


class FakeApify:
    """Mimics start run -> poll -> dataset items. Returns 3 tweets inside the run's window."""

    def __init__(self):
        self.headers, self.runs, self.calls = {}, {}, 0

    def post(self, url, json=None, params=None, timeout=None):
        self.calls += 1
        rid = f"run{len(self.runs)}"
        q = json.get("twitterContent", "")
        since = int(q.split("since_time:")[1].split()[0])
        self.runs[rid] = since
        return FakeResp({"data": {"id": rid, "status": "RUNNING", "defaultDatasetId": f"ds-{rid}"}})

    def get(self, url, params=None, timeout=None):
        if "/actor-runs/" in url:
            rid = url.rsplit("/", 1)[1]
            return FakeResp({"data": {"id": rid, "status": "SUCCEEDED", "defaultDatasetId": f"ds-{rid}"}})
        rid = url.split("/datasets/ds-")[1].split("/")[0]
        t0 = pd.Timestamp(self.runs[rid], unit="s", tz="UTC")
        items = [{"id": f"{rid}-{k}", "text": f"bitcoin looks bullish number {k} today",
                  "createdAt": (t0 + pd.Timedelta(hours=k + 1)).strftime("%a %b %d %H:%M:%S +0000 %Y"),
                  "likeCount": k, "retweetCount": 0, "author": {"userName": "u", "followers": 5}}
                 for k in range(3)]
        items.append({"id": "old", "text": "out of window mock tweet text here",
                      "createdAt": "Mon Jan 01 00:00:00 +0000 2018"})
        return FakeResp(items)


def test_scrape_is_windowed_and_resumable(cfg, monkeypatch):
    fake = FakeApify()
    monkeypatch.setattr(apify_x, "_session", lambda token: fake)
    cfg.apify_actor_id = KAITO_ACTOR
    n = apify_x.scrape(cfg)
    con = connect(cfg.db_path)
    tw = read(con, "SELECT * FROM tweets")
    runs = read(con, "SELECT * FROM scrape_runs")
    assert n == 9 and len(tw) == 9 and fake.calls == 3           # 3 days x 3 tweets
    assert "old" not in set(tw.tweet_id)                          # out-of-window item filtered
    assert set(runs.status) == {"SUCCEEDED"} and runs.run_id.notna().all()
    assert apify_x.scrape(cfg) == 0 and fake.calls == 3           # nothing re-scraped


def test_tavily_news(cfg, monkeypatch):
    payload = {"answer": "Bitcoin rose on ETF demand.", "results": [
        {"url": "https://a", "title": "Bitcoin surges on ETF inflows", "content": "rally", "score": 0.9,
         "published_date": "Mon, 02 Nov 2026 10:00:00 GMT"},
        {"url": "https://b", "title": "Bitcoin crash fears", "content": "selloff", "score": 0.5}]}
    monkeypatch.setattr(tavily_news.requests, "post", lambda *a, **k: FakeResp(payload))
    cfg.news_queries = ("q1", "q2")
    assert tavily_news.fetch_news(cfg) == 2
    news = read(connect(cfg.db_path), "SELECT * FROM news ORDER BY url")
    assert news.loc[0, "sent_compound"] > 0 > news.loc[1, "sent_compound"]


def test_coinbase_fallback(cfg, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("yahoo down")
    monkeypatch.setattr(prices, "_yfinance", boom)
    today = pd.Timestamp.now(tz="UTC").normalize()
    rows = [[int((today - pd.Timedelta(days=k)).timestamp()), 90, 110, 95, 100 + k, 5] for k in range(0, 10)]
    monkeypatch.setattr(prices.requests, "get", lambda *a, **k: FakeResp(rows))
    prices.fetch_prices(cfg, start=today - pd.Timedelta(days=10))
    p = read(connect(cfg.db_path), "SELECT * FROM prices ORDER BY date_utc")
    assert len(p) == 9                                  # today's still-forming candle excluded
    assert p["date_utc"].max() < today.strftime("%Y-%m-%d")


# ------------------------------------------------------------------ historical CSVs
def test_historical_kaggle_comma_format(cfg, tmp_path):
    csv = tmp_path / "Bitcoin_tweets.csv"
    pd.DataFrame({
        "user_name": ["a", "b", "c"], "user_followers": ["100", "2000", "x"],
        "user_verified": ["False", "True", "False"],
        "date": ["2021-02-10 23:59:04", "2021-02-11 00:01:00", "bad date"],
        "text": ["Bitcoin to the moon", "BTC dump incoming", "ignored"],
    }).to_csv(csv, index=False)
    assert historical.load_csv(cfg, csv) == 2
    t = read(connect(cfg.db_path), "SELECT * FROM tweets ORDER BY date_utc")
    assert list(t.date_utc) == ["2021-02-10", "2021-02-11"] and t.author_verified.tolist() == [0, 1]


def test_historical_semicolon_format_with_per_day_cap(cfg, tmp_path):
    csv = tmp_path / "tweets.csv"
    lines = ["id;user;fullname;url;timestamp;replies;likes;retweets;text"]
    for k in range(5):
        lines.append(f"{k};u{k};U;;2018-03-01 1{k}:00:00+00;0;{k};0;bitcoin tweet number {k}")
    csv.write_text("\n".join(lines))
    assert historical.load_csv(cfg, csv, per_day=2) == 2
    t = read(connect(cfg.db_path), "SELECT tweet_id FROM tweets")
    assert set(t.tweet_id) == {"3", "4"}                 # top-2 by engagement kept
