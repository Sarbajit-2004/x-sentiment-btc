"""Daily aggregation, X-sentiment/price join, and leakage-safe features.

Timing convention (the most important design decision in the project)
--------------------------------------------------------------------
Row `n` is the CURRENT day (UTC). We stand at its open (00:00 UTC), so we know
every post and every price bar up to the close of day n-1, and nothing after.

  * features[n] are built only from days <= n-1           (every input is shift(1))
  * target y_h[n] = log(close[n+h] / close[n-1]),  h = 0..5
  * predicted price for n+h = close[n-1] * exp(y_hat_h)

So 'n+0' is a genuine forecast of today's close, and n+5 is the close five
days after today. Inputs are sentiment and price only (no trading volume, no
on-chain or macro data), as the brief requires.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import Config
from .db import connect, read, replace_table

log = logging.getLogger(__name__)

SENT_FEATURES = [
    "s_ew", "s_ew_z", "s_mean_z", "s_fw_z", "s_infl_z", "s_bb", "s_std",
    "s_ew_d1", "s_mom", "s_ew_ma7", "vol_z", "eng_z", "s_x_eng",
]
PRICE_FEATURES = [
    "r_l1", "r_l2", "r_l3", "ret_5", "ret_10", "ret_20", "rv_7", "rv_20",
    "rsi_14", "ma_gap_20", "hl_l1", "dow_sin", "dow_cos",
]


# -------------------------------------------------------------------------
# 1. daily X aggregates
# -------------------------------------------------------------------------
def daily_sentiment(tweets: pd.DataFrame) -> pd.DataFrame:
    t = tweets[(tweets["is_spam"].fillna(0) == 0) & tweets["sent_compound"].notna()].copy()
    if t.empty:
        return pd.DataFrame(columns=["date_utc"])
    eng = t["likes"] + 2 * t["retweets"] + t["replies"] + t["quotes"]
    t["w_eng"] = 1 + np.log1p(eng.clip(lower=0))
    t["w_fol"] = 1 + np.log1p(t["author_followers"].clip(lower=0))
    t["eng"] = eng
    t["bull"] = (t["sent_compound"] >= 0.05).astype(float)
    t["bear"] = (t["sent_compound"] <= -0.05).astype(float)
    t["infl"] = t["author_followers"] >= 10_000

    def agg(g: pd.DataFrame) -> pd.Series:
        c = g["sent_compound"]
        infl = g.loc[g["infl"], "sent_compound"]
        return pd.Series({
            "n_posts": len(g),
            "n_authors": g["author"].nunique(),
            "sent_mean": c.mean(),
            "sent_ew": np.average(c, weights=g["w_eng"]),
            "sent_fw": np.average(c, weights=g["w_fol"]),
            "sent_infl": infl.mean() if len(infl) >= 3 else np.nan,
            "sent_std": c.std(ddof=0),
            "bull_share": g["bull"].mean(),
            "bear_share": g["bear"].mean(),
            "eng_total": float(g["eng"].sum()),
            "views_total": float(g["views"].sum()),
        })

    d = t.groupby("date_utc").apply(agg, include_groups=False).reset_index()
    d["sent_infl"] = d["sent_infl"].fillna(d["sent_fw"])
    return d.sort_values("date_utc")


def build_daily(cfg: Config) -> pd.DataFrame:
    """Join daily X aggregates onto the (continuous) daily price calendar and store it."""
    con = connect(cfg.db_path)
    tweets = read(con, "SELECT date_utc, text, likes, retweets, replies, quotes, views, author, "
                       "author_followers, sent_compound, is_spam FROM tweets")
    prices = read(con, "SELECT date_utc, open, high, low, close FROM prices ORDER BY date_utc")
    if prices.empty:
        raise SystemExit("No prices in the database - run the 'prices' step first.")
    sent = daily_sentiment(tweets)
    daily = prices.merge(sent, on="date_utc", how="left")
    daily["has_x"] = daily["n_posts"].fillna(0) >= 10
    replace_table(con, "daily", daily)
    con.close()
    log.info("daily table: %d days, %d with X coverage", len(daily), int(daily["has_x"].sum()))
    return daily


# -------------------------------------------------------------------------
# 2. features + targets
# -------------------------------------------------------------------------
def _rz(x: pd.Series, win: int = 30, minp: int = 10) -> pd.Series:
    """Rolling z-score using only the window up to and including the row."""
    m = x.rolling(win, min_periods=minp).mean()
    s = x.rolling(win, min_periods=minp).std()
    return ((x - m) / s.replace(0, np.nan)).fillna(0.0)


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def make_features(daily: pd.DataFrame, horizons=(0, 1, 2, 3, 4, 5)) -> pd.DataFrame:
    df = daily.sort_values("date_utc").drop_duplicates("date_utc").copy()
    # put rows on a gap-free daily calendar, so "shift(1)" always means "the previous day"
    full = pd.date_range(df["date_utc"].min(), df["date_utc"].max(), freq="D").strftime("%Y-%m-%d")
    df = df.set_index("date_utc").reindex(full).rename_axis("date_utc").reset_index()
    df["date"] = pd.to_datetime(df["date_utc"])

    # ---- day-level quantities (known at the close of that day) ----------
    lc = np.log(df["close"])
    r = lc.diff()                                   # return of day d
    day = pd.DataFrame(index=df.index)
    day["r"] = r
    day["ret_5"] = r.rolling(5).sum()
    day["ret_10"] = r.rolling(10).sum()
    day["ret_20"] = r.rolling(20).sum()
    day["rv_7"] = r.rolling(7).std()
    day["rv_20"] = r.rolling(20).std()
    day["rsi_14"] = _rsi(df["close"]) / 100 - 0.5
    day["ma_gap_20"] = lc - np.log(df["close"].rolling(20).mean())
    day["hl"] = np.log(df["high"] / df["low"])

    # sentiment: computed on covered days only, then put back on the calendar
    cov = df["has_x"].astype("boolean").fillna(False).astype(bool)
    s = df.loc[cov].copy()
    sd = pd.DataFrame(index=s.index)
    if len(s):
        sd["s_ew"] = s["sent_ew"]
        sd["s_ew_z"] = _rz(s["sent_ew"])
        sd["s_mean_z"] = _rz(s["sent_mean"])
        sd["s_fw_z"] = _rz(s["sent_fw"])
        sd["s_infl_z"] = _rz(s["sent_infl"])
        sd["s_bb"] = s["bull_share"] - s["bear_share"]
        sd["s_std"] = s["sent_std"]
        sd["s_ew_d1"] = s["sent_ew"].diff()
        sd["s_mom"] = s["sent_ew"].ewm(span=3).mean() - s["sent_ew"].ewm(span=7).mean()
        sd["s_ew_ma7"] = s["sent_ew"].rolling(7, min_periods=3).mean()
        sd["vol_z"] = _rz(np.log1p(s["n_posts"]))
        sd["eng_z"] = _rz(np.log1p(s["eng_total"]))
        sd["s_x_eng"] = sd["s_ew_z"] * sd["eng_z"]
    for c in SENT_FEATURES:
        day[c] = sd[c] if c in sd else np.nan
    day["cov"] = cov

    # ---- shift everything by one day: row n sees only days <= n-1 -------
    feat = pd.DataFrame({"date_utc": df["date_utc"], "date": df["date"]})
    lag = day.shift(1)
    feat["r_l1"] = lag["r"]
    feat["r_l2"] = day["r"].shift(2)
    feat["r_l3"] = day["r"].shift(3)
    for c in ["ret_5", "ret_10", "ret_20", "rv_7", "rv_20", "rsi_14", "ma_gap_20"]:
        feat[c] = lag[c]
    feat["hl_l1"] = lag["hl"]
    dow = df["date"].dt.day_of_week          # calendar of day n is known in advance
    feat["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    feat["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    for c in SENT_FEATURES:
        feat[c] = lag[c]
    feat["x_cov_l1"] = day["cov"].astype(float).shift(1).fillna(0).astype(bool)

    # ---- reference price and targets --------------------------------------
    feat["ref_close"] = df["close"].shift(1)          # last known close (day n-1)
    feat["close"] = df["close"]
    for h in horizons:
        feat[f"y_{h}"] = lc.shift(-h) - lc.shift(1)   # log(close[n+h] / close[n-1])
        feat[f"close_{h}"] = df["close"].shift(-h)
    feat["r_today"] = r                                # realised return of day n (backtest P&L)
    return feat


def build_features(cfg: Config) -> pd.DataFrame:
    daily = build_daily(cfg)
    feat = make_features(daily, cfg.horizons)
    con = connect(cfg.db_path)
    replace_table(con, "features", feat.drop(columns=["date"]))
    con.close()
    return feat
