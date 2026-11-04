"""One-page sentiment outlook (Markdown) built from the latest X data, Tavily news and the live forecast."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .db import connect, read


def _pct(x: float) -> str:
    return "n/a" if pd.isna(x) else f"{x * 100:+.1f}%"


def _regime(z: float) -> str:
    if pd.isna(z):
        return "Unknown"
    if z >= 1.0:
        return "Bullish (stretched)"
    if z >= 0.35:
        return "Mildly bullish"
    if z > -0.35:
        return "Neutral"
    if z > -1.0:
        return "Mildly bearish"
    return "Bearish (washed-out)"


def _short(t: str, n: int = 150) -> str:
    t = " ".join(str(t).split())
    t = t.replace("|", "/")
    return t if len(t) <= n else t[: n - 1] + "…"


def build_outlook(cfg, daily: pd.DataFrame, live: pd.DataFrame, demo: bool = False) -> str:
    con = connect(cfg.db_path)
    d = daily.sort_values("date_utc").reset_index(drop=True)
    dx = d[d["has_x"].astype(bool)]
    asof = d["date_utc"].iloc[-1]
    last = d["close"].iloc[-1]
    r7 = last / d["close"].iloc[-8] - 1 if len(d) > 8 else np.nan
    r30 = last / d["close"].iloc[-31] - 1 if len(d) > 31 else np.nan

    s = dx["sent_ew"]
    s7, s_prev7 = s.tail(7).mean(), s.iloc[-14:-7].mean() if len(s) >= 14 else np.nan
    hist = s.rolling(7).mean().dropna()
    z = (s7 - hist.mean()) / hist.std() if len(hist) > 10 and hist.std() > 0 else np.nan
    pctile = (hist < s7).mean() if len(hist) else np.nan
    bull7, bear7 = dx["bull_share"].tail(7).mean(), dx["bear_share"].tail(7).mean()
    eng = np.log1p(dx["eng_total"])
    eng_z = (eng.tail(7).mean() - eng.mean()) / eng.std() if eng.std() > 0 else np.nan
    n_posts = int(dx["n_posts"].sum())
    regime = _regime(z)

    wk_start = (pd.Timestamp(asof) - pd.Timedelta(days=6)).strftime("%Y-%m-%d")
    top = read(con, """SELECT text, author, author_followers, likes, retweets, sent_compound
                       FROM tweets WHERE date_utc >= ? AND is_spam = 0 AND sent_compound IS NOT NULL""",
               (wk_start,))
    top["eng"] = top["likes"] + 2 * top["retweets"]
    bulls = top[top.sent_compound >= 0.3].nlargest(3, "eng")
    bears = top[top.sent_compound <= -0.3].nlargest(3, "eng")
    news = read(con, "SELECT title, url, published_date, sent_compound, answer, score FROM news "
                     "ORDER BY fetched_at DESC, score DESC")
    con.close()

    # ---- narrative (rule-based, so it is reproducible) ------------------
    lines = []
    dir_s = "improved" if s7 > s_prev7 else "deteriorated"
    if not pd.isna(s_prev7):
        lines.append(f"Crowd tone on X has **{dir_s}** week-on-week (7-day engagement-weighted score "
                     f"{s7:+.3f} vs {s_prev7:+.3f}), placing it at the **{pctile * 100:.0f}th percentile** "
                     f"of the last {len(hist)} days.")
    if not pd.isna(r7):
        agree = (r7 > 0) == (s7 > s_prev7)
        lines.append(f"Price moved {_pct(r7)} over the same week, so sentiment and price are "
                     f"**{'moving together' if agree else 'diverging'}**"
                     + ("." if agree else " — a divergence worth watching."))
    if not pd.isna(eng_z):
        lines.append(f"Attention (engagement) is {'above' if eng_z > 0 else 'below'} its average "
                     f"(z = {eng_z:+.2f}); {'louder crowds amplify the sentiment signal' if eng_z > 0 else 'a quiet tape makes the sentiment reading less reliable'}.")
    if not live.empty:
        p1 = live.iloc[min(1, len(live) - 1)]
        p5 = live.iloc[-1]
        lines.append(f"The model's central path is **{_pct(p1.pred_close / p1.ref_close - 1)}** by "
                     f"{p1.horizon} and **{_pct(p5.pred_close / p5.ref_close - 1)}** by {p5.horizon}; "
                     f"the 80% band at {p5.horizon} is {p5.lo80:,.0f} – {p5.hi80:,.0f}, i.e. the "
                     "uncertainty is much larger than the predicted move.")

    md = []
    if demo:
        md.append("> **DEMO OUTPUT — synthetic data generated offline to test the pipeline. Not a real market view.**\n")
    md.append(f"# {cfg.asset_name} — X Sentiment Outlook\n")
    md.append(f"*As of the {asof} UTC daily close · {n_posts:,} filtered X posts over "
              f"{len(dx)} days · news via Tavily*\n")
    md.append(f"## Verdict: **{regime}**\n")
    md.append("\n".join(f"- {l}" for l in lines) + "\n")
    md.append("| Metric | Value |\n|---|---|")
    md.append(f"| Last close | ${last:,.0f} |")
    md.append(f"| 7-day / 30-day return | {_pct(r7)} / {_pct(r30)} |")
    md.append(f"| 7-day sentiment (engagement-weighted) | {s7:+.3f} (z = {z:+.2f}) |")
    md.append(f"| Bullish / bearish share of posts (7d) | {bull7 * 100:.0f}% / {bear7 * 100:.0f}% |")
    md.append(f"| Engagement vs average (z) | {eng_z:+.2f} |\n")
    if not live.empty:
        from .charts import MODEL_LABEL
        md.append(f"**Model forecast — {MODEL_LABEL.get(cfg.primary_model, cfg.primary_model)}, "
                  "sentiment + price**\n")
        md.append("| Horizon | Date | Forecast | 80% band |\n|---|---|---|---|")
        for _, r in live.iterrows():
            md.append(f"| {r.horizon} | {r.target_date} | ${r.pred_close:,.0f} "
                      f"({_pct(r.pred_close / r.ref_close - 1)}) | ${r.lo80:,.0f} – ${r.hi80:,.0f} |")
        md.append("")
    if len(bulls) or len(bears):
        md.append("**Most-engaged posts this week**\n")
        for lab, g in (("Bull", bulls), ("Bear", bears)):
            for _, r in g.iterrows():
                md.append(f"- *{lab}* · @{r.author} ({int(r.author_followers):,} followers, "
                          f"{int(r.likes):,} likes): “{_short(r.text)}”")
        md.append("")
    if not news.empty:
        md.append("**News backdrop (Tavily)**\n")
        ans = news["answer"].dropna()
        if len(ans):
            md.append(f"> {_short(ans.iloc[0], 400)}\n")
        for _, r in news.drop_duplicates("title").head(5).iterrows():
            tone = "positive" if r.sent_compound > 0.05 else "negative" if r.sent_compound < -0.05 else "neutral"
            md.append(f"- [{_short(r.title, 110)}]({r.url}) — {r.published_date or 'recent'} · tone {tone}")
        md.append("")
    md.append("*Sentiment is a noisy, crowd-driven signal; the forecast band shows how little of the next "
              "week is predictable. This is research output, not investment advice.*")
    return "\n".join(md)
