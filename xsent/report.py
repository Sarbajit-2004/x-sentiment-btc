"""Evaluation report (Markdown, optional PDF) with model metrics, Sharpe / Sortino / Max Drawdown."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .charts import MODEL_LABEL
from .db import connect, read
from .features import PRICE_FEATURES, SENT_FEATURES


def _f(x, kind="num"):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    if kind == "pct":
        return f"{x * 100:+.1f}%"
    if kind == "pct0":
        return f"{x * 100:.1f}%"
    if kind == "int":
        return f"{int(x):,}"
    if kind == "usd":
        return f"${x:,.0f}"
    return f"{x:.2f}"


def _table(df: pd.DataFrame, cols: list[tuple[str, str, str]]) -> str:
    head = "| " + " | ".join(c[1] for c in cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    body = []
    for _, r in df.iterrows():
        cells = []
        for key, _, kind in cols:
            v = r[key]
            if kind == "model":
                v = MODEL_LABEL.get(v, v)
            elif kind != "str":
                v = _f(v, kind)
            cells.append(str(v))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, sep] + body)


def data_summary(cfg) -> dict:
    con = connect(cfg.db_path)
    t = read(con, "SELECT source, COUNT(*) n, SUM(is_spam) spam, MIN(date_utc) d0, MAX(date_utc) d1, "
                  "COUNT(DISTINCT date_utc) days FROM tweets GROUP BY source")
    p = read(con, "SELECT COUNT(*) n, MIN(date_utc) d0, MAX(date_utc) d1, MAX(source) src FROM prices")
    # a window that failed once and succeeded on a retry counts as succeeded
    runs = read(con, """SELECT actor_id, status, COUNT(*) n, SUM(n_items) items FROM (
                          SELECT actor_id, window_start, MAX(n_items) n_items,
                                 CASE WHEN SUM(status='SUCCEEDED') > 0 THEN 'SUCCEEDED'
                                      ELSE MAX(status) END status
                          FROM scrape_runs GROUP BY actor_id, window_start)
                        GROUP BY actor_id, status""")
    news = read(con, "SELECT COUNT(*) n FROM news")["n"].iloc[0]
    con.close()
    return {"tweets": t, "prices": p, "runs": runs, "news": int(news)}


def write_report(cfg, fm: pd.DataFrame, bt: pd.DataFrame, live: pd.DataFrame, charts: dict,
                 eval_range: tuple[str, str], demo: bool = False) -> str:
    ds = data_summary(cfg)
    pm = cfg.primary_model
    prim_bt = bt[(bt.model == pm)].sort_values("h")
    bh = bt[bt.model == "buy_and_hold"].iloc[0]
    b0 = prim_bt[prim_bt.h == 0].iloc[0] if (prim_bt.h == 0).any() else None
    f_prim = fm[fm.model == pm].sort_values("h")
    f_rw = fm[fm.model == "rw"].set_index("h")

    md = []
    if demo:
        md.append("> **DEMO REPORT — produced from synthetic offline data to verify the pipeline end to end. "
                  "These numbers say nothing about Bitcoin. Run `python run_pipeline.py all` with your API "
                  "keys to produce the real report.**\n")
    md.append(f"# {cfg.asset_name} X-Sentiment Price Prediction — Evaluation Report\n")
    md.append(f"Out-of-sample evaluation window: **{eval_range[0]} → {eval_range[1]}** (UTC daily bars). "
              f"Primary model chosen before testing: **{MODEL_LABEL.get(pm, pm)}**. "
              f"Transaction cost {cfg.cost_bps:.0f} bp per unit of turnover; risk-free rate 0; "
              "annualisation √365.\n")

    # ---------------------------------------------------------- headline
    md.append("## 1. Headline results\n")
    if b0 is not None:
        md.append(f"- **Trading the n+0 forecast (long/flat):** Sharpe **{_f(b0.sharpe)}**, "
                  f"Sortino **{_f(b0.sortino)}**, Max drawdown **{_f(b0.max_drawdown, 'pct')}**, "
                  f"total return {_f(b0.total_return, 'pct')} with {_f(b0.exposure, 'pct0')} time in market.")
    md.append(f"- **Buy & hold over the same days:** Sharpe {_f(bh.sharpe)}, Sortino {_f(bh.sortino)}, "
              f"Max drawdown {_f(bh.max_drawdown, 'pct')}, total return {_f(bh.total_return, 'pct')}.")
    if len(f_prim):
        ratios = [r.rmse_ret_bp / f_rw.loc[r.h, "rmse_ret_bp"] for r in f_prim.itertuples()]
        md.append(f"- **Forecast skill vs random walk:** RMSE ratio ranges {min(ratios):.3f}–{max(ratios):.3f} "
                  f"across n+0…n+5 (<1 beats the random walk); directional accuracy "
                  f"{f_prim.dir_acc.min() * 100:.1f}%–{f_prim.dir_acc.max() * 100:.1f}%.")
        sig = f_prim[(f_prim.dm_p < 0.05) & (f_prim.dm_stat_vs_rw < 0)]
        worse = f_prim[(f_prim.dm_p < 0.05) & (f_prim.dm_stat_vs_rw > 0)]
        md.append(f"- **Statistical significance:** the Diebold–Mariano test rejects 'no better than random "
                  f"walk' at 5% for {len(sig)} of {len(f_prim)} horizons"
                  + (f" ({', '.join(sig.horizon)})" if len(sig) else "")
                  + f"; it finds the model significantly *worse* than the random walk at {len(worse)}"
                  + (f" ({', '.join(worse.horizon)})." if len(worse) else "."))
    abl = fm[fm.model.isin(["ridge", "ridge_price"])].pivot(index="h", columns="model", values="rmse_ret_bp")
    if {"ridge", "ridge_price"} <= set(abl.columns):
        better = int((abl["ridge"] < abl["ridge_price"]).sum())
        md.append(f"- **Does X sentiment add information?** Adding sentiment features to the same Ridge model "
                  f"lowered out-of-sample error at {better} of {len(abl)} horizons (ablation vs price-only).")
    md.append("")

    # ---------------------------------------------------------- data
    md.append("## 2. Data\n")
    t = ds["tweets"]
    md.append("| Source | Posts | Flagged spam | Days | First | Last |\n|---|---|---|---|---|---|")
    for _, r in t.iterrows():
        md.append(f"| `{r.source}` | {_f(r.n, 'int')} | {_f(r.spam or 0, 'int')} | {_f(r.days, 'int')} | "
                  f"{r.d0} | {r.d1} |")
    p = ds["prices"].iloc[0]
    md.append(f"\nPrices: {_f(p.n, 'int')} daily bars ({p.d0} → {p.d1}), source `{p.src}`, "
              f"ticker `{cfg.price_ticker}`. News: {ds['news']} Tavily articles.\n")
    if not ds["runs"].empty:
        md.append("Apify runs:\n")
        for _, r in ds["runs"].iterrows():
            md.append(f"- actor `{r.actor_id}` — {r.status}: {_f(r.n, 'int')} windows, "
                      f"{_f(r.get('items') or 0, 'int')} posts")
        md.append("")
    md.append(f"X query: `{cfg.x_query}` (ticker, cashtag and keyword in one advanced search), language "
              f"`{cfg.apify_lang}`, {cfg.window_hours}-hour windows capped at {cfg.tweets_per_window} posts, "
              f"sort `{cfg.apify_query_type}`. Posts are bucketed by UTC date to match the 00:00–24:00 UTC "
              "daily bar; an America/New_York timestamp is stored alongside for reference.\n")

    # ---------------------------------------------------------- method
    md.append("## 3. Method\n")
    md.append("**Timing (no look-ahead).** Row *n* is the current day; the forecast is made at its 00:00 UTC "
              "open using posts and prices up to the close of day *n−1* only. Targets are "
              "`log(close[n+h] / close[n−1])` for h = 0…5, so *n+0* is today's close and *n+5* the close five "
              "days later. Forecast prices are `close[n−1] · exp(ŷ)`.\n")
    md.append("**Sentiment.** VADER (built for social-media text) extended with a crypto lexicon "
              "(moon, rekt, HODL, FUD, dump, ETF inflows/outflows, …). URLs and @mentions are stripped; "
              f"cashtag-stuffed shill posts (> {cfg.max_cashtags} cashtags), same-day duplicates and near-empty "
              "posts, and giveaway / engagement-farming posts (\"repost & reply with your BTC address\") "
              "are flagged as spam and excluded.\n")
    md.append(f"**Features — sentiment ({len(SENT_FEATURES)}):** engagement-weighted score and its rolling "
              "30-day z-score, follower-weighted and influencer (≥10k followers) z-scores, bull−bear share, "
              "dispersion, 1-day change, 3/7-day EMA momentum, 7-day mean, post-volume and engagement z-scores, "
              f"sentiment×engagement. **Price ({len(PRICE_FEATURES)}):** 1/2/3-day lagged returns, 5/10/20-day "
              "momentum, 7/20-day realised volatility, RSI-14, gap to the 20-day mean, high-low range, "
              "day-of-week. No traded volume, on-chain or macro inputs.\n")
    md.append("**Models.** One model per horizon: Ridge (regularisation chosen by time-ordered CV with a "
              "gap of h days) and depth-2 gradient-boosted trees on "
              "sentiment + price, their average (the *ensemble*, fixed in advance as primary), a price-only "
              "Ridge as an ablation, and random-walk / drift baselines.\n")
    md.append(f"**Walk-forward validation.** Expanding window, minimum {cfg.min_train_days} training days, "
              f"refit every {cfg.refit_every} day(s). For horizon *h*, training rows are purged so every "
              "training target is already realised at the forecast time (row *i* is used only if "
              "*i + h ≤ n − 1*). Every prediction in this report is out of sample.\n")
    md.append("**Backtest.** At each open, go long if the forecast log-return > "
              f"{cfg.signal_threshold:g}, otherwise flat{' (or short)' if cfg.allow_short else ''}; hold one "
              "day; pay costs on every change of position. Each horizon's forecast is traded as its own "
              "signal so positions never overlap.\n")

    # ---------------------------------------------------------- results
    md.append("## 4. Trading performance (out of sample, after costs)\n")
    show = bt[bt.model.isin([pm, "ridge_price", "buy_and_hold"])].copy()
    show["sig"] = show.apply(lambda r: f"{MODEL_LABEL.get(r.model, r.model)}"
                             + ("" if r.horizon == "-" else f" · {r.horizon}"), axis=1)
    md.append(_table(show, [("sig", "Signal", "str"), ("sharpe", "Sharpe", "num"),
                            ("sortino", "Sortino", "num"), ("max_drawdown", "Max DD", "pct"),
                            ("total_return", "Total return", "pct"), ("cagr", "CAGR", "pct"),
                            ("ann_vol", "Ann. vol", "pct0"), ("calmar", "Calmar", "num"),
                            ("exposure", "Exposure", "pct0"), ("hit_rate", "Hit rate", "pct0"),
                            ("trades", "Trades", "int")]))
    md.append(f"\n![Equity curve]({charts['paths']['equity_curve'].relative_to(cfg.out_dir).as_posix()})\n")

    md.append("## 5. Forecast accuracy by horizon\n")
    fmx = fm[fm.model.isin([pm, "ridge", "gbm", "ridge_price", "rw"])].copy()
    fmx["rmse_ratio"] = fmx.apply(lambda r: r.rmse_ret_bp / f_rw.loc[r.h, "rmse_ret_bp"], axis=1)
    md.append(_table(fmx.sort_values(["h", "model"]),
                     [("horizon", "Horizon", "str"), ("model", "Model", "model"), ("n", "N", "int"),
                      ("mape_pct", "MAPE %", "num"), ("rmse_price", "RMSE $", "usd"),
                      ("rmse_ratio", "RMSE ÷ RW", "num"), ("dir_acc", "Dir. acc.", "pct0"),
                      ("dir_p", "p (dir)", "num"), ("ic_spearman", "IC", "num"),
                      ("dm_stat_vs_rw", "DM stat", "num"), ("dm_p", "DM p", "num")]))
    md.append("\n*RMSE ÷ RW < 1 means lower error than the random walk. DM = Diebold–Mariano test on "
              "squared errors vs random walk (Newey–West, lag h); negative = model better. "
              "p (dir) = one-sided binomial test of the hit rate against 50%.*\n")
    for k in ["skill_by_horizon", "forecast_vs_actual"]:
        md.append(f"![{k}]({charts['paths'][k].relative_to(cfg.out_dir).as_posix()})\n")

    md.append("## 6. Sentiment vs price\n")
    for k in ["sentiment_vs_price", "correlation_matrix", "rolling_correlation"]:
        md.append(f"![{k}]({charts['paths'][k].relative_to(cfg.out_dir).as_posix()})\n")

    md.append("## 7. Live forecast (from the latest close)\n")
    if not live.empty:
        md.append(_table(live, [("horizon", "Horizon", "str"), ("target_date", "Date", "str"),
                                ("pred_close", "Forecast", "usd"), ("lo80", "80% low", "usd"),
                                ("hi80", "80% high", "usd"), ("rw_close", "Random walk", "usd")]))
    md.append("")

    md.append("## 8. Limitations\n")
    n_days = int(bh["days"])
    se = (365 / max(n_days, 1)) ** 0.5
    md.append(f"- Only {n_days} out-of-sample trading days: an annualised Sharpe ratio estimated on that "
              f"sample has a standard error of roughly ±{se:.1f} (≈√(365/N)), so a Sharpe ratio within "
              f"about ±{1.96 * se:.1f} of zero (or of buy & hold) is not statistically distinguishable from it.\n"
              "- Each day's sentiment is a capped sample of posts (top or latest), not the full X firehose; "
              "post-count features therefore measure sampling more than true volume.\n"
              "- Lexicon sentiment misses sarcasm and context; FinBERT and CryptoBERT are available via "
              "`SENTIMENT_ENGINE=finbert` or `cryptobert` for comparison.\n"
              "- Results depend on one asset and one period; no hyper-parameters were tuned on the test window.\n")
    if (cfg.out_dir / "ADDENDUM_VARIANTS.md").exists():
        md.append("**Variants:** a run trained with extra 2021–23 Kaggle tweets, the check that ruled out the "
                  "re-dated \"2025–2026\" Kaggle set, and a CryptoBERT sentiment run, each compared on the same "
                  "test days, are in `ADDENDUM_VARIANTS.md` / `.pdf`.\n")
    md.append("## 9. Reproduce\n")
    md.append("```bash\npip install -r requirements.txt\ncp .env.example .env   # add APIFY_TOKEN and TAVILY_API_KEY\n"
              "python run_pipeline.py all\n```\n")
    text = "\n".join(md)
    (cfg.out_dir / "REPORT.md").write_text(text, encoding="utf-8")
    metrics = {
        "eval_start": eval_range[0], "eval_end": eval_range[1], "primary_model": pm,
        "strategy_n0": None if b0 is None else {k: float(b0[k]) for k in
                                                ["sharpe", "sortino", "max_drawdown", "total_return"]},
        "buy_and_hold": {k: float(bh[k]) for k in ["sharpe", "sortino", "max_drawdown", "total_return"]},
    }
    (cfg.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return text


def to_pdf(md_path, pdf_path) -> bool:
    """Best-effort Markdown -> PDF (needs `markdown` + `playwright`). Returns False if unavailable."""
    try:
        import markdown
        from playwright.sync_api import sync_playwright
    except ImportError:
        return False
    html = markdown.markdown(md_path.read_text(encoding="utf-8"), extensions=["tables", "fenced_code"])
    css = ("body{font-family:Segoe UI,Helvetica,Arial,sans-serif;font-size:11px;max-width:900px;margin:auto;"
           "color:#0b0b0b}table{border-collapse:collapse;font-size:9.5px;margin:8px 0}td,th{border:1px solid "
           "#e1e0d9;padding:3px 6px;text-align:right}th{background:#f0efec}td:first-child,th:first-child"
           "{text-align:left}img{max-width:100%}blockquote{border-left:3px solid #eb6834;margin:0;"
           "padding:4px 10px;background:#fdf3ee}h1{font-size:20px}h2{font-size:15px;margin-top:18px}")
    page = f"<html><head><meta charset='utf-8'><style>{css}</style></head><body>{html}</body></html>"
    tmp = md_path.with_suffix(".html")
    tmp.write_text(page, encoding="utf-8")
    try:
        with sync_playwright() as p:
            b = p.chromium.launch()
            pg = b.new_page()
            pg.goto(tmp.resolve().as_uri())
            pg.pdf(path=str(pdf_path), format="A4", margin={"top": "12mm", "bottom": "12mm",
                                                            "left": "10mm", "right": "10mm"})
            b.close()
    except Exception:
        return False
    finally:
        tmp.unlink(missing_ok=True)
    return True
