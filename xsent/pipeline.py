"""Orchestrates: ingest -> sentiment -> features -> walk-forward -> backtest -> charts -> reports."""
from __future__ import annotations

import logging

import pandas as pd

from . import charts
from .backtest import backtest, forecast_metrics
from .config import Config
from .dashboard import build_dashboard
from .db import connect, replace_table
from .features import build_features
from .model import live_forecast, walk_forward
from .outlook import build_outlook
from .report import to_pdf, write_report
from .sentiment import score_db

log = logging.getLogger(__name__)


def analyse(cfg: Config, demo: bool = False, pdf: bool = True) -> dict:
    cfg.ensure_dirs()
    score_db(cfg)
    feat = build_features(cfg)
    con = connect(cfg.db_path)
    daily = pd.read_sql_query("SELECT * FROM daily ORDER BY date_utc", con)
    con.close()

    # evaluation window: the most recent EVAL_DAYS days. Anything older (e.g. historical
    # Kaggle tweets) is used for training only.
    last = pd.Timestamp(feat.loc[feat["y_0"].notna(), "date_utc"].max())
    start = (last - pd.Timedelta(days=cfg.eval_days - 1)).strftime("%Y-%m-%d")
    oos = walk_forward(feat, cfg, test_start=start)
    if oos.empty:
        raise SystemExit("Not enough overlapping X + price history to run walk-forward. "
                         f"Need > {cfg.min_train_days} covered days.")
    ev = oos.copy()
    fm = forecast_metrics(ev)
    bt, curves = backtest(ev, cfg.cost_bps, cfg.signal_threshold, cfg.allow_short)
    live = live_forecast(feat, daily, oos, cfg)

    out = cfg.out_dir
    oos.to_csv(out / "oos_predictions.csv", index=False)
    fm.to_csv(out / "forecast_metrics.csv", index=False)
    bt.to_csv(out / "backtest_metrics.csv", index=False)
    live.to_csv(out / "live_forecast.csv", index=False)
    con = connect(cfg.db_path)
    replace_table(con, "predictions", oos)
    con.close()

    ch = charts.make_all(daily, feat, ev, fm, curves, out, cfg.asset_name, cfg.primary_model)
    ch["corr_matrix"].to_csv(out / "sentiment_return_matrix.csv")
    eval_range = (ev["date_utc"].min(), ev["date_utc"].max())
    write_report(cfg, fm, bt, live, ch, eval_range, demo=demo)
    tiles = []
    b0 = bt[(bt.model == cfg.primary_model) & (bt.h == 0)]
    bh = bt[bt.model == "buy_and_hold"]
    if len(b0):
        b0 = b0.iloc[0]
        tiles = [["Last close", f"${daily['close'].iloc[-1]:,.0f}"],
                 ["Strategy Sharpe (n+0)", f"{b0.sharpe:.2f}"],
                 ["Strategy Sortino", f"{b0.sortino:.2f}"],
                 ["Strategy max DD", f"{b0.max_drawdown * 100:.1f}%"],
                 ["Buy & hold Sharpe", f"{bh.iloc[0].sharpe:.2f}" if len(bh) else "–"]]
    build_dashboard(daily, live, out / "dashboard.html", cfg.asset_name, demo=demo, tiles=tiles)
    outlook = build_outlook(cfg, daily, live, demo=demo)
    (out / "SENTIMENT_OUTLOOK.md").write_text(outlook, encoding="utf-8")
    if pdf:
        for name in ["REPORT", "SENTIMENT_OUTLOOK"]:
            ok = to_pdf(out / f"{name}.md", out / f"{name}.pdf")
            log.info("%s.pdf %s", name, "written" if ok else "skipped (pip install markdown playwright "
                     "&& playwright install chromium)")
    log.info("done -> %s", out)
    return {"oos": oos, "fm": fm, "bt": bt, "live": live, "feat": feat, "daily": daily}
