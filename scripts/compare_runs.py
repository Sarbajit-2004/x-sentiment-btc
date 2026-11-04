"""Compare saved pipeline runs on the out-of-sample days they all share.

Runs with more training history start testing earlier, so their headline numbers cover a
different (longer) window. This script restricts every run to the common dates, per horizon,
and recomputes the forecast and trading metrics with the same functions the pipeline uses.

    python scripts/compare_runs.py outputs/runs/apify_only outputs/runs/hist_2021_23 \
        --out outputs/RUN_COMPARISON.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from xsent.backtest import backtest, forecast_metrics  # noqa: E402
from xsent.config import Config  # noqa: E402

MODELS = ["ensemble", "ridge", "gbm", "ridge_price"]


def common_window(runs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    keys = None
    for oos in runs.values():
        k = set(map(tuple, oos[["horizon", "date_utc"]].drop_duplicates().values))
        keys = k if keys is None else keys & k
    idx = pd.MultiIndex.from_tuples(sorted(keys), names=["horizon", "date_utc"])
    return {name: oos.set_index(["horizon", "date_utc"]).loc[lambda d: d.index.isin(idx)].reset_index()
            for name, oos in runs.items()}


def summarise(name: str, oos: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    fm = forecast_metrics(oos)
    bt, _ = backtest(oos, cfg.cost_bps, cfg.signal_threshold, cfg.allow_short)
    rw = fm[fm.model == "rw"].set_index("h")["rmse_ret_bp"]
    fm = fm[fm.model.isin(MODELS)].assign(run=name, rmse_ratio=lambda d: d.rmse_ret_bp / d.h.map(rw))
    bt = bt[bt.model.isin(MODELS + ["buy_and_hold"])].assign(run=name)
    return fm, bt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run folders that contain oos_predictions.csv")
    ap.add_argument("--out", default=None, help="write the Markdown tables here")
    a = ap.parse_args()
    cfg = Config()
    runs = {Path(r).name: pd.read_csv(Path(r) / "oos_predictions.csv", dtype={"date_utc": str}) for r in a.runs}
    runs = common_window(runs)
    d0 = next(iter(runs.values()))
    d0 = d0[(d0.horizon == 0) & (d0.model == "rw")]["date_utc"]
    fms, bts = zip(*(summarise(n, o, cfg) for n, o in runs.items()))
    fm, bt = pd.concat(fms), pd.concat(bts)

    md = [f"Common out-of-sample window: **{d0.min()} → {d0.max()}** ({len(d0)} days at n+0). "
          f"Costs {cfg.cost_bps:g} bp per unit turnover, long/flat.\n"]
    md.append("### Forecast skill (RMSE ÷ random walk; < 1 beats it)\n")
    piv = fm.pivot_table(index=["run", "model"], columns="horizon", values="rmse_ratio")
    md.append("| Run | Model | " + " | ".join(piv.columns) + " |")
    md.append("|---|---|" + "---|" * len(piv.columns))
    for (run, m), row in piv.iterrows():
        md.append(f"| {run} | {m} | " + " | ".join(f"{v:.3f}" for v in row.values) + " |")
    md.append("\n### Directional accuracy\n")
    piv = fm.pivot_table(index=["run", "model"], columns="horizon", values="dir_acc")
    md.append("| Run | Model | " + " | ".join(piv.columns) + " |")
    md.append("|---|---|" + "---|" * len(piv.columns))
    for (run, m), row in piv.iterrows():
        md.append(f"| {run} | {m} | " + " | ".join(f"{v * 100:.1f}%" for v in row.values) + " |")
    md.append("\n### Trading the n+0 forecast\n")
    md.append("| Run | Signal | Sharpe | Sortino | Max DD | Total return | Exposure |")
    md.append("|---|---|---|---|---|---|---|")
    for r in bt[bt.h.isin([-1, 0])].itertuples():
        md.append(f"| {r.run} | {r.model} | {r.sharpe:.2f} | {r.sortino:.2f} | {r.max_drawdown * 100:.1f}% "
                  f"| {r.total_return * 100:+.1f}% | {r.exposure * 100:.0f}% |")
    text = "\n".join(md) + "\n"
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
    print(text)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
