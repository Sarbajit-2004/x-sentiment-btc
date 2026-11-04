"""Static charts (PNG) showing X sentiment against price over time, plus model diagnostics.

Design rules: one y-axis per panel (no dual axes - price and sentiment get
stacked panels on a shared time axis), fixed categorical colour order,
diverging blue/red with a grey midpoint for signed quantities, recessive grid.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

log = logging.getLogger(__name__)

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
POS, NEG, MID = "#2a78d6", "#e34948", "#f0efec"
DIVERGING = LinearSegmentedColormap.from_list("bl_rd", [NEG, MID, POS])
MODEL_COLOR = {"ridge_price": SERIES[1], "ridge": SERIES[2], "gbm": SERIES[3], "ensemble": SERIES[0]}
MODEL_LABEL = {"ridge_price": "Ridge (price only)", "ridge": "Ridge (sentiment + price)",
               "gbm": "GBM (sentiment + price)", "ensemble": "Ensemble (primary)",
               "drift": "Drift", "rw": "Random walk", "buy_and_hold": "Buy & hold"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelsize": 9, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 8, "font.family": "DejaVu Sans",
    "lines.linewidth": 1.6, "figure.dpi": 110,
})


def _datefmt(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))


def _save(fig, path: Path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("chart -> %s", path)


# ----------------------------------------------------------------------------
def sentiment_vs_price(daily: pd.DataFrame, path: Path, asset: str, days: int | None = None):
    d = daily[daily["has_x"].astype(bool)].copy()
    d["date"] = pd.to_datetime(d["date_utc"])
    if days:
        d = d[d["date"] >= d["date"].max() - pd.Timedelta(days=days)]
    ma = d["sent_ew"].rolling(7, min_periods=3).mean()
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.2), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1.6, 1]})
    ax = axes[0]
    ax.plot(d["date"], d["close"], color=SERIES[0])
    ax.set_title(f"{asset}: daily close vs X sentiment", pad=10)
    ax.set_ylabel("Close (USD)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    ax = axes[1]
    colors = np.where(d["sent_ew"] >= 0, POS, NEG)
    ax.bar(d["date"], d["sent_ew"], color=colors, width=0.8, alpha=0.55, linewidth=0)
    ax.plot(d["date"], ma, color=INK, linewidth=1.6, label="7-day average")
    ax.axhline(0, color=AXIS, linewidth=0.8)
    ax.set_ylabel("Engagement-weighted\nsentiment (VADER)")
    ax.legend(loc="upper left")

    ax = axes[2]
    ax.bar(d["date"], np.log10(1 + d["eng_total"]), color=SERIES[2], width=0.8, linewidth=0)
    ax.set_ylabel("log10 engagement\n(likes+2·RT+replies)")
    _datefmt(ax)
    fig.align_ylabels(axes)
    _save(fig, path)


def correlation_matrices(feat: pd.DataFrame, path: Path):
    """Left: sentiment feature (known at open of n) vs forward return y_h. Right: lead/lag."""
    ok = feat["x_cov_l1"].astype(bool)
    f = feat[ok]
    rows = {"s_ew": "Sentiment (eng.-weighted)", "s_ew_z": "Sentiment z-score",
            "s_bb": "Bull − bear share", "s_mom": "Sentiment momentum",
            "s_infl_z": "Influencer sentiment z", "eng_z": "Engagement z", "vol_z": "Post volume z"}
    hs = [c for c in f.columns if c.startswith("y_")]
    m1 = pd.DataFrame({h.replace("y_", "n+"): [f[r].corr(f[h], method="spearman") for r in rows]
                       for h in hs}, index=list(rows.values()))
    # lead/lag: corr(sentiment of day d-k, return of day d); k<0 means price leads sentiment
    s_day = f["s_ew"].shift(-1)            # undo the 1-day feature lag -> sentiment of day n
    r_day = f["r_today"]
    lags = range(-5, 6)
    m2 = pd.Series({k: s_day.shift(k).corr(r_day, method="spearman") for k in lags})

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw={"width_ratios": [1.5, 1]})
    lim = max(0.15, np.nanmax(np.abs(m1.values)))
    im = a1.imshow(m1.values, cmap=DIVERGING, vmin=-lim, vmax=lim, aspect="auto")
    a1.set_xticks(range(len(m1.columns)), m1.columns)
    a1.set_yticks(range(len(m1.index)), m1.index)
    a1.grid(False)
    for i in range(m1.shape[0]):
        for j in range(m1.shape[1]):
            v = m1.values[i, j]
            a1.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=8,
                    color=INK if abs(v) < 0.6 * lim else "#ffffff")
    a1.set_title("Sentiment at open of day n vs return to close n+h\n(Spearman ρ)", fontsize=11)
    fig.colorbar(im, ax=a1, fraction=0.035, pad=0.02)

    colors = [POS if v >= 0 else NEG for v in m2.values]
    a2.bar(list(m2.index), m2.values, color=colors, width=0.7, linewidth=0, zorder=3)
    a2.axhline(0, color=AXIS, linewidth=0.8)
    a2.axvline(0, color=MUTED, linewidth=0.8, linestyle=":")
    a2.set_xticks(list(lags))
    a2.set_xlabel("k (days): sentiment of day d−k vs return of day d\n"
                  "k<0: price leads sentiment · k>0: sentiment leads price")
    a2.set_ylabel("Spearman ρ")
    a2.set_title("Lead/lag structure", fontsize=11)
    n = int(s_day.notna().sum())
    if n > 3:
        band = 1.96 / np.sqrt(n)
        a2.axhspan(-band, band, color=GRID, alpha=0.7, linewidth=0, zorder=1, label="95% noise band (±1.96/√N)")
        a2.legend(loc="upper left")
    fig.tight_layout()
    _save(fig, path)
    return m1, m2


def rolling_correlation(feat: pd.DataFrame, path: Path, window: int = 30):
    f = feat[feat["x_cov_l1"].astype(bool)].copy()
    f["date"] = pd.to_datetime(f["date_utc"])
    rc = f["s_ew"].rolling(window, min_periods=window // 2).corr(f["r_today"])
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.plot(f["date"], rc, color=SERIES[0])
    ax.axhline(0, color=AXIS, linewidth=0.8)
    ax.set_title(f"Rolling {window}-day correlation: yesterday's X sentiment vs today's return", pad=8)
    ax.set_ylabel("Pearson r")
    _datefmt(ax)
    _save(fig, path)


def forecast_vs_actual(oos: pd.DataFrame, model: str, path: Path):
    hs = sorted(oos["horizon"].unique())
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.6), sharex=True)
    for ax, h in zip(axes.ravel(), hs):
        g = oos[(oos.horizon == h) & (oos.model == model)].dropna(subset=["close_true"]).copy()
        tgt = pd.to_datetime(g["date_utc"]) + pd.to_timedelta(h, unit="D")  # date being forecast
        ax.plot(tgt, g["close_true"], color=INK2, linewidth=1.4, label="Actual close")
        ax.plot(tgt, g["close_pred"], color=SERIES[0], linewidth=1.4, label="Forecast")
        mape = (abs(g["close_true"] - g["close_pred"]) / g["close_true"]).mean() * 100
        ax.set_title(f"n+{h}  ·  MAPE {mape:.2f}%", fontsize=10)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:,.0f}k"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    h1, l1 = axes[0, 0].get_legend_handles_labels()
    fig.suptitle(f"Walk-forward forecasts vs actual close — {MODEL_LABEL.get(model, model)}",
                 x=0.01, ha="left", fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.legend(h1, l1, loc="upper right", ncol=2, bbox_to_anchor=(0.995, 1.0))
    _save(fig, path)


def equity_curves(curves: dict, model: str, path: Path, horizons=(0, 1)):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    bh = None
    for i, h in enumerate(horizons):
        if (h, model) not in curves:
            continue
        g = curves[(h, model)].copy()
        g["date"] = pd.to_datetime(g["date_utc"])
        eq = (1 + g["strat_ret"]).cumprod()
        c = SERIES[i]
        a1.plot(g["date"], eq, color=c, label=f"Strategy on n+{h} forecast")
        a2.plot(g["date"], eq / eq.cummax() - 1, color=c)
        if bh is None:
            bh = g
    if bh is not None:
        eqb = (1 + bh["bh_ret"]).cumprod()
        a1.plot(bh["date"], eqb, color=INK2, linewidth=1.3, linestyle="--", label="Buy & hold")
        a2.plot(bh["date"], eqb / eqb.cummax() - 1, color=INK2, linewidth=1.1, linestyle="--")
    a1.axhline(1, color=AXIS, linewidth=0.8)
    a1.set_ylabel("Growth of $1")
    a1.set_title(f"Out-of-sample equity — {MODEL_LABEL.get(model, model)}, long/flat, after costs", pad=26)
    a1.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, borderaxespad=0.2)
    a2.set_ylabel("Drawdown")
    a2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    _datefmt(a2)
    fig.align_ylabels((a1, a2))
    _save(fig, path)


def skill_by_horizon(fm: pd.DataFrame, path: Path):
    rw = fm[fm.model == "rw"].set_index("h")["rmse_ret_bp"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    for m in ["ridge_price", "ridge", "gbm", "ensemble"]:
        g = fm[fm.model == m].set_index("h").sort_index()
        if g.empty:
            continue
        ratio = g["rmse_ret_bp"] / rw.reindex(g.index)
        a1.plot(g.index, ratio, marker="o", markersize=5, color=MODEL_COLOR[m], label=MODEL_LABEL[m])
        a2.plot(g.index, g["dir_acc"] * 100, marker="o", markersize=5, color=MODEL_COLOR[m])
    a1.axhline(1, color=INK2, linewidth=0.9, linestyle="--")
    a1.text(1.0, 1.0, "random walk ", va="bottom", ha="right", color=INK2, fontsize=8,
            transform=a1.get_yaxis_transform())
    a1.set_xlabel("Horizon h (days)")
    a1.set_ylabel("RMSE ÷ random-walk RMSE  (<1 is better)")
    a1.set_title("Forecast error relative to random walk", fontsize=11, pad=44)
    a1.legend(loc="lower left", bbox_to_anchor=(0, 1.08), ncol=2, borderaxespad=0)
    a2.axhline(50, color=INK2, linewidth=0.9, linestyle="--")
    a2.set_xlabel("Horizon h (days)")
    a2.set_ylabel("Directional accuracy (%)")
    a2.set_title("Direction hit rate (50% = coin flip)", fontsize=11, pad=44)
    for a in (a1, a2):
        a.set_xticks(sorted(fm["h"].unique()), [f"n+{h}" for h in sorted(fm["h"].unique())])
    fig.tight_layout()
    _save(fig, path)


def make_all(daily, feat, oos, fm, curves, out_dir: Path, asset: str, model: str) -> dict:
    cdir = out_dir / "charts"
    cdir.mkdir(parents=True, exist_ok=True)
    paths = {k: cdir / f"{k}.png" for k in
             ["sentiment_vs_price", "correlation_matrix", "rolling_correlation",
              "forecast_vs_actual", "equity_curve", "skill_by_horizon"]}
    sentiment_vs_price(daily, paths["sentiment_vs_price"], asset)
    m1, m2 = correlation_matrices(feat, paths["correlation_matrix"])
    rolling_correlation(feat, paths["rolling_correlation"])
    forecast_vs_actual(oos, model, paths["forecast_vs_actual"])
    equity_curves(curves, model, paths["equity_curve"])
    skill_by_horizon(fm, paths["skill_by_horizon"])
    return {"paths": paths, "corr_matrix": m1, "leadlag": m2}
