"""Forecast-accuracy metrics and a simple trading backtest (Sharpe, Sortino, Max Drawdown).

Trading rule (per model, per horizon h):
    decide at the open of day n using y_hat_h[n]
    position[n] = +1 if y_hat > threshold, -1 if y_hat < -threshold and shorting allowed, else 0
    P&L[n]      = position[n] * simple_return(day n) - cost * |position[n] - position[n-1]|
Positions are held for one day and re-decided daily, so horizons never overlap.
Crypto trades every day -> annualisation factor 365. Risk-free rate = 0.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

ANN = 365


# ---------------------------------------------------------------- risk metrics
def sharpe(r: pd.Series) -> float:
    r = r.dropna()
    sd = r.std(ddof=1)
    return float(r.mean() / sd * math.sqrt(ANN)) if sd and sd > 0 else float("nan")


def sortino(r: pd.Series) -> float:
    r = r.dropna()
    dd = math.sqrt((np.minimum(r, 0) ** 2).mean()) if len(r) else 0
    return float(r.mean() / dd * math.sqrt(ANN)) if dd > 0 else float("nan")


def max_drawdown(r: pd.Series) -> float:
    eq = (1 + r.fillna(0)).cumprod()
    return float((eq / eq.cummax() - 1).min()) if len(eq) else float("nan")


def perf(r: pd.Series) -> dict:
    r = r.fillna(0)
    n = len(r)
    total = float((1 + r).prod() - 1) if n else float("nan")
    cagr = (1 + total) ** (ANN / n) - 1 if n and total > -1 else float("nan")
    mdd = max_drawdown(r)
    return {
        "total_return": total, "cagr": cagr,
        "ann_vol": float(r.std(ddof=1) * math.sqrt(ANN)) if n > 1 else float("nan"),
        "sharpe": sharpe(r), "sortino": sortino(r), "max_drawdown": mdd,
        "calmar": cagr / abs(mdd) if mdd and mdd < 0 else float("nan"),
        "days": n,
    }


# ---------------------------------------------------------------- forecast metrics
def _binom_p(k: int, n: int) -> float:
    """One-sided P(X >= k) for X~Bin(n, 0.5) (normal approximation with continuity correction)."""
    if n == 0:
        return float("nan")
    z = (k - 0.5 - n / 2) / math.sqrt(n / 4)
    return 0.5 * math.erfc(z / math.sqrt(2))


def diebold_mariano(e_model: np.ndarray, e_bench: np.ndarray, h: int) -> tuple[float, float]:
    """DM test on squared errors with Newey-West variance (lag h). Negative stat = model better."""
    d = e_model ** 2 - e_bench ** 2
    d = d[~np.isnan(d)]
    n = len(d)
    if n < 10:
        return float("nan"), float("nan")
    dbar = d.mean()
    g0 = ((d - dbar) ** 2).mean()
    var = g0
    for k in range(1, h + 1):
        gk = ((d[k:] - dbar) * (d[:-k] - dbar)).mean()
        var += 2 * (1 - k / (h + 1)) * gk
    if var <= 0:
        return float("nan"), float("nan")
    stat = dbar / math.sqrt(var / n)
    p = math.erfc(abs(stat) / math.sqrt(2))  # two-sided
    return float(stat), float(p)


def forecast_metrics(oos: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (h, m), g in oos.dropna(subset=["y_true"]).groupby(["horizon", "model"]):
        e = g["y_true"] - g["y_pred"]
        pe = g["close_true"] - g["close_pred"]
        rw = oos[(oos.horizon == h) & (oos.model == "rw")].set_index("date_utc").reindex(g["date_utc"])
        nz = g["y_pred"] != 0
        hits = int((np.sign(g.loc[nz, "y_pred"]) == np.sign(g.loc[nz, "y_true"])).sum())
        dm, dmp = diebold_mariano(e.values, (rw["y_true"] - rw["y_pred"]).values, h) if m != "rw" \
            else (float("nan"), float("nan"))
        ic = g["y_pred"].corr(g["y_true"], method="spearman") if g["y_pred"].std() > 0 else float("nan")
        rows.append({
            "horizon": f"n+{h}", "h": h, "model": m, "n": len(g),
            "rmse_price": float(np.sqrt((pe ** 2).mean())), "mae_price": float(pe.abs().mean()),
            "mape_pct": float((pe.abs() / g["close_true"]).mean() * 100),
            "rmse_ret_bp": float(np.sqrt((e ** 2).mean()) * 1e4),
            "dir_acc": hits / int(nz.sum()) if nz.sum() else float("nan"),
            "dir_p": _binom_p(hits, int(nz.sum())),
            "ic_spearman": float(ic), "dm_stat_vs_rw": dm, "dm_p": dmp,
        })
    return pd.DataFrame(rows).sort_values(["h", "model"]).reset_index(drop=True)


# ---------------------------------------------------------------- backtest
def strategy_returns(g: pd.DataFrame, cost_bps: float, thr: float, allow_short: bool) -> pd.DataFrame:
    g = g.sort_values("date_utc").copy()
    ret = np.expm1(g["r_today"].astype(float))
    pos = np.where(g["y_pred"] > thr, 1.0, np.where((g["y_pred"] < -thr) & allow_short, -1.0, 0.0))
    turn = np.abs(np.diff(np.r_[0.0, pos]))
    g["position"] = pos
    g["bh_ret"] = ret.values
    g["strat_ret"] = pos * ret.values - turn * cost_bps / 1e4
    return g


def backtest(oos: pd.DataFrame, cost_bps: float, thr: float, allow_short: bool) -> tuple[pd.DataFrame, dict]:
    rows, curves = [], {}
    for (h, m), g in oos.dropna(subset=["r_today"]).groupby(["horizon", "model"]):
        if m == "rw":
            continue  # zero forecast -> always flat
        g = strategy_returns(g, cost_bps, thr, allow_short)
        p = perf(g["strat_ret"])
        p.update({"horizon": f"n+{h}", "h": h, "model": m,
                  "exposure": float((g["position"] != 0).mean()),
                  "trades": int((np.diff(np.r_[0, g["position"].values]) != 0).sum()),
                  "hit_rate": float((g.loc[g.position != 0, "strat_ret"] > 0).mean())
                  if (g.position != 0).any() else float("nan")})
        rows.append(p)
        curves[(h, m)] = g[["date_utc", "position", "bh_ret", "strat_ret"]]
    bt = pd.DataFrame(rows)
    # buy & hold over the same evaluation days (from the n+0 test set)
    ref = oos[(oos.horizon == oos.horizon.min()) & (oos.model == "rw")].sort_values("date_utc")
    bh = perf(pd.Series(np.expm1(ref["r_today"].astype(float)).values))
    bh.update({"horizon": "-", "h": -1, "model": "buy_and_hold", "exposure": 1.0, "trades": 1,
               "hit_rate": float((ref["r_today"] > 0).mean())})
    bt = pd.concat([bt, pd.DataFrame([bh])], ignore_index=True)
    return bt.sort_values(["h", "model"]).reset_index(drop=True), curves
