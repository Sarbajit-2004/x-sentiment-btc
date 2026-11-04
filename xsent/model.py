"""Per-horizon forecasting models with purged, expanding-window walk-forward validation.

For every test day n and horizon h the model is trained only on rows i whose
target is fully observed by the close of day n-1, i.e. i + h <= n - 1.
Without this purge an h-day target would overlap the test period (look-ahead).

Models (all predict y_h = log(close[n+h] / close[n-1])):
  rw          random walk, y_hat = 0 (tomorrow's price = last close)  <- the bar to beat
  drift       mean of the training targets
  ridge_price Ridge on price features only (ablation: what does sentiment add?)
  ridge       Ridge on sentiment + price features (alpha by purged time-series CV)
  gbm         shallow gradient-boosted trees on sentiment + price
  ensemble    average of ridge and gbm (chosen a priori as the primary model,
              so no model is picked by looking at test results)
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import Config
from .features import PRICE_FEATURES, SENT_FEATURES

log = logging.getLogger(__name__)
FULL = SENT_FEATURES + PRICE_FEATURES
MODELS = ["rw", "drift", "ridge_price", "ridge", "gbm", "ensemble"]


def _ridge(h: int = 0):
    # alpha chosen by time-ordered CV with a gap of h rows, so overlapping h-day targets
    # cannot leak between the inner train/validation folds (plain LOO/GCV under-regularises).
    return make_pipeline(StandardScaler(),
                         RidgeCV(alphas=np.logspace(-1, 5, 15), cv=TimeSeriesSplit(n_splits=3, gap=h)))


def _gbm(seed: int):
    # heavily regularised: ~150-day training sets are mostly noise. Settings were chosen on
    # synthetic data (with and without a planted signal), never on the real test window.
    return HistGradientBoostingRegressor(max_depth=2, learning_rate=0.02, max_iter=100,
                                         min_samples_leaf=25, l2_regularization=10.0,
                                         max_features=0.5, random_state=seed)


def usable_rows(feat: pd.DataFrame) -> pd.Series:
    """Rows with a full feature vector (requires X coverage on day n-1)."""
    return feat[FULL].notna().all(axis=1) & feat["x_cov_l1"].astype(bool) & feat["ref_close"].notna()


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, ycol: str, seed: int) -> dict:
    y = train[ycol].values
    h = int(ycol.split("_")[1])
    out = {"rw": np.zeros(len(test)), "drift": np.full(len(test), y.mean())}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = _ridge(h).fit(train[PRICE_FEATURES].values, y)
        out["ridge_price"] = m.predict(test[PRICE_FEATURES].values)
        m = _ridge(h).fit(train[FULL].values, y)
        out["ridge"] = m.predict(test[FULL].values)
        g = _gbm(seed).fit(train[FULL].values, y)
        out["gbm"] = g.predict(test[FULL].values)
    out["ensemble"] = 0.5 * (out["ridge"] + out["gbm"])
    return out


def _walk_horizon(feat: pd.DataFrame, ok: np.ndarray, in_test: np.ndarray, h: int,
                  cfg: Config) -> list[dict]:
    pos = np.arange(len(feat))
    ycol = f"y_{h}"
    has_y = feat[ycol].notna().values
    cum_train = np.cumsum(ok & has_y)                     # trainable rows up to index i
    test_idx = [n for n in pos[ok & has_y & in_test]
                if n - 1 - h >= 0 and cum_train[n - 1 - h] >= cfg.min_train_days]
    if not test_idx:
        log.warning("h=%d: not enough history for walk-forward (need %d training days)",
                    h, cfg.min_train_days)
        return []
    records = []
    ntr = []
    for k in range(0, len(test_idx), cfg.refit_every):
        block = test_idx[k:k + cfg.refit_every]
        n0 = block[0]
        train_mask = (pos <= n0 - 1 - h) & ok & has_y        # purge: target realised by n0-1
        p = fit_predict(feat[train_mask], feat.loc[block], ycol, cfg.random_seed)
        ntr.append(int(train_mask.sum()))
        for j, n in enumerate(block):
            row = feat.loc[n]
            for m in MODELS:
                records.append({
                    "date_utc": row["date_utc"], "horizon": h, "model": m,
                    "y_true": row[ycol], "y_pred": float(p[m][j]), "ref_close": row["ref_close"],
                    "close_true": row[f"close_{h}"],
                    "close_pred": row["ref_close"] * np.exp(p[m][j]),
                    "r_today": row["r_today"], "n_train": ntr[-1],
                })
    log.info("h=%d: %d walk-forward predictions (train %d..%d rows)", h, len(test_idx), min(ntr), max(ntr))
    return records


def walk_forward(feat: pd.DataFrame, cfg: Config, test_start: str | None = None) -> pd.DataFrame:
    """Return long table: date_utc, horizon, model, y_true, y_pred, ref_close, close_true, close_pred.

    Rows before `test_start` (e.g. years of historical tweets) are only ever used for
    training; out-of-sample predictions are produced from `test_start` onwards.
    Horizons are independent, so they run in parallel (N_JOBS, default all cores).
    """
    feat = feat.reset_index(drop=True)
    ok = usable_rows(feat).values
    in_test = ((feat["date_utc"] >= test_start).values if test_start is not None
               else np.ones(len(feat), dtype=bool))
    log.info("walk-forward: %d horizons, %d candidate test days, refit every %d day(s)%s",
             len(cfg.horizons), int((ok & in_test).sum()), cfg.refit_every,
             "" if cfg.n_jobs == 1 else " (parallel)")
    if cfg.n_jobs == 1 or len(cfg.horizons) == 1:
        parts = [_walk_horizon(feat, ok, in_test, h, cfg) for h in cfg.horizons]
    else:
        from joblib import Parallel, delayed
        parts = Parallel(n_jobs=min(len(cfg.horizons), cfg.n_jobs if cfg.n_jobs > 0 else 64))(
            delayed(_walk_horizon)(feat, ok, in_test, h, cfg) for h in cfg.horizons)
    out = pd.DataFrame([r for part in parts for r in part])
    if not out.empty:
        cnt = out[out.model == "rw"].groupby("horizon").size()
        log.info("walk-forward done: %s", ", ".join(f"n+{h}: {c} preds" for h, c in cnt.items()))
    return out


def live_forecast(feat: pd.DataFrame, daily: pd.DataFrame, oos: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Forecast n+0..n+5 for n = today (the day after the last complete daily bar)."""
    from .features import make_features

    last = pd.Timestamp(daily["date_utc"].max())
    today = (last + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    # append an empty 'today' row so features are computed from data up to `last`
    ext = pd.concat([daily, pd.DataFrame({"date_utc": [today]})], ignore_index=True)
    ext["has_x"] = ext["has_x"].astype("boolean").fillna(False).astype(bool)
    for c in ["open", "high", "low", "close"]:
        ext[c] = ext[c].ffill()  # placeholder only; today's own values are never used as inputs
    f_ext = make_features(ext, cfg.horizons)
    x_today = f_ext.iloc[[-1]]
    if not usable_rows(f_ext).iloc[-1]:
        log.warning("latest day lacks X coverage; live forecast uses zero-filled sentiment")
        x_today = x_today.fillna(0)
    ok = usable_rows(feat)
    rows = []
    for h in cfg.horizons:
        ycol = f"y_{h}"
        tr = feat[ok & feat[ycol].notna()]
        if len(tr) < cfg.min_train_days:
            continue
        p = fit_predict(tr, x_today, ycol, cfg.random_seed)[cfg.primary_model][0]
        res = oos[(oos.horizon == h) & (oos.model == cfg.primary_model)]
        err = (res["y_true"] - res["y_pred"]).dropna()
        sd = err.std() if len(err) > 5 else tr[ycol].std()
        ref = float(daily["close"].iloc[-1])
        rows.append({
            "target_date": (last + pd.Timedelta(days=1 + h)).strftime("%Y-%m-%d"),
            "horizon": f"n+{h}", "ref_close": ref, "pred_logret": p,
            "pred_close": ref * np.exp(p),
            "lo80": ref * np.exp(p - 1.2816 * sd), "hi80": ref * np.exp(p + 1.2816 * sd),
            "rw_close": ref,
        })
    return pd.DataFrame(rows)
