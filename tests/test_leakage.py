"""The two properties that make the evaluation honest: no look-ahead in features, purged training."""
import numpy as np
import pandas as pd

import xsent.model as M
from xsent.features import PRICE_FEATURES, SENT_FEATURES, make_features


def _daily(n=150, seed=0):
    rng = np.random.default_rng(seed)
    close = 50000 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    return pd.DataFrame({
        "date_utc": pd.date_range("2026-01-01", periods=n).strftime("%Y-%m-%d"),
        "open": close, "high": close * 1.01, "low": close * 0.99, "close": close,
        "n_posts": rng.integers(50, 100, n), "n_authors": 50,
        "sent_mean": rng.normal(0, 0.2, n), "sent_ew": rng.normal(0, 0.2, n),
        "sent_fw": rng.normal(0, 0.2, n), "sent_infl": rng.normal(0, 0.2, n),
        "sent_std": rng.uniform(0.2, 0.5, n), "bull_share": rng.uniform(0.3, 0.6, n),
        "bear_share": rng.uniform(0.1, 0.4, n), "eng_total": rng.uniform(1e3, 1e4, n),
        "views_total": 1e5, "has_x": True,
    })


def test_features_use_only_past_data():
    d = _daily()
    f1 = make_features(d)
    n = 100
    d2 = d.copy()
    future = d2.index >= n           # change day n and everything after it
    d2.loc[future, ["close", "high", "low", "open"]] *= 1.5
    d2.loc[future, ["sent_ew", "sent_mean", "sent_fw", "sent_infl", "bull_share"]] = 0.99
    d2.loc[future, ["eng_total", "n_posts"]] = 1e9
    f2 = make_features(d2)
    cols = SENT_FEATURES + PRICE_FEATURES + ["ref_close"]
    pd.testing.assert_frame_equal(f1.loc[:n, cols], f2.loc[:n, cols])
    # ...while the targets for row n DO change (they are the future)
    assert not np.isclose(f1.loc[n, "y_0"], f2.loc[n, "y_0"])


def test_target_definition():
    d = _daily()
    f = make_features(d)
    n = 80
    for h in range(6):
        assert np.isclose(f.loc[n, f"y_{h}"], np.log(d.loc[n + h, "close"] / d.loc[n - 1, "close"]))
    assert np.isclose(f.loc[n, "ref_close"], d.loc[n - 1, "close"])


def test_walk_forward_purges_overlapping_targets(cfg, monkeypatch):
    d = _daily(160)
    f = make_features(d).reset_index(drop=True)
    seen = []
    real = M.fit_predict

    def spy(train, test, ycol, seed):
        h = int(ycol.split("_")[1])
        seen.append((h, train.index.max(), test.index.min()))
        return real(train, test, ycol, seed)

    monkeypatch.setattr(M, "fit_predict", spy)
    cfg.min_train_days = 60
    cfg.refit_every = 7
    cfg.n_jobs = 1          # in-process so the spy sees every fit
    oos = M.walk_forward(f, cfg)
    assert len(seen) > 0 and not oos.empty
    for h, train_max, test_min in seen:
        assert train_max + h <= test_min - 1, (h, train_max, test_min)
    # every out-of-sample row is predicted exactly once per model/horizon
    assert not oos.duplicated(["date_utc", "horizon", "model"]).any()


def test_missing_price_day_does_not_misalign_lags():
    d = _daily(60).drop(index=30).reset_index(drop=True)     # one calendar day missing
    f = make_features(d)
    assert len(f) == 60                                        # gap re-inserted
    assert f.loc[30, ["close", "ref_close"]].isna().all() or np.isnan(f.loc[30, "close"])
    assert np.isnan(f.loc[31, "ref_close"])                    # previous day unknown -> no forecast
