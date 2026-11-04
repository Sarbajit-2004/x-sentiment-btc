import math

import numpy as np
import pandas as pd

from xsent.backtest import (diebold_mariano, max_drawdown, perf, sharpe, sortino,
                            strategy_returns)


def test_max_drawdown_known_path():
    # 100 -> 120 -> 60 -> 90 : peak 120, trough 60 => -50%
    r = pd.Series([0.2, -0.5, 0.5])
    assert math.isclose(max_drawdown(r), -0.5, rel_tol=1e-9)


def test_sharpe_and_sortino():
    r = pd.Series([0.01, -0.005, 0.02, -0.01, 0.015])
    assert math.isclose(sharpe(r), r.mean() / r.std(ddof=1) * math.sqrt(365))
    dd = math.sqrt((np.minimum(r, 0) ** 2).mean())
    assert math.isclose(sortino(r), r.mean() / dd * math.sqrt(365))
    assert math.isnan(sharpe(pd.Series([0.0, 0.0, 0.0])))


def test_perf_total_return():
    p = perf(pd.Series([0.1, -0.1]))
    assert math.isclose(p["total_return"], 1.1 * 0.9 - 1)


def test_strategy_costs_and_positions():
    g = pd.DataFrame({"date_utc": ["d1", "d2", "d3"], "y_pred": [0.01, -0.01, 0.02],
                      "r_today": np.log([1.02, 0.98, 1.01])})
    out = strategy_returns(g, cost_bps=10, thr=0.0, allow_short=False)
    assert list(out["position"]) == [1, 0, 1]
    # day1: +2% - 10bp entry ; day2: flat, 10bp exit ; day3: +1% - 10bp entry
    np.testing.assert_allclose(out["strat_ret"], [0.02 - 0.001, -0.001, 0.01 - 0.001], atol=1e-12)


def test_dm_detects_better_model():
    rng = np.random.default_rng(1)
    y = rng.normal(0, 1, 400)
    good = y + rng.normal(0, 0.3, 400)
    stat, p = diebold_mariano(y - good, y - 0, h=0)
    assert stat < 0 and p < 0.01
