"""Whole pipeline on synthetic data + the key sanity check: a planted signal is found, noise is not."""
from xsent.backtest import forecast_metrics
from xsent.features import build_features
from xsent.ingest.demo import generate
from xsent.model import walk_forward
from xsent.pipeline import analyse
from xsent.sentiment import score_db, score_texts


def test_sentiment_lexicon_direction():
    s = score_texts(["BTC to the moon, so bullish", "Bitcoin dump, got rekt", "Bitcoin price today"])
    assert s.compound[0] > 0.3 and s.compound[1] < -0.3 and abs(s.compound[2]) < 0.2


def _skill(cfg, signal, seed):
    generate(cfg, days=300, signal=signal, seed=seed)
    score_db(cfg)
    feat = build_features(cfg)
    cfg.horizons, cfg.refit_every = (0,), 10
    fm = forecast_metrics(walk_forward(feat, cfg)).set_index("model")
    return fm.loc["ridge", "rmse_ret_bp"] / fm.loc["rw", "rmse_ret_bp"], fm.loc["ridge", "ic_spearman"]


def test_planted_signal_is_recovered(cfg):
    ratio, ic = _skill(cfg, signal=1.0, seed=3)
    assert ratio < 0.95 and ic > 0.2


def test_no_signal_gives_no_false_skill(cfg):
    ratio, ic = _skill(cfg, signal=0.0, seed=3)
    assert ratio > 0.97          # cannot beat the random walk on pure noise (no leakage)


def test_full_demo_pipeline_writes_all_outputs(cfg):
    cfg.refit_every = 10
    generate(cfg, days=200)
    res = analyse(cfg, demo=True, pdf=False)
    for f in ["REPORT.md", "SENTIMENT_OUTLOOK.md", "oos_predictions.csv", "forecast_metrics.csv",
              "backtest_metrics.csv", "live_forecast.csv", "metrics.json"]:
        assert (cfg.out_dir / f).exists(), f
    for c in ["sentiment_vs_price", "correlation_matrix", "rolling_correlation", "forecast_vs_actual",
              "equity_curve", "skill_by_horizon"]:
        assert (cfg.out_dir / "charts" / f"{c}.png").stat().st_size > 10_000
    assert len(res["live"]) == 6
    rep = (cfg.out_dir / "REPORT.md").read_text(encoding="utf-8")
    assert "Sharpe" in rep and "Sortino" in rep and "Max drawdown" in rep and "DEMO" in rep


def test_giveaway_and_duplicate_posts_are_flagged():
    import pandas as pd
    from xsent.sentiment import spam_flags
    df = pd.DataFrame({"date_utc": ["d1"] * 5, "text": [
        "$500 BITCOIN GIVEAWAY! To Enter: Repost this post, reply with your BTC address",
        "like this post and follow us, $100 in $BTC will go to one of you today",
        "Bitcoin steadies as ETF outflows weigh on crypto markets",
        "Bitcoin steadies as ETF outflows weigh on crypto markets",       # same-day duplicate
        "$BTC $ETH $SOL $XRP $DOGE $ADA $PEPE moon soon",                 # cashtag stuffing
    ]})
    assert spam_flags(df, max_cashtags=5).tolist() == [1, 1, 0, 1, 1]
