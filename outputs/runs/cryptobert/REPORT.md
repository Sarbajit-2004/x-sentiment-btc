# Bitcoin X-Sentiment Price Prediction — Evaluation Report

Out-of-sample evaluation window: **2026-07-10 → 2026-11-03** (UTC daily bars). Primary model chosen before testing: **Ensemble (primary)**. Transaction cost 10 bp per unit of turnover; risk-free rate 0; annualisation √365.

## 1. Headline results

- **Trading the n+0 forecast (long/flat):** Sharpe **-1.40**, Sortino **-2.17**, Max drawdown **-17.3%**, total return -10.8% with 40.2% time in market.
- **Buy & hold over the same days:** Sharpe 1.88, Sortino 3.47, Max drawdown -11.5%, total return +22.4%.
- **Forecast skill vs random walk:** RMSE ratio ranges 1.041–1.115 across n+0…n+5 (<1 beats the random walk); directional accuracy 40.2%–45.0%.
- **Statistical significance:** the Diebold–Mariano test rejects 'no better than random walk' at 5% for 0 of 6 horizons; it finds the model significantly *worse* than the random walk at 4 (n+0, n+1, n+2, n+3).
- **Does X sentiment add information?** Adding sentiment features to the same Ridge model lowered out-of-sample error at 1 of 6 horizons (ablation vs price-only).

## 2. Data

| Source | Posts | Flagged spam | Days | First | Last |
|---|---|---|---|---|---|
| `apify:kaitoeasyapi/twitter-x-data-tweet-scraper-pay-per-result-cheapest` | 9,842 | 344 | 181 | 2026-05-08 | 2026-11-04 |

Prices: 240 daily bars (2026-03-09 → 2026-11-03), source `yfinance`, ticker `BTC-USD`. News: 44 Tavily articles.

Apify runs:

- actor `kaitoeasyapi/twitter-x-data-tweet-scraper-pay-per-result-cheapest` — SUCCEEDED: 180 windows, 9,842 posts

X query: `(bitcoin OR BTC OR $BTC OR #bitcoin OR #BTC)` (ticker, cashtag and keyword in one advanced search), language `en`, 24-hour windows capped at 80 posts, sort `Top`. Posts are bucketed by UTC date to match the 00:00–24:00 UTC daily bar; an America/New_York timestamp is stored alongside for reference.

## 3. Method

**Timing (no look-ahead).** Row *n* is the current day; the forecast is made at its 00:00 UTC open using posts and prices up to the close of day *n−1* only. Targets are `log(close[n+h] / close[n−1])` for h = 0…5, so *n+0* is today's close and *n+5* the close five days later. Forecast prices are `close[n−1] · exp(ŷ)`.

**Sentiment.** VADER (built for social-media text) extended with a crypto lexicon (moon, rekt, HODL, FUD, dump, ETF inflows/outflows, …). URLs and @mentions are stripped; cashtag-stuffed shill posts (> 5 cashtags), same-day duplicates and near-empty posts, and giveaway / engagement-farming posts ("repost & reply with your BTC address") are flagged as spam and excluded.

**Features — sentiment (13):** engagement-weighted score and its rolling 30-day z-score, follower-weighted and influencer (≥10k followers) z-scores, bull−bear share, dispersion, 1-day change, 3/7-day EMA momentum, 7-day mean, post-volume and engagement z-scores, sentiment×engagement. **Price (13):** 1/2/3-day lagged returns, 5/10/20-day momentum, 7/20-day realised volatility, RSI-14, gap to the 20-day mean, high-low range, day-of-week. No traded volume, on-chain or macro inputs.

**Models.** One model per horizon: Ridge (regularisation chosen by time-ordered CV with a gap of h days) and depth-2 gradient-boosted trees on sentiment + price, their average (the *ensemble*, fixed in advance as primary), a price-only Ridge as an ablation, and random-walk / drift baselines.

**Walk-forward validation.** Expanding window, minimum 60 training days, refit every 1 day(s). For horizon *h*, training rows are purged so every training target is already realised at the forecast time (row *i* is used only if *i + h ≤ n − 1*). Every prediction in this report is out of sample.

**Backtest.** At each open, go long if the forecast log-return > 0, otherwise flat; hold one day; pay costs on every change of position. Each horizon's forecast is traded as its own signal so positions never overlap.

## 4. Trading performance (out of sample, after costs)

| Signal | Sharpe | Sortino | Max DD | Total return | CAGR | Ann. vol | Calmar | Exposure | Hit rate | Trades |
|---|---|---|---|---|---|---|---|---|---|---|
| Buy & hold | 1.88 | 3.47 | -11.5% | +22.4% | +87.8% | 37.2% | 7.66 | 100.0% | 49.6% | 1 |
| Ensemble (primary) · n+0 | -1.40 | -2.17 | -17.3% | -10.8% | -29.9% | 23.4% | -1.73 | 40.2% | 42.6% | 36 |
| Ridge (price only) · n+0 | -1.62 | -2.41 | -12.0% | -9.2% | -25.9% | 17.6% | -2.16 | 24.8% | 37.9% | 16 |
| Ensemble (primary) · n+1 | -2.22 | -2.92 | -15.6% | -13.6% | -37.1% | 20.0% | -2.38 | 47.0% | 42.6% | 26 |
| Ridge (price only) · n+1 | -1.93 | -2.26 | -7.7% | -7.7% | -22.3% | 12.7% | -2.92 | 23.5% | 44.4% | 4 |
| Ensemble (primary) · n+2 | -0.20 | -0.32 | -12.3% | -2.8% | -8.6% | 27.1% | -0.70 | 56.6% | 45.3% | 8 |
| Ridge (price only) · n+2 | -1.97 | -2.30 | -7.3% | -7.3% | -21.8% | 12.1% | -2.97 | 22.1% | 40.0% | 10 |
| Ensemble (primary) · n+3 | -0.05 | -0.08 | -12.2% | -1.5% | -4.9% | 27.3% | -0.41 | 56.8% | 46.0% | 10 |
| Ridge (price only) · n+3 | -1.92 | -2.28 | -7.8% | -7.8% | -23.4% | 13.4% | -3.01 | 30.6% | 41.2% | 2 |
| Ensemble (primary) · n+4 | -0.68 | -0.97 | -10.9% | -5.6% | -17.6% | 24.3% | -1.61 | 54.1% | 47.5% | 7 |
| Ridge (price only) · n+4 | -1.94 | -2.31 | -7.8% | -7.8% | -23.8% | 13.5% | -3.05 | 32.1% | 40.0% | 2 |
| Ensemble (primary) · n+5 | -0.75 | -1.07 | -10.3% | -6.1% | -19.2% | 24.4% | -1.87 | 54.2% | 48.3% | 5 |
| Ridge (price only) · n+5 | -2.14 | -2.44 | -10.1% | -9.8% | -29.8% | 15.9% | -2.96 | 34.6% | 43.2% | 1 |

![Equity curve](charts/equity_curve.png)

## 5. Forecast accuracy by horizon

| Horizon | Model | N | MAPE % | RMSE $ | RMSE ÷ RW | Dir. acc. | p (dir) | IC | DM stat | DM p |
|---|---|---|---|---|---|---|---|---|---|---|
| n+0 | Ensemble (primary) | 117 | 1.33 | $1,503 | 1.04 | 44.4% | 0.90 | -0.24 | 2.84 | 0.00 |
| n+0 | GBM (sentiment + price) | 117 | 1.38 | $1,530 | 1.06 | 42.7% | 0.95 | -0.27 | 3.24 | 0.00 |
| n+0 | Ridge (sentiment + price) | 117 | 1.31 | $1,484 | 1.03 | 45.3% | 0.87 | -0.18 | 2.13 | 0.03 |
| n+0 | Ridge (price only) | 117 | 1.30 | $1,479 | 1.03 | 44.4% | 0.90 | -0.14 | 1.82 | 0.07 |
| n+0 | Random walk | 117 | 1.26 | $1,450 | 1.00 | – | – | – | – | – |
| n+1 | Ensemble (primary) | 115 | 1.99 | $2,168 | 1.06 | 41.7% | 0.97 | -0.18 | 2.63 | 0.01 |
| n+1 | GBM (sentiment + price) | 115 | 2.08 | $2,217 | 1.08 | 45.2% | 0.87 | -0.19 | 3.00 | 0.00 |
| n+1 | Ridge (sentiment + price) | 115 | 1.94 | $2,138 | 1.04 | 44.3% | 0.90 | -0.19 | 2.40 | 0.02 |
| n+1 | Ridge (price only) | 115 | 1.94 | $2,139 | 1.05 | 44.3% | 0.90 | -0.18 | 2.13 | 0.03 |
| n+1 | Random walk | 115 | 1.86 | $2,061 | 1.00 | – | – | – | – | – |
| n+2 | Ensemble (primary) | 113 | 2.52 | $2,779 | 1.08 | 42.5% | 0.95 | -0.24 | 2.58 | 0.01 |
| n+2 | GBM (sentiment + price) | 113 | 2.63 | $2,849 | 1.11 | 43.4% | 0.93 | -0.21 | 2.79 | 0.01 |
| n+2 | Ridge (sentiment + price) | 113 | 2.43 | $2,740 | 1.07 | 40.7% | 0.98 | -0.30 | 2.52 | 0.01 |
| n+2 | Ridge (price only) | 113 | 2.36 | $2,705 | 1.05 | 46.9% | 0.77 | -0.24 | 2.12 | 0.03 |
| n+2 | Random walk | 113 | 2.27 | $2,590 | 1.00 | – | – | – | – | – |
| n+3 | Ensemble (primary) | 111 | 2.92 | $3,361 | 1.09 | 45.0% | 0.87 | -0.16 | 2.02 | 0.04 |
| n+3 | GBM (sentiment + price) | 111 | 3.13 | $3,463 | 1.12 | 45.9% | 0.83 | -0.12 | 2.20 | 0.03 |
| n+3 | Ridge (sentiment + price) | 111 | 2.78 | $3,315 | 1.07 | 48.6% | 0.65 | -0.28 | 2.14 | 0.03 |
| n+3 | Ridge (price only) | 111 | 2.73 | $3,280 | 1.06 | 49.5% | 0.58 | -0.27 | 1.84 | 0.07 |
| n+3 | Random walk | 111 | 2.63 | $3,125 | 1.00 | – | – | – | – | – |
| n+4 | Ensemble (primary) | 109 | 3.42 | $3,952 | 1.10 | 41.3% | 0.97 | -0.13 | 1.77 | 0.08 |
| n+4 | GBM (sentiment + price) | 109 | 3.72 | $4,131 | 1.15 | 46.8% | 0.78 | -0.10 | 1.95 | 0.05 |
| n+4 | Ridge (sentiment + price) | 109 | 3.23 | $3,869 | 1.08 | 47.7% | 0.72 | -0.31 | 1.95 | 0.05 |
| n+4 | Ridge (price only) | 109 | 3.18 | $3,841 | 1.07 | 45.0% | 0.87 | -0.28 | 1.78 | 0.08 |
| n+4 | Random walk | 109 | 3.03 | $3,626 | 1.00 | – | – | – | – | – |
| n+5 | Ensemble (primary) | 107 | 3.88 | $4,477 | 1.11 | 40.2% | 0.98 | -0.13 | 1.63 | 0.10 |
| n+5 | GBM (sentiment + price) | 107 | 4.24 | $4,742 | 1.18 | 40.2% | 0.98 | -0.12 | 1.85 | 0.06 |
| n+5 | Ridge (sentiment + price) | 107 | 3.65 | $4,336 | 1.08 | 38.3% | 0.99 | -0.28 | 1.71 | 0.09 |
| n+5 | Ridge (price only) | 107 | 3.65 | $4,328 | 1.08 | 37.4% | 1.00 | -0.27 | 1.67 | 0.10 |
| n+5 | Random walk | 107 | 3.40 | $4,063 | 1.00 | – | – | – | – | – |

*RMSE ÷ RW < 1 means lower error than the random walk. DM = Diebold–Mariano test on squared errors vs random walk (Newey–West, lag h); negative = model better. p (dir) = one-sided binomial test of the hit rate against 50%.*

![skill_by_horizon](charts/skill_by_horizon.png)

![forecast_vs_actual](charts/forecast_vs_actual.png)

## 6. Sentiment vs price

![sentiment_vs_price](charts/sentiment_vs_price.png)

![correlation_matrix](charts/correlation_matrix.png)

![rolling_correlation](charts/rolling_correlation.png)

## 7. Live forecast (from the latest close)

| Horizon | Date | Forecast | 80% low | 80% high | Random walk |
|---|---|---|---|---|---|
| n+0 | 2026-11-04 | $77,261 | $75,313 | $79,259 | $77,345 |
| n+1 | 2026-11-05 | $77,220 | $74,406 | $80,140 | $77,345 |
| n+2 | 2026-11-06 | $77,303 | $73,690 | $81,093 | $77,345 |
| n+3 | 2026-11-07 | $77,596 | $73,222 | $82,231 | $77,345 |
| n+4 | 2026-11-08 | $77,630 | $72,480 | $83,146 | $77,345 |
| n+5 | 2026-11-09 | $77,764 | $71,951 | $84,047 | $77,345 |

## 8. Limitations

- Only 117 out-of-sample trading days: an annualised Sharpe ratio estimated on that sample has a standard error of roughly ±1.8 (≈√(365/N)), so a Sharpe ratio within about ±3.5 of zero (or of buy & hold) is not statistically distinguishable from it.
- Each day's sentiment is a capped sample of posts (top or latest), not the full X firehose; post-count features therefore measure sampling more than true volume.
- Lexicon sentiment misses sarcasm and context; FinBERT and CryptoBERT are available via `SENTIMENT_ENGINE=finbert` or `cryptobert` for comparison.
- Results depend on one asset and one period; no hyper-parameters were tuned on the test window.

## 9. Reproduce

```bash
pip install -r requirements.txt
cp .env.example .env   # add APIFY_TOKEN and TAVILY_API_KEY
python run_pipeline.py all
```
