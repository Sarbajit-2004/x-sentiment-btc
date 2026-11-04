# Addendum — Variants: Historical Training Data and CryptoBERT

The main report (`REPORT.md`) is the **Apify-only** run with VADER sentiment. This addendum reports two variants
**alongside** it, not instead of it. Each changes exactly one thing:

- **Sections 1–5: historical training data.** The brief suggests historical tweet datasets "to learn / teach the model",
  so this variant adds 2021–23 Kaggle tweets as extra training rows.
- **Section 6: CryptoBERT.** The same Apify data, scored by a crypto-specific transformer instead of VADER.

Models, features, hyper-parameters and the primary model (the ensemble) are unchanged in both.

## 1. Data validation: the "2025–2026" Kaggle set is re-dated

The best-matching dataset on the list, **"Bitcoin Tweets Dataset (2025–2026)"** (`pokeash/bitcoin-tweets-dataset-20252026`,
v1, 2.10 GB), failed a date sanity check and was **not used**:

| Check | Expected if the dates are real | `pokeash` "2025–26" | `kaushiksuresh147` 2021–23 |
|---|---|---|---|
| Newest account-creation date | ≈ tweet era | **2023-01-09** (tweets stamped 2025-03 → 2026-03) | 2023-01-09 |
| Years mentioned in tweet text | mostly the stamped years | **2021: 96k, 2022: 139k**; 2025: 2.6k, 2026: 0.6k | same |
| "Happy Monday" / "#Friday" vs stamped weekday | most agree | **0.9 %** agree (36 / 4,210) | **88.2 %** agree |

Both files have the same 13 columns and nearly the same size. Their first rows are the same tweet, stamped 2021-02 in one
file and 2026-02 in the other. The "2025–26" file is a copy of the 2021–23 dataset with shifted timestamps. Training on it
would have paired 2021–22 sentiment with 2025 prices. Reproduce with `python scripts/validate_tweet_dates.py <csv>`.

## 2. What was added

- **Dataset:** `kaushiksuresh147/bitcoin-tweets`, `Bitcoin_tweets.csv` (CC0, genuine dates, 2021-02 → 2023-01).
- **Loading:** `historical` step, capped at 300 posts per UTC day. The dataset has no like or repost counts, so the cap keeps
  the posts from the largest accounts and engagement weights fall back to follower counts.
- **Result:** **66,300 posts on 222 days.** The dataset was collected in bursts, so the days are scattered. 20 % of posts
  were flagged as spam, against 3.5 % in the Apify "Top" sample.
- **Separate database:** the variant ran on its own copy of the database (`DATA_DIR=data/hist_2021_23`). The primary
  `data/xsent.db` and `outputs/` are untouched.
- **Training rows per refit:** 220–398 days, against 60–176 days for Apify only.
- **Prices:** extended back to 2020-12 so the price features have history.
- **Test window:** the evaluation is still the most recent 180 days, all Apify data. Historical rows are used for training
  only. The purged walk-forward is unchanged.

## 3. Results on the same days (2026-07-10 → 2026-11-03, 117 days at n+0)

With more training history, the variant can start testing in May, while the Apify-only run can only start on 07-10. The
fair comparison therefore restricts both runs to the days they share (`scripts/compare_runs.py`, full tables in
`RUN_COMPARISON.md`).

**Trading the n+0 forecast** (long/flat, 10 bp costs):

| Signal | Apify only: Sharpe | Sortino | Max DD | Return | + 2021–23 history: Sharpe | Sortino | Max DD | Return |
|---|---|---|---|---|---|---|---|---|
| **Ensemble (primary)** | **−3.16** | −3.57 | −14.3 % | −12.8 % | **0.04** | 0.07 | −16.9 % | −1.1 % |
| Ridge (sentiment + price) | −2.77 | −3.14 | −11.7 % | −10.5 % | 1.11 | 1.95 | −11.5 % | +10.4 % |
| GBM (sentiment + price) | −1.24 | −1.89 | −15.5 % | −8.5 % | 0.52 | 0.95 | −14.1 % | +3.6 % |
| Ridge (price only) | −1.62 | −2.41 | −12.0 % | −9.2 % | −0.08 | −0.13 | −9.3 % | −2.0 % |
| Buy & hold | 1.88 | 3.47 | −11.5 % | +22.4 % | 1.88 | 3.47 | −11.5 % | +22.4 % |

**Forecast skill of the ensemble:**

| | Apify only | + 2021–23 history |
|---|---|---|
| RMSE ÷ random walk, n+0 … n+5 | 1.031 – 1.100 | **0.996 – 1.025** |
| Diebold–Mariano: significantly *worse* than random walk (5 %) | n+0, n+1, n+2 | n+3 |
| Diebold–Mariano: significantly *better* than random walk (5 %) | none | none |
| Directional accuracy | 39.8 – 50.5 % | 42.2 – 51.3 % |

## 4. Results on the variant's full window (2026-05-09 → 2026-11-03, 179 days)

This window adds the falling May–July market, where buy & hold lost money.

| Signal | Sharpe | Sortino | Max DD | Return | Exposure |
|---|---|---|---|---|---|
| **Ensemble (primary)** | **−0.70** | −1.10 | −16.9 % | −10.1 % | 37 % |
| Ridge (sentiment + price) | −1.02 | −1.49 | −27.9 % | −16.3 % | 66 % |
| GBM (sentiment + price) | −0.14 | −0.24 | −14.1 % | −3.1 % | 35 % |
| Ridge (price only) | −1.66 | −2.29 | −24.5 % | −21.8 % | 67 % |
| Buy & hold | −0.01 | −0.02 | −28.7 % | −3.5 % | 100 % |

- **Ensemble RMSE ÷ random walk:** 0.998–1.033.
- **Diebold–Mariano:** no horizon significantly better; significantly worse at n+4 and n+5.
- **Sentiment ablation:** adding sentiment to the Ridge lowered error at 1 of 6 horizons.
- Full report: `runs/hist_2021_23/REPORT.md`.

## 5. Interpretation

1. **More training history helps, but only up to "no worse than a random walk".** The 2021–23 data moves the primary model
   from significantly worse than the random walk (n+0 to n+2) to indistinguishable from it. The trading Sharpe on the
   common window rises from −3.16 to 0.04. No horizon is significantly better than the random walk, and no signal beats
   buy & hold over the rally.
2. **The improvement is mainly regularisation, not a sentiment signal.** The price-only Ridge improves almost as much as
   the sentiment models (−1.62 → −0.08). Sentiment lowered error at only 1 of 6 horizons on the full window. With 60–175
   training days, the Apify-only models learned the May–July mean reversion; with 220–400 days, that pattern is diluted.
3. **The best-looking variant is unstable.** The sentiment Ridge reaches Sharpe 1.11 on the common window but −1.02 on the
   full window, which only adds two months. That swing is why the primary model was fixed in advance and is not replaced
   after seeing results.

**Caveats of the historical data:**

- The history ends 3.3 years before the test window.
- It was collected differently: a keyword stream versus Apify's "Top" sort.
- It has no engagement data.
- 2021–22 was a different market regime (the 69k peak and the 2022 bear market).

Rolling z-scored sentiment features reduce the level shift between the two sources, but do not remove it.

**Bottom line:** across both runs, X sentiment from these sources does not give a statistically reliable daily edge on
BTC. More data mainly stops the model from overfitting.

## 6. Sentiment engine: CryptoBERT instead of VADER

[`ElKulako/cryptobert`](https://huggingface.co/ElKulako/cryptobert) is a transformer fine-tuned on crypto social-media
posts. Each post's score is P(bullish) − P(bearish). The data, spam filter, features and models are identical to the main
run. The comparison was planned before its result was known, and VADER stays the primary engine either way.

| 2026-07-10 → 2026-11-03 (117 days) | VADER (main run) | CryptoBERT |
|---|---|---|
| Ensemble n+0: Sharpe / Sortino / Max DD / Return | −3.16 / −3.57 / −14.3 % / −12.8 % | −1.40 / −2.17 / −17.3 % / −10.8 % |
| Ensemble RMSE ÷ random walk, n+0 … n+5 | 1.031 – 1.100 | 1.041 – 1.115 (worse at every horizon) |
| Diebold–Mariano: significantly *worse* than random walk | n+0, n+1, n+2 | n+0 … n+3 |
| Sentiment ablation: horizons where sentiment lowers Ridge error | 3 of 6 | 1 of 6 |
| Ensemble IC, n+0 … n+5 | −0.12 to −0.25 | −0.13 to −0.24 |
| Buy & hold Sharpe | 1.88 | 1.88 |

**Mixed and not better.** The less negative Sharpe comes with higher forecast error at every horizon and a weaker
ablation, so no conclusion should be drawn from the Sharpe alone. Its standard error over 117 days is ≈ ±1.8. The
price-only Ridge is identical in both runs, which confirms that only the sentiment input changed.

**The two engines barely agree:**

- Post by post, they correlate at **ρ = 0.15**.
- Daily engagement-weighted scores correlate at 0.44.
- CryptoBERT is far more bullish: mean +0.34, against +0.12 for VADER.
- Each has visible failures. VADER reads "Satoshi lit a fire in cyberspace" as bearish; CryptoBERT fixes that, but still
  reads "21,000,000. Never more." as bearish.

A "sentiment signal" this sensitive to the measuring instrument is itself evidence that there is little stable
information in daily crypto-Twitter tone. Full report: `runs/cryptobert/REPORT.md`.

## Reproduce

```bat
python scripts\validate_tweet_dates.py path\to\Bitcoin_tweets.csv
mkdir data\hist_2021_23 && copy data\xsent.db data\hist_2021_23\xsent.db
set DATA_DIR=data/hist_2021_23
set OUT_DIR=outputs/runs/hist_2021_23
python run_pipeline.py historical --csv path\to\Bitcoin_tweets.csv --end 2026-05-08
python run_pipeline.py analyse

rem CryptoBERT (needs: pip install torch transformers). Clear the VADER scores in a copy so every post is re-scored.
mkdir data\cryptobert && copy data\xsent.db data\cryptobert\xsent.db
python -c "import sqlite3; c=sqlite3.connect('data/cryptobert/xsent.db'); c.execute('UPDATE tweets SET sent_compound=NULL'); c.commit()"
set DATA_DIR=data/cryptobert
set OUT_DIR=outputs/runs/cryptobert
set SENTIMENT_ENGINE=cryptobert
python run_pipeline.py analyse

set DATA_DIR=
set OUT_DIR=
set SENTIMENT_ENGINE=
python scripts\compare_runs.py outputs\runs\apify_only outputs\runs\cryptobert outputs\runs\hist_2021_23 --out outputs\RUN_COMPARISON.md
```
