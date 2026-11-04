Common out-of-sample window: **2026-07-10 → 2026-11-03** (117 days at n+0). Costs 10 bp per unit turnover, long/flat.

### Forecast skill (RMSE ÷ random walk; < 1 beats it)

| Run | Model | n+0 | n+1 | n+2 | n+3 | n+4 | n+5 |
|---|---|---|---|---|---|---|---|
| apify_only | ensemble | 1.031 | 1.057 | 1.074 | 1.065 | 1.088 | 1.100 |
| apify_only | gbm | 1.046 | 1.079 | 1.100 | 1.082 | 1.112 | 1.138 |
| apify_only | ridge | 1.025 | 1.043 | 1.054 | 1.060 | 1.079 | 1.076 |
| apify_only | ridge_price | 1.026 | 1.047 | 1.053 | 1.059 | 1.070 | 1.077 |
| cryptobert | ensemble | 1.041 | 1.060 | 1.082 | 1.086 | 1.102 | 1.115 |
| cryptobert | gbm | 1.060 | 1.085 | 1.109 | 1.119 | 1.152 | 1.178 |
| cryptobert | ridge | 1.028 | 1.043 | 1.065 | 1.069 | 1.077 | 1.078 |
| cryptobert | ridge_price | 1.026 | 1.047 | 1.053 | 1.059 | 1.070 | 1.077 |
| hist_2021_23 | ensemble | 0.999 | 0.996 | 1.025 | 1.022 | 1.018 | 1.022 |
| hist_2021_23 | gbm | 1.006 | 1.003 | 1.053 | 1.054 | 1.052 | 1.063 |
| hist_2021_23 | ridge | 1.003 | 0.999 | 1.006 | 0.999 | 0.997 | 0.995 |
| hist_2021_23 | ridge_price | 1.004 | 1.016 | 1.005 | 0.998 | 0.996 | 0.995 |

### Directional accuracy

| Run | Model | n+0 | n+1 | n+2 | n+3 | n+4 | n+5 |
|---|---|---|---|---|---|---|---|
| apify_only | ensemble | 43.6% | 42.6% | 39.8% | 49.5% | 50.5% | 45.8% |
| apify_only | gbm | 44.4% | 43.5% | 43.4% | 50.5% | 55.0% | 44.9% |
| apify_only | ridge | 44.4% | 44.3% | 45.1% | 48.6% | 43.1% | 38.3% |
| apify_only | ridge_price | 44.4% | 44.3% | 46.9% | 49.5% | 45.0% | 37.4% |
| cryptobert | ensemble | 44.4% | 41.7% | 42.5% | 45.0% | 41.3% | 40.2% |
| cryptobert | gbm | 42.7% | 45.2% | 43.4% | 45.9% | 46.8% | 40.2% |
| cryptobert | ridge | 45.3% | 44.3% | 40.7% | 48.6% | 47.7% | 38.3% |
| cryptobert | ridge_price | 44.4% | 44.3% | 46.9% | 49.5% | 45.0% | 37.4% |
| hist_2021_23 | ensemble | 50.4% | 51.3% | 47.8% | 44.1% | 42.2% | 46.7% |
| hist_2021_23 | gbm | 49.6% | 53.0% | 48.7% | 41.4% | 44.0% | 47.7% |
| hist_2021_23 | ridge | 50.4% | 47.8% | 46.9% | 45.0% | 43.1% | 44.9% |
| hist_2021_23 | ridge_price | 44.4% | 47.0% | 47.8% | 45.0% | 43.1% | 44.9% |

### Trading the n+0 forecast

| Run | Signal | Sharpe | Sortino | Max DD | Total return | Exposure |
|---|---|---|---|---|---|---|
| apify_only | buy_and_hold | 1.88 | 3.47 | -11.5% | +22.4% | 100% |
| apify_only | ensemble | -3.16 | -3.57 | -14.3% | -12.8% | 27% |
| apify_only | gbm | -1.24 | -1.89 | -15.5% | -8.5% | 38% |
| apify_only | ridge | -2.77 | -3.14 | -11.7% | -10.5% | 23% |
| apify_only | ridge_price | -1.62 | -2.41 | -12.0% | -9.2% | 25% |
| cryptobert | buy_and_hold | 1.88 | 3.47 | -11.5% | +22.4% | 100% |
| cryptobert | ensemble | -1.40 | -2.17 | -17.3% | -10.8% | 40% |
| cryptobert | gbm | -1.68 | -2.57 | -14.5% | -12.6% | 44% |
| cryptobert | ridge | -3.45 | -3.77 | -14.3% | -14.0% | 26% |
| cryptobert | ridge_price | -1.62 | -2.41 | -12.0% | -9.2% | 25% |
| hist_2021_23 | buy_and_hold | 1.88 | 3.47 | -11.5% | +22.4% | 100% |
| hist_2021_23 | ensemble | 0.04 | 0.07 | -16.9% | -1.1% | 48% |
| hist_2021_23 | gbm | 0.52 | 0.95 | -14.1% | +3.6% | 45% |
| hist_2021_23 | ridge | 1.11 | 1.95 | -11.5% | +10.4% | 72% |
| hist_2021_23 | ridge_price | -0.08 | -0.13 | -9.3% | -2.0% | 71% |
