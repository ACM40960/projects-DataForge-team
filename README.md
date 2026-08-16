[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/-bKyY6qM)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-2e0aaae1b6195c2367325f4f02e2d04e9abb55f0b24a779b69b11b9e10269abc.svg)](https://classroom.github.com/online_ide?assignment_repo_id=23948756&assignment_repo_type=AssignmentRepo)

<h1 align="center">A Hybrid ARIMA-XGBoost Model for Forecasting Volatile AI Semiconductor Stocks</h1>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/statsmodels-0.14-orange.svg" alt="statsmodels">
  <img src="https://img.shields.io/badge/xgboost-2.0-green.svg" alt="xgboost">
  <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License">
  <img src="https://img.shields.io/badge/runtime-~8%20min-lightgrey.svg" alt="Runtime">
</p>

Two-stage forecasting of daily closing prices for eight AI semiconductor stocks.
ARIMA models the linear trend, XGBoost learns the errors ARIMA leaves behind, and
the two predictions are added together.

**Headline result: the hybrid does not beat rolling ARIMA alone, and neither beats a
random walk. We report why, and where the predictable structure actually sits.**

---

## Table of Contents

- [Motivation](#motivation)
- [Methodology](#methodology)
- [Results](#results)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Reproduction](#reproduction)
- [Directory Structure](#directory-structure)
- [References](#references)
- [Authors](#authors)
- [License](#license)

---

## Motivation

Wang & Guo (2020) propose a two-stage design in which the price is split into a
linear part and a leftover:

```
y  =  L  +  N

L = the linear trend, predicted by ARIMA
N = the leftover ARIMA misses, predicted by XGBoost
```

The bet is that **N contains learnable structure**. Their results, and those of
three related papers, suggest it does. None of the four tested the method on AI
semiconductor stocks, and none reported a naive benchmark.

We tested it on eight tickers covering the whole chip supply chain: NVDA and AMD
(GPUs), SMCI (servers), ARM (architecture), MU and 000660.KS / SK Hynix (memory),
AVGO (networking silicon), TSM (fabrication). Daily closes, 2019 to 2026.

---

## Methodology

<img src="arima_xgboost_project/flow_diagram.png" alt="Flow diagram" width="500"/>

**Collecting and cleaning the data.** Daily bars come from Yahoo Finance with
`auto_adjust=True`, so splits and dividends are applied backwards through the
series. Without it, NVDA's June 2024 ten-for-one split appears as a 90% overnight
crash and ARIMA would model it as a real event. Cleaning found nothing to remove:
zero missing values across all eight tickers.

**Looking at the series before modelling.** None of them is stationary, which
rules out feeding raw prices to ARIMA. Returns cluster in volatility — calm
stretches alternating with turbulent ones — and the tails are fat, with excess
kurtosis up to 17.21 for ARM. Across the differenced series, the strongest
autocorrelation anywhere is **0.136**, and six of eight tickers have nothing above 0.14:

![ACF summary](arima_xgboost_project/results/plots/acf_summary/acf_summary.png)

There is almost no linear structure for ARIMA to find, and correspondingly
nothing left in the residuals for XGBoost to learn.

**Choosing how much to difference.** ADF tests decide `d` per ticker rather than
assuming a value. Seven need one difference; MU needs two.

**Selecting the ARIMA order.** A grid search over 36 combinations of p and q
picks the best fit on **BIC** rather than AIC, because AIC over-parameterises
when a series is close to white noise, which the ACF plots show several of these are.

**Checking what the model missed.** Two Ljung-Box tests separate the questions
that matter: is the *direction* of the errors patterned, and is their *size*?

| Test | What it asks | Rejects on |
|---|---|---|
| Ljung-Box on residuals | is the **direction** of the error patterned? | 5 of 8 |
| Ljung-Box on squared residuals | is the **size** of the error patterned? | 8 of 8, p < 0.0001 |

Direction is close to unpredictable. Magnitude is strongly predictable — volatility
clustering: turbulent days follow turbulent days regardless of which way the price
goes. XGBoost was trained on the signed residual, so it was aimed at the half of
the problem with almost no signal.

**Correcting the residuals.** XGBoost trains on ARIMA's training-period errors
using features ARIMA cannot see: lagged returns, rolling volatility, moving
average ratios, volume, intraday range, momentum, and ARIMA residual lags. All
features are lagged one day to prevent look-ahead bias, and the correction is
added to the rolling ARIMA forecast.

**The evaluation protocol changed mid-project**, and it matters. Our first design
forecast all 374 test days in one call, which produces a flat line and MAPE
between 20.9% and 49.4%. Switching to rolling one-step-ahead dropped that to
2.87%, which looked like success until we checked what a random walk scores under
the same protocol: **2.86%**. The improvement was the shorter horizon, not the model.
Both protocols remain in the code and both are reported.

---

## Results

Mean MAPE across the eight tickers, same 80/20 chronological split:

| Model | MAPE | Notes |
|---|---|---|
| ARIMA multi-step | 31.15% | one forecast for 374 days, no updating |
| **Naive random walk** | **2.86%** | tomorrow's price is today's price |
| Rolling ARIMA | 2.87% | forecast one day, observe, repeat |
| Hybrid (ARIMA + XGBoost) | 3.21% | worse than rolling ARIMA on all 8 tickers |

The naive benchmark is the whole story. Moving from multi-step to rolling looks
like a 28-point improvement, but a model with no parameters — which simply repeats
yesterday's price — scores 2.86% under the same protocol. That accounts for 99.9%
of the apparent gain. What improved was the forecast horizon, not the forecasting.

Per ticker:

| Ticker | Multi-Step MAPE | Rolling ARIMA MAPE | Hybrid MAPE |
|---|---|---|---|
| TSM | 27.69% | **1.93%** | 2.22% |
| NVDA | 20.92% | **2.04%** | 2.21% |
| AVGO | 36.59% | **2.28%** | 2.55% |
| AMD | 32.16% | **2.73%** | 2.95% |
| 000660.KS | 49.35% | **3.05%** | 3.37% |
| MU | 39.35% | **3.31%** | 3.41% |
| ARM | 22.18% | **3.44%** | 4.37% |
| SMCI | 20.98% | **4.20%** | 4.61% |

**XGBoost contributes −0.34 percentage points on average.** The correction makes
the forecast worse on every ticker. Directional accuracy confirms it — across the
eight tickers the hybrid calls the next day's direction correctly **45% to 54%**
of the time, which is a coin flip. A 2.87% MAPE sounds accurate only because
daily moves are small.

---

## Limitations

- **One test window.** December 2024 to June 2026 was a sector-wide uptrend. A
  single regime is not enough to claim anything general about the method.
- **ARM has a third of the history.** It listed in September 2023, giving 689
  observations against roughly 1,870 for the others. Its ARIMA(0,1,0) selection
  may reflect the short sample as much as the data.
- **MU is over-differenced.** ADF forces `d=2`, which produces a lag-1
  autocorrelation of −0.445, the standard signature of differencing too far.
- **Coefficients are frozen after training.** `refit=False` keeps the rolling
  forecast honest but means the model never adapts to the test period.
- **No wavelet decomposition.** Wang & Guo apply one before ARIMA. Leaving it out
  is the most likely reason our results diverge from theirs.
- **XGBoost trains on in-sample residuals**, which are systematically smaller
  than out-of-sample errors. We tested a 60/20/20 split to fix this and it made
  matters worse, so we kept the simpler design and documented the trade-off.

---

## Future Work

- **Target the variance instead of the mean.** ARCH tests reject on all eight
  tickers, so volatility is predictable even though direction is not. GARCH or
  EGARCH on the residuals is the natural next step, benchmarked against HAR-RV.
- **Force `d=1` for MU.** Its ADF test fails at one difference, but `d=2`
  produces a lag-1 autocorrelation of −0.445, the signature of over-differencing.
- **Add the wavelet step.** Wang & Guo apply a discrete wavelet transform before
  ARIMA. We omitted it, and it is the most likely reason our results differ from theirs.
- **Walk-forward validation** across several windows, rather than a single test
  period that happens to be a sector-wide uptrend.

---

## Reproduction

Clone the repository:

```bash
git clone https://github.com/ACM40960/projects-DataForge-team.git
cd projects-DataForge-team/arima_xgboost_project
```

Install the dependencies:

```bash
pip install pandas numpy statsmodels xgboost matplotlib scikit-learn yfinance
```

Run the pipeline in order (`arima_forecasting.py` must come before the last two,
since they read the orders it selects):

```bash
python data_collection.py       # ~1 min, needs internet. optional, CSVs are committed
python exploratory_analysis.py  # ~20 sec
python arima_forecasting.py     # ~6 min
python acf_summary_plot.py      # ~5 sec
python hybrid_model.py          # ~1.5 min
```

About 8 minutes end to end. `END_DATE` in `data_collection.py` is pinned to
`2026-06-14` so the dataset stays reproducible.

**Note on reproducibility.** XGBoost results vary in the third decimal place
across machines even with `random_state=42`, because thread scheduling changes
floating-point summation order. Expect the hybrid mean between 3.15% and 3.25%.
Multi-step, naive, and rolling are stable to two decimal places.

---

## Directory Structure

```
arima_xgboost_project/
├── data_collection.py          downloads 8 tickers from Yahoo Finance, cleans them,
│                               writes one CSV per stock
├── exploratory_analysis.py     plots closing prices and daily returns, computes
│                               rolling volatility and summary statistics
├── arima_forecasting.py        tests stationarity, picks the differencing order,
│                               grid searches p and q on BIC, fits each model,
│                               runs the residual diagnostics
├── acf_summary_plot.py         measures the strongest autocorrelation per ticker
│                               and puts all eight on one chart
├── hybrid_model.py             builds the rolling forecast, trains XGBoost on the
│                               residuals, compares the three models
├── flow_diagram.png            the pipeline figure used above
├── data/                       8 cleaned CSVs, one per ticker
├── residuals/                  ARIMA output files (Date, Actual, Predicted, Residual)
└── results/
    ├── tables/                 summary statistics, ADF results, model comparison
    └── plots/
        ├── closing_prices/     price series with event markers
        ├── daily_returns/      returns and 21-day rolling volatility
        ├── acf_pacf_plots/     ACF/PACF and forecast comparison per ticker
        ├── acf_summary/        strongest autocorrelation, all 8 tickers
        └── forecast_comparison/ multi-step vs rolling vs hybrid
```

---

## References

1. Adebiyi, A.A., Adewumi, A.O., Ayo, C.K. (2014). *Stock Price Prediction Using
   the ARIMA Model.* 16th UKSim-AMSS International Conference on Computer
   Modelling and Simulation, IEEE.
2. Wang, Y., Guo, Y. (2020). *Forecasting Method of Stock Market Volatility Based
   on Mixed Model of ARIMA and XGBoost.* China Communications, IEEE.
3. Somkunwar, R.K., Pimpalkar, A., Srivastava, V. (2024). *A Novel Approach for
   Accurate Stock Market Forecasting by Integrating ARIMA and XGBoost.* IEEE SCEECS.
4. Zhang, F., Chen, L., Yu, J. (2022). *A Two-Stage ARIMA Model via Machine
   Learning and its Application in Stock Price Prediction.* BCP Business &
   Management, 26, 400–408.

Price data from [Yahoo Finance](https://finance.yahoo.com) via the `yfinance` package.

---

## Authors

**Soham Barve** — data collection, exploratory analysis, ARIMA modelling and diagnostics  
[soham.barve@ucdconnect.ie](mailto:soham.barve@ucdconnect.ie)

**Asawari Lad** — residual correction with XGBoost and hybrid evaluation  
[asawarilad2105@gmail.com](mailto:asawarilad2105@gmail.com)

ACM 40960 Mathematical Modelling, University College Dublin  
Supervisor: Dr. Sarp Akcay

**Disclaimer.** This project is for academic purposes only. Nothing here
constitutes financial advice, and the results explicitly show that these models
do not beat a random walk.

---

## License

MIT. See [LICENSE](LICENSE).
