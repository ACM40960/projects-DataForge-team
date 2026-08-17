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

A two stage forecasting model applied to eight AI semiconductor stocks. ARIMA
handles the linear part of the price series, XGBoost is trained on the errors
ARIMA leaves behind, and the two predictions are added together.

The short version of what we found: the hybrid does not beat rolling ARIMA on its
own, and neither of them beats simply guessing that tomorrow's price equals
today's. The rest of this file explains how we got there and why it happens.

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

Wang and Guo (2020) propose splitting the series into two parts:

```
y  =  L  +  N

L = the linear component is handled by ARIMA model
N = the nonlinear leftover is handled by XGBoost model
```

The assertion is that N remains learnable, and the three other studies corroborate
this claim. The common problem with all the four studies is that none of them has
tested their designs using AI semiconductor stocks, which is the most volatile
type of equities for today's particular period. None of the four studies provided
us a naive benchmark either to support our idea.

We picked eight tickers spanning the whole chip supply chain so the result would
not depend on one business model only: NVDA and AMD (GPUs), SMCI (servers), ARM
(architecture), MU and 000660.KS / SK Hynix (memory), AVGO (networking silicon),
TSM (fabrication). Daily closes, 2019 to 2026.

The reason why AI semiconductor stocks are selected is that they have been identified 
as the highest volatile class of equities in the present cycle. The cycles of demand for 
GPU and memory shortages, along with earning surprises, result in volatility that is hard 
for regular statistical models to capture. Both ARM and SMCI stocks have a daily return 
standard deviation of 4.80%, which is more than twice of TSM’s 2.36%. 
The value of excess kurtosis is 17.21% in the case of ARM.

<img width="1440" height="720" alt="volatility_comparison" src="https://github.com/user-attachments/assets/ad42a86f-078b-4afd-ba8a-36459a61852f" />


Volatility is precisely the key. A good hybrid design should be easier to prove where 
this degree of volatility exists within the series, and harder to falsify, since any 
flat or inactive forecast gets punished right away.

---

## Methodology

<img src="arima_xgboost_project/flow_diagram.png" alt="Flow diagram" width="500"/>

**Data collection and cleaning.** The daily bars were obtained from Yahoo Finance
using `auto_adjust=True` so that any splits and dividends are backward-adjusted
throughout the whole of the series. Otherwise, the ten for one split of NVDA in
June 2024 would be seen as a 90% crash overnight and would be classified by ARIMA
as a true event. No data cleaning was required: no rows were dropped and no NA
values were observed in any of the eight tickers.

**Examining the series prior to modelling.** Nothing remains motionless. In this
instance, neither the variance nor the mean stay constant when each series grows
by one to two orders of magnitude. Because of this, ARIMA cannot accept raw
pricing at this time. The returns cluster in volatility, with calm intervals
within 3% interspersed with periods when 10% days are normal. The tails are fat,
with excess kurtosis reaching 17.21 for ARM. The greatest autocorrelation in the
differenced series is 0.136, whereas six out of eight tickers show no
autocorrelation greater than 0.14:

![ACF summary](arima_xgboost_project/results/plots/acf_summary/acf_summary.png)

There is almost no linear structure for ARIMA to find and correspondingly nothing
was left in the residuals for XGBoost to learn.

**Choosing how much to difference.** Augmented Dickey Fuller tests set the `d`
variable by ticker instead of using a fixed `d` parameter. The null hypothesis
here is "non-stationary", so a small p-value is the desired result, which works
counter to people's expectations and needs to be stated explicitly. Seven tickers
become stationary at one difference. MU fails at `d = 1` with p = 0.5515 and
succeeds at `d = 2`.

**Choosing the ARIMA sequence.** Using the BIC criterion instead of the AIC
criterion, a grid search of all 36 possible combinations of p and q, ranging from
0 to 5, for every ticker. The ACF and PACF plots demonstrate that the penalty term
associated with AIC is insufficient when a time series is almost identical to
white noise, which is why this decision was made. Because of this the AIC
frequently selects larger models that have marginally better historical fitting
but marginally predict worse. ARM chooses the random walk, ARIMA(0,1,0), meaning
BIC determined that no ARMA structure was worth estimating.

**Examining what the model overlooked.** There are two Ljung-Box examinations that
pose truly different questions.

| Test | What it asks | Rejects on |
|---|---|---|
| Ljung-Box on residuals | is the **direction** of the error patterned? | 5 of 8 |
| Ljung-Box on squared residuals | is the **size** of the error patterned? | 8 out of 8, p < 0.0001 |

The gap between the two rows represents the finding of the project. The direction
is nearly unforeseeable. The formal definition of volatility clustering is that
magnitude is highly predictable: tumultuous days follow turbulent days regardless
of the direction of the price. Since XGBoost was trained on the signed residual,
it was targeted at the nearly signal-free half of the problem.

**Adjusting the residuals.** XGBoost uses features to train on ARIMA's training
period errors. Since ARIMA only considers historical price values, it is
fundamentally blind to lagged returns, rolling volatility, moving average ratios,
volume, intraday range, momentum, and day of week. The row for day `t` only
includes what was known at the end of `t-1` because each feature is moved ahead by
one day. Without that change, the model reads the response from the question
paper, making the results useless. The rolling ARIMA forecast is then updated with
the revision.

**It is significant that the evaluation process was altered in the middle of the
project.** With a flat line and MAPE ranging from 20.9% to 49.4%, our first design
anticipated all 374 test days in a single call. That reduced to 2.87% when we
switched to rolling one step forward, which appeared to be a success until we
looked at what a random walk scored using the same protocol: 2.86%. The shorter
horizon, rather than the model, was the improvement. Both protocols are reported
and are still included in the code.

---

## Results

Mean MAPE across the eight tickers, same 80/20 chronological split:

| Model | MAPE | Notes |
|---|---|---|
| ARIMA multi-step | 31.15% | one forecast for 374 days, no updating |
| **Naive random walk** | **2.86%** | tomorrow's price is today's price |
| Rolling ARIMA | 2.87% | forecast one day, observe, repeat |
| Hybrid (ARIMA + XGBoost) | 3.15% | worse than rolling ARIMA on all 8 tickers |

The naive benchmark is the whole story.

Moving from multi-step to rolling looks like a 28 point improvement, but a model
with no parameters, which simply repeats yesterday's price, scores 2.86% under the
same protocol. That accounts for 99.9% of the apparent gain. What improved was the
forecast horizon, not the forecasting.

ARM provides an example here. For ARM, the rolling ARIMA MAPE and the naive
benchmark MAPE are equal at **3.444295%**, identical to six decimal points,
because BIC selected ARIMA(0,1,0) and that is the random walk model. Out of 36
possible order combinations, it independently decided "do not model this".

Per ticker:

| Ticker | Multi-Step MAPE | Rolling ARIMA MAPE | Hybrid MAPE |
|---|---|---|---|
| TSM | 27.69% | **1.93%** | 2.17% |
| NVDA | 20.92% | **2.04%** | 2.18% |
| AVGO | 36.59% | **2.28%** | 2.66% |
| AMD | 32.16% | **2.73%** | 2.93% |
| 000660.KS | 49.35% | **3.05%** | 3.34% |
| MU | 39.35% | **3.31%** | 3.51% |
| ARM | 22.18% | **3.44%** | 3.92% |
| SMCI | 20.98% | **4.20%** | 4.49% |
| **Mean** | **31.15%** | **2.87%** | **3.15%** |

The average contribution from XGBoost is **-0.28 percentage points**. On every
ticker, the outlook gets worse due to the correction. It is confirmed by
directional accuracy: the hybrid accurately predicts the direction of the
following day 45% to 52% of the time, averaging 49.2%, which is a coin flip,
across the eight tickers. Only because daily moves are tiny does a 2.87% MAPE seem
accurate.

The variance of ARIMA's residuals has structure, whereas the mean has very little.
The extent of tomorrow's mistake is really predictable because ARCH tests reject on
8 out of 8 tickers at p < 0.0001. However, direction is where the signal is
weakest, therefore XGBoost was trained to predict the signed residual, which is
direction plus magnitude. As a result, the model focused on the nearly empty half
of the problem, but it never addressed the other half. The reason the correction is
negative rather than just zero is because it fitted noise in the training residuals
and carried that noise into the test period.

---

## Limitations

- **One test window.** From December 2024 to June 2026, the trend within the
  sector was upward. One regime is insufficient to make any generalisations about
  the technique.
- **ARM has the least historical data**, at a third of the others. It listed in
  September 2023, giving 689 observations against roughly 1870 for the others, and
  only 137 test days. Its ARIMA(0,1,0) selection could be attributed to the
  limited sample size.
- **MU is over differenced.** ADF imposes `d = 2`, which produces a lag 1
  autocorrelation of -0.445, the standard signature of differencing too far. The
  test result and the diagnostic plot are inconsistent, and we stuck with the
  former.
- **Model parameters remain fixed.** `refit=False` assures us of a fair rolling
  forecast. However, it means that the model does not adapt throughout the entire
  18 month test period.
- **There is no wavelet decomposition.** Before entering the series into ARIMA,
  Wang and Guo used discrete wavelet decomposition to extract the component that
  best suited each model. We skipped the step, which is most likely the only
  reason for the disparity.
- Compared to the out-of-sample errors it is supposed to correct, **XGBoost trains
  on in-sample residuals**, which are consistently smaller and better behaved. In
  order to allow the model to learn from truly out-of-sample residuals, we
  experimented with a 60/20/20 split. We decided to stick with the simpler design
  because the coverage decreased from 80% to 65% and the residual correlation
  decreased from 0.205 to 0.130.

---

## Future Work

- **Focus on the variance rather than the mean.** This is the instant result of
  the diagnostics. Since ARCH effects reject on all eight tickers, volatility is
  predictable yet direction is unknown. Fitting GARCH or EGARCH to the ARIMA
  residuals models the amount that is genuinely predictable, benchmarked against
  HAR-RV.
- **Force `d = 1` for MU** and compare with the ADF-driven `d = 2` to ascertain
  the actual cost of over-differencing in forecast terms.
- **It is necessary to include the wavelet step.** Wang and Guo employ a discrete
  wavelet transform before ARIMA. Because we omitted it, our results most likely
  differ from theirs.
- **Walk-forward validation** is carried out across many non-overlapping periods
  that contain at least one decline, as opposed to a single test period that
  conveniently corresponds with a sector-wide increase.

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

Run the scripts in this order. `arima_forecasting.py` has to come before the last
two, since both read the orders it selects from `arima_fit_summary.csv`:

```bash
python data_collection.py        # ~1 min, needs internet. optional, the CSVs are committed
python exploratory_analysis.py   # ~20 sec
python arima_forecasting.py      # ~6 min, this is the grid search
python acf_summary_plot.py       # ~5 sec
python hybrid_model.py           # ~1.5 min
```

A total of almost eight minutes. To prevent the dataset from drifting, `END_DATE`
in `data_collection.py` is set to 2026-06-14. You can fully exclude
`data_collection.py` since the cleaned CSVs have been committed and re-downloading
carries the risk of Yahoo Finance altering a historical bar.

**Reproducibility note.** The ACF summary, BIC ordering, ADF results, and
multi-step and rolling MAPE figures all replicate exactly. Hybrid MAPE varies by
up to 0.15 percentage points per ticker between machines because thread scheduling
changes the order of floating point summing within XGBoost and the optimiser
arrives at slightly different locations. The mean stays at 3.15% and the ranking
stays the same.

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
        ├── acf_pacf_plots/     ACF and PACF per ticker
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
   Learning and its Application in Stock Price Prediction.* BCP Business and
   Management, 26, 400 to 408.

Price data from [Yahoo Finance](https://finance.yahoo.com) via the `yfinance` package.

---

## Authors
**Asawari Lad**  
Module: ACM 40960 — Mathematical Modelling  
University College Dublin

**Soham Barve**  
Module: ACM 40960 — Mathematical Modelling  
University College Dublin

**Disclaimer.** This project is for academic purposes only. Nothing here
constitutes financial advice, and the results explicitly show that these models do
not beat a random walk.

---

## License

MIT. See [LICENSE](LICENSE).
