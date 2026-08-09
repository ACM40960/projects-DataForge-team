# ARIMA Output Files - Notes for XGBoost Stage

Each `{ticker}_arima_output.csv` in this folder has 4 columns: `Date`, `Actual`, `Predicted`, `Residual`.

`Predicted` is ARIMA's linear component (the `L` in y = L + N). `Residual` is `Actual - Predicted` - the leftover nonlinear part (the `N`) that the XGBoost stage is meant to model.

## Which tickers actually need XGBoost

These tickers still have a detectable pattern left in their residuals (Ljung-Box test failed) - there's real structure here for XGBoost to learn:

- **AMD** (order ARIMA(2,1,2), ljung-box p=0.0001)
- **SMCI** (order ARIMA(0,1,4), ljung-box p=0.0003)
- **MU** (order ARIMA(2,2,4), ljung-box p=0.0000)
- **TSM** (order ARIMA(0,1,1), ljung-box p=0.0000)
- **000660.KS** (order ARIMA(5,1,3), ljung-box p=0.0000)

These tickers' residuals already look like random noise - ARIMA pretty much captured everything explainable, so XGBoost likely won't add much here (not a bug, just not much left to model):

- **NVDA** (order ARIMA(0,1,4), ljung-box p=0.8561)
- **ARM** (order ARIMA(0,1,0), ljung-box p=0.1863)
- **AVGO** (order ARIMA(4,1,0), ljung-box p=0.1946)

## Volatility clustering (ARCH effect)

All of these show volatility clustering in the residuals - periods of calm vs turbulent stretches - which is a variance pattern ARIMA structurally can't fix (it only models the mean): NVDA, AMD, SMCI, ARM, MU, AVGO, TSM, 000660.KS


## Forecast accuracy reference (30-day holdout)

| Ticker | Order | RMSE | MAE | MAPE |
|---|---|---|---|---|
| NVDA | ARIMA(0,1,4) | 15.89 | 13.52 | 6.17% |
| AMD | ARIMA(2,1,2) | 116.51 | 105.87 | 22.08% |
| SMCI | ARIMA(0,1,4) | 10.91 | 8.78 | 22.00% |
| ARM | ARIMA(0,1,0) | 105.38 | 78.50 | 22.80% |
| MU | ARIMA(2,2,4) | 311.61 | 281.81 | 32.15% |
| AVGO | ARIMA(4,1,0) | 24.51 | 16.74 | 3.96% |
| TSM | ARIMA(0,1,1) | 23.56 | 18.84 | 4.44% |
| 000660.KS | ARIMA(5,1,3) | 713397.29 | 642615.56 | 31.57% |