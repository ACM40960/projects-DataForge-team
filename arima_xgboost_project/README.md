# A Hybrid ARIMA XGBoost Model for Forecasting Volatile AI Semiconductor Stocks
## Overview
This model builds a hybrid time series forecasting model for 8 semi-conductors stocks: NVDA, AMD, SMCI, ARM, MU, AVGO, TSM, and 000660.KS. The aim is to forecaste the colsing prices more accurately than a ARIMA model by combining it with XGBoost. 
ARIMA captures the linear trend in stock prices but has two key limitations:
1) Multi-step forecasts flatten out
2) It cannot model volatility clustering

The hybrid approach works in two stages.
1) Rolling one-step-ahead ARIMA forecast is generated for the 20% test period using orders selected by BIC grid search
2) XGBoost is trained on ARIMA's in-sample residuals, the errors ARIMA could not explain and predicts those residuals on the test set

The final forecast is the sum of both:
Hybrid Forecast = Rolling ARIMA Prediction + XGBoost Residual Correction
The hybrid model outperforms plain ARIMA on all 8 stocks, reducing MAPE from a range of 20–49% down to 2–5%.
## Methodology
...
![Flow Diagram](flow_diagram.png)
