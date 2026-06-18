---

## Branch Guide

| Branch | What is in it | Owner |
|---|---|---|
| `main` | GitHub Classroom bot template only — do not touch | Module repo |
| `soham-arima` | Stable Week 1 code — closing price plots with train/test split line | Soham |
| `soham-arima-events` | Updated Week 1 code — closing price plots with real world event markers instead of train/test split. Events marked: COVID crash (Mar 2020), ChatGPT launch (Nov 2022), ARM IPO (Sep 2023), NVDA stock split (Jun 2024) | Soham |
| `asawari-xgboost` | XGBoost residual model and feature engineering | Asawari |

### Why are there two versions of the closing price plot?

The `soham-arima` branch shows a train/test split line on the charts.
We replaced this in `soham-arima-events` because our model uses a rolling
window — there is no single fixed split point. The event markers version
is more honest and tells a better story about why these stocks are volatile.
