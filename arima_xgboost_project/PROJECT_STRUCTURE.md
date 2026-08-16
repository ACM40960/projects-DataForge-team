# Project Structure

```
arima_xgboost_project/
│
├── data_collection.py          # Downloads & cleans Yahoo Finance data for 8 tickers
├── exploratory_analysis.py     # Closing price plots + daily return / volatility plots
├── arima_forecasting.py        # ADF stationarity tests + ARIMA fitting (BIC grid search)
├── hybrid_model.py             # Rolling ARIMA + XGBoost hybrid model + comparison plots
│
├── data/                       # Raw cleaned CSVs (one per ticker)
│   ├── NVDA.csv
│   ├── AMD.csv
│   ├── SMCI.csv
│   ├── ARM.csv
│   ├── MU.csv
│   ├── AVGO.csv
│   ├── TSM.csv
│   └── 000660.KS.csv
│
├── residuals/                  # ARIMA output files (handoff to XGBoost stage)
│   ├── NVDA_arima_output.csv   # Columns: Date, Actual, Predicted, Residual
│   ├── AMD_arima_output.csv
│   ├── SMCI_arima_output.csv
│   ├── ARM_arima_output.csv
│   ├── MU_arima_output.csv
│   ├── AVGO_arima_output.csv
│   ├── TSM_arima_output.csv
│   └── 000660.KS_arima_output.csv
│
├── results/
│   ├── tables/                 # All CSVs produced by the pipeline
│   │   ├── cleaning_report.csv
│   │   ├── closing_price_summary.csv
│   │   ├── daily_returns_summary.csv
│   │   ├── adf_test_results.csv
│   │   ├── adf_second_difference_results.csv
│   │   ├── adf_final_d_selection.csv
│   │   ├── arima_fit_summary.csv
│   │   └── model_comparison.csv
│   │
│   └── plots/
│       ├── closing_prices/         # 8 closing price charts with event markers
│       ├── daily_returns/          # 8 daily return + rolling volatility charts
│       ├── acf_pacf_plots/         # 8 ACF/PACF diagnostic plots per ticker
│       └── forecast_comparison/    # 8 forecast comparison charts (3 models)
│
├── README.md
├── ARIMA_MODELING_REPORT.md    # Detailed ARIMA methodology and results
├── exploratory_analysis_notes.md
├── notes_stock_analysis.md
└── flow_diagram.png            # Pipeline diagram
```

## Run Order

1. `data_collection.py`      — fetch and clean data
2. `exploratory_analysis.py` — generate EDA plots and summary tables
3. `arima_forecasting.py`    — run ADF tests and fit ARIMA models
4. `hybrid_model.py`         — run hybrid model and produce final comparison
