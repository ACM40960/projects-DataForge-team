import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    from statsmodels.tsa.arima.model import ARIMA
except (ImportError, ModuleNotFoundError):
    from statsmodels.tsa.arima_model import ARIMA
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

TICKERS = [
    "NVDA", "AMD", "SMCI", "ARM",
    "MU", "AVGO", "TSM", "000660.KS"
]

DATA_DIR = "data"
TABLES_DIR = os.path.join("results", "tables")
PLOTS_DIR = os.path.join("results", "plots", "task6_hybrid_forecast")

TEST_SIZE = 0.2
ROLLING_WINDOW = 21
LAG_DAYS = [1, 2, 3, 5, 10]
MA_WINDOWS = [5, 10, 20]


def load_series(ticker):
    path = os.path.join(DATA_DIR, f"{ticker}.csv")
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def split_train_test(df, test_size=TEST_SIZE):
    n_test = int(len(df) * test_size)
    train = df.iloc[:-n_test].reset_index(drop=True)
    test = df.iloc[-n_test:].reset_index(drop=True)
    return train, test


def load_arima_order(ticker):
    # reads Soham's best order from arima_model.py's BIC grid search
    # so we don't run a second grid search here
    path = os.path.join(TABLES_DIR, "arima_fit_summary.csv")
    if not os.path.exists(path):
        print(f"  arima_fit_summary.csv not found — defaulting to (1,1,1) for {ticker}")
        return (1, 1, 1)

    summary = pd.read_csv(path)
    row = summary[summary["ticker"] == ticker]
    if row.empty:
        print(f"  {ticker} not in arima_fit_summary.csv — defaulting to (1,1,1)")
        return (1, 1, 1)

    p = int(row.iloc[0]["order_p"])
    d = int(row.iloc[0]["order_d"])
    q = int(row.iloc[0]["order_q"])
    return (p, d, q)


def fit_arima(train_close, order):
    # fits ARIMA on training period only using Soham's BIC-selected order
    return ARIMA(train_close, order=order).fit()


def rolling_arima_forecast(train_close, test_close, order):
    """One-step-ahead rolling forecast through the test period.

    Fits ARIMA once on training data, then for each test day:
      - forecasts 1 step ahead using everything seen so far
      - appends the actual observed price before moving to the next day

    This avoids the flat-line problem of multi-step forecasting.
    Uses refit=False (parameters fixed, model state updated) for speed.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = ARIMA(train_close, order=order).fit()

    preds = []
    current_fit = fitted

    for i, actual_price in enumerate(test_close):
        pred = float(current_fit.forecast(steps=1).iloc[0])
        preds.append(pred)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            current_fit = current_fit.append([actual_price], refit=False)

        if (i + 1) % 50 == 0:
            print(f"    rolling forecast: {i + 1}/{len(test_close)} steps done")

    return pd.Series(preds).reset_index(drop=True), fitted


def build_features(df):
    feat = pd.DataFrame(index=df.index)

    for lag in LAG_DAYS:
        feat[f"return_lag_{lag}"] = df["Close"].pct_change(lag) * 100

    feat["rolling_vol"] = df["Close"].pct_change().rolling(ROLLING_WINDOW).std() * 100

    for window in MA_WINDOWS:
        ma = df["Close"].rolling(window).mean()
        feat[f"ma_{window}"] = ma
        feat[f"price_to_ma_{window}"] = df["Close"] / ma

    if "Volume" in df.columns:
        feat["volume_change"] = df["Volume"].pct_change() * 100
        feat["volume_ma_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()

    if "High" in df.columns and "Low" in df.columns:
        feat["hl_range_pct"] = (df["High"] - df["Low"]) / df["Close"] * 100

    feat["momentum"] = df["Close"].pct_change(5) - df["Close"].pct_change(10)

    # shift by 1 so features for day t only use data up to day t-1
    feat = feat.shift(1)
    feat["day_of_week"] = df["Date"].dt.dayofweek

    return feat


def mape(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)
    return np.mean(np.abs((actual - predicted) / actual)) * 100


def rmse(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)
    return np.sqrt(np.mean((actual - predicted) ** 2))


def mae(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)
    return np.mean(np.abs(actual - predicted))


def run_ticker(ticker):
    df = load_series(ticker)
    train_df, test_df = split_train_test(df)

    # Stage 1: ARIMA — order comes from Soham's arima_model.py BIC grid search
    order = load_arima_order(ticker)
    print(f"  Using ARIMA{order}")

    # Stage 1a: ARIMA multi-step baseline — forecasts all test days at once
    # this is the standard ARIMA benchmark (flat line on volatile stocks)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        arima_model = fit_arima(train_df["Close"].reset_index(drop=True), order)

    arima_test_pred = pd.Series(
        arima_model.forecast(steps=len(test_df))
    ).reset_index(drop=True)

    # in-sample residuals on training period = what XGBoost learns to correct
    arima_train_pred = arima_model.fittedvalues.reset_index(drop=True)
    train_residuals = train_df["Close"].reset_index(drop=True) - arima_train_pred

    # Stage 1b: rolling one-step-ahead ARIMA for the hybrid base
    # must run before feature building so rolling errors can be used as features
    rolling_pred, _ = rolling_arima_forecast(
        train_df["Close"].reset_index(drop=True),
        test_df["Close"].values,
        order,
    )

    # Stage 2: XGBoost trained on ARIMA's training residuals
    full_features = build_features(df)
    full_features = full_features.replace([np.inf, -np.inf], np.nan)

    # ARIMA residual lag features — directly targets volatility clustering
    test_residuals = pd.Series(
        test_df["Close"].values - rolling_pred.values
    )
    full_residuals = pd.concat(
        [train_residuals.reset_index(drop=True), test_residuals],
        ignore_index=True
    )
    full_features["resid_lag_1"] = full_residuals.shift(1).values
    full_features["resid_lag_2"] = full_residuals.shift(2).values
    full_features["resid_rolling_std_5"] = full_residuals.rolling(5).std().shift(1).values

    train_features = full_features.iloc[:len(train_df)].reset_index(drop=True)
    test_features = full_features.iloc[len(train_df):].reset_index(drop=True)

    valid_rows = train_features.dropna().index

    xgb_residual_model = XGBRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        random_state=42,
    )
    xgb_residual_model.fit(
        train_features.loc[valid_rows],
        train_residuals.loc[valid_rows],
    )
    xgb_residual_pred = xgb_residual_model.predict(test_features)

    # standalone XGBoost on raw prices — kept as a comparison baseline
    xgb_price_model = XGBRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        random_state=42,
    )
    train_price = train_df["Close"].reset_index(drop=True)
    xgb_price_model.fit(
        train_features.loc[valid_rows],
        train_price.loc[valid_rows],
    )
    xgb_only_pred = xgb_price_model.predict(test_features)

    # hybrid = rolling ARIMA (strong trend base) + XGBoost residual correction
    hybrid_pred = rolling_pred.values + xgb_residual_pred

    actual = test_df["Close"].values

    # plot
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.figure(figsize=(12, 5))
    plt.plot(test_df["Date"], actual, label="Actual", color="black", linewidth=1.2)
    plt.plot(test_df["Date"], arima_test_pred.values,
             label=f"ARIMA{order} multi-step (baseline)", linestyle="--", color="steelblue")
    plt.plot(test_df["Date"], hybrid_pred,
             label="Hybrid (Rolling ARIMA + XGBoost)", linestyle="--", color="darkorange")
    plt.title(f"{ticker} — ARIMA multi-step vs Hybrid (Rolling ARIMA + XGBoost)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{ticker}_forecast_comparison.png"), dpi=150)
    plt.close()

    return {
        "ticker": ticker,
        "arima_order": order,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "arima_rmse": rmse(actual, arima_test_pred.values),
        "arima_mae": mae(actual, arima_test_pred.values),
        "arima_mape": mape(actual, arima_test_pred.values),
        "xgb_rmse": rmse(actual, xgb_only_pred),
        "xgb_mae": mae(actual, xgb_only_pred),
        "xgb_mape": mape(actual, xgb_only_pred),
        "hybrid_rmse": rmse(actual, hybrid_pred),
        "hybrid_mae": mae(actual, hybrid_pred),
        "hybrid_mape": mape(actual, hybrid_pred),
        "improvement_pct": mape(actual, arima_test_pred.values) - mape(actual, hybrid_pred),
    }


def main():
    results = []

    for ticker in TICKERS:
        print(f"\nRunning {ticker}...")
        result = run_ticker(ticker)
        results.append(result)
        print(f"  ARIMA{result['arima_order']} MAPE: {result['arima_mape']:.2f}%   "
              f"Hybrid MAPE: {result['hybrid_mape']:.2f}%")

    results_df = pd.DataFrame(results).sort_values("improvement_pct", ascending=False)

    os.makedirs(TABLES_DIR, exist_ok=True)
    out_path = os.path.join(TABLES_DIR, "model_comparison.csv")
    results_df.to_csv(out_path, index=False)

    print(f"\nSaved -> {out_path}")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
