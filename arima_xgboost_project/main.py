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

warnings.filterwarnings("ignore")  # ARIMA throws convergence warnings on some orders — expected, not a bug

TICKERS = [
    "NVDA", "AMD", "SMCI", "ARM",
    "MU", "AVGO", "TSM", "000660.KS"
]

DATA_DIR = "data"
PLOTS_DIR = os.path.join("results", "plots", "task6_hybrid_forecast")
TABLES_DIR = os.path.join("results", "tables")

# We spit the data into train and test sets, 
# using the last 20% of each ticker's history for testing. 
# The rolling window for feature engineering is set to 21 days, 
# and we consider lag features for 1, 2, 3, 5, and 10 days. 
# A small set of ARIMA orders is defined for grid search per ticker to find the best model.
TEST_SIZE = 0.2          # last 20% of each ticker's history is used for testing, 
ROLLING_WINDOW = 21      # defined rolling window
LAG_DAYS = [1, 2, 3, 5, 10]  # defined lag days
MA_WINDOWS = [5, 10, 20]    # defined moving average windows


# instead of guessing one order for all 8 stocks, we try a few and pick the best per ticker based on AIC 
# so that we don't overfit to one stock's history and get poor results on others
ARIMA_ORDERS = [(1, 1, 0), (2, 1, 0), (0, 1, 1), (1, 1, 1), (2, 1, 1), (2, 1, 2)]


# Load the time series data for a given ticker
def load_series(ticker):
    path = os.path.join(DATA_DIR, f"{ticker}.csv")
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df

# Split the data into training and testing sets based on the specified test size
def split_train_test(df, test_size=TEST_SIZE):
    # Calculate the number of test samples
    n_test = int(len(df) * test_size)
    # Split the DataFrame into training and testing sets
    train = df.iloc[:-n_test].reset_index(drop=True)
    test = df.iloc[-n_test:].reset_index(drop=True)
    return train, test

# Fit the best ARIMA model for the training data based on AIC
def fit_best_arima(train_close):
    best_aic = np.inf
    best_model = None
    best_order = None

    # Try different ARIMA orders and select the one with the lowest AIC
    for order in ARIMA_ORDERS:
        try:
            # fitting the arima model
            model = ARIMA(train_close, order=order, trend='t').fit()
        except Exception:
            continue  # some orders fail to converge for some tickers — skip and try the next

        # Compare the AIC of the current model with the best AIC found
        if model.aic < best_aic:
            # Update the best AIC, model, and order
            best_aic = model.aic
            best_model = model
            best_order = order

    return best_model, best_order

# Build lag and rolling features for the XGBoost model based on the original time series data
def build_features(df):
    feat = pd.DataFrame(index=df.index)

    #Creating Lag Features
    for lag in LAG_DAYS:
        # Calculate the percentage change in the "Close" price for defined lag days 
        feat[f"return_lag_{lag}"] = df["Close"].pct_change(lag) * 100

    # Creating Rolling Volatility Feature
    feat["rolling_vol"] = df["Close"].pct_change().rolling(ROLLING_WINDOW).std() * 100

    # Creating Moving Average Features
    for window in MA_WINDOWS:
        # Calculate the moving average of the "Close" price for defined windows
        ma = df["Close"].rolling(window).mean()
        feat[f"ma_{window}"] = ma
        feat[f"price_to_ma_{window}"] = df["Close"] / ma

    # everything above is derived from Close[t], so shift the whole block forward
    # by one day - the feature row lined up with day t must only reflect what was
    # knowable through day t-1, otherwise it's peeking at the price it's predicting
    feat = feat.shift(1)
    feat["day_of_week"] = df["Date"].dt.dayofweek

    return feat

# Train an XGBoost model on the residuals from the ARIMA model
def train_xgb_on_residuals(train_features, train_residuals):
    model = XGBRegressor(
        # Use a small number of estimators 
        n_estimators=200,
        # Use a small max depth to prevent overfitting
        max_depth=3,
        # Use a small learning rate for better convergence
        learning_rate=0.05,
        # Set a random state for reproducibility
        random_state=42,
    )
    # Fit the XGBoost model on the training features and residuals
    model.fit(train_features, train_residuals)
    return model

# defining function to train an XGBoost model 
def train_xgb_on_price(train_features, train_price):
    model = XGBRegressor(
        # Use a small number of estimators to prevent overfitting
        n_estimators=200,
        # Use a small max depth to prevent overfitting
        max_depth=3,
        # Use a small learning rate for better convergence
        learning_rate=0.05,
        # Set a random state for reproducibility
        random_state=42,
    )
    model.fit(train_features, train_price)
    return model

# Calculating the Mean Absolute Percentage Error between actual and predicted values
def mape(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)
    return np.mean(np.abs((actual - predicted) / actual)) * 100

# Calculating the Root Mean Squared Error between actual and predicted values
def rmse(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)
    return np.sqrt(np.mean((actual - predicted) ** 2))

# Calculating the Mean Absolute Error between actual and predicted values
def mae(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)
    return np.mean(np.abs(actual - predicted))

def run_ticker(ticker):
    df = load_series(ticker)
    # Split the data into training and testing sets
    train_df, test_df = split_train_test(df)

    # Stage 1: Fit ARIMA model on the training data
    arima_model, order = fit_best_arima(train_df["Close"])
    # Generating ARIMA forecasting
    arima_test_pred = pd.Series(arima_model.forecast(steps=len(test_df))).reset_index(drop=True)

    # in-sample fitted values on the training period = ARIMA's own predictions of the past
    arima_train_pred = arima_model.fittedvalues.reset_index(drop=True)
    train_residuals = train_df["Close"].reset_index(drop=True) - arima_train_pred

    # ---- stage 2: XGBoost on ARIMA's leftover error ----
    # built on the FULL series before splitting, so rolling/lag features in the
    # test period correctly use history from the train period — no leakage, no gaps
    full_features = build_features(df)
    train_features = full_features.iloc[:len(train_df)].reset_index(drop=True)
    test_features = full_features.iloc[len(train_df):].reset_index(drop=True)

   # Remove rows with NaN values
    valid_rows = train_features.dropna().index
    # Train the XGBoost model on the residuals of the ARIMA model
    xgb_model = train_xgb_on_residuals(
        train_features.loc[valid_rows],
        train_residuals.loc[valid_rows],
    )

    # Predict the residuals using the trained XGBoost model
    xgb_residual_pred = xgb_model.predict(test_features)

    # XGBoost on the original price series, for comparison
    train_price = train_df["Close"].reset_index(drop=True)
    xgb_price_model = train_xgb_on_price(
        train_features.loc[valid_rows],
        train_price.loc[valid_rows],
    )
    xgb_only_pred = xgb_price_model.predict(test_features)
    
    # combine ARIMA and XGBoost prediction 
    hybrid_pred = arima_test_pred.values + xgb_residual_pred
    # evaluate the models using RMSE, MAE, and MAPE metrics
    actual = test_df["Close"].values
    
    # evaluate the ARIMAmodels using RMSE, MAE, and MAPE metrics
    arima_rmse_val  = rmse(actual, arima_test_pred.values)
    arima_mae_val   = mae(actual, arima_test_pred.values)
    arima_mape_val  = mape(actual, arima_test_pred.values)

    # evaluate the XGBoost models using RMSE, MAE, and MAPE metrics
    xgb_rmse_val    = rmse(actual, xgb_only_pred)
    xgb_mae_val     = mae(actual, xgb_only_pred)
    xgb_mape_val    = mape(actual, xgb_only_pred)

    # evaluate the hybrid models using RMSE, MAE, and MAPE metrics
    hybrid_rmse_val = rmse(actual, hybrid_pred)
    hybrid_mae_val  = mae(actual, hybrid_pred)
    hybrid_mape_val = mape(actual, hybrid_pred)

    # plot: actual vs ARIMA vs hybrid, test period only ----
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.figure(figsize=(12, 5))
    plt.plot(test_df["Date"], actual, label="Actual", color="black", linewidth=1.2)
    plt.plot(test_df["Date"], arima_test_pred.values, label=f"ARIMA{order}", linestyle="--")
    plt.plot(test_df["Date"], hybrid_pred, label="ARIMA + XGBoost", linestyle="--")
    plt.title(f"{ticker} — Actual vs ARIMA vs Hybrid (test period)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{ticker}_forecast_comparison.png"), dpi=150)
    plt.close()    

    #---- return results ----
    return {
        "ticker": ticker,
        "arima_order": order,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "arima_rmse": arima_rmse_val,
        "arima_mae": arima_mae_val,
        "arima_mape": arima_mape_val,
        "xgb_rmse": xgb_rmse_val,
        "xgb_mae": xgb_mae_val,
        "xgb_mape": xgb_mape_val,
        "hybrid_rmse": hybrid_rmse_val,
        "hybrid_mae": hybrid_mae_val,
        "hybrid_mape": hybrid_mape_val,
        "improvement_pct": arima_mape_val - hybrid_mape_val,
    }

def main():
    results = []

    for ticker in TICKERS:
        print(f"Running {ticker}...")
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