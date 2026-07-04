# arima_model.py
# this is where all the actual ARIMA stuff is (task 6 onward),
# this file is just for plotting/looking at the data, not testing/modeling it.



import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

plt.style.use("seaborn-v0_8-whitegrid")

TICKERS = [
    "NVDA", "AMD", "SMCI", "ARM",
    "MU", "AVGO", "TSM", "000660.KS"
]

DATA_DIR = "data"
TABLES_DIR = os.path.join("results", "tables")
PLOTS_DIR = os.path.join("results", "plots")
ARIMA_FITS_DIR = os.path.join(PLOTS_DIR, "task7_14_arima_fits")
PREDICTIONS_DIR = "forecasts"  # kept for future use (e.g. real future-date forecasts)
RESIDUALS_DIR = "residuals"

# task 7 = NVDA, task 8 = AMD, ... task 14 = 000660.KS
TASK_TICKER_MAP = {
    7: "NVDA",
    8: "AMD",
    9: "SMCI",
    10: "ARM",
    11: "MU",
    12: "AVGO",
    13: "TSM",
    14: "000660.KS",
}

# TASK 6 - ADF stationarity test


def run_adf(series, label):
    # null hyp = non-stationary, p < 0.05 means reject that
    result = adfuller(series.dropna(), autolag="AIC")

    adf_stat = result[0]
    p_value = result[1]
    n_lags = result[2]
    n_obs = result[3]
    crit_1pct = result[4]["1%"]
    crit_5pct = result[4]["5%"]
    crit_10pct = result[4]["10%"]

    is_stationary = p_value < 0.05

    return {
        "series": label,
        "adf_statistic": adf_stat,
        "p_value": p_value,
        "n_lags_used": n_lags,
        "n_obs": n_obs,
        "critical_1pct": crit_1pct,
        "critical_5pct": crit_5pct,
        "critical_10pct": crit_10pct,
        "is_stationary": is_stationary,
    }


def adf_test():

    os.makedirs(TABLES_DIR, exist_ok=True)

    all_results = []

    for ticker in TICKERS:

        print(f"\nRunning ADF test for {ticker}...")

        path = os.path.join(DATA_DIR, f"{ticker}.csv")
        df = pd.read_csv(path, parse_dates=["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

        # raw price - should fail basically every time
        raw_result = run_adf(df["Close"], "raw_close")
        raw_result["ticker"] = ticker
        all_results.append(raw_result)

        print(
            f"  Raw Close      -> ADF stat: {raw_result['adf_statistic']:.4f}, "
            f"p-value: {raw_result['p_value']:.4f}, "
            f"stationary: {raw_result['is_stationary']}"
        )

        # first difference - tells us if d=1 works
        diff_series = df["Close"].diff()
        diff_result = run_adf(diff_series, "first_difference")
        diff_result["ticker"] = ticker
        all_results.append(diff_result)

        print(
            f"  1st Difference -> ADF stat: {diff_result['adf_statistic']:.4f}, "
            f"p-value: {diff_result['p_value']:.4f}, "
            f"stationary: {diff_result['is_stationary']}"
        )

    results_df = pd.DataFrame(all_results)

    cols = ["ticker", "series"] + [
        c for c in results_df.columns if c not in ("ticker", "series")
    ]
    results_df = results_df[cols]

    save_path = os.path.join(TABLES_DIR, "adf_test_results.csv")
    results_df.to_csv(save_path, index=False)
    print(f"\nSaved -> {save_path}")

    return results_df


# only runs second difference for tickers that failed at first
def adf_test_second_difference(results_df):

    os.makedirs(TABLES_DIR, exist_ok=True)

    failed_mask = (
        (results_df["series"] == "first_difference")
        & (results_df["is_stationary"] == False)
    )
    failed_tickers = results_df.loc[failed_mask, "ticker"].tolist()

    if not failed_tickers:
        print("\nEvery ticker passed at first difference, no second difference needed.")
        return pd.DataFrame()

    print(f"\nTickers that need a second difference check: {failed_tickers}")

    second_diff_results = []

    for ticker in failed_tickers:

        path = os.path.join(DATA_DIR, f"{ticker}.csv")
        df = pd.read_csv(path, parse_dates=["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

        diff2_series = df["Close"].diff().diff()
        result = run_adf(diff2_series, "second_difference")
        result["ticker"] = ticker
        second_diff_results.append(result)

        print(
            f"  {ticker} 2nd Difference -> ADF stat: {result['adf_statistic']:.4f}, "
            f"p-value: {result['p_value']:.4f}, "
            f"stationary: {result['is_stationary']}"
        )

    second_df = pd.DataFrame(second_diff_results)

    cols = ["ticker", "series"] + [
        c for c in second_df.columns if c not in ("ticker", "series")
    ]
    second_df = second_df[cols]

    save_path = os.path.join(TABLES_DIR, "adf_second_difference_results.csv")
    second_df.to_csv(save_path, index=False)
    print(f"\nSaved -> {save_path}")

    return second_df


# picks the final d for each ticker based on whatever passed
def determine_final_d(results_df, second_df):

    final_rows = []

    for ticker in TICKERS:

        raw_row = results_df[
            (results_df["ticker"] == ticker) & (results_df["series"] == "raw_close")
        ].iloc[0]

        diff1_row = results_df[
            (results_df["ticker"] == ticker) & (results_df["series"] == "first_difference")
        ].iloc[0]

        if raw_row["is_stationary"]:
            chosen_d = 0
            basis_p = raw_row["p_value"]
            note = "raw price was already stationary"

        elif diff1_row["is_stationary"]:
            chosen_d = 1
            basis_p = diff1_row["p_value"]
            note = "stationary after first differencing"

        else:
            diff2_rows = (
                second_df[second_df["ticker"] == ticker]
                if not second_df.empty else pd.DataFrame()
            )

            if not diff2_rows.empty and diff2_rows.iloc[0]["is_stationary"]:
                chosen_d = 2
                basis_p = diff2_rows.iloc[0]["p_value"]
                note = "needed a second difference to become stationary"
            else:
                chosen_d = None
                basis_p = (
                    diff2_rows.iloc[0]["p_value"] if not diff2_rows.empty
                    else diff1_row["p_value"]
                )
                note = "still not stationary, needs manual review"

        final_rows.append({
            "ticker": ticker,
            "recommended_d": chosen_d,
            "p_value_at_chosen_d": basis_p,
            "note": note,
        })

    final_df = pd.DataFrame(final_rows)

    save_path = os.path.join(TABLES_DIR, "adf_final_d_selection.csv")
    final_df.to_csv(save_path, index=False)
    print(f"\nSaved -> {save_path}")

    return final_df


# TASK 7-14 - fit ARIMA for each ticker, same function reused


def get_d_for_ticker(ticker):
    # grabs the d we already found in task 6
    path = os.path.join(TABLES_DIR, "adf_final_d_selection.csv")

    if not os.path.exists(path):
        print(f"  (no d table found, defaulting to d=1 for {ticker})")
        return 1

    d_table = pd.read_csv(path)
    row = d_table[d_table["ticker"] == ticker]

    if row.empty:
        print(f"  ({ticker} not in d table, defaulting to d=1)")
        return 1

    raw_d = row.iloc[0]["recommended_d"]

    if pd.isna(raw_d):
        print(f"  ({ticker} had no valid d, defaulting to d=1 - worth a manual look)")
        return 1

    return int(raw_d)


def plot_acf_pacf(series, ticker, d, chosen_p=None, chosen_q=None):
    # the plot we actually look at to guess p and q
    os.makedirs(ARIMA_FITS_DIR, exist_ok=True)

    clean_series = series.dropna()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))

    plot_acf(clean_series, lags=30, ax=ax1, color="black")
    ax1.set_title("ACF")
    ax1.set_xlabel("Lag (days)")

    plot_pacf(clean_series, lags=30, ax=ax2, method="ywm", color="black")
    ax2.set_title("PACF")
    ax2.set_xlabel("Lag (days)")

    # mark the lag we actually picked, so the plot and the model
    # choice are visually tied together instead of living in
    # separate places
    if chosen_q is not None and chosen_q > 0:
        ax1.axvline(chosen_q, color="darkorange", linestyle="--", linewidth=1.5, label=f"chosen q={chosen_q}")
        ax1.legend(fontsize=8)

    if chosen_p is not None and chosen_p > 0:
        ax2.axvline(chosen_p, color="darkorange", linestyle="--", linewidth=1.5, label=f"chosen p={chosen_p}")
        ax2.legend(fontsize=8)

    order_str = f"ARIMA({chosen_p},{d},{chosen_q})" if chosen_p is not None else f"d={d}"
    fig.suptitle(f"{ticker} - ACF & PACF  ({order_str})", fontsize=12, fontweight="bold")

    # quick plain-english note if basically nothing crosses the
    # confidence band - saves the reader wondering if the plot
    # is broken when it's actually just a flat, low-order case
    from statsmodels.tsa.stattools import acf as _acf, pacf as _pacf
    n = len(clean_series)
    conf_band = 1.96 / np.sqrt(n)
    acf_vals = _acf(clean_series, nlags=30)[1:]
    pacf_vals = _pacf(clean_series, nlags=30, method="ywm")[1:]
    n_significant = np.sum(np.abs(acf_vals) > conf_band) + np.sum(np.abs(pacf_vals) > conf_band)

    if n_significant == 0:
        fig.text(0.5, -0.02, "No lags outside the confidence band - little autocorrelation detected",
                  ha="center", fontsize=9, style="italic", color="grey")

    plt.tight_layout()

    save_path = os.path.join(ARIMA_FITS_DIR, f"{ticker}_acf_pacf.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved -> {save_path}")


def find_best_order(close_series, d, max_p=5, max_q=5):
    # brute force grid search, 36 combos, keep lowest BIC
    #
    # using BIC instead of AIC here on purpose - AIC's penalty for
    # extra parameters is too weak when the series is close to
    # white noise (which several of these tickers are, per the
    # flat ACF/PACF plots). AIC ends up picking high-order models
    # that fit the historical data slightly better but forecast
    # worse on new data. BIC's penalty scales with sample size and
    # pushes back much harder against unnecessary p/q, which
    # matches what the ACF/PACF plots are actually showing
    #
    # also skip anything that didn't actually converge
    best_bic = float("inf")
    best_order = None
    best_model = None
    n_skipped = 0

    for p in range(max_p + 1):
        for q in range(max_q + 1):

            try:
                model = ARIMA(close_series, order=(p, d, q))
                fitted = model.fit()

                converged = fitted.mle_retvals.get("converged", True)
                bic_is_valid = not np.isnan(fitted.bic)

                if not converged or not bic_is_valid:
                    n_skipped += 1
                    continue

                if fitted.bic < best_bic:
                    best_bic = fitted.bic
                    best_order = (p, d, q)
                    best_model = fitted

            except Exception:
                n_skipped += 1
                continue

    if n_skipped > 0:
        print(f"  ({n_skipped} of {(max_p+1)*(max_q+1)} order combos failed or didn't converge, skipped)")

    return best_order, best_model


def evaluate_forecast(ticker, close_series, order, test_size=30):
    # hold out last 30 days, fit on the rest, see how off the forecast was
    train = close_series[:-test_size]
    test = close_series[-test_size:]

    try:
        model = ARIMA(train, order=order)
        fitted = model.fit()
        forecast = fitted.forecast(steps=test_size)
    except Exception as e:
        print(f"  Could not evaluate {ticker} forecast: {e}")
        return None

    errors = test.values - forecast.values

    rmse = np.sqrt(np.mean(errors ** 2))
    mae = np.mean(np.abs(errors))
    # mape is the one we compare across tickers with (works fine
    # even though 000660.KS is priced in KRW, not USD)
    mape = np.mean(np.abs(errors / test.values)) * 100

    print(f"  Forecast check (last {test_size} days) -> RMSE: {rmse:.2f}, MAE: {mae:.2f}, MAPE: {mape:.2f}%")

    return {
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
    }


def check_residuals_look_random(fitted_model, ticker):
    # if residuals still have a pattern, the model missed something
    residuals = fitted_model.resid.dropna()

    lb_result = acorr_ljungbox(residuals, lags=[10], return_df=True)
    lb_pvalue = lb_result["lb_pvalue"].iloc[0]
    looks_random = lb_pvalue > 0.05

    if not looks_random:
        print(f"  (heads up: {ticker}'s residuals still show some pattern, ljung-box p={lb_pvalue:.4f})")

    # same test but on squared residuals - catches volatility
    # clustering instead of a missed mean pattern. ARIMA can't
    # fix this by changing p/q, that's a job for xgboost
    squared_residuals = residuals ** 2
    arch_result = acorr_ljungbox(squared_residuals, lags=[10], return_df=True)
    arch_pvalue = arch_result["lb_pvalue"].iloc[0]
    has_arch_effect = arch_pvalue < 0.05

    if has_arch_effect:
        print(f"  (also: {ticker} shows volatility clustering, p={arch_pvalue:.4f} - not fixable via p/q)")

    return looks_random, lb_pvalue, has_arch_effect, arch_pvalue


def save_arima_handoff(ticker, df, fitted_model, d):
    # one combined file per ticker instead of two - residual is
    # just actual minus predicted, so keeping them in separate
    # files was pure duplication. this is the file asawari's
    # xgboost stage actually needs: Date, Actual, Predicted (the
    # "L" / linear component), Residual (the "N" / nonlinear
    # leftover she's trying to predict)
    os.makedirs(RESIDUALS_DIR, exist_ok=True)

    fitted_values = fitted_model.fittedvalues.iloc[d:]
    residuals = fitted_model.resid.iloc[d:]

    handoff_df = pd.DataFrame({
        "Date": df.loc[fitted_values.index, "Date"].values,
        "Actual": df.loc[fitted_values.index, "Close"].values,
        "Predicted": fitted_values.values,
        "Residual": residuals.values,
    })

    save_path = os.path.join(RESIDUALS_DIR, f"{ticker}_arima_output.csv")
    handoff_df.to_csv(save_path, index=False)
    print(f"  Saved -> {save_path}")

    # in-sample metrics - how well ARIMA fit everything it already
    # saw (different from the 30-day holdout test further down,
    # which checks how well it forecasts data it DIDN'T see)
    errors = handoff_df["Actual"] - handoff_df["Predicted"]
    rmse_in = np.sqrt(np.mean(errors ** 2))
    mae_in = np.mean(np.abs(errors))
    mape_in = np.mean(np.abs(errors / handoff_df["Actual"])) * 100

    print(f"  In-sample fit -> RMSE: {rmse_in:.2f}, MAE: {mae_in:.2f}, MAPE: {mape_in:.2f}%")

    return handoff_df, {"rmse_insample": rmse_in, "mae_insample": mae_in, "mape_insample": mape_in}


def fit_arima_for_ticker(ticker, task_number):

    print(f"\nTask {task_number}: Fitting ARIMA model for {ticker}...")

    path = os.path.join(DATA_DIR, f"{ticker}.csv")

    if not os.path.exists(path):
        print(f"  Could not find {path}, skipping {ticker}")
        return None

    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    if "Close" not in df.columns or df["Close"].isna().all():
        print(f"  {ticker}.csv has no usable Close column, skipping")
        return None

    close_series = df["Close"]
    d = get_d_for_ticker(ticker)
    print(f"  Using d={d} (from task 6 results)")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        best_order, best_model = find_best_order(close_series, d)

    if best_order is None or best_model is None:
        print(f"  Could not fit any ARIMA model for {ticker} - every order combo failed")
        return None

    print(f"  Best order: ARIMA{best_order}, AIC: {best_model.aic:.2f}")

    # difference d times just for the ACF/PACF plot - the actual
    # fit above handles differencing internally via order=(p,d,q)
    diffed = close_series.copy()
    for _ in range(d):
        diffed = diffed.diff()

    plot_acf_pacf(diffed, ticker, d, chosen_p=best_order[0], chosen_q=best_order[2])

    # save the combined predictions+residuals file, get in-sample metrics back
    handoff_df, insample_metrics = save_arima_handoff(ticker, df, best_model, d)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        looks_random, lb_pvalue, has_arch_effect, arch_pvalue = check_residuals_look_random(best_model, ticker)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eval_metrics = evaluate_forecast(ticker, close_series, best_order)

    result = {
        "ticker": ticker,
        "order_p": best_order[0],
        "order_d": best_order[1],
        "order_q": best_order[2],
        "aic": best_model.aic,
        "bic": best_model.bic,
        "log_likelihood": best_model.llf,
        "n_params": len(best_model.params),
        "residuals_look_random": looks_random,
        "ljung_box_pvalue": lb_pvalue,
        "has_arch_effect": has_arch_effect,
        "arch_pvalue": arch_pvalue,
    }

    if eval_metrics is not None:
        result.update(eval_metrics)
    else:
        result.update({"rmse": None, "mae": None, "mape": None})

    result.update(insample_metrics)

    return result


def fit_all_arima_models():

    os.makedirs(TABLES_DIR, exist_ok=True)

    all_fits = []
    failed_tickers = []

    # one line per ticker on purpose, matches the task sheet
    # (task 7 = NVDA, task 8 = AMD, etc)

    # Task 7: NVDA
    result = fit_arima_for_ticker("NVDA", 7)
    all_fits.append(result) if result else failed_tickers.append("NVDA")

    # Task 8: AMD
    result = fit_arima_for_ticker("AMD", 8)
    all_fits.append(result) if result else failed_tickers.append("AMD")

    # Task 9: SMCI
    result = fit_arima_for_ticker("SMCI", 9)
    all_fits.append(result) if result else failed_tickers.append("SMCI")

    # Task 10: ARM
    result = fit_arima_for_ticker("ARM", 10)
    all_fits.append(result) if result else failed_tickers.append("ARM")

    # Task 11: MU
    result = fit_arima_for_ticker("MU", 11)
    all_fits.append(result) if result else failed_tickers.append("MU")

    # Task 12: AVGO
    result = fit_arima_for_ticker("AVGO", 12)
    all_fits.append(result) if result else failed_tickers.append("AVGO")

    # Task 13: TSM
    result = fit_arima_for_ticker("TSM", 13)
    all_fits.append(result) if result else failed_tickers.append("TSM")

    # Task 14: 000660.KS (SK Hynix)
    result = fit_arima_for_ticker("000660.KS", 14)
    all_fits.append(result) if result else failed_tickers.append("000660.KS")

    fits_df = pd.DataFrame(all_fits)

    save_path = os.path.join(TABLES_DIR, "arima_fit_summary.csv")
    fits_df.to_csv(save_path, index=False)
    print(f"\nSaved -> {save_path}")

    if failed_tickers:
        print(f"\nWARNING - could not fit these tickers at all: {failed_tickers}")

    if not fits_df.empty and "residuals_look_random" in fits_df.columns:
        flagged = fits_df[fits_df["residuals_look_random"] == False]["ticker"].tolist()
        if flagged:
            print(f"WARNING - these tickers fit fine but residuals still show a pattern: {flagged}")

    if not fits_df.empty and "has_arch_effect" in fits_df.columns:
        arch_flagged = fits_df[fits_df["has_arch_effect"] == True]["ticker"].tolist()
        if arch_flagged:
            print(f"\nNOTE - volatility clustering detected in: {arch_flagged}")
            print("(exactly the kind of pattern xgboost is meant to pick up)")

    return fits_df


if __name__ == "__main__":
    print("Task 6: Running ADF stationarity tests...")
    results_df = adf_test()

    print("\nChecking second difference for any tickers that failed...")
    second_df = adf_test_second_difference(results_df)

    print("\nPicking the final d for every ticker...")
    final_df = determine_final_d(results_df, second_df)
    print(final_df)

    print("\nDone. Check results/tables/ for all 3 csvs.")

    print("\n" + "=" * 60)
    print("Tasks 7-14: Fitting ARIMA models for all 8 tickers...")
    print("=" * 60)
    fits_df = fit_all_arima_models()
    print(fits_df)

    print("\nDone. Check results/tables/arima_fit_summary.csv and")
    print("results/plots/task7_14_arima_fits/ for the ACF/PACF plots.")