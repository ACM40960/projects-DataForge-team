# arima_model.py

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
PREDICTIONS_DIR = "forecasts"

#  for task 7 = NVDA to task 14 = 000660.KS
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

# Task 6   -> run_adf, adf_test, adf_test_second_difference, determine_final_d
# work out how much differencing each ticker needs (d) [ d from the arima(p,d,q) model ]


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

# task 6 proper. two tests per ticker: raw price, then first difference.
# raw should fail every time (prices trend), first difference should pass for almost everything - that's what tells us d=1 is enough

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


# only runs second difference for tickers that failed at first time.
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


# picks the final "d" for each ticker based on whatever has passed
def determine_final_d(results_df, second_df):
    # the smallest d that works because every extra difference throws away a row and adds noise
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


# Task 7-14 -> get_d_for_ticker, plot_acf_pacf, find_best_order,
# check_residuals_look_random, insample_fit_metrics,
# fit_arima_for_ticker, fit_all_arima_models


def get_d_for_ticker(ticker):
    # grabs the d we already found in task 6
    # reads back what task 6 decided instead of recalculating.
    # keeps the two stages from ever disagreeing, and means the d choice can be audited in csv rather

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

    # ACF  = how related today's change is to the change N days ago, including everything in between

    # PACF = same thing with the indirect effects stripped out -> hints at p
    # the shaded band is the "could just be randomness". bars inside it mean nothing, bars poking out are real signal.
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
    # choice are visually tied together instead of living in a
    # separate places
    if chosen_q is not None and chosen_q > 0:
        ax1.axvline(chosen_q, color="darkorange", linestyle="--", linewidth=1.5, label=f"chosen q={chosen_q}")
        ax1.legend(fontsize=8)

    if chosen_p is not None and chosen_p > 0:
        ax2.axvline(chosen_p, color="darkorange", linestyle="--", linewidth=1.5, label=f"chosen p={chosen_p}")
        ax2.legend(fontsize=8)

    order_str = f"ARIMA({chosen_p},{d},{chosen_q})" if chosen_p is not None else f"d={d}"
    fig.suptitle(f"{ticker} - ACF & PACF  ({order_str})", fontsize=12, fontweight="bold")
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
    # using brute force grid search and having 36 combos keeping lowest BIC

    # Also using BIC instead of AIC here on purpose - AIC's penalty for extra parameters is too weak when the series is close to
    # white noise (which several of these tickers are, per the  flat ACF/PACF plots). AIC ends up picking high-order models
    # that fit the historical data slightly better but forecast
    # worse on new data.

    # BIC's penalty scales with sample size and
    # pushes back much harder against unnecessary p/q, which
    # matches what the ACF/PACF plots are actually showing

    best_bic = float("inf")
    best_order = None
    best_model = None
    n_skipped = 0 # skip what is not converged or has NaN BIC as a result

    for p in range(max_p + 1):
        for q in range(max_q + 1):

            try:
                model = ARIMA(close_series, order=(p, d, q))
                fitted = model.fit()

                converged = fitted.mle_retvals.get("converged", True)
                bic_is_valid = not np.isnan(fitted.bic)

                # the converged flag on its own isn't enough - we hit a case
                # where a model came back converged=True with a nonsense BIC
                # and residuals bigger than the whole price range. if the
                # errors are wider than the spread of the data the fit is
                # broken no matter what the flag says
                resid_std = fitted.resid.std()
                fit_is_sane = np.isfinite(resid_std) and resid_std < close_series.std()

                if not converged or not bic_is_valid or not fit_is_sane:
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

    # same test but on squared residuals - catches volatility clustering instead of a missed mean pattern.
    # ARIMA can't fix this by changing p/q, that's a job for xgboost

    squared_residuals = residuals ** 2
    arch_result = acorr_ljungbox(squared_residuals, lags=[10], return_df=True)
    arch_pvalue = arch_result["lb_pvalue"].iloc[0]
    has_arch_effect = arch_pvalue < 0.05

    if has_arch_effect:
        print(f"  (also: {ticker} shows volatility clustering, p={arch_pvalue:.4f} - not fixable via p/q)")

    return looks_random, lb_pvalue, has_arch_effect, arch_pvalue


def insample_fit_metrics(ticker, df, fitted_model, d):
    # how well ARIMA fit everything it already saw. this is NOT a forecast -
    # the model had these days during fitting, so the numbers are flattering
    # by construction. the honest ones are in main.py, which forecasts a
    # held-out 20% it never touched.
    #
    # drop the first d rows: differencing eats them, so there is no real
    # prediction there and the "residual" would just be the raw price
    fitted_values = fitted_model.fittedvalues.iloc[d:]
    residuals = fitted_model.resid.iloc[d:]

    fit_df = pd.DataFrame({
        "Date": df.loc[fitted_values.index, "Date"].values,
        "Actual": df.loc[fitted_values.index, "Close"].values,
        "Predicted": fitted_values.values,
        "Residual": residuals.values,
    })

    errors = fit_df["Actual"] - fit_df["Predicted"]
    rmse_in = np.sqrt(np.mean(errors ** 2))
    mae_in = np.mean(np.abs(errors))
    mape_in = np.mean(np.abs(errors / fit_df["Actual"])) * 100

    print(f"  In-sample fit -> RMSE: {rmse_in:.2f}, MAE: {mae_in:.2f}, MAPE: {mape_in:.2f}%")

    return fit_df, {"rmse_insample": rmse_in, "mae_insample": mae_in, "mape_insample": mape_in}


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

    # difference d times just for the ACF/PACF plot - the actual fit above handles differencing internally via order=(p,d,q)

    diffed = close_series.copy()
    for _ in range(d):
        diffed = diffed.diff()

    plot_acf_pacf(diffed, ticker, d, chosen_p=best_order[0], chosen_q=best_order[2])

    # in-sample fit numbers for the summary table
    fit_df, insample_metrics = insample_fit_metrics(ticker, df, best_model, d)

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


# the project flow diagram. lives here rather than in its own file so the
# repo stays at five scripts.
#
# needs graphviz, which is two separate installs on windows: "pip install
# graphviz" for the python wrapper, then the actual graphviz program from
# graphviz.org added to PATH. if either is missing we just skip it - a
# missing picture shouldn't take the whole ARIMA run down with it. the png
# is committed anyway, so this only matters if you want to change it.
def make_flow_diagram():

    try:
        from graphviz import Digraph
    except ImportError:
        print("  (graphviz not installed, skipping flow diagram - pip install graphviz)")
        return

    g = Digraph("pipeline", format="png")

    # splines="polyline" rather than "ortho": ortho refuses to place edge
    # labels and leaves them floating in the middle of nowhere
    g.attr(rankdir="TB", splines="polyline", nodesep="0.4", ranksep="0.55",
           bgcolor="white", fontname="Helvetica", compound="true")
    g.attr("node", shape="box", style="rounded,filled", fontname="Helvetica",
           fontsize="11", margin="0.18,0.10")
    g.attr("edge", fontname="Helvetica", fontsize="9", color="#444444")

    BLUE, PINK, PURPLE, GREEN = "#dbeafe", "#fce7f3", "#e0e7ff", "#d1fae5"

    with g.subgraph(name="cluster_in") as c:
        c.attr(label="  Input   data_pipeline.py", style="rounded,filled",
               fillcolor="#f0f7ff", color="#93c5fd", fontsize="12",
               fontname="Helvetica-Bold", labeljust="l")
        c.node("dl", "Yahoo Finance\n8 tickers, 2019 to 2026\nauto_adjust=True", fillcolor=BLUE)
        c.node("csv", "data/*.csv\n1872 rows each\n0 missing, 0 dropped", fillcolor=BLUE)
        c.node("split", "80 / 20 split\nchronological, no shuffle", fillcolor=BLUE)
        c.edge("dl", "csv")
        c.edge("csv", "split")

    with g.subgraph(name="cluster_eda") as c:
        c.attr(label="  Tasks 4 and 5   eda.py", style="rounded,filled",
               fillcolor="#fffbeb", color="#fcd34d", fontsize="12",
               fontname="Helvetica-Bold", labeljust="l")
        c.node("eda", "Closing prices + COVID marker\nDaily returns + 21-day volatility",
               fillcolor="#fef3c7")
        c.node("eda2", "Finding: volatility clusters\nTails are fat (kurtosis up to 17)",
               fillcolor="#fef3c7", shape="note")
        c.edge("eda", "eda2", style="dashed")

    with g.subgraph(name="cluster_s1") as c:
        c.attr(label="  Stage 1   arima_model.py   Tasks 6 to 14",
               style="rounded,filled", fillcolor="#fdf2f8", color="#f9a8d4",
               fontsize="12", fontname="Helvetica-Bold", labeljust="l")
        c.node("adf", "ADF tests\npick d\n7 tickers d=1, MU d=2", fillcolor=PINK)
        c.node("bic", "BIC grid search\n36 combinations of p and q", fillcolor=PINK)
        c.node("fit", "Fit ARIMA\non training 80% only", fillcolor=PINK)
        c.node("ms", "Multi-step forecast\nall 374 days at once\ngoes flat", fillcolor=PINK)
        c.node("roll", "Rolling one-step forecast\nforecast 1 day, observe actual,\nappend, repeat",
               fillcolor=PINK)
        c.node("resid", "In-sample residuals\nActual minus ARIMA fitted", fillcolor=PINK)
        c.node("diag", "Ljung-Box on residuals: 5 of 8 reject\n"
                       "ARCH on squared residuals: 8 of 8 reject",
               fillcolor="#fbcfe8", shape="note")
        c.edge("adf", "bic")
        c.edge("bic", "fit")
        c.edge("fit", "ms")
        c.edge("fit", "roll")
        c.edge("fit", "resid")
        c.edge("resid", "diag", style="dashed")

    with g.subgraph(name="cluster_s2") as c:
        c.attr(label="  Stage 2   main.py", style="rounded,filled",
               fillcolor="#eef2ff", color="#a5b4fc", fontsize="12",
               fontname="Helvetica-Bold", labeljust="l")
        c.node("feat", "Feature engineering\nlag returns, rolling volatility, MA ratios\n"
                       "volume, high-low range, momentum\nday of week, ARIMA residual lags",
               fillcolor=PURPLE)
        c.node("xgb", "Train XGBoost\non training residuals", fillcolor=PURPLE)
        c.node("pred", "Predict test residuals", fillcolor=PURPLE)
        c.edge("feat", "xgb")
        c.edge("xgb", "pred")

    with g.subgraph(name="cluster_out") as c:
        c.attr(label="  Output and evaluation", style="rounded,filled",
               fillcolor="#ecfdf5", color="#6ee7b7", fontsize="12",
               fontname="Helvetica-Bold", labeljust="l")
        c.node("hyb", "Hybrid forecast\nrolling ARIMA + residual correction", fillcolor=GREEN)
        c.node("cmp", "Compare all five on the same 20% test set", fillcolor=GREEN)
        c.node("res", "ARIMA multi-step   31.15%\n"
                      "XGBoost alone      21.95%\n"
                      "Hybrid              3.18%\n"
                      "Rolling ARIMA       2.87%\n"
                      "Naive benchmark     2.86%   <-- wins",
               fillcolor="#a7f3d0", fontname="Courier-Bold", fontsize="11")
        c.edge("hyb", "cmp")
        c.edge("cmp", "res")

    # the wiring between lanes - this is the part that actually explains the
    # project, because it shows both ARIMA branches feeding the comparison
    g.edge("csv", "eda", style="dashed", constraint="false", label="  same data  ")
    g.edge("split", "adf", label=" train ")
    g.edge("resid", "feat", label=" train residuals ")
    g.edge("roll", "hyb", label=" base forecast ")
    g.edge("pred", "hyb", label=" correction ")
    g.edge("ms", "cmp", style="dashed", label=" baseline ")
    g.edge("roll", "cmp", style="dashed")

    try:
        g.render("flow_diagram", cleanup=True)
        print("  Saved -> flow_diagram.png")
    except Exception as e:
        # graphviz python package present but the "dot" binary isn't on PATH
        print(f"  (couldn't render flow diagram: {e})")
        print("  install graphviz from graphviz.org and add it to PATH")


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

    print("\nRebuilding the project flow diagram...")
    make_flow_diagram()

    print("\nDone. Check results/tables/arima_fit_summary.csv and")
    print("results/plots/task7_14_arima_fits/ for the ACF/PACF plots.")
