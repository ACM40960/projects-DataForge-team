# eda.py
# Task 4 - plotting closing prices for all 8 stocks + marking some
# real world events on the chart so we can see how price reacted

# Task 5 - daily returns + rolling volatility (basically how "jumpy"
# the stock is on a day to day basis)

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
plt.style.use("seaborn-v0_8-whitegrid")

TICKERS = [
    "NVDA", "AMD", "SMCI", "ARM",
    "MU", "AVGO", "TSM", "000660.KS"
]

DATA_DIR = "data"
PLOTS_DIR = os.path.join("results", "plots")
TABLES_DIR = os.path.join("results", "tables")

# giving each task its own folder inside plots so it doesn't turn into a
# mess of 40+ random pngs once we add more tasks later
CLOSING_PRICE_DIR = os.path.join(PLOTS_DIR, "task4_closing_prices")
DAILY_RETURNS_DIR = os.path.join(PLOTS_DIR, "task5_daily_returns")

# SK Hynix trades in korea so it's priced in won, not dollars. was
# labelling every chart "Price (USD)" which is plainly wrong for that one
# - 2.15 million dollars a share would be quite something.
# no won symbol in the stats box on purpose: matplotlib's default font
# has no glyph for it and it renders as an empty square. the axis label
# already says KRW so the bare number is fine
CURRENCY = {"000660.KS": ("KRW", "")}
DEFAULT_CURRENCY = ("USD", "$")

# events list, reusing this in both task 4 and task 5 charts so
# they line up and tell the same story.
#
# trimmed this down to just COVID. checked the other three and they
# weren't earning their place - chatgpt moved the price a lot but not the
# volatility, arm ipo did nothing outside arm itself, and the nvda split
# can't show up at all because auto_adjust=True already applies it
# backwards through the whole series.
#
# COVID is the one that survives testing: realised volatility in the 60
# days after was 1.13x to 1.94x each stock's own baseline, on all 7
# tickers that existed then. that's the chart-level version of the
# volatility clustering the ARCH tests pick up in tasks 7-14 (ljung-box
# on squared residuals rejects for 5 of the 8 tickers, at p < 1e-100 for
# NVDA and SMCI)
EVENTS = [
    ("2020-03-16", "COVID\nCrash", "red"),
]


# TASK 4 Close Plotting with Event Markers

def plot_closing_prices():

    os.makedirs(CLOSING_PRICE_DIR, exist_ok=True)

    for ticker in TICKERS:

        print(f"Plotting {ticker}...")

        path = os.path.join(DATA_DIR, f"{ticker}.csv")

        df = pd.read_csv(path, parse_dates=["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

        y_min = df["Close"].min()
        y_max = df["Close"].max()
        y_range = y_max - y_min

        fig, ax = plt.subplots(figsize=(14, 5))

        # main price line
        ax.plot(
            df["Date"],
            df["Close"],
            color="black",
            linewidth=1.2,
            label=f"{ticker} Closing Price"
        )

        df_start = df["Date"].min()
        df_end = df["Date"].max()

        for date_str, label, colour in EVENTS:

            event_date = pd.Timestamp(date_str)

            # only draw the event line if it actually falls inside
            # this stock's date range (ARM won't have COVID on it
            # since it didn't even exist yet)
            if df_start <= event_date <= df_end:

                # matplotlib wants dates as floats not timestamps
                event_x = float(mdates.date2num(event_date))

                ax.axvline(
                    x=event_x,
                    color=colour,
                    linestyle="--",
                    linewidth=1.2,
                    alpha=0.8,
                    label=label.replace("\n", " ")
                )

                ax.text(
                    event_x,
                    y_max - (y_range * 0.05),
                    label,
                    color=colour,
                    fontsize=7.5,
                    ha="center",
                    va="top",
                    fontweight="bold"
                )

        # x axis just showing years, looks cleaner than full dates
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        plt.title(
            f"{ticker} — Daily Closing Price (2019–2026)\n"
            f"with COVID Crash marked",
            fontsize=12,
            fontweight="bold"
        )

        # pull the right currency for this ticker - everything is USD
        # except SK Hynix
        code, symbol = CURRENCY.get(ticker, DEFAULT_CURRENCY)

        plt.xlabel("Date")
        plt.ylabel(f"Price ({code})")

        plt.grid(True, alpha=0.3, linestyle="--")
        plt.legend(loc="upper left", fontsize=8)

        min_price = df["Close"].min()
        max_price = df["Close"].max()
        last_price = df["Close"].iloc[-1]

        # little stats box in the corner, just min/max/latest.
        # no decimals for won, those prices are in the millions and
        # 2150000.00 just looks daft
        dp = 0 if code == "KRW" else 2
        stats = (
            f"Min: {symbol}{min_price:,.{dp}f}\n"
            f"Max: {symbol}{max_price:,.{dp}f}\n"
            f"Latest: {symbol}{last_price:,.{dp}f}"
        )

        ax.text(
            0.98,
            0.05,
            stats,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                alpha=0.85
            )
        )

        save_path = os.path.join(
            CLOSING_PRICE_DIR,
            f"{ticker}_closing_price.png"
        )

        plt.savefig(
            save_path,
            dpi=150,
            bbox_inches="tight"
        )

        plt.close()

        print(f"Saved -> {save_path}")


# quick numbers version of task 4 - basically the same stats that show up
# in the corner box on the plot, just dumped into one csv so i don't have
# to open 8 pngs and squint at the numbers

def closing_price_summary():

    os.makedirs(TABLES_DIR, exist_ok=True)

    rows = []

    for ticker in TICKERS:

        path = os.path.join(DATA_DIR, f"{ticker}.csv")

        df = pd.read_csv(path, parse_dates=["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

        first_price = df["Close"].iloc[0]
        last_price = df["Close"].iloc[-1]

        # how much the stock grew (or fell) from the very first
        # day of data to the latest day, as a percentage
        total_return_pct = ((last_price - first_price) / first_price) * 100

        rows.append({
            "ticker": ticker,
            "start_date": df["Date"].iloc[0].date(),
            "end_date": df["Date"].iloc[-1].date(),
            "n_trading_days": len(df),
            "min_price": df["Close"].min(),
            "max_price": df["Close"].max(),
            "first_price": first_price,
            "latest_price": last_price,
            "total_return_pct": total_return_pct,
            "mean_price": df["Close"].mean(),
            # flagging the currency in the table too, otherwise SK Hynix's
            # numbers look like a typo next to the USD ones
            "currency": CURRENCY.get(ticker, DEFAULT_CURRENCY)[0],
        })

    summary_df = pd.DataFrame(rows)

    # putting the biggest gainers on top, easier to see who
    # actually did well vs who's been flat/down
    summary_df = summary_df.sort_values(
        "total_return_pct", ascending=False
    ).reset_index(drop=True)

    save_path = os.path.join(TABLES_DIR, "closing_price_summary.csv")
    summary_df.to_csv(save_path, index=False)

    print(f"Saved -> {save_path}")

    return summary_df


# TASK 5 daily returns + rolling volatility

def plot_daily_returns():

    os.makedirs(DAILY_RETURNS_DIR, exist_ok=True)

    for ticker in TICKERS:

        print(f"Plotting daily returns for {ticker}...")

        path = os.path.join(DATA_DIR, f"{ticker}.csv")

        df = pd.read_csv(path, parse_dates=["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

        # % change from yesterday's close to today's close
        # this is basically the "how much did it move today" number,
        # which matters more than the raw price for the volatility side.
        # also puts every ticker on the same scale, so SK Hynix's won
        # prices stop being a problem here
        df["Return"] = df["Close"].pct_change() * 100

        # first row is always NaN since there's no "yesterday" for it
        df = df.dropna(subset=["Return"]).reset_index(drop=True)

        # rolling std over last 21 days (~1 trading month), gives us
        # a smoothed volatility line instead of just noisy daily spikes.
        # this is basically showing where the stock had calm periods
        # vs crazy periods, which normal ARIMA can't really pick up on
        df["RollingVol"] = df["Return"].rolling(window=21).std()

        r_min = df["Return"].min()
        r_max = df["Return"].max()
        r_range = r_max - r_min

        fig, (ax1, ax2) = plt.subplots(
            2, 1,
            figsize=(14, 7),
            sharex=True,
            gridspec_kw={"height_ratios": [2, 1]}
        )

        # top graph - the actual daily returns
        ax1.plot(
            df["Date"],
            df["Return"],
            color="black",
            linewidth=0.6,
            label=f"{ticker} Daily Return (%)"
        )

        ax1.axhline(0, color="grey", linewidth=0.8, alpha=0.6)

        df_start = df["Date"].min()
        df_end = df["Date"].max()

        for date_str, label, colour in EVENTS:

            event_date = pd.Timestamp(date_str)

            if df_start <= event_date <= df_end:

                event_x = float(mdates.date2num(event_date))

                ax1.axvline(
                    x=event_x,
                    color=colour,
                    linestyle="--",
                    linewidth=1.2,
                    alpha=0.8,
                    label=label.replace("\n", " ")
                )

                ax1.text(
                    event_x,
                    r_max - (r_range * 0.05),
                    label,
                    color=colour,
                    fontsize=7.5,
                    ha="center",
                    va="top",
                    fontweight="bold"
                )

        ax1.set_title(
            f"{ticker} — Daily Returns (2019–2026)\n"
            f"with COVID Crash marked",
            fontsize=12,
            fontweight="bold"
        )
        ax1.set_ylabel("Daily Return (%)")
        ax1.grid(True, alpha=0.3, linestyle="--")
        ax1.legend(loc="upper left", fontsize=8)

        vol_mean = df["Return"].std()
        max_up = df["Return"].max()
        max_down = df["Return"].min()

        stats = (
            f"Std Dev: {vol_mean:.2f}%\n"
            f"Max Gain: {max_up:.2f}%\n"
            f"Max Drop: {max_down:.2f}%"
        )

        ax1.text(
            0.98,
            0.05,
            stats,
            transform=ax1.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                alpha=0.85
            )
        )

        # bottom graph - the rolling volatility. this is honestly the more
        # important panel since it actually shows the "clustering" pattern
        # (calm patch, then a burst of crazy days, then calm again) which
        # is the thing ARIMA structurally can't model - it assumes the
        # error variance is constant
        ax2.plot(
            df["Date"],
            df["RollingVol"],
            color="darkorange",
            linewidth=1.0,
            label="21-Day Rolling Volatility"
        )

        ax2.fill_between(
            df["Date"],
            df["RollingVol"],
            color="darkorange",
            alpha=0.2
        )

        ax2.set_ylabel("Rolling Std Dev (%)")
        ax2.set_xlabel("Date")
        ax2.grid(True, alpha=0.3, linestyle="--")
        ax2.legend(loc="upper left", fontsize=8)

        ax2.xaxis.set_major_locator(mdates.YearLocator())
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        plt.tight_layout()

        save_path = os.path.join(
            DAILY_RETURNS_DIR,
            f"{ticker}_daily_returns.png"
        )

        plt.savefig(
            save_path,
            dpi=150,
            bbox_inches="tight"
        )

        plt.close()

        print(f"Saved -> {save_path}")


# numbers version of task 5. same stats as the plot's text box, plus skew
# and kurtosis which are useful for the writeup when talking about fat
# tails / non-normal returns
def daily_returns_summary():

    os.makedirs(TABLES_DIR, exist_ok=True)

    rows = []

    for ticker in TICKERS:

        path = os.path.join(DATA_DIR, f"{ticker}.csv")

        df = pd.read_csv(path, parse_dates=["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

        df["Return"] = df["Close"].pct_change() * 100
        df = df.dropna(subset=["Return"]).reset_index(drop=True)

        # same window as the plot so the numbers actually match
        # what you'd see if you looked at the chart
        df["RollingVol"] = df["Return"].rolling(window=21).std()

        rows.append({
            "ticker": ticker,
            "mean_daily_return_pct": df["Return"].mean(),
            "std_dev_pct": df["Return"].std(),
            "max_gain_pct": df["Return"].max(),
            "max_drop_pct": df["Return"].min(),
            # pandas .kurt() gives EXCESS kurtosis (normal = 0, not 3).
            # worth knowing before quoting these in the report
            "skewness": df["Return"].skew(),
            "kurtosis": df["Return"].kurt(),
            "peak_rolling_vol_pct": df["RollingVol"].max(),
            "latest_rolling_vol_pct": df["RollingVol"].iloc[-1],
        })

    summary_df = pd.DataFrame(rows)

    # most volatile stock on top, makes it easy to spot the
    # crazy ones (looking at you SMCI and ARM)
    summary_df = summary_df.sort_values(
        "std_dev_pct", ascending=False
    ).reset_index(drop=True)

    save_path = os.path.join(TABLES_DIR, "daily_returns_summary.csv")
    summary_df.to_csv(save_path, index=False)

    print(f"Saved -> {save_path}")

    return summary_df


if __name__ == "__main__":
    print("Task 4: Plotting closing prices with event markers...")
    plot_closing_prices()
    print("\nTask 4: Computing closing price summary table...")
    close_summary = closing_price_summary()
    print(close_summary)
    print("\nTask 5: Plotting daily returns with volatility clustering...")
    plot_daily_returns()
    print("\nTask 5: Computing daily returns summary table...")
    returns_summary = daily_returns_summary()
    print(returns_summary)
    print("\nDone. Check results/plots/task4_closing_prices/, results/plots/task5_daily_returns/, results/tables/closing_price_summary.csv, and results/tables/daily_returns_summary.csv")
