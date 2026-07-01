# ============================================================
# FILE: eda.py
# TASK 4: Plot closing prices for all 8 stocks
# with real world event markers
# ============================================================

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


def plot_closing_prices():

    os.makedirs(PLOTS_DIR, exist_ok=True)

    events = [
        ("2020-03-16", "COVID\nCrash", "red"),
        ("2022-11-30", "ChatGPT\nLaunch", "green"),
        ("2023-09-14", "ARM\nIPO", "blue"),
        ("2024-06-10", "NVDA\nSplit", "purple"),
    ]

    for ticker in TICKERS:

        print(f"Plotting {ticker}...")

        path = os.path.join(DATA_DIR, f"{ticker}.csv")

        df = pd.read_csv(path, parse_dates=["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

        y_min = df["Close"].min()
        y_max = df["Close"].max()
        y_range = y_max - y_min

        

        fig, ax = plt.subplots(figsize=(14, 5))

        # Closing price line
        ax.plot(
            df["Date"],
            df["Close"],
            color="black",
            linewidth=1.2,
            label=f"{ticker} Closing Price"
        )

        df_start = df["Date"].min()
        df_end = df["Date"].max()

        for date_str, label, colour in events:

            event_date = pd.Timestamp(date_str)

            if df_start <= event_date <= df_end:

                # Convert datetime to matplotlib float
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

        # Format x-axis
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        plt.title(
            f"{ticker} — Daily Closing Price (2019–2026)\n"
            f"with Key Market Events",
            fontsize=12,
            fontweight="bold"
        )

        plt.xlabel("Date")
        plt.ylabel("Price (USD)")

        plt.grid(True, alpha=0.3, linestyle="--")
        plt.legend(loc="upper left", fontsize=8)

        min_price = df["Close"].min()
        max_price = df["Close"].max()
        last_price = df["Close"].iloc[-1]

        stats = (
            f"Min: ${min_price:.2f}\n"
            f"Max: ${max_price:.2f}\n"
            f"Latest: ${last_price:.2f}"
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
            PLOTS_DIR,
            f"{ticker}_closing_price.png"
        )

        plt.savefig(
            save_path,
            dpi=150,
            bbox_inches="tight"
        )

        plt.close()

        print(f"Saved -> {save_path}")


def plot_daily_returns():
    pass


if __name__ == "__main__":
    print("Task 4: Plotting closing prices with event markers...")
    plot_closing_prices()
    print("\nDone. Check results/plots/ folder.")