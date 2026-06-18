# ============================================================
# FILE: eda.py
# TASK 4: Plot closing prices for all 8 stocks
# with real world event markers
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

TICKERS = [
    "NVDA", "AMD", "SMCI", "ARM",
    "MU", "AVGO", "TSM", "000660.KS"
]
DATA_DIR  = "data"
PLOTS_DIR = os.path.join("results", "plots")


def plot_closing_prices():

    # make sure the plots folder exists before saving into it
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # real world events that explain why these stocks move so sharply
    # we mark these instead of a train/test split line because our model
    # uses a rolling window — there is no single fixed split point
    events = [
        ("2020-03-16", "COVID\nCrash",    "red"),
        ("2022-11-30", "ChatGPT\nLaunch", "green"),
        ("2023-09-14", "ARM\nIPO",        "blue"),
        ("2024-06-10", "NVDA\nSplit",     "purple"),
    ]

    for ticker in TICKERS:

        print(f"  Plotting {ticker}...")

        # 1: LOAD
        # load the cleaned csv that data_pipeline.py already saved
        path = os.path.join(DATA_DIR, f"{ticker}.csv")
        df   = pd.read_csv(path, parse_dates=["Date"])
        df   = df.sort_values("Date").reset_index(drop=True)

    
        y_min = df["Close"].min()
        y_max = df["Close"].max()
        y_range = y_max - y_min

        # 2: SETUP
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.figure(figsize=(14, 5))

        # 3: PLOT
        # draw the closing price line
        plt.plot(df["Date"], df["Close"],
                 color="black", linewidth=1.2,
                 label=f"{ticker} closing price")

        # drawn a vertical line for each real world event
        # only drawn if the event and the date is within this stock's range
        # ARM only starts Sept 2023 so COVID and ChatGPT wont appear on it
        for date_str, label, colour in events:

            # convert to python datetime so matplotlib handles it cleanly
            event_date = pd.Timestamp(date_str).to_pydatetime()
            df_start   = df["Date"].min().to_pydatetime()
            df_end     = df["Date"].max().to_pydatetime()

            #  drawn if this event falls within the stock's date range
            if df_start <= event_date <= df_end:

                # draw the vertical dashed line
                plt.axvline(x=event_date,
                            color=colour,
                            linestyle="--",
                            linewidth=1.2,
                            alpha=0.8,
                            label=label.replace("\n", " "))

                # add text label near the top of the chart
                # y_max - y_range * 0.05 puts it just below the top
                plt.text(event_date,
                         y_max - y_range * 0.05,
                         label,
                         color=colour,
                         fontsize=7.5,
                         ha="center",
                         va="top",
                         fontweight="bold")

        # 4: FORMAT
        # format the x axis to show years only, not months/days     
        plt.title(f"{ticker} — Daily Closing Price (2019–2026) "
                  f"with Key Market Events",
                  fontsize=12, fontweight="bold")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.grid(True, alpha=0.3, linestyle="--")
        plt.legend(loc="upper left", fontsize=8)

        # add min max latest price as a small text box bottom right
        min_price  = df["Close"].min()
        max_price  = df["Close"].max()
        last_price = df["Close"].iloc[-1]

        stats = (f"Min:    ${min_price:.2f}\n"
                 f"Max:    ${max_price:.2f}\n"
                 f"Latest: ${last_price:.2f}")

        plt.gca().text(0.98, 0.05, stats,
                       transform=plt.gca().transAxes,
                       ha="right", va="bottom", fontsize=9,
                       bbox=dict(boxstyle="round,pad=0.3",
                                 facecolor="white", alpha=0.85))

        # 5: SAVE 
        save_path = os.path.join(PLOTS_DIR,
                                 f"{ticker}_closing_price.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

        # 6: CLOSE
        plt.close()

        print(f"  Saved -> {save_path}")


# only runs when you execute: python eda.py directly
if __name__ == "__main__":
    print("Task 4: Plotting closing prices with event markers...")
    plot_closing_prices()
    print("\nDone. Check results/plots/ folder.")