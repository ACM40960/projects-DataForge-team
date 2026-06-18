# ============================================================
# FILE: eda.py
# TASK 4: Plot closing prices for all 8 stocks
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt

# removed yfinance import — we dont need it here
# data is already downloaded and saved as CSVs by data_pipeline.py

TICKERS  = [
    "NVDA", "AMD", "SMCI", "ARM",
    "MU", "AVGO", "TSM", "000660.KS"
]
DATA_DIR  = "data"
PLOTS_DIR = os.path.join("results", "plots")


def plot_closing_prices():

    # make sure the plots folder exists before we try to save into it
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # loop over all 8 stocks one by one
    for ticker in TICKERS:

        print(f"  Plotting {ticker}...")

        # ── STEP 1: LOAD ──────────────────────────────────
        # load the cleaned CSV that data_pipeline.py already saved
        path = os.path.join(DATA_DIR, f"{ticker}.csv")
        df   = pd.read_csv(path, parse_dates=["Date"])
        df   = df.sort_values("Date").reset_index(drop=True)

        # calculate the train/test split point
        # 80% of rows for training, last 20% for testing
        # this is where ARIMA will stop seeing data
        n_train    = int(len(df) * 0.8)
        split_date = df["Date"].iloc[n_train]
        end_date   = df["Date"].iloc[-1]

        # ── STEP 2: CREATE ────────────────────────────────
        plt.figure(figsize=(14, 5))

        # ── STEP 3: DRAW ──────────────────────────────────
        # draw the closing price line
        plt.plot(df["Date"], df["Close"],
                 color="black", linewidth=1.2,
                 label=f"{ticker} closing price")

        # shade the test period in light orange
        plt.axvspan(split_date, end_date,
                    alpha=0.12, color="orange",
                    label="test period (last 20%)")

        # red dashed line at the train/test split
        plt.axvline(x=split_date,
                    color="red", linestyle="--",
                    linewidth=1.0, label="train/test split")

        # ── STEP 4: DECORATE ──────────────────────────────
        # f"{ticker}" puts the stock name in the title automatically
        # so NVDA gets "NVDA — Daily Closing Price", AMD gets its own, etc
        plt.title(f"{ticker} — Daily Closing Price (2019–2026)",
                  fontsize=13, fontweight="bold")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.grid(True, alpha=0.3, linestyle="--")
        plt.legend(loc="upper left", fontsize=9)

        # add min, max, latest price as a small text box
        # this gives the reader key numbers without opening the CSV
        min_price  = df["Close"].min()
        max_price  = df["Close"].max()
        last_price = df["Close"].iloc[-1]

        stats = (f"Min:    ${min_price:.2f}\n"
                 f"Max:    ${max_price:.2f}\n"
                 f"Latest: ${last_price:.2f}")

        # place the text box in the bottom right corner of the chart
        # transform=plt.gca().transAxes means 0 to 1 scale not price scale
        # so 0.98, 0.05 means 98% across and 5% up from the bottom
        plt.gca().text(0.98, 0.05, stats,
                       transform=plt.gca().transAxes,
                       ha="right", va="bottom", fontsize=9,
                       bbox=dict(boxstyle="round,pad=0.3",
                                 facecolor="white", alpha=0.85))

        # ── STEP 5: SAVE ──────────────────────────────────
        # save one PNG per stock e.g. results/plots/NVDA_closing_price.png
        save_path = os.path.join(PLOTS_DIR, f"{ticker}_closing_price.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

        # ── STEP 6: CLOSE ─────────────────────────────────
        # always close after saving — if we dont, the next stock
        # will draw on top of this one and the charts will be a mess
        plt.close()

        print(f"  Saved -> {save_path}")


# only runs when you execute: python eda.py directly
if __name__ == "__main__":
    print("Task 4: Plotting closing prices...")
    plot_closing_prices()
    print("\nDone. Check results/plots/ folder.")