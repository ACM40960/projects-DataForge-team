import os
import pandas as pd
import yfinance as yf

# Configuration
TICKERS = [
    "NVDA",      # NVIDIA
    "AMD",       # Advanced Micro Devices
    "SMCI",      # Super Micro Computer
    "ARM",       # Arm Holdings
    "MU",        # Micron Technology
    "AVGO",      # Broadcom
    "TSM",       # Taiwan Semiconductor Manufacturing Company
    "000660.KS" # SK hynix
]

START_DATE = "2019-01-01"
END_DATE = "2026-06-14"

OUTPUT_DIR = "data"


def fetch_yfinance(
    ticker: str,
    start: str,
    end: str
) -> pd.DataFrame | None: 

    try:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=True,
            progress=False
        )
    except Exception as e:
        print(f"Error downloading {ticker}: {e}")
        return None

    if df is None or df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    return df


def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for ticker in TICKERS:

        print(f"\n=== {ticker} ===")

        df = fetch_yfinance(
            ticker,
            START_DATE,
            END_DATE
        )

        if df is None:
            print(f"FAILED: no data for {ticker}")
            continue

        out_path = os.path.join(
            OUTPUT_DIR,
            f"{ticker}.csv"
        )

        df.to_csv(out_path, index=False)

        print(
            f"Saved {ticker} -> {out_path} "
            f"({len(df)} rows)"
        )


if __name__ == "__main__":
    main()