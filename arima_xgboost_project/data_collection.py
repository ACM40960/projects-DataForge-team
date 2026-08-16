import os
import pandas as pd
import yfinance as yf

TICKERS = [
    "NVDA",       # NVIDIA — primary AI GPU manufacturer
    "AMD",        # Advanced Micro Devices — GPU competitor
    "SMCI",       # Super Micro Computer — AI server infrastructure
    "ARM",        # Arm Holdings — chip architecture
    "MU",         # Micron Technology — memory chips
    "AVGO",       # Broadcom — semiconductors and infrastructure
    "TSM",        # Taiwan Semiconductor — world's largest chip foundry
    "000660.KS",  # SK Hynix — Korean memory chip manufacturer (KRW, not USD)
]

START_DATE = "2019-01-01"
END_DATE   = "2026-06-14"
OUTPUT_DIR = "data"


def fetch_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame | None:
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

    return df.reset_index()


def clean_data(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.drop_duplicates(subset=["Date"], keep="first")

    missing_before = df.isnull().sum().sum()
    rows_before    = len(df)

    df = df.dropna().sort_values("Date").reset_index(drop=True)

    rows_after    = len(df)
    missing_after = df.isnull().sum().sum()

    print(f"  {ticker}: {rows_before} -> {rows_after} rows "
          f"({rows_before - rows_after} removed, {missing_before} missing values)")
    print(f"  Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")

    return df


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join("results", "tables"), exist_ok=True)

    cleaning_report = []

    for ticker in TICKERS:
        print(f"\n=== {ticker} ===")

        df = fetch_yfinance(ticker, START_DATE, END_DATE)

        if df is None:
            print(f"  FAILED: no data for {ticker}")
            continue

        rows_before    = len(df)
        missing_before = df.isnull().sum().sum()

        df = clean_data(df, ticker)

        rows_after    = len(df)
        missing_after = df.isnull().sum().sum()

        cleaning_report.append({
            "Ticker":         ticker,
            "Rows Before":    rows_before,
            "Rows After":     rows_after,
            "Rows Removed":   rows_before - rows_after,
            "Missing Before": missing_before,
            "Missing After":  missing_after,
            "Date From":      df["Date"].min().date(),
            "Date To":        df["Date"].max().date(),
        })

        out_path = os.path.join(OUTPUT_DIR, f"{ticker}.csv")
        df.to_csv(out_path, index=False)
        print(f"  Saved -> {out_path} ({rows_after} rows)")

    report_df   = pd.DataFrame(cleaning_report)
    report_path = os.path.join("results", "tables", "cleaning_report.csv")
    report_df.to_csv(report_path, index=False)

    print(f"\n{'='*50}")
    print("CLEANING SUMMARY")
    print("=" * 50)
    print(report_df.to_string(index=False))
    print(f"\nSaved report -> {report_path}")


if __name__ == "__main__":
    main()
