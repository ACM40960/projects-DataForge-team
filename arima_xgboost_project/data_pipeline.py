import os
import pandas as pd
import yfinance as yf

# Configuration
TICKERS = [
    "NVDA",       # NVIDIA — primary AI GPU manufacturer
    "AMD",        # Advanced Micro Devices — GPU competitor
    "SMCI",       # Super Micro Computer — AI server infrastructure
    "ARM",        # Arm Holdings — chip architecture
    "MU",         # Micron Technology — memory chips
    "AVGO",       # Broadcom — semiconductors and infrastructure
    "TSM",        # Taiwan Semiconductor — world's largest chip foundry
    # NOTE: SK Hynix is listed on the Korea Stock Exchange (KRX).
    # Prices are in Korean Won (KRW), not USD.
    # Trading hours differ from US exchanges so some dates will not overlap.
    # This will be addressed in the methodology section of the report.
    "000660.KS",  # SK Hynix — Korean memory chip manufacturer
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

def clean_data(df, ticker):
    
    # count missing values before
    missing_before = df.isnull().sum().sum()
    
    # count rows before
    rows_before = len(df)
    
    # remove rows with missing values
    df = df.dropna()
    
    # sort by date oldest to newest — ARIMA requires this
    df = df.sort_values("Date")
    
    # reset row numbers so they go 0, 1, 2, 3 again
    df = df.reset_index(drop=True)
    
    # count what we have after cleaning
    missing_after = df.isnull().sum().sum()
    rows_after = len(df)
    
    # report everything
    print(f"  {ticker}: missing values {missing_before} -> {missing_after}")
    print(f"  {ticker}: rows {rows_before} -> {rows_after} "
          f"({rows_before - rows_after} removed)")
    print(f"  Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    
    return df 


def main():

    # make the data folder if it doesnt exist already
    # exist_ok=True means dont crash if the folder is already there
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # also make the results/tables folder because thats where
    # we save the cleaning report at the end
    os.makedirs(os.path.join("results", "tables"), exist_ok=True)

    # this is an empty list where we will keep adding
    # one summary row for each stock after we clean it
    # at the end we turn this into a proper table and save it
    cleaning_report = []

    # go through each stock one by one
    # ticker will be NVDA first, then AMD, then SMCI, etc
    for ticker in TICKERS:

        print(f"\n=== {ticker} ===")

        # download the data for this stock from yahoo finance
        # if something goes wrong, df will be None
        df = fetch_yfinance(ticker, START_DATE, END_DATE)

        # if we got nothing back, skip this stock and move to next one
        # continue means jump straight to the next loop iteration
        if df is None:
            print(f"  FAILED: no data for {ticker}")
            continue

        # count rows and missing values BEFORE we clean
        # we save these numbers so we can compare them after cleaning
        rows_before    = len(df)
        missing_before = df.isnull().sum().sum()

        # now actually clean the data
        # this removes missing rows and sorts by date
        df = clean_data(df, ticker)

        # count rows and missing values AFTER cleaning
        # ideally missing_after should be 0
        rows_after    = len(df)
        missing_after = df.isnull().sum().sum()

        # add one row to our cleaning report for this stock
        # this captures everything that happened during cleaning
        # so we have a proper record for the report
        cleaning_report.append({
            "Ticker":         ticker,
            "Rows Before":    rows_before,
            "Rows After":     rows_after,
            "Rows Removed":   rows_before - rows_after,  # how many rows got deleted
            "Missing Before": missing_before,
            "Missing After":  missing_after,
            "Date From":      df["Date"].min().date(),   # earliest date in data
            "Date To":        df["Date"].max().date(),   # latest date in data
        })

        # save the cleaned data as a csv file in the data folder
        # one file per stock, named after the ticker e.g. NVDA.csv
        out_path = os.path.join(OUTPUT_DIR, f"{ticker}.csv")
        df.to_csv(out_path, index=False)
        print(f"  Saved -> {out_path} ({rows_after} rows)")

    # now that all stocks are done, turn the list into a proper table
    # pd.DataFrame() converts a list of dictionaries into a table
    report_df = pd.DataFrame(cleaning_report)

    # save the cleaning report as a csv so we have it permanently
    # this is evidence that we cleaned the data properly
    report_path = os.path.join("results", "tables", "cleaning_report.csv")
    report_df.to_csv(report_path, index=False)

    # print a nice summary at the end so we can see everything at once
    print(f"\n{'='*50}")
    print("CLEANING SUMMARY")
    print("="*50)
    print(report_df.to_string(index=False))
    print(f"\nSaved report -> {report_path}")


# this means only run main() if we directly run this file
# if another file imports this one, main() wont run automatically
if __name__ == "__main__":
    main()