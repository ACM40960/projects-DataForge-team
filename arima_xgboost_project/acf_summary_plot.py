# acf_summary_plot.py

# the per-ticker ACF/PACF plots in task7_14_arima_fits/ are the proper
# box-jenkins identification tool and they stay. but they're a bad poster
# graphic - 16 panels, and the reader has to know what a confidence band is
# before any of it means anything.
#
# this collapses the same information into one chart: for every ticker, how
# big is the strongest autocorrelation in the differenced series. that's the
# number that decides whether there's anything for ARIMA to model.
#
# the 0.20 line is a rough practical threshold. with ~1870 observations the
# statistical band sits at only 0.045, so a correlation of 0.05 comes out
# "significant" while meaning nothing in practice. size matters more than
# significance once the sample is this big.

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf

plt.style.use("seaborn-v0_8-whitegrid")

TICKERS = ["NVDA", "AMD", "SMCI", "ARM", "MU", "AVGO", "TSM", "000660.KS"]
DATA_DIR = "data"
TABLES_DIR = os.path.join("results", "tables")
OUT_DIR = os.path.join("results", "plots", "acf_summary")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    d_table = pd.read_csv(os.path.join(TABLES_DIR, "adf_final_d_selection.csv")
                          ).set_index("ticker")

    rows = []
    for t in TICKERS:
        df = pd.read_csv(os.path.join(DATA_DIR, f"{t}.csv"))
        s = df["Close"].copy()
        d = int(d_table.loc[t, "recommended_d"])
        for _ in range(d):
            s = s.diff()
        s = s.dropna()

        a = acf(s, nlags=30)[1:]
        p = pacf(s, nlags=30, method="ywm")[1:]
        rows.append({
            "ticker": t, "d": d, "n": len(s),
            "band": 1.96 / np.sqrt(len(s)),
            "max_abs_acf": np.abs(a).max(),
            "acf_lag1": a[0],
            "n_over_020": int((np.abs(np.concatenate([a, p])) > 0.20).sum()),
        })

    r = pd.DataFrame(rows).sort_values("max_abs_acf")
    r.to_csv(os.path.join(TABLES_DIR, "acf_summary.csv"), index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5),
                                   gridspec_kw={"width_ratios": [1.15, 1]})

    # left: strongest correlation per ticker 
    # MU gets its own colour because its bar is an artefact, not a finding
    colours = ["#c0392b" if t == "MU" else "#4a7ba7" for t in r.ticker]
    ax1.barh(r.ticker, r.max_abs_acf, color=colours)
    ax1.axvline(0.20, color="grey", ls="--", lw=1.2)
    ax1.text(0.205, -0.65, "0.20 — rough threshold for\nstructure worth modelling",
             fontsize=8, color="grey", va="bottom")
    ax1.axvline(r.band.mean(), color="lightsteelblue", ls=":", lw=1.2)
    ax1.text(r.band.mean() + 0.006, 4.5, "statistical\nband (0.045)",
             fontsize=7.5, color="steelblue", va="center", ha="left",
             bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))

    for i, (v, t) in enumerate(zip(r.max_abs_acf, r.ticker)):
        ax1.text(v + 0.008, i, f"{v:.3f}", va="center", fontsize=8.5,
                 fontweight="bold" if t == "MU" else "normal")

    ax1.set_xlim(0, 0.52)
    ax1.set_xlabel("Largest absolute autocorrelation in the differenced series")
    ax1.set_title("Six of eight have nothing above 0.14", fontweight="bold")

    # right: MU's lag-1, the over-differencing tell
    mu = r[r.ticker == "MU"].iloc[0]
    others = r[r.ticker != "MU"]

    ax2.scatter(others.d, [abs(x) for x in others.acf_lag1],
                s=70, color="#4a7ba7", zorder=3, label="d = 1 (seven tickers)")
    ax2.scatter([mu.d], [abs(mu.acf_lag1)], s=140, color="#c0392b",
                zorder=3, marker="D", label="d = 2 (MU)")
    ax2.axhline(0.5, color="#c0392b", ls="--", lw=1.2)
    ax2.text(1.5, 0.515,
             "0.50 = what double-differencing a random walk\nproduces on its own",
             fontsize=8, color="#c0392b", ha="center")
    ax2.annotate(f"MU: {mu.acf_lag1:.3f}", xy=(2, abs(mu.acf_lag1)),
                 xytext=(1.62, 0.35), fontsize=9, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.2))

    ax2.set_xticks([1, 2])
    ax2.set_xlim(0.7, 2.3)
    ax2.set_ylim(0, 0.62)
    ax2.set_xlabel("Differencing order d")
    ax2.set_ylabel("|lag-1 autocorrelation|")
    ax2.set_title("MU's structure is an artefact of d = 2", fontweight="bold")
    ax2.legend(fontsize=8, loc="upper left")

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "acf_summary.png")
    plt.savefig(path, dpi=150)
    plt.close()

    pd.set_option("display.width", 200)
    print(r.round(3).to_string(index=False))
    print(f"\nSaved -> {path}")
    print(f"Saved -> {TABLES_DIR}/acf_summary.csv")


if __name__ == "__main__":
    main()
