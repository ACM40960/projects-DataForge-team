
import os
import pandas as pd

TABLES_DIR = os.path.join("results", "tables")
OUTPUT_PATH = os.path.join("residuals", "README.md")


def generate_readme():

    summary_path = os.path.join(TABLES_DIR, "arima_fit_summary.csv")

    if not os.path.exists(summary_path):
        print(f"Can't find {summary_path} - run arima_model.py first")
        return

    df = pd.read_csv(summary_path)

    needs_xgboost = df[df["residuals_look_random"] == False]["ticker"].tolist()
    already_clean = df[df["residuals_look_random"] == True]["ticker"].tolist()

    lines = []
    lines.append("# ARIMA Output Files - Notes for XGBoost Stage\n")
    lines.append(
        "Each `{ticker}_arima_output.csv` in this folder has 4 columns: "
        "`Date`, `Actual`, `Predicted`, `Residual`.\n"
    )
    lines.append(
        "`Predicted` is ARIMA's linear component (the `L` in y = L + N). "
        "`Residual` is `Actual - Predicted` - the leftover nonlinear part "
        "(the `N`) that the XGBoost stage is meant to model.\n"
    )

    lines.append("## Which tickers actually need XGBoost\n")
    lines.append(
        f"These tickers still have a detectable pattern left in their residuals "
        f"(Ljung-Box test failed) - there's real structure here for XGBoost to learn:\n"
    )
    for t in needs_xgboost:
        row = df[df["ticker"] == t].iloc[0]
        lines.append(f"- **{t}** (order ARIMA({row['order_p']},{row['order_d']},{row['order_q']}), ljung-box p={row['ljung_box_pvalue']:.4f})")

    lines.append(
        f"\nThese tickers' residuals already look like random noise - ARIMA "
        f"pretty much captured everything explainable, so XGBoost likely "
        f"won't add much here (not a bug, just not much left to model):\n"
    )
    for t in already_clean:
        row = df[df["ticker"] == t].iloc[0]
        lines.append(f"- **{t}** (order ARIMA({row['order_p']},{row['order_d']},{row['order_q']}), ljung-box p={row['ljung_box_pvalue']:.4f})")

    arch_tickers = df[df["has_arch_effect"] == True]["ticker"].tolist()
    if arch_tickers:
        lines.append("\n## Volatility clustering (ARCH effect)\n")
        lines.append(
            f"All of these show volatility clustering in the residuals - "
            f"periods of calm vs turbulent stretches - which is a variance "
            f"pattern ARIMA structurally can't fix (it only models the mean): "
            f"{', '.join(arch_tickers)}\n"
        )

    lines.append("\n## Forecast accuracy reference (30-day holdout)\n")
    lines.append("| Ticker | Order | RMSE | MAE | MAPE |")
    lines.append("|---|---|---|---|---|")
    for _, row in df.iterrows():
        lines.append(
            f"| {row['ticker']} | ARIMA({row['order_p']},{row['order_d']},{row['order_q']}) | "
            f"{row['rmse']:.2f} | {row['mae']:.2f} | {row['mape']:.2f}% |"
        )

    os.makedirs("residuals", exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        f.write("\n".join(lines))

    print(f"Saved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    generate_readme()