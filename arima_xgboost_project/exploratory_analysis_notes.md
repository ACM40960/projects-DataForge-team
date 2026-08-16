# Exploratory Data Analysis

This covers the first two sections of the project: closing prices and
daily returns / volatility. Both sections come with a plot per ticker and a
summary CSV so the numbers don't just live inside the chart images.

Code: `exploratory_analysis.py`
Plots: `results/plots/closing_prices/`, `results/plots/daily_returns/`
Tables: `results/tables/closing_price_summary.csv`, `results/tables/daily_returns_summary.csv`

---

## Closing Prices

### What we did

For each of the 8 tickers, we plotted the daily closing price from 2019 to mid-2026,
with four real-world events marked as vertical lines:

- **COVID Crash** (Mar 2020)
- **ChatGPT Launch** (Nov 2022)
- **ARM IPO** (Sep 2023)
- **NVDA Split** (Jun 2024)

The point of the event markers isn't just decoration — it lets us visually connect
price moves to things that were actually happening in the world, instead of a plain
price line that doesn't explain anything on its own.

We also built a summary table (`closing_price_summary.csv`) with each stock's
start/end date, min/max price, and total return over the whole period.

### Results

| Ticker | Total Return | Trading Days |
|---|---|---|
| NVDA | **5,983%** | 1,872 |
| 000660.KS (SK Hynix) | 3,720% | 1,824 |
| MU | 2,973% | 1,872 |
| AMD | 2,617% | 1,872 |
| SMCI | 1,917% | 1,872 |
| AVGO | 1,723% | 1,872 |
| TSM | 1,264% | 1,872 |
| ARM | 499% | 689 |

**Note on ARM:** it only IPO'd in Sept 2023, so it has ~689 trading days compared to
~1,872 for the rest. Its lower total return isn't a fair like-for-like comparison
against stocks that had 3x longer to grow — just something worth flagging rather
than reading too much into.

**Takeaway:** all 8 stocks grew a lot over this window (makes sense — this is the
AI/semiconductor boom period), but the spread between them is huge. NVDA returned
roughly 12x what TSM did. That kind of spread is actually useful for our project —
if every stock had grown the same predictable amount, a simple model would probably
be enough. The fact that they're all so different from each other is part of why a
more flexible forecasting approach is worth building.

---

## Daily Returns & Volatility

### What we did

Closing price tells you *what the stock is worth*. It doesn't tell you *how much it
moves around day to day* — a stock can be at an all-time high and still be wildly
jumpy on a daily basis. That's what this task is for.

For each ticker we computed:

- **Daily return** — % change from yesterday's close to today's close
- **21-day rolling volatility** — the standard deviation of daily returns over a
  trailing ~1 month window, so we get a smoothed line showing calm periods vs.
  turbulent periods, instead of one flat number for the whole 7 years

Each plot has two panels: daily returns on top, rolling volatility underneath.

We also built a numbers table (`daily_returns_summary.csv`) with mean return, std
dev, max gain/drop, skewness, and kurtosis for each ticker.

### Why "21 days" specifically

21 trading days is roughly one calendar month (markets are open ~252 days a year,
252 / 12 ≈ 21). It's a standard window size used in finance — long enough to smooth
out single-day noise, short enough to still clearly show something like COVID as a
distinct spike rather than disappearing into a flat average.

### Results — sorted by volatility (most volatile first)

| Ticker | Std Dev | Max Gain | Max Drop | Skew | Kurtosis |
|---|---|---|---|---|---|
| ARM | 4.80% | 47.9% | -19.5% | 1.93 | 17.2 |
| SMCI | 4.80% | 35.9% | -33.3% | 0.58 | 11.7 |
| AMD | 3.49% | 23.8% | -17.3% | 0.63 | 4.7 |
| MU | 3.29% | 19.3% | -19.8% | 0.18 | 3.8 |
| NVDA | 3.20% | 24.4% | -18.5% | 0.30 | 4.7 |
| 000660.KS | 2.88% | 15.9% | -11.5% | 0.43 | 2.7 |
| AVGO | 2.68% | 24.4% | -19.9% | 0.30 | 10.4 |
| TSM | 2.36% | 12.7% | -14.0% | 0.20 | 3.2 |

**What std dev means:** on a "normal" day, how much this stock typically moves.
TSM barely moves day to day (2.36%), ARM and SMCI can swing twice as much (4.80%).

**What skewness means:** whether the extreme days lean more toward gains or losses.
Close to 0 = roughly balanced. ARM's 1.93 stands out hard — it's had extreme days
that lean much more toward big gains than big losses (its 47.9% single-day max gain
is almost certainly one earnings/news spike pulling this number up).

**What kurtosis means:** how likely extreme "shock" days are, compared to a normal
bell-curve distribution. Higher = more prone to sudden big moves. ARM (17.2) and
SMCI (11.7) are way out ahead of everyone else here.

**Why this matters for the actual project:** every single ticker has kurtosis well
above what a normal distribution would have. That's the evidence that stock returns
don't behave like a nice smooth bell curve — there are more extreme days than a
"well-behaved" statistical model expects. ARIMA's math leans on an assumption close
to constant, roughly-normal noise. This data says that assumption doesn't really
hold, especially for ARM and SMCI. That's the concrete reason we're not just using
ARIMA alone — we need a second model (XGBoost) that can pick up on this kind of
irregular, clustered volatility pattern that ARIMA's assumptions can't capture.

---

## Return and Risk Together

| Ticker | Total Return (T4) | Volatility (T5) | Pattern |
|---|---|---|---|
| NVDA | 5,983% | 3.20% (mid) | Huge return, moderate risk |
| 000660.KS | 3,720% | 2.88% (low) | Huge return, low risk |
| MU | 2,973% | 3.29% (mid) | Huge return, moderate risk |
| AMD | 2,617% | 3.49% (mid-high) | Big return, moderate-high risk |
| SMCI | 1,917% | 4.80% (highest, tied) | Big return, high risk |
| AVGO | 1,723% | 2.68% (low) | Solid return, low risk |
| TSM | 1,264% | 2.36% (lowest) | Smallest return, lowest risk |
| ARM | 499% | 4.80% (highest, tied) | Smallest return, highest risk |

Normally you'd expect "more risk = more reward" to roughly hold. Here it mostly
doesn't:

- **NVDA and SK Hynix are the standout cases** — massive growth without needing
  huge daily volatility to get there.
- **ARM is the opposite** — took on the most day-to-day chaos and got the smallest
  reward for it (though its shorter trading history is part of that story).
- **SMCI is the closest fit to the textbook pattern** — high risk, high reward —
  though some of its volatility traces back to its 2024 accounting scandal rather
  than pure AI-market movement, worth remembering when interpreting the number.
- **TSM is the steady, low-drama stock** — lowest risk, lowest reward, consistent
  with being an already-massive, mature company rather than a fast mover.

The fact that these 8 stocks don't all follow the same risk-reward pattern is part
of the argument for this project: a single simple model probably can't handle all 8
tickers equally well, since they behave so differently from each other. A model
that can adapt per-ticker (which is what the hybrid ARIMA-XGBoost approach is meant
to do) makes more sense here than trying to force one static approach onto all of
them.