# Stock Analysis Notes — What the Charts Are Actually Telling Us
## ACM 40960 | Soham Barve | June 2026

---

## Why I Made These Notes

When I plotted the closing prices for all 8 stocks I noticed some
really interesting things happening at specific points in time.
These notes are basically me writing down what I see in the charts
and trying to understand why each stock moved the way it did.
This will also help when writing the methodology section later.

---

## The 4 Events We Marked on the Charts and What Actually Happened

---

### COVID Crash — March 2020

So basically in March 2020 the whole world went into lockdown and
every stock market crashed. But what is interesting here is that
the semiconductor stocks did not crash as badly as other sectors
and actually recovered really fast. The reason is simple — when
everyone went home they bought laptops, monitors, gaming consoles,
and companies rushed to move everything to the cloud. All of that
needs chips.

What I see in the charts:

- **NVDA** — small dip but recovered in like 2 months. Data centers
  kept buying GPUs even during COVID so NVIDIA barely felt it.
- **AMD** — similar small dip then fast recovery. Gaming demand
  actually went up during lockdowns.
- **SMCI** — honestly you can barely see any reaction. The company
  was so small and unknown at this point that the market did not
  care much about it.
- **MU** — moderate dip. Memory demand was uncertain for a bit
  but recovered as PC buying surged.
- **AVGO, TSM, 000660.KS** — all had moderate dips in line with
  the broader market, all recovered within a few months.
- **ARM** — no data here, ARM only went public in September 2023.

---

### ChatGPT Launch — November 2022

This is honestly the most important event on all 8 charts.
You can literally see the moment AI became real for the stock market.
OpenAI released ChatGPT on November 30 2022 and within weeks
every investor realised that training AI models at scale needs
an insane amount of GPU compute. And who makes the best GPUs?
NVIDIA. So the whole sector went crazy.

What I see in the charts:

- **NVDA** — this is where the chart just goes vertical. Before
  this event NVDA was around $10-15. After this it never looked
  back. The ChatGPT launch is the single biggest reason NVDA
  reached $235 by 2026.
- **AMD** — was actually at its lowest point right around the
  ChatGPT launch after a brutal 2022 selloff. The launch gave
  AMD a reason to recover as investors bet AMD could compete
  in AI chips too.
- **SMCI** — the reaction was not immediate. It took a few months
  for the market to realise that someone needs to actually build
  the servers that hold these GPUs. SMCI does exactly that. So
  the explosion came a bit later in 2023.
- **MU** — also a delayed reaction. AI chips need a lot of high
  bandwidth memory but it took the market 6 to 12 months to
  connect those dots.
- **AVGO** — steady acceleration. Every AI data center needs
  networking chips and Broadcom makes those.
- **TSM** — makes every single AI chip in existence including
  NVIDIA GPUs. So the AI boom is directly good for TSMC.
- **000660.KS** — also delayed but then explosive. SK Hynix
  makes the HBM memory that goes inside NVIDIA GPUs. Once that
  connection became clear the stock went crazy.

---

### ARM IPO — September 2023

ARM Holdings went public in September 2023. It was the biggest
tech IPO of 2023 which is why we marked it. This is also why
our ARM data only starts from this date — there is no price
history before the company was listed.

What I see in the charts:

- **ARM** — started around $60 at IPO, was volatile for the
  first few months like most new listings. Then climbed to
  $190 as people realised ARM chip designs are in almost
  every AI device being built. Pulled back in 2025 then
  exploded to $411 in 2026.
- **Other stocks** — the ARM IPO did not directly move the
  other stocks. It is marked just for context.

One thing worth noting — ARM only has 689 rows of data
compared to 1872 for the other stocks. This is going to
affect how the ARIMA model performs on ARM since it has
much less history to learn from.

---

### NVDA Stock Split — June 2024

NVIDIA did a 10-for-1 stock split in June 2024. This means
if you had 1 share worth $1200 you now had 10 shares worth
$120 each. Total value the same, just more affordable for
smaller investors.

What I see in the charts:

- **NVDA** — you might expect to see a sudden drop here but
  our code uses auto_adjust=True in yfinance which means all
  historical prices are already adjusted for the split. So
  the chart shows a smooth continuous line without any
  artificial jump or drop. This is correct behaviour.
- **Other stocks** — no direct impact. The split is a NVIDIA
  internal decision.

---

## What These Charts Are Actually Telling Us About Our Model

### Why predicting these stocks is genuinely hard

Looking at all 8 charts together here is what stands out:

**SMCI is the hardest stock to predict by far.** It was completely
flat for years then went up 80x in one year then crashed 75%.
No model in the world could have predicted that from price data
alone. The accounting scandal that caused the crash was completely
invisible in the historical prices.

**The stocks move together but not identically.** They all
benefited from the AI boom but at different speeds and magnitudes.
NVDA moved first, SMCI followed months later, MU and SK Hynix
followed after that. This lag structure is interesting.

**Volatility comes in clusters.** Look at any of the charts and
you can see that big moves do not happen randomly — they happen
in groups. A big earnings announcement leads to days of big moves
around it. This is called volatility clustering and it is exactly
the kind of pattern that ARIMA cannot capture but XGBoost can.

### Why ARIMA alone will not be good enough

ARIMA is really good at modelling the smooth overall trend.
If a stock is going up steadily, ARIMA will follow it well.
But ARIMA completely misses things like:

- The sudden $5 to $120 explosion in SMCI in 2023
- The clusters of big daily moves after earnings
- The structural break that happened after ChatGPT launched

This is exactly why we pass the ARIMA errors (residuals) to
XGBoost. The residuals are basically everything ARIMA got wrong.
And XGBoost is really good at finding patterns in messy nonlinear
data. Together they should do better than either one alone.

### What I expect the results to show

Based on just looking at the charts:

- NVDA and AVGO should have the lowest MAPE because they
  trend fairly smoothly upward without too many crazy jumps
- SMCI should have the highest MAPE because it is genuinely
  unpredictable from price history alone
- The hybrid model should beat ARIMA alone on every stock
  but the improvement will be bigger on volatile stocks
  like SMCI and AMD than on smoother ones like AVGO and TSM

---

## Quick Summary of Each Stock

| Stock | What is the story | How volatile |
|---|---|---|
| NVDA | The AI boom stock. Steady strong growth driven by GPU demand | High |
| AMD | AI competitor to NVDA. More volatile, higher risk higher reward | Very High |
| SMCI | The most dramatic story. Boom then accounting scandal crash | Extreme |
| ARM | New listing in 2023. AI chip architecture play | High |
| MU | Memory chip maker. Delayed but explosive AI benefit | High |
| AVGO | Steady data center networking chips. Less exciting but consistent | Moderate |
| TSM | Makes every AI chip in the world. Steady beneficiary | Moderate |
| 000660.KS | Korean HBM memory leader. Prices in KRW not USD | Very High |

---

## Things to Remember When Writing the Methodology Section

- All 8 stocks show a clear change in behaviour after late 2022
  when the AI boom started. This is called a structural break.
- SMCI has the most extreme volatility and is the strongest
  argument for why we need a two stage model not just ARIMA.
- SK Hynix prices are in Korean Won so the numbers look huge
  compared to USD stocks. This is normal and expected.
- ARM only has 689 trading days of data so its ARIMA model
  will have less training data than the others.
- NVDA historical prices are split adjusted so the chart
  looks continuous even though a 10-for-1 split happened.