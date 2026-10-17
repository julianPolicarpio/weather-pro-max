# Weather Pro Max

Probabilistic forecasts for daily high and low temperatures in 24 US cities, tested against the prediction
market that trades them (Kalshi's 48 daily temperature series), with a paper-trading sandbox and a fine-tuned
small language model as a bet/skip gate.

The question this project answers honestly: **can a forecasting model beat the market's own prices?** So far the
answer is no, and the repository shows how that was measured without fooling ourselves.

> Research and paper trading only. Nothing here places real orders or is financial advice.

## What it does

| Part | What it is |
|---|---|
| `highcast` | The original forecaster for Los Angeles (KLAX) and Miami (KMIA): day-ahead and intraday probability ladders for the NWS Daily Climate Report value each market settles on, with a live dashboard. |
| `citycast` | The 24-city pipeline: a 5-year data store (NWS climate reports, MOS forecasts, airport observations, every Kalshi market and hourly price), leak-free features, LightGBM quantile models, walk-forward evaluation, live predictions. |
| `sandbox` + **Weather Pro Max** | A paper-trading environment with six $5 accounts (one per strategy) that snapshots live market prices every 15 minutes, settles bets on the official report, and only unlocks a strategy for real money after it passes statistical tests. Its dashboard follows a trading-app layout: markets as probability ladders, per-bot portfolios, an unlock checklist. |
| Decision gate | Contenders that decide whether to take a bet: a simple edge rule, a LightGBM meta-labeler, and [Laya](https://huggingface.co/convaiinnovations/laya) (a 421M-parameter decision model) fine-tuned on Kaggle GPUs. The fine-tune lives in the companion repo `laya-weather-gate`. |

## Results

All numbers come from the reports in [`docs/reports/`](docs/reports/).

**Forecasting skill.** In walk-forward testing on April to September 2026 (data the models never trained on), the
models beat the plain MOS forecast in all 48 city/metric pairs:

| Metric | Model error (RPS) | MOS baseline | Error reduction |
|---|---|---|---|
| Daily high | 1.07 | 1.59 | 17 to 45% by city |
| Daily low | 0.88 | 1.40 | 14 to 50% by city |

**Against the market.** In every city with prices, the market's own probability was more accurate than the model
(Brier score), with most confidence intervals entirely on the market's side.

**Decision gate, once-only final test** (July to September 2026, 21,953 contracts, profit per contract seen):

| Contender | Profit | Note |
|---|---|---|
| Never bet | 0.0¢ | the winner |
| LightGBM meta-labeler | −0.27¢ | |
| Fine-tuned Laya | −0.32¢ | Brier 0.4591 zero-shot to 0.0860 after fine-tuning; cut the simple rule's losses by about 60% |
| Simple edge rule | −0.82¢ | |
| Laya zero-shot | −4.39¢ | |

Every contender's loss had a 95% confidence interval below zero, so the gate's answer is "don't bet". A breakdown
of validation losses points at cheap long-shot contracts, the favourite/long-shot bias, as the main leak.

## How the testing was kept honest

- **No look-ahead.** Every feature is built only from data published before the decision time. Forecast
  publication delays were measured live (MEX guidance took up to 17.8 hours to appear in the archive), and the
  first full walk-forward was thrown away unscored once the measured delays proved the defaults too short.
- **Walk-forward with an embargo.** Quarterly refits train only on earlier data, with a one-day gap at each
  boundary; the spread calibration uses earlier quarters only.
- **A once-only test.** The final comparison can run once; a second run needs an explicit force flag. A code
  fingerprint warns if the modelling code changes after the test period has been seen.
- **Fair baselines.** The MOS baseline gets the same observed-temperature information the model gets on
  same-day forecasts, and a winning strategy must beat "never bet", not just a weaker rule.
- **Clustered confidence intervals.** Bootstrap by city-day or by event, so correlated contracts are not counted
  as independent evidence.
- **Paper before money.** A strategy needs at least 100 settled bets, a profit interval above zero and better
  accuracy than the market, then a second confirmation window, before real money is even considered.

## Tech

Python 3.12 · LightGBM · pandas / NumPy · SQLite · FastAPI + Jinja · pytest (about 1,800 tests) · launchd jobs
for live collection · Kaggle T4 GPUs and Hugging Face for the Laya fine-tune · data from the Iowa Environmental
Mesonet (NWS reports, MOS, METAR) and Kalshi's public market API.

## Running it

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
.venv/bin/python -m pytest -q
```

Paper sandbox and the Weather Pro Max dashboard:

```bash
sandbox init
sandbox cycle
sandbox serve
```

City models (the first backfill downloads several GB from public sources):

```bash
citycast init
citycast backfill all
citycast dataset --metric high
citycast walkforward --metric high
citycast evaluate
```

`highcast` needs a station config: copy `config/klax.example.yaml` to `config/klax.yaml` first. See
`highcast --help`, `citycast --help` and `sandbox --help` for every command.

## Repository map

- `src/highcast/`: two-city forecaster, settlement parsing, live dashboard
- `src/citycast/`: 24-city data store, features, models, walk-forward, evaluation, decision gate
- `src/sandbox/`: paper trading, bots, settlement, scoreboard, Weather Pro Max dashboard
- `docs/reports/`: evaluation and final-test reports
