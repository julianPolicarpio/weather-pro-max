# highcast

A daily-high temperature probability forecaster for Los Angeles International
Airport (KLAX), built to match the exact series the Kalshi **LAHIGH** contract
settles on — the NWS Daily Climate Report (CLI) high, not any raw sensor
reading. It produces a calibrated probability for every 1°F bucket, both
day-ahead and continuously updated intraday as the day's observations come
in, and serves the result on a small local web dashboard. See the design spec
at `docs/superpowers/specs/2026-07-17-lax-high-temp-predictor-design.md` and
the Kalshi contract rules in `LAXHIGH.pdf` (repo root) for the settlement
details this project is built around.

## Setup

```
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

Optional: a Synoptic Data API token (https://synopticdata.com/) in
`config/klax.yaml` as `synoptic_token: "..."` unlocks the 5-minute-resolution
live obs feed, giving the intraday model a sharper running-max signal in the
gaps between METAR reports. Note that Synoptic access is a **paid**
subscription beyond the initial free trial — treat it as a later upgrade, not
a setup step. Everything in this project — backfill, day-ahead prediction,
intraday prediction, backtesting, the dashboard — works correctly without a
token, and all backtest numbers quoted in this README were measured
METAR-only; `fivemin.fetch_recent` simply returns no rows and the pipeline
uses METAR obs alone.

Then build the historical dataset and initialize the database:

```
highcast init-db && highcast backfill --years 5
```

This pulls ~5 years of CLI reports, VTEC alerts, METAR archive, and forecast
archives (Open-Meteo lead-1, NWS MOS). It's a one-time pull; day-to-day
freshness after this is handled by `wake` (see Scheduling below).

## Commands

- `highcast fetch` — one live snapshot: current METAR/5-min obs, Open-Meteo
  and NWS and MOS forecasts, active alerts, and yesterday's CLI report if not
  already stored.
- `highcast wake` — gap-fills any METAR hole since the last stored
  observation (bootstraps a week back if the database is empty), then runs
  `fetch`, then runs the auto prediction cycle (day-ahead once daily, intraday
  through the climate day). This is what the scheduled job below runs.
- `highcast predict [--date YYYY-MM-DD] [--live]` — issues a prediction and
  stores it. Without `--live` it's the day-ahead model (features known before
  the climate day starts); with `--live` it's the intraday nowcast, floored at
  the observed running max so far.
- `highcast backtest --model baseline|ml [--refit-every N]` — walk-forward
  backtest over the full historical dataset, printing MAE/CRPS/coverage for the
  chosen model (`baseline` = seasonal blend, `ml` = the LightGBM day-ahead
  model). `--refit-every` defaults to 30 (~840 fits/run); raise it for a quick
  look, keep 30 for a number you intend to quote.
- `highcast verify-hour --metric high|low --hour H [--arms ...]` — **the
  decision-hour comparison.** Walk-forwards each arm (`intraday`,
  `intraday-residual`, `dayahead`) restricted to the days common to all of them,
  then paired-bootstraps each against the first arm and flags whether the CI
  clears zero. `--hour` is **station-local wall clock**, printed back in the
  header so PDT never gets confused with LST. This is how the "which model at
  which hour" claims in `HANDOFF.md` are produced — re-run it after any model
  change rather than trusting the recorded number.
- `highcast export-bets [--metric high|low|both] [--kind manual|real|shadow]
  [--out bets.csv]` — CSV of your logged bets for the tracking spreadsheet:
  prediction / date placed / time placed / bucket / bet amount / winnings if
  correct / hit Y-N, plus the LA climate day, the settled actual, price and
  contract count so a row can be audited against the climate report. Date and
  time are in `display_tz` (Manila — when you actually placed it); the LA
  climate day is a separate column on purpose. An unsettled bet leaves the
  hit column **blank**, never `N`. Log bets with `highcast my-bet` (or the
  dashboard's MY BETS panel) and this stays in sync automatically.
- `highcast reparse-obs` — re-runs the METAR parser over every stored raw
  report and updates the derived columns in place. No network: it re-derives
  from text already in the database. Idempotent. Run it after any change to
  `parse_metar` (this is how `six_min_c`, the ASOS 6-hour minimum, was
  backfilled across 5 years of history).
- `highcast verify-quantization` — checks the "hidden spike" hypothesis (CLI
  high vs. visible running max) and spot-fetches 1-minute ASOS archive rows
  for manual comparison. ⚠️ The 1-minute archive lags ~3 days and is gappy —
  see `ACCURACY_IMPROVEMENTS.md` §0α; it is *not* a usable nowcast source.
- `highcast serve` — runs the FastAPI dashboard at `http://127.0.0.1:8712`
  (today's buckets, day-ahead vs. live distributions, the Manila information
  clock, and history).

Note: `python -m highcast.cli` is a no-op (no `if __name__ == "__main__"`
guard) — always use the `highcast` console script installed by `pip install
-e`, not `python -m highcast.cli`.

## Scheduling

The Mac this runs on lives in Asia/Manila and sleeps through LA's afternoon
(peak heating happens 03:00–07:00 PHT — the middle of the night, Manila
time). Rather than fight that, the schedule leans on `highcast wake`'s
gap-fill: whenever the machine wakes up, it backfills every METAR hour it
slept through before running the normal fetch + auto-predict cycle. A launchd
agent runs `wake` every 5 minutes while awake, and once immediately on every
load (including wake-from-sleep), so a nap of any length is caught up in one
cycle rather than left as a permanent hole.

To install it:

```
sed "s|__PROJECT__|$PWD|g" launchd/com.julian.highcast.plist \
  > ~/Library/LaunchAgents/com.julian.highcast.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.julian.highcast.plist
```

Run that from the project root so `$PWD` resolves correctly. To check it's
running: `launchctl list | grep highcast` should list `com.julian.highcast`
(a `0` in the last-exit-status column means the last run succeeded — the job
writes directly to `data/launchd.log` / `data/launchd.err`, not through a
shell pipe, so a non-zero exit is never masked). After the first load, and
after every subsequent wake from sleep, `data/launchd.log` should show a
`wake` cycle within a few seconds (`RunAtLoad` fires immediately; after that
it's every 300 seconds via `StartInterval`).

To uninstall: `launchctl bootout gui/$(id -u)/com.julian.highcast`.

## Manila cheat sheet

The dashboard's information clock (`/api/clock`) renders this per-day, DST
aware; here's the shape of it for a July (PDT) day:

| Event | LA time | Manila time |
|---|---|---|
| 00Z model runs land | 20:00 (prev day) | 11:00 |
| Climate day starts (midnight LST) | 01:00 | 16:00 |
| Morning obs window opens | 06:00 | 21:00 |
| 12Z model runs land / day-ahead sharpest | 08:00 | 23:00 |
| Peak heating begins | 12:00 | 03:00 (next day) |
| Peak heating ends | 16:00 | 07:00 (next day) |
| Preliminary climate report | 16:35 | 07:35 (next day) |
| Market close | 23:59 | 14:59 (next day) |
| Climate day ends | 01:00 (next day) | 16:00 (next day) |
| Settlement (10:00 AM ET) | 07:00 (next day) | 22:00 (next day) |

**Takeaway:** ~21:00–24:00 PHT (the morning obs window through the 12Z model
land time) is historically the sharpest high-information window before the
day's outcome is effectively locked in — the forecast distribution moves the
most per hour of waiting right there, and comparatively little after.

## Notes and caveats

- **The `yesterday_high` feature has no `fetched_at` bound.** The previous
  day's CLI report typically lands from IEM overnight, well before it's
  needed as a day-ahead feature, so `features.py` doesn't restrict which
  `fetched_at` snapshot of it to use the way it does for same-day forecast
  runs. This is an accepted design risk, not an oversight: if IEM's CLI feed
  is ever delayed past the next day's feature-build time, `yesterday_high`
  would silently be missing/stale for that one day rather than fail loudly.
- **Morning-hour intraday backtest CRPS is slightly optimistic.** The
  intraday model's `base_p50` feature comes from a `BaselineBlend` fit on the
  *full* historical dataset, including days that later show up as test days
  in the walk-forward acceptance check — a plan-mandated simplification for
  that check, not a leak in the live path. In production, `highcast predict
  --live` always fits the baseline on history strictly before the target
  date, so the live numbers are honest; only the backtest's own reported
  early-hour CRPS (hour 8, etc.) should be read as a mild lower bound rather
  than the exact number to expect live.
- **Don't schedule `--live` between local midnight and ~01:00.** In that
  window `predict --live` correctly reports "no observations yet for the
  current climate day" (the new climate day hasn't started, so there's
  nothing to nowcast from) — `auto_cycle`'s `in_climate_day` gate already
  keeps the scheduled job from calling `--live` there, since the climate day
  itself doesn't start until 01:00 local standard time.

## Sandbox (paper trading on every Kalshi temperature market)

Spec: `docs/superpowers/specs/2026-10-04-weather-sandbox-design.md`. Paper only —
it never imports the real-money client and never writes to `data/highcast.db`.

```
sandbox init                 # creates data/sandbox.db and four $5 accounts
sandbox cycle --force-decide # one snapshot + settle + bot decisions now
sandbox serve                # dashboard at http://127.0.0.1:8713
sandbox report               # persisted LOCKED/PROVISIONAL/UNLOCKED evidence
```

Schedule (every 15 minutes; bots decide once per UTC hour; a cycle that finds another
still running prints `{"status": "locked_out"}` and exits 0):

```
sed "s|__PROJECT__|$PWD|g" launchd/com.julian.sandbox.cycle.plist > ~/Library/LaunchAgents/com.julian.sandbox.cycle.plist
sed "s|__PROJECT__|$PWD|g" launchd/com.julian.sandbox.serve.plist > ~/Library/LaunchAgents/com.julian.sandbox.serve.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.julian.sandbox.cycle.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.julian.sandbox.serve.plist
```

A failed cycle exits 1 (fetch failure, stale snapshot, a bot error or a bot skipped as
stale). launchd keeps that as the job's last exit status — see it with
`launchctl print gui/$(id -u)/com.julian.sandbox.cycle`; it is not written to the log
files.

Uninstall:

```
launchctl bootout gui/$(id -u)/com.julian.sandbox.cycle
launchctl bootout gui/$(id -u)/com.julian.sandbox.serve
rm ~/Library/LaunchAgents/com.julian.sandbox.cycle.plist ~/Library/LaunchAgents/com.julian.sandbox.serve.plist
```

Each paper account first reaches PROVISIONAL, then UNLOCKED only after a fresh confirmation
window independently passes ALL of:

1. **≥100 settled bets.**
2. **≥100 Brier samples** — one per unique (event, snapshot); several bets on one event
   at one snapshot score once.
3. **≥30 distinct events**, counted both among the settled bets and among the scored
   Brier samples (`events` and `scored-events` in the report).
4. **Profit test:** profit after fees, cluster-bootstrap (by event) 95% CI lower bound > 0.
5. **Score test:** multiclass Brier of (bot − de-vigged market at the bet's own snapshot),
   cluster-bootstrap (by event) 95% CI upper bound < 0. A sample is scored only when the
   market quoted every bucket the bot priced.

The first passing window records the latest order ID. Confirmation uses only later orders
and only city-days new to its window (an order on an event the account already traded at or
before that marker is dropped). It is graded once, when the fresh window holds all four count
floors (100 settled bets, 30 events, 100 scored samples, 30 scored events), which never
depend on outcomes. Failure records its timestamp, counts a failed attempt and starts another
attempt after the latest order ID; the report and dashboard show the count. UNLOCKED remains
fixed. This two-stage heuristic does not control lifetime false positives across unlimited
attempts, and the sandbox verdict never authorizes or submits a real order. Reports and
dashboard GETs are read-only and say so when the database is not initialised.

No price ceiling by design (Julian, 2026-10-04). Reports show pre-day and same-day settled
results separately; confirmation uses the combined sample. Highcast covers KLAX/KMIA and
uses the desk's decision-hour forecast-kind preference, with no predictions newer than its
snapshot; MOS covers all 24 cities.

## Disclaimers

This project produces informational probability estimates, not financial or
investment advice, and Fable/Claude did not and cannot place, size, or advise
on any trade. If you intend to use this alongside Kalshi's LAHIGH market,
verify your own eligibility to trade there first — Kalshi is a US-regulated
exchange and access rules for residents outside the US, including the
Philippines, are Kalshi's to state and can change; this project makes no
claim about them. Finally, NWS climate reports can be revised after initial
publication (see LAXHIGH.pdf: a preliminary report is issued mid-afternoon
and a final report follows after the climate day closes) — this project's
training and verification always target the **final** CLI report, matching
how the contract itself settles, but any number you see intraday should be
read as provisional until that final report lands.
