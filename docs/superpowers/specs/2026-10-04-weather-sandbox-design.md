# Weather Desk Sandbox — design

**Date:** 2026-10-04 (Manila). **Status:** approved; revised 2026-10-04 during planning with
live measurements (see "Planning corrections" at the end).
**Sub-project 1 of 3** in the "all weather markets + Laya" plan:
1. **Sandbox** (this spec) — live paper trading on every Kalshi daily temperature market.
2. City forecasts — per-city models beyond the MOS baseline.
3. Laya — fine-tune `convaiinnovations/laya` as the **bet/skip gate** (meta-labeler) on
   sandbox history and enter it as a bot (see §9).

## 1. Purpose and the rule it enforces

Julian's Kalshi demo account does not work for him, and the live desk's own record says the
market price beats our model at the commit hour (QUEUE item 1: multiclass Brier market 0.3107 vs
model 0.3881, n=72, CI [+0.0030, +0.1537]). The sandbox replaces the demo account with paper
trading against **real live prices and real order books**, for every Kalshi daily temperature
series with open markets (48 series across 24 US cities, measured 2026-10-04), and produces the
evidence that decides whether any model may bet real money.

**Unlock rule (decided):** no real-money bets from any model until its sandbox account has
**≥100 settled bets** AND passes **both**:
1. **Profit test** — profit after fees, bootstrap 95% CI entirely above zero.
2. **Score test** — multiclass Brier of its probabilities beats the de-vigged market ladder at
   each bet's own timestamp, paired-bootstrap 95% CI of (bot − market) entirely below zero.

**Tightened 2026-10-04 (Task 9 review):** the score test also needs ≥100 Brier samples, one per
unique (event, snapshot) — several bets on one event at one snapshot score once. Both CIs use a
**cluster bootstrap by event** (resample whole city-days), because bets on the same event share
one outcome and are not independent evidence. The bets must also span **≥30 distinct events**,
so the cluster bootstrap has enough clusters to be trusted (30 is a judgement call, not a
measurement).

**Tightened 2026-10-04 (final review):** the event floor applies to the score test as well:
`brier_events ≥ 30` distinct events among the scored Brier samples (shown as `scored-events`).
A Brier sample is scored only when the market ladder at that snapshot priced every bucket in the
bot's pmf; otherwise the market would be charged a full miss on a bucket it never quoted.

**Two-stage confirmation (Julian, 2026-10-04):** the displayed state starts LOCKED.
Passing every gate above on the current attempt's orders makes it PROVISIONAL and records
`MAX(orders.id)` as a marker. Only orders after that marker count toward confirmation.
Stage 2 is graded once, when that fresh window holds all four count floors: 100 settled bets,
30 events, 100 scored Brier samples and 30 scored events. It is then evaluated against ALL the
same gates. None of those counts depends on who won (whether a bet settled, whether a
candidate row exists, whether the market quoted every bucket), so waiting for them keeps the
stopping rule blind to outcomes. A bot that fills several buckets of one event in the same
hour gets one Brier sample for them, so it needs more than 100 settled bets to reach the
scored floor. A pass permanently sets UNLOCKED; a failure returns LOCKED with the failure time
and a new attempt starting after the current maximum order ID. A window counts only city-days
that are new to it: an order whose event already had any order from the same account at or
before the window's marker is dropped from that window. Stages 1 and 2 therefore share no
orders and no events, and a failed attempt's events never enter the next attempt.
`failed_attempts` counts failures, survives every transition, and is shown in every state once it
is above zero (`PROVISIONAL (…, after k failed attempts)`, `LOCKED — confirmation failed … (k failed
attempts)`, `UNLOCKED (after k failed attempts)`; singular for k = 1). This is a two-stage evidence heuristic,
not lifetime false-positive control over unlimited attempts. Confirmation advances after
settlement, before decisions, once per decision hour for all accounts, and an error in one
account is logged and reported in the cycle summary without stopping the others or trading;
reading the dashboard or report never advances it or writes the database.

The sandbox displays this verdict. It never places real orders (enforced by test, §7).

## 2. Scope

**In:** every series whose ticker starts `KXHIGH` or `KXLOWT` and has open markets, discovered
dynamically each cycle; simulated taker buys filled by walking the real order book; per-account
ledgers with fees; settlement on Kalshi's own market result; snapshot recording; three bots
(`favorite`, `mos`, `highcast`); one manual account; scoreboard with the unlock verdict; a local
dashboard.

**Out (later):** real order placement; rain, snow and hurricane markets; selling before
settlement and resting limit orders (Julian's 99¢ exits — timing, not EV); the `laya` bot;
multi-contract sizing rules; periodic full order-book recording (books are stored only at fill
time in v1).

## 3. Placement

New package `src/sandbox/` beside `src/highcast/`, own SQLite file `data/sandbox.db`.

- **Reuses (import, never copy):** `highcast.fees` (`taker_fee`, `fee_per_contract`),
  `highcast.buckets` (`pmf_mass`, `pmf_from_buckets`), `highcast.fetchers.kalshi.parse_markets`,
  `highcast.fetchers.forecasts` (`parse_mos`, `IEM_MOS`), `highcast.http.get_json`,
  `highcast.orderbot.MIN_EDGE`.
- **Settlement station per series** is parsed from each market's `rules_primary` (`(CLIXXX)` →
  ICAO `KXXX`); verified for all 48 live series on 2026-10-04. `highcast.cities.ROSTER` is stale
  (12 cities) and is not used.
- **Never writes** to highcast's database. Reads `predictions` read-only for the `highcast` bot.
- **Never imports** `highcast.kalshi_client` (the real-money client).
- Git repo since 2026-10-07: commit only when Julian asks (project rule in `CLAUDE.md`).

## 4. Components

| Unit | Responsibility |
|---|---|
| `stations` | Station from rules text; metric from series ticker; tz table for 24 stations |
| `kalshi_public` | Paced, unauthenticated Kalshi reads (series, open/settled markets, order book) |
| `markets` | Parse a series payload into `Market` rows (buckets, quotes, sizes, close time) |
| `book` | Parse an order book; **pure** fill walk |
| `pricing` | De-vig a ladder; multiclass Brier |
| `db` + `recorder` | Schema; snapshots, quotes, series/markets upserts, candidate rows |
| `ledger` | Accounts, orders, cash, holdings |
| `bots` | Strategy interface; `favorite`, `mos`, `highcast` |
| `settle` | Record Kalshi results; pay orders; label candidates |
| `scoreboard` | Profit CI, Brier vs market, unlock verdict |
| `cycle` | One scheduled run |
| `web` | Local FastAPI dashboard, own port |
| `cli` + launchd | `sandbox init / cycle / serve / report` |

## 5. Data flow

**Every 15 minutes:**
1. **Discover** — `GET /series?category=Climate and Weather`, keep `KXHIGH*`/`KXLOWT*`, then one
   `GET /markets?series_ticker=…&status=open&limit=200` per series (≈48 requests).
2. **Record** — one snapshot row plus one quote row per open market (best bid/ask and their sizes,
   from the markets payload). Snapshots are append-only. The snapshot's `taken_at` is the time
   discovery **started**, so every quote in it was observed at or after `taken_at` and its age is
   never understated.

**On the first cycle of each hour, additionally:**
3. **Settle** — §5 step 5.
4. **Decide** — each bot receives the latest snapshot at or before the decision time. **A bot never
   receives data stamped after its decision time.** Every bucket a forecasting bot (`mos`,
   `highcast`) evaluates is logged as a **candidate** — placed or skipped — with its context (§9).
5. **Fill** — for each intent, fetch a fresh order book and walk it. Kalshi's book lists resting
   **bids** only (`orderbook_fp.yes_dollars`, `orderbook_fp.no_dollars`, price and quantity as
   decimal strings — verified live 2026-10-04, public, no auth). Buying YES at price p consumes NO
   bids at (1 − p), best first; buying NO mirrors it. Partial fills are recorded as partial. Fee =
   `taker_fee(c, p)` per price level taken (conservative: rounds up per level). The book used is
   stored with the order. Manual bets take the same path at click time. `orders.decided_at` is
   the fill time (the clock read just before that intent's book fetch), not the snapshot time.
6. **Settle** — markets Kalshi reports `finalized`/`settled` with `result` yes/no pay $1 per
   winning contract; no settlement fee. All positions are held to settlement in v1.

**Pacing:** at most 5 requests per second (Kalshi's authenticated Basic tier allows 20/s at 10
tokens per request; the unauthenticated limit is undocumented, so stay well under).

**Failure handling:** a failed discovery fetch skips the whole cycle (no snapshot, no bets); a
snapshot older than 10 minutes makes every bot sit out; a book that moved between decision and
fill uses the fresh book. Staleness is checked against the real clock when deciding starts,
**again before each bot** (a stale bot is skipped and listed in `stale_bots`) and **again before
each intent** (that intent and every later one get `stale_snapshot`; no order rows). A cycle with
a non-`ok` status, a bot error or a stale bot exits 1. One cycle runs at a time
(`<db>.cycle.lock`); an overlapping start prints `{"status": "locked_out"}` and exits 0.

**Time:** stored as UTC ISO strings with `+00:00`; displayed Manila first; a market's day is
Kalshi's event date (the station's climate day).

## 6. Bots, accounts and dashboard

**No price ceiling by design (Julian, 2026-10-04); pre-day/same-day shown separately.**
Settled orders before station-local standard midnight count as pre-day; all others,
including unknown stations, count as same-day. The split is display-only and never
affects the combined confirmation verdict.

All bots: **1 contract per bet**, YES side, at most one bet per bucket ticker (a bot may hold
several buckets of the same city-day), $5 starting cash each. Each hour a bot's intents are
**funded by priority** (forecast bots: edge after fee; `favorite`: the favorite's probability;
ties by ticker). An intent the account cannot cover at its price cap plus fee is skipped as
`insufficient_cash` with no order row and no book fetch; a cheaper later intent may still fill.

| Bot | Rule |
|---|---|
| `favorite` | Buy YES on the highest de-vigged bucket of each event at the ask. Control: "follow the crowd" after fees. **Pre-day only** (the station's climate day has not started). If the favorite bucket has no usable ask the event is skipped — never a runner-up. No candidates logged. |
| `mos` | Every series whose station is in the tz table (all 24 cities). Bucket pmf = Normal(MOS MEX max/min for the event date, σ = 3.0 °F) over integer outcomes. Labelled **uncalibrated** on the dashboard (σ is a prior, not a measurement; calibration is sub-project 2). Bet when p − ask − fee per contract ≥ `MIN_EDGE`. **Pre-day only**: MOS has no observation input, so an event whose climate day (local standard midnight) has begun gets no MOS fetch and no candidates; an unknown station counts as started. After 3 consecutive MOS fetch failures the cycle stops fetching MOS. |
| `highcast` | KLAX and KMIA (high and low) only. Desk-preferred prediction via `highcast.gating.kind_at_hour` at station-local decision time (HIGH before its knee on the event date prefers dayahead; otherwise intraday), falling back to the other allowed kind `dayahead` or `intraday` (an allowlist: `evening` and any new kind are never read) issued at or before the snapshot's `taken_at`, read from highcast `predictions`; same edge rule. Not pre-day gated. |
| `manual` | Julian's hand bets from the dashboard (YES or NO). |

`MIN_EDGE` is `highcast.orderbot.MIN_EDGE` (0.03), imported.

**Dashboard** ("Weather Desk Sandbox", no Kalshi branding, localhost only, port 8713; highcast uses
8712):
1. **Markets** — every open series, each bucket's price and size on offer.
2. **Bet** — manual bet form that previews the exact fill and fee from the live book before confirming.
3. **Scoreboard** — per account: cash, profit after fees with CI, Brier vs market, two rows
   for pre-day/same-day settled counts, profit/CI and Brier difference/CI, and the persisted
   LOCKED / PROVISIONAL / UNLOCKED banner (PROVISIONAL shows the fresh settled, event, scored and
   scored-event counts; LOCKED after failure shows the failure time, as
   `2026-10-04 18:00 Manila`, and the number of failed attempts). Sample gates remain visible:
   `LOCKED: settled n/100 · scored n/100 · events n/30 · scored-events n/30 · profit test ✓/✗ ·
   score test ✓/✗`.
4. **Log** — every order with its snapshot and book levels, for audit.

## 7. Testing and acceptance

- Unit tests on saved real Kalshi payloads: YES-via-NO-bids walk, NO walk, partial fill, price
  cap, per-level fee rounding, settlement payout, de-vig, bootstrap CIs, unlock verdict.
- **No-lookahead test:** a bot is never handed a snapshot newer than its decision time.
- **No-real-money test:** importing every `sandbox` module must not import `highcast.kalshi_client`.
- **Candidate log test:** skipped candidates receive an outcome and as-if profit at settlement,
  identical in form to placed ones; no candidate row is ever updated except its outcome fields.
- **Acceptance:** 3 days of live cycles; 10 orders reconciled by hand against stored books and
  Kalshi settlements; no cycle placed a bet on a stale snapshot; `data/sandbox.db` growth
  measured and recorded.

## 8. Feeds the Laya sub-project

The recorder's snapshots, the candidate log, stored books and settled results are the training
and evaluation set for sub-project 3. Laya enters as one more bot under the same unlock rule.
**Verified for sub-project 3:** Kalshi serves public hourly candlesticks (`yes_bid`/`yes_ask`
OHLC) per market and paginated settled-market history, so candidates can be backfilled.

## 9. Candidate log and Laya's role (decided 2026-10-04)

**Candidate log (built in this sub-project).** One append-only row per (forecasting bot, bucket,
decision cycle) evaluated, whether placed or skipped:

- identity: bot, ticker, series, station, metric, event date, snapshot id, decision timestamp
  (the real clock when the bot decided; `minutes_to_close` is measured from it);
- forecast: bot probability for the bucket, and the full event pmf (`{ticker: p}`);
- market: de-vigged market probability, best bid/ask, spread, size at the ask, minutes to close;
- economics: fee per contract at the ask, edge after fees;
- context: `gates_json` and `regime_json` columns exist but are NULL in v1 (populated in
  sub-project 3); `forecast_json` records the forecast behind the pmf — `mos`:
  `{"source": "mos_mex", "mu": …, "sigma": …}`, `highcast`:
  `{"source": "highcast", "kind": …, "issued_at_utc": …}`;
- decision: placed or skipped, and why (`edge`, `below_min_edge`, `no_ask`, `already_held`,
  `no_fill`, `insufficient_cash`, `stale_snapshot`, `bot_error` — the last for a candidate whose
  intent never ran because the bot failed part-way);
- outcome (filled at settlement): bucket result, and profit per contract after fees **as if
  bought** YES at the recorded ask — filled for skipped candidates too, so skips are labelled.

**Laya's role (sub-project 3, designed separately).** Laya is the bet/skip gate:
- **Input:** a candidate rendered as text. **Output:** calibrated P(this bet profits after fees).
- **Fixed formula, not learned:** expected value = P(win)·$1 − price − fee per contract; place only
  if EV ≥ `MIN_EDGE`; stake stays at 1 contract until sizing is designed against sandbox evidence.
  Laya never chooses stake size.
- **Must beat, in the sandbox, on identical candidates:** (a) the existing hand-written gates and
  (b) a LightGBM meta-labeler on the same features. Same unlock rule as every bot.
- **Data caution:** candidates from the same city-day share one outcome, so the effective sample
  size is city-days, not rows. Training/validation splits are by date, never random.

## Planning corrections (2026-10-04, measured live)

1. **48 series / 24 cities, not 24 / 12.** Kalshi lists 96 `KXHIGH*`/`KXLOWT*` series; 48 had open
   markets (all US, 12 buckets each across 2 event dates = 576 markets). Discovery is dynamic.
2. **Cadence 15 min (snapshots) and hourly (decisions, settlement),** not 5 min, to keep
   `sandbox.db` small (28 GB free on this Mac; `data/` is already 4.1 GB).
3. **No periodic full-book recording;** books are fetched and stored at fill time only. Top-of-book
   sizes come free in the markets payload.
4. **`mos` σ is a fixed 3.0 °F prior** for every city, labelled uncalibrated, instead of per-station
   historical error (highcast has history for only 2 stations).
