# KMIA HIGH order bot — design

**Date:** 2026-08-01
**Status:** approved in conversation; DRY-RUN build, arming deferred ~2 weeks
**Supersedes** the phase-2/phase-4 sketches in
`2026-07-31-accuracy-first-decision-hours-design.md` §7, which predate the
measurements below.

---

## 1. Why this market, and why now

KMIA HIGH is the only market that clears every gate the accuracy work produced:

| | |
|---|---|
| walk-forward hit at h14, n=2,122 | 79.3 % |
| **August**, n=155 | **84.2 %** [77.6, 89.1], MAE 0.376 °F |
| calibration | **−1.6 %** — under-states, the safe direction |
| ≥80 % band, n=244 | stated 87.8 %, hits **93.4 %** [89.6, 95.8] |
| `CONFIDENCE_FLOOR` | **0.70** — the only market with an admissible floor |
| seasonal stability | 70–84 % all year, no cliff |
| stale CLI rows | 0 of 570 days |

It also fits Julian's actual mechanic. He does not hold to settlement: he rests a
sell limit near 99 ¢ and exits when the market reprices to near-certainty. That
only works if the price reaches 99 ¢ **while the market is still open**:

| market | reached 99 ¢ | when |
|---|---|---|
| **KMIA HIGH** | **4/4 days** | 03:31–04:57 Manila |
| KLAX HIGH | 7/8 | ~05:00–07:00 Manila |
| KLAX LOW | 1/7 | — |
| KMIA LOW | **0/4** | — |

A HIGH locks once the afternoon peak passes and the 6-hour max group files,
leaving hours of open market to reprice. A LOW's climate day runs to 01:00 local,
so it is never certain until the window is closing. **The strategy structurally
requires a HIGH**, and KMIA is the accurate one.

## 2. THE CONSTRAINT THAT DECIDES PROFIT — the price ceiling

Exiting at 99 ¢ changes *timing*, not expected value. What decides profit is
entry price against true hit rate:

| buy at | break-even win rate |
|---|---|
| 0.70 | 73.5 % |
| **0.80** | **83.7 %** |
| 0.90 | 92.9 % |
| 0.95 | 98.0 % |

At 84.2 % measured accuracy, **0.81 is the highest price that can profit.** The
bot therefore carries a hard ceiling of **0.80**.

This is not a nicety. In a simulated week KLAX HIGH went **6 for 6** and still
had an expected value of **−20.5 ¢/contract**, because the average entry was
0.91 against 72.6 % accuracy. A high win rate above 85 ¢ is a losing system that
feels like a winning one — the single most important thing this bot must not do.

## 3. What it does

Once per day, at KMIA's commit hour **h14 (02:00 Manila)**:

1. Read the live market for KMIA HIGH.
2. Take the model's modal bucket and its calibrated probability (`band_calib`).
3. Ask `gating.decision_gates(...)` for a verdict. Any refusal → stop.
4. Apply the bot's own rails (§4). Any failure → stop.
5. Place **at most one** buy order, **at most $1**, at a limit no worse than the
   ceiling.
6. Record what it did — or would have done — in `signals`/`app_bets`.

It does **not** manage the exit. The 99 ¢ sell limit stays Julian's, placed
manually. Adding an exit leg is a separate decision with its own gate.

## 4. Rails, all of which must pass

| rail | value | why |
|---|---|---|
| market | KMIA HIGH only | the only market with an admissible floor |
| hour | h14 station-local | where every number above was measured |
| **price ceiling** | **≤ 0.80** | §2 — above this, 84.2 % loses money |
| max spend | **$1.00/day** | Julian's cap |
| max orders | **1 per calendar day** | idempotent under a 5-minute wake loop |
| confidence | `confidence_floor_ok` ≥ 0.70 | measured, on the calibrated number |
| gates | `decision_gates(...).ok` | freshness, band, NBS — fails closed |
| spread | existing `max_spread` + crossing-cost rail | already shipped |
| **dry-run** | **default TRUE** | see §5 |

Every rail fails CLOSED. A missing price, an exception, an unreachable API, an
unparseable response — all mean "do not order".

## 5. Dry-run is the default, and arming is a deliberate act

`ARMED` is false unless an environment variable is explicitly set to a specific
literal. Absent, malformed, or unset ⇒ dry-run. The bot then does everything
except the final POST, and writes the order it *would* have sent.

**Arming criterion, agreed up front:** ~2 weeks of dry-run at h14, then check
(a) does the entry price it picks actually sit at or below 0.80, and (b) does
live accuracy at h14 track the 84.2 % it was built on. Both, or it stays dry.

Rationale: h14 is one day old in production, and the profitability case rests on
**four** observed entry prices against thousands of accuracy days. The dry-run
closes precisely that gap and costs only time.

## 6. Credentials — the boundary

Kalshi's trading API requires an API key ID plus an RSA private key, with each
request signed (RSA-PSS over `timestamp + method + path`).

- Credentials come from the environment. **Never** committed, never logged, never
  echoed in an error, never placed in a config file that is read into the repo.
- Claude writes this code. **Claude does not run the live path, does not handle
  or enter the credentials, and does not place trades.** Julian installs the key
  and arms the bot himself.
- `test_freeze_contracts.py` currently asserts NO trading capability exists in
  `src/`. That test must be **narrowed, not deleted** — it should keep banning
  trading code everywhere except the one new module, so the freeze still protects
  the rest of the codebase.

## 7. Components

| file | responsibility |
|---|---|
| `src/highcast/kalshi_client.py` (new) | auth, request signing, portfolio read, order placement. Dry-run aware. Knows nothing about weather. |
| `src/highcast/orderbot.py` (new) | the decision: should we order today, which bucket, what price, what size. Knows nothing about HTTP. |
| `cli.py` | one new subcommand to run the bot once; the wake cycle calls it at h14 |

The split matters: the client is testable without a model, the bot is testable
without a network. Neither is testable if they are one file.

## 8. Testing

- Every rail gets a refusal test: over-ceiling price, second order same day,
  gate refusal, below floor, missing price, API exception.
- **A test that dry-run places nothing** — asserting no POST is attempted.
- **A test that arming requires the exact literal**, so a typo cannot arm it.
- Signing tested against a known vector, so a silent auth change is caught.
- The freeze test narrowed to exempt only `kalshi_client.py`.

## 9. Honest limits

- Accuracy is measured on 2,122 walk-forward days; **entry price on four**. The
  ceiling is arithmetic, not measurement.
- h14 has one live day behind it.
- 84.2 % is one station-month (n=155). The annual figure is 79.3 %.
- 0 of 12 buckets that reached 99 ¢ later lost — so the exit is currently costing
  ~2 ¢/contract for capital efficiency, not saving losses. That is fine, but it
  is not the protection it may feel like.
- ~~Kalshi's settlement source for `KXHIGHMIA` is assumed to be the NWS CLI~~
  **CONFIRMED 2026-08-01.** `KXHIGHMIA` settles on the **NWS Climatological
  Report (Daily) for Miami International Airport** — the same source
  `cli_reports` stores and every accuracy number here is measured against. The
  day runs midnight–23:59 **Local Standard Time**, i.e. 01:00–00:59 local during
  DST, which is exactly this project's KMIA climate day (05:00Z→05:00Z).

  Two consequences:
  1. The measurement basis is correct; nothing moves.
  2. The rules specify the **FINAL** CLI. That is precisely what
     `backfill.cli_row_is_final()` now enforces after a preliminary report
     mis-graded a bet on 2026-07-30 — so that fix was directly load-bearing for
     settlement correctness, not merely for scorekeeping.

  Sources: help.kalshi.com/markets/popular-markets/weather-markets;
  kalshi.com/markets/kxhighmia.

---

## 10. Amendment — 2026-08-01, late: two ways this would have shipped inert

Both found by asking the question none of the safety work asks. Every guard in
§4–6 proves an order is **not** sent; nothing proved one **can** be. So the
first evidence either way would have arrived at the commit hour on the one day
of the fortnight the bot finally wanted to trade — and both failures would have
been logged in a shape indistinguishable from a gate refusal.

### 10.1 The ticker was constructed, and the construction was wrong

`_ticker()` built `f"{series}-{target}-{bucket}"`:

| | value |
|---|---|
| what it built | `KXHIGHMIA-2026-08-01-94–95` |
| what Kalshi uses | `KXHIGHMIA-26AUG01-B94.5` |

Two independent defects — the date format, and an **en dash** carried in from
the bucket label, a character no exchange ticker has ever contained. Every order
rejected, 100 % of the time.

The old docstring said the suffix *"must be confirmed against a live event
listing rather than guessed"*, and then guessed. The listing was already being
fetched every cycle: `parse_markets` simply dropped the `ticker` field. It now
keeps it, `botrunner` carries it through, and `_ticker()` reads it instead of
building one. When it is absent the bot **refuses** — a refusal we can name
beats a rejection we would have to diagnose at 02:00 Manila.

Verified against the live ladder, not a fixture:

```
  ≤87    KXHIGHMIA-26AUG01-T88      92–93  KXHIGHMIA-26AUG01-B92.5
  88–89  KXHIGHMIA-26AUG01-B88.5    94–95  KXHIGHMIA-26AUG01-B94.5
  90–91  KXHIGHMIA-26AUG01-B90.5    ≥96    KXHIGHMIA-26AUG01-T95
```

The test fixture had **omitted the ticker entirely**, so no test could have
caught this: the fixture taught the code that ladders have no tickers. Same
class as the two fixtures already corrected during this build — a fixture
missing a field production always supplies is a test passing for the wrong
reason.

### 10.2 Arming would have been a no-op

`place_order` answered `"armed but no transport configured"` whenever nobody
injected a transport — and production injects nothing. So at the end of the
two-week dry-run, setting the arming literal would have produced a log saying
**ARMED** and an exchange seeing nothing. This project's least favourite failure
mode: configured-looking and inert, the same shape as the launchd env-var trap
§6 was written to avoid.

Fixed by `default_transport()` — lazily constructed, and only ever reached after
an `armed` check, so dry-run still builds no HTTP client at all (asserted).

### 10.3 A safety claim that expired

§5 and the module docstring claimed dry-run was *physically* incapable of
ordering, resting on two accidents: `cryptography` absent, and no transport.
Both are now false — the signing library went in so `creds-check` could verify a
real key, and 10.2 removed the other. **The claim is retracted rather than
quietly left standing.** What actually stops an order now:

1. `is_armed` — one exact literal, tested against every truthy near-miss.
2. `orderbot` caps — KMIA HIGH only, one/day, ≤ $1, ceiling 0.80, confidence floor.
3. The commit-hour check — inert 23 hours a day.

Weaker than "impossible", and strong enough. A decayed guarantee left in a
docstring is worse than one never claimed.

### 10.4 `order-test` — the missing direction of evidence

`highcast order-test [--demo]` places 1 contract at 1 ¢, reads it back, and
cancels it. Worst case in the whole procedure is one cent, and the cancel runs
**even if the read-back fails** — an abandoned resting order is the only
genuinely bad outcome available.

- `--demo` → `demo.kalshi.co`, fake money, a **separate account** with its own
  key in `~/.highcast/credentials-demo`, read with `use_env=False` so an
  exported production key id cannot leak into a demo run.
- Production requires the **same arming literal** as the bot. One way to enable
  real orders, not two — and the manual test rehearses the exact ritual.

### 10.5 `--at-market` — the one thing 1 ¢ cannot prove

Julian asked for a real $1 order rather than the demo sandbox, so `order-test`
gained a second mode. The distinction is not cosmetic:

| | `order-test` | `order-test --at-market` |
|---|---|---|
| price | 1 ¢, **cannot fill** | current ask, **meant to fill** |
| cost | nothing | up to **$1.00**, hard-capped |
| cancelled | always, even if read-back fails | no — a fill cannot be cancelled |
| result | proof the mechanics work | **a real position** |

The 1 ¢ round trip proves every mechanical step: auth, signing, ticker format,
order body, order id, read-back, cancel. `--at-market` proves the only thing it
cannot — that a fill happens — and the price of that proof is a position.

Three rails, all tested:

- **The dollar cap lives in `kalshi_client`, not the CLI.** `--dollars 50` buys
  $1.00. A number that guards money belongs next to the code that spends it, not
  where a flag can move it.
- **Size gives way to the budget; price never does.** A cheaper price is a
  different order that may not fill, which is the thing being tested.
- **The gate verdict is printed with the pick, including when it refuses.**
  Placing manually against a refusal is a legitimate choice; doing it without
  being told is not. The order is also not written to `bot_orders` — a manual
  trade must not contaminate the dry-run evidence the arming decision reads.
