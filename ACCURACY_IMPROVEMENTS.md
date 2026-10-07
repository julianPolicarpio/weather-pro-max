# Accuracy Plan — highcast (KLAX HIGH/LOW)

Rewritten 2026-07-25 around one goal: **consistent, well-calibrated accuracy at
the specific hours Julian can actually act.** Not best-average MAE, not maximum
edge — accuracy that holds up at a fixed decision time, day after day.

Read with `HANDOFF.md` and `README.md`. Every item below is a hypothesis with a
falsification test. Nothing is asserted to work.

---

## 0ω. THE ACCEPTANCE BAR — and the bias that nearly got past it

**Ground truth is the settled NWS CLI value. The market is a benchmark, never an
oracle.** Every model change must clear: out-of-sample skill against settled CLI
values, paired bootstrap, plus the `+RANDOM` noise control (must come back
non-significant). Nothing else counts.

**Explicitly disallowed as evidence:** "the model now agrees with the market",
"it agrees with the guidance ensemble", "the distribution looks more sensible",
"the spurious edge went away". A process that accepts changes moving *toward* the
market and rejects those moving *away* converges on a market-follower with **zero
edge by construction**, while appearing validated at every step. §11 found no
edge; adopting the market as a correctness oracle guarantees there never will be.

- Disagreeing with the market is **not an error** — it is the only place edge can
  live. It is a reason to *check*, never a reason to *change*.
- A 40–70 point edge against a liquid book is a **magnitude prior** that something
  is broken, and a trigger to investigate. It is not a verdict, and resolving it
  requires a truth-based measurement.
- The favoured bucket is a **mode, not a certainty**: 65% is wrong ~35% of the
  time. Single outcomes validate nothing in either direction — only calibration
  over many days does.
- A Kalshi price is **not a probability** (vig + spread; the six buckets routinely
  sum to >1.00). De-vig before treating it as the market's belief, as
  `signals.edge_report` does. Ad-hoc price comparisons do not.

This section exists because on 2026-07-26 the write-up of a genuinely good fix
(matched-hour training, justified against settled CLI values) cited "the model now
agrees with the market's favoured bucket" as corroboration. The fix stood; the
published reasoning was contaminated. Julian caught it. See
`HANDOFF-2026-07-26.md` § BIAS DISCIPLINE.

---

## 0α. AUDIT 2026-07-26 — the LOW's ratchet was reading the wrong channel

*This section supersedes nothing; it adds the largest measured LOW win found so
far, plus two closed negative results. All numbers below are measured on
`data/highcast.db`, no network.*

### ⭐ THE FIND: the ASOS broadcasts a 6-hour MINIMUM and the parser threw it away

`fetchers/metar.py` parses the 6-hour **maximum** remark group (`1sTTT`) into
`obs.six_max_c`, and `latentmax.visible_running_extremum` folds it into the
HIGH's ratchet. **The mirrored 6-hour minimum group (`2sTTT`) was never
parsed.** So the HIGH's ratchet saw continuously-sampled data while the LOW's
saw only hourly spot readings — and the LOW is the metric whose entire model is
essentially the ratchet.

The data was already in the database: **14.1% of the 57,570 stored raw METARs
carry a `2sTTT` group, and all 2,029 climate days have at least one usable one.**

Rounded ratchet vs the settled CLI low, 2,029 climate days:

| ratchet source | exact | MAE | bias |
|---|---|---|---|
| hourly spot min (what shipped) | 55.5% | 0.568 | **−0.536** |
| spot + 6-hr-min groups | **85.4%** | **0.244** | **−0.041** |

Jun–Aug: 74.0% → 88.7%. 2026 only: 55.1% → **88.8%**, bias −0.576 → **0.000**.
Day-by-day on the last 9 settled days: **9/9 exact**. For scale, the whole LOW
nowcast scores MAE 0.765 at its decision hour — the *measurement error in its
own bound* was 0.568 of that, and it was a one-sided warm bias.

**CORRECTNESS RULE (do not regress).** A 6-hourly group in the `HH:53Z` report
belongs to synoptic hour `HH+1` and covers `(synoptic_end − 6h, synoptic_end]`.
The KLAX climate day is 08:00Z→08:00Z, so:

| report | window | usable? |
|---|---|---|
| 18Z | 12–18Z | ✅ wholly inside |
| 00Z | 18–00Z | ✅ wholly inside |
| 06Z | 00–06Z | ✅ wholly inside |
| **12Z** | **06–12Z** | ❌ **straddles the 08:00Z start — must be ignored** |

Using the straddling group imports a temperature from the previous climate day.
**The same guard was missing on `six_max_c`** — the HIGH's running max has been
folding in the 12Z report's 06–12Z window all along. Same bug, opposite sign;
both are now guarded.

**Operational limit, stated plainly:** at the LOW's decision hour (06:00 PDT =
13:00Z) the only 6-hourly group yet published is the 12Z one — the straddling,
unusable one. So this fix does **not** sharpen the bound *at* the decision hour;
it sharpens (a) every training row at hours ≥ 18Z, (b) the residual target the
nowcast regresses, and (c) settlement/verification certainty from 11:00 PDT on.

### SHIPPED AND VERIFIED (2026-07-26)

Migration `ensure_six_min_column` + `highcast reparse-obs`, which re-derives the
derived columns from the raw text already stored — **no network**. Run against
`data/highcast.db`: scanned 57,571 reports, **updated 8,096**; a second run
reported **0 changed** (idempotent). `tmpf`/`tmpf_precise` unchanged.

Effect on the ratchet's own residual, `cli_low − visible_min` over 2,029 days:

| | 0 °F | −1 °F | −2 °F | tail |
|---|---|---|---|---|
| before | 52.4% | 36.6% | 7.9% | to −7 |
| **after** | **86.1%** | **12.4%** | **0.7%** | **gone** |

**HIGH regression check (the `six_max_c` guard is a behaviour change):**
`cli_high − visible_max` is **99.31% exact** with **0.00% mass in the impossible
direction** (`cli_high < visible_max`) — the guard did not break the HIGH.

Live confirmation: the nowcast's 2026-07-25 LOW distribution sharpened from
70 °F @ 63.3% to **70 °F @ 67.5%**, and the settled CLI low was **70**.
Suite: **269 passed, 0 failures** (was 252).

### ⚠️ MEASURED: this does NOT improve the LOW model at its decision hour

Re-ran `verify-hour --metric low --hour 6` after the fix. The result is
**byte-identical** to the pre-fix baseline — MAE 0.753, CRPS 0.517, cov90 0.79,
and the same bootstrap CIs.

That is not a bug, and the reason matters more than the number. Re-measured
through the shipped `visible_running_extremum` (N=2,030 climate days):

| scope | spot-only | shipped | bound moved | fixed | broke |
|---|---|---|---|---|---|
| all | 55.6% | **86.5%** | 674 (33.2%) | 640 | 13 |
| Jun–Aug | 74.1% | 89.1% | 87 (17.0%) | 82 | 5 |
| 2026 | 55.3% | 88.8% | 69 (33.5%) | 69 | 0 |

So the bound really is far better — **over the whole climate day.** But the
6-hourly groups arrive at 18Z/00Z/06Z, and at the LOW's decision hour (13:00Z)
the only one published is the 12Z straddler the guard correctly rejects. Spot
check at training hour 12 local (19:00Z, where only the 18Z group is live): the
bound moved on **1 of 25** July days. Hence no measurable model change.

**Where this fix actually pays:**
1. **Settlement and verification certainty from 11:00 PDT on** — after the 18Z
   report the day's minimum is known essentially exactly. `settle_signals`,
   `paper_report`, and the history panel all get a correct truth value sooner.
2. **`hidden_spike_pmf` is now accurate** — it is a dashboard diagnostic *and* a
   sanity floor for the intraday model, and it was reading a 36.6% one-degree
   undercut that was mostly measurement error, not weather (now 12.4%).
3. **Any future decision-hour-indexed model (§3) verified at hours ≥ 12 local**
   inherits a nearly-exact target.

**What it does NOT do:** improve the 06:00 PDT LOW recommendation. Do not claim
it does. The genuine remaining LOW error at that hour is still the undercut, and
it is still bounded by §1b's information curve.

A worked example of a real save — 2026-07-23: every spot reading bottomed at
69.08 °F (→ 69), the 18Z group read 20.0 °C = **68.0 °F**, and the settled CLI
low was **68**. Conversely 2026-07-01's minimum fell late in the climate day and
the spot readings already had it (62.60 → 63 = CLI); the groups sat higher and
were correctly ignored.

### ❌ CLOSED: IEM 1-minute ASOS is not a usable sub-hourly source

`latentmax.fetch_asos1min` exists and works, so the obvious idea was to ingest
it live. **Measured and rejected:**
- The archive **lags ~3 days** (on 2026-07-25 it ended 2026-07-22T23:13Z) — it
  cannot serve a nowcast at all.
- Coverage is gappy (30–80% of minutes), and the gaps hide the extremum: over
  48 usable days the full-day 1-min minimum reproduced the CLI low **64.6%** of
  the time versus **85.4%** for the hourly METAR minimum. On 2026-07-15 it read
  71 °F against a settled low of 68.

It is *worse* than what we already had. The `2sTTT` group is the correct
sub-hourly channel; it is authoritative, free, already stored, and complete.

### ⚠️ The LOW nowcast is asked to predict at an hour it was never trained on

`models.intraday.build_checkpoint_frame` defaults to `hours=(8,10,12,14,16,18,20)`
(station-local wall clock). `cli.cmd_predict --live` trains on that default and
then predicts with `hours=(now.hour,)`. **The LOW's decision hour is 06:00 PDT —
outside the trained range**, so LightGBM clamps `hour` to its leftmost split
(≈8) and applies the *post-dawn, already-locked* residual at a moment when the
minimum may still be forming. The published "LOW intraday @6am MAE 0.765" was
measured with train and test hours matched, which production does not do.

`highcast verify-hour` (added this session) exists to measure exactly this.

### The TOMORROW card runs an untrained regime — now with a visible symptom

Confirmed live for 2026-07-26: the HIGH's evening-before day-ahead has
`morn_tmpf/dwpf/onshore/sky/dtd/rate` **all NaN** and returns p10 66.0 / p90
81.3 — a **15 °F** band, versus ~4 °F for the LOW's. The LOW's tomorrow card is
not NaN (its `_morning_features` uses a 24 h lookback), but its cutoff is
resolved into the *future* when the card is generated before 20:00 PDT, so it
silently substitutes the latest available observation for the intended one.

---

## 0a. VERIFIED FINDINGS (audit 2026-07-25) — read before planning anything

Everything in this section was **measured**, not argued. It changes the priority
order below.

### Overfitting: the model is NOT overfitted — tested with a noise control
Paired bootstrap over **1,664 walk-forward days**, baseline vs `+morning`
(shipped) vs `+RANDOM` (two pure-noise features injected as a control):

| variant | ΔMAE vs baseline | 95% CI | verdict |
|---|---|---|---|
| `+morning` (shipped) | **−0.0515** | [−0.0724, −0.0314] | **significant** |
| `+RANDOM` (control) | −0.0017 | [−0.0144, +0.0114] | not significant |

Seam (N=76): baseline CRPS 1.686 → `+morning` **1.634** → `+RANDOM` 1.760.
**The noise control is the discriminator.** Search-driven overfitting would have
made random features "win" too; they didn't. The morning-feature gain is real —
and small (0.05 °F). Any future feature claim must clear this same bar.

### The backtest is internally valid but measures a DIFFERENT SYSTEM than production
No leakage (obs strictly ≤ cutoff, verified). But feature availability across the
backtest window (2022-01-01 → 2026-07-24, N=1,664):

| feature | non-null | | feature | non-null |
|---|---|---|---|---|
| `fc_mos_mex` | 100% | | `fc_nbm` | 39.2% |
| `fc_gfs` | 98.8% | | `fc_aifs` | 31.3% |
| `fc_hrrr` | 55.0% | | `fc_nam` | 12.4% |
| `fc_icon` | 54.9% | | **`fc_nws`** | **0.4%** |
| `fc_ecmwf` | 54.1% | | | |

- **Era split:** 2022-01→2024-01 (N=761) averages **36%** `fc_*` availability;
  2024-02→now (N=903) averages **78%**. The headline MAE blends two different
  systems, so **the live model is probably better than the backtest reports.**
- **`fc_nws` is dead weight** at 0.4%, and `backtest.summarize`'s `mae_nws`
  benchmark (prints "4.0") is computed on ~7 days — **it is noise reported as a
  benchmark and must not be cited.**

### Clock drift is real (§0 claim confirmed by measurement)
`_cutoff_utc(cfg, d, 9)` resolves to **LST 08:00 in summer (PDT)** and **LST
09:00 in winter (PST)** — the cutoff *moves an hour with DST*. Conservative, not
leakage, but the model's information state is inconsistent across seasons.

### Train/serve gap is narrower than feared, but real at the edges
At the **scheduled** issue time (≥09:00 PDT) `morn_rate` is NaN on **0/60** days —
production is fine. It *is* NaN for (a) manual/early `predict` runs and (b) the
**evening-before TOMORROW card**, where all `morn_*` are NaN by construction —
a regime the backtest never measured.

### §5a inversion was TESTED and was a WASH — but only in its weak form
IGRA v2.2 NKX (San Diego, USM00072293) parsed to **1,923 daily 12Z 850 hPa temps
(2021→2026, 95% coverage)**; sanity-matched against U. Wyoming soundings.
Ablation on top of the shipped model:

| variant | seam MAE | seam CRPS | overall MAE | overall CRPS |
|---|---|---|---|---|
| baseline (shipped) | 2.16 | 1.613 | 1.58 | 1.140 |
| `+t850` | 2.13 | **1.612** | 1.58 | 1.142 |

**Wash.** Interpretation: `morn_rate`/`morn_dtd` already encode the *local* marine
state (the effect); 850 hPa temp alone is the *regional cause* — redundant.
**Caveat that keeps §5a alive:** only **850 hPa temperature** was tested, NOT
**inversion base height** or **inversion strength (ΔT across the cap)**, which are
the physically meaningful quantities §5a actually proposes. The weak proxy failed;
the strong form is still untested.

Also note: Open-Meteo/ERA5 serve **no** 850 hPa archive for KLAX (confirmed null
across all model variants) — IGRA is the only historical route. A live-only
`forecast_aloft` capture pipeline is shipped and accumulating.

### NEW: Kalshi historical market data is available, unauthenticated
This unblocks measuring **edge** instead of only accuracy.

```
GET /trade-api/v2/historical/cutoff                       -> rolling-window boundary
GET /trade-api/v2/historical/markets?event_ticker=KXHIGHLAX-{YYMMMDD}
                                                          -> 6 buckets + result=yes/no (settlement)
GET /trade-api/v2/historical/markets?series_ticker=KXHIGHLAX&limit=N
GET /trade-api/v2/series/KXHIGHLAX/markets/{ticker}/candlesticks
        ?start_ts&end_ts&period_interval                  -> hourly OHLC + yes_bid/yes_ask
```

- **Depth: back to ~2025-03** (`25MAR01` returns data, `24DEC01` empty) ⇒ **~500
  settled KLAX days with prices.**
- Prices are `*_dollars` **strings** (`price.close_dollars`, `yes_bid`, `yes_ask`,
  `volume_fp`). `fetchers/kalshi.py` already reads the current field names.
- **Post-settlement summaries show a stale book** (`last=0.01, bid=0, ask=1`) —
  a usable price must come from **candlesticks at the decision hour**, not the
  market summary.

**Consequence:** "does the model beat the market?" is now a *backtestable*
question over ~500 days, not something to wait weeks for. See §11.

---

## 0c. THE DUAL-METRIC ROADMAP — accuracy plan for HIGH and LOW

*Written 2026-07-26 from measured evidence, replacing the implicit assumption
that one plan serves both markets. It does not: they are different problems and
they need different work.*

### The structural split (measured, not asserted)

| | **HIGH** (decision ~9am PDT) | **LOW** (decision ~6am PDT / 5am LST) |
|---|---|---|
| has the extremum happened? | **No** — max forms 11am–2pm | **Largely yes** — min is mostly behind us |
| best model at decision hour | **day-ahead** MAE 1.509 / CRPS 1.080 | **nowcast** MAE **0.806** / CRPS **0.561** |
| the losing model | nowcast 1.620 / 1.178 | day-ahead 1.469 / 1.058 |
| nature of the problem | **forecasting** (burn-off timing) | **measurement + small residual** |
| naive benchmark | — | "low = running min" already gives MAE **1.071** |
| dominant error | seam 80–84 warm bias; hot tail | the *undercut* after the decision hour |

**The LOW is already the better-predicted market by a wide margin** (0.806 vs
1.509 MAE). That inverts the intuition the project started with, and it matches
Julian's timezone advantage — the LOW forms during his evening.

### What is left to gain, per metric

**LOW — one clear idea, then diminishing returns.**
The nowcast is strong because the ratchet bound is nearly the answer. The
remaining error is the *undercut*: how far below the observed running minimum the
settled CLI value lands. Measured at LST 05:00 over 2,029 days: **0 °F 36% ·
1 °F 38% · 2 °F 14% · 3 °F 6%**, median 1 °F. That is a small, tightly
distributed, non-negative integer — a far easier target than temperature.
→ **L1 (below).** Beyond that, the LOW's residual ~16% late-evening undercut is
largely irreducible by waiting (§1b), so expect small gains.

**HIGH — the harder problem, and the physics is still missing.**
Everything tried so far either failed or was redundant with the morning
observations (calibration ✗, 850 hPa temp ✗, ML-anchoring ✗). The one class of
information never tested is **offshore/compression flow**, which is what produces
the hot days the model still cannot reach. → **H1 (below).**

### Ranked queue (each item states its gate up front)

| # | metric | item | why it is next | gate |
|---|---|---|---|---|
| **L1** | LOW | **Residual/undercut target for the nowcast** — regress the remaining move from the ratchet bound instead of the temperature (`IntradayModel(residual=True)`, implemented) | attacks the only remaining LOW error, on the model that *is* the LOW recommendation | beat **MAE 0.806 / CRPS 0.561** at the decision hour, paired bootstrap, N≈1300 |
| **H1** | HIGH | **§5b Santa-Ana / offshore pressure gradient** (LAX−KDAG SLP from IEM, decades free) | the only untested *physical* signal; aimed at the hot tail the GBM still misses even with the §7 hybrid | 84+ bias toward 0 **and** no CRPS regression <80; must clear a random-noise control |
| **H2** | HIGH | **§3 decision-hour-indexed models** | makes the model correct *by construction* at the hour it is used; subsumes cutoff tweaks | per-hour CRPS curve monotone in information; beat the fixed-cutoff model at 9am |
| **B1** | both | ~~Daily candlestick capture~~ — **already satisfied, verified 2026-07-26** | `signals.market_json` already stores all 6 buckets with `yes_bid`/`yes_ask` on *every wake* (~142 snapshots/day, both metrics). No new pipeline needed. | n/a — only *pre-logging* days are unrecoverable |
| **L2** | LOW | Extend cutoff 21:00 → 22:00 Manila | §1b: buys ~4 pp of "min already locked" | measure, don't assume |

### Reading the §11 result correctly

§11 found **no market edge**, and that does **not** make this roadmap pointless —
it re-scopes it. Accuracy and edge are different quantities:

- Accuracy is **necessary but not sufficient**. The market prices the same public
  forecasts, so being accurate only pays where the model is accurate *and the
  market is not*.
- Therefore the queue above is ordered by **where the model is worst relative to
  its own achievable floor** (hot tail, undercut) — those are the places a gap
  versus the market is even possible.
- And **B1 is non-negotiable**: without accumulating price history, the edge
  question can never be answered with a tighter CI than the ±0.09 we have now.

---

## 0b. Hygiene items (small, mechanical, do alongside anything)

Each is cheap and independently verifiable; none is a research question.

1. **Drop `fc_nws` from `FEATURES`.** 0.4% non-null over the backtest — it is a
   dead column that costs a split dimension at 899 effective rows. (Keep
   *fetching* NWS; it is only the model feature that is useless.)
2. **Delete or fix `mae_nws` in `backtest.summarize`.** Computed over ~7 days, it
   is printed next to real metrics and invites false comparison. Either compute it
   only over days where `fc_nws` exists **and print N**, or remove it.
3. **Make the cutoff LST-consistent.** Resolve the cutoff against the station's
   **standard** offset (`std_utc_offset`), not `America/Los_Angeles` wall time, so
   it stops drifting an hour across DST. This is a behaviour change to features —
   re-run the §0a bar (paired bootstrap + noise control) before shipping.
4. **Guard the TOMORROW regime.** The evening-before card runs the day-ahead model
   with every `morn_*` NaN. Either train a dedicated no-morning variant or label
   the card explicitly as a different (wider) product.

---

## 0. Decision times (the design constraint everything follows from)

| Market | Decision window (Manila) | = LA local (PDT) | = LST (climate-day clock) |
|---|---|---|---|
| **HIGH** | up to **01:00** | up to **10:00** | up to **09:00** |
| **LOW** | evening → **01:00** | 03:00 → **10:00** | 02:00 → **09:00** |

**Clock discipline — a live source of confusion in this codebase.** The climate
day is midnight-to-midnight **LST (UTC−8)**, but `features._cutoff_utc` resolves
its cutoff in **`America/Los_Angeles` local time**, which is PDT (UTC−7) for most
of the year. So the HIGH's configured `cutoff_local_hour = 9` is **LST 08:00**,
not LST 09:00. Every hour stated in this document is labelled LST or PDT
explicitly. Any new code must do the same.

**Current mismatch:** the HIGH model cuts features at PDT 09:00 but the decision
is at PDT 10:00. The LOW model cuts features at PDT 20:00 *the prior evening* but
the decision is 7–14 hours later. **Neither model is trained for the moment it is
used.** That is the central problem this plan addresses.

---

## 1. Measured: what is actually knowable at each hour

Computed from `data/highcast.db` — 2,031 climate days with ≥20 METAR
observations, binned by hour of the **daily extremum in LST**.

### 1a. When the daily MINIMUM occurs

```
LST 00  ##########################################   302  14.9%   <- first hour of climate day
LST 01  ########################                     176   8.7%
LST 02  ####################################         262  12.9%
LST 03  ############################################ 322  15.9%
LST 04  ############################################ 358  17.6%
LST 05  ########################                     177   8.7%
LST 06  ###########                                   84   4.1%
LST 07  #                                             14   0.7%
LST 08                                                 5   0.2%
...
LST 21  ######                                        50   2.5%
LST 22  #########                                     70   3.4%
LST 23  ###############                              114   5.6%   <- last hour of climate day
```

**The minimum is not a dawn event.** Only 31.4% fall in the LST 04:00–08:59
"dawn" window. The mass sits across LST 00:00–05:00, and **11.5% of days set
their minimum in the final three hours (LST 21:00–23:59)** — a late-evening
marine push or frontal drop that undercuts the overnight low.

### 1b. The LOW information curve — the operationally important result

Cumulative probability that the minimum has **already occurred** by a given
decision time:

| Manila | PDT | LST | Min already locked |
|---|---|---|---|
| 18:00 | 03:00 | 02:00 | 36.5% |
| 19:00 | 04:00 | 03:00 | 52.4% |
| 20:00 | 05:00 | 04:00 | 70.0% |
| **21:00** | **06:00** | **05:00** | **78.7%**  ← current cutoff |
| 22:00 | 07:00 | 06:00 | 82.8% |
| 23:00 | 08:00 | 07:00 | 83.5% |
| 00:00 | 09:00 | 08:00 | 83.7% |
| 01:00 | 10:00 | 09:00 | 83.8% |

Two conclusions, both actionable:

1. **Every hour from Manila 18:00→21:00 is expensive to give up** (36% → 79%).
   Deciding early is costly in a way the current backtest never shows.
2. **The curve is flat after Manila 22:00.** Pushing the LOW cutoff from 21:00 to
   01:00 buys ~5 percentage points. **Your current cutoff is already near
   optimal — do not extend it.** Manila 21:00–22:00 is the right window.

The residual ~16% is *irreducible by waiting*. It is dominated by the
late-evening undercut, and it must be modelled rather than waited out (§4).

### 1c. When the daily MAXIMUM occurs

```
LST 08  ####                                          49   2.4%
LST 09  ####################                         213  10.5%
LST 10  ########################################     423  20.8%
LST 11  ##################################################  525  25.8%
LST 12  ###########################################  454  22.4%
LST 13  ######################                       231  11.4%
LST 14  #######                                       76   3.7%
```

At the HIGH decision time (**LST 09:00**), the maximum has already occurred on
only **13.9%** of days. **On ~86% of days the high is still genuinely in the
future**, and it forms in the LST 10:00–13:00 window — after the decision.

**This is the key asymmetry between the two markets:**

- The **LOW is mostly an observation problem.** ~79–84% of the answer is already
  in the METAR record at decision time. Accuracy comes from correctly handling
  what remains.
- The **HIGH is a genuine forecast problem.** Waiting does not solve it — the
  peak is hours away. Accuracy comes from *physics*, specifically burn-off
  timing.

These require different architectures. Treating both with one generic quantile
regressor is the structural error in the current design.

---

## 2. Retracted — band-aids from the previous draft

Removed deliberately. They tune the machine instead of fixing what it lacks, and
their expected effect is within noise of a 1.53 MAE baseline:

- **Hyperparameter tuning** (`min_child_samples`, `num_leaves`, half-life).
  Symptom management. The tail starves because the *target parameterization* is
  wrong (§6), not because the leaf floor is 20.
- **`year_index` ablation / `fc_spread` → std.** Correct hygiene, real
  non-stationarity, but not solutions. Fold into whatever change ships next; do
  not schedule them as work.
- **LightGBM → XGBoost.** No evidence to justify it at 899 effective rows.
  Answered separately; not a lever.
- **Extending the HIGH cutoff to LST 09:30–10:00.** Superseded by §3, which
  handles decision-time properly instead of hard-coding a better guess. The
  measured gain was ~10% of days' maxima plus one hour of burn-off signal — real,
  but it falls out of §3 for free.

---

## 3. Real solution 1 — model the decision hour, don't hard-code it

**Problem.** One model is trained at one cutoff (`metric.cutoff_local_hour`) and
used at another. The intraday nowcast partially covers the gap but is a
*different model with different features*, and the headline backtest reports
day-ahead numbers that describe **no moment Julian ever trades at**.

**Fix.** Make issuance time an explicit dimension of the dataset and the
verification.

- For each historical climate day, build feature rows at **every candidate
  decision hour** (LOW: LST 02:00–09:00; HIGH: LST 04:00–09:00), each respecting
  a strict `fetched_at <= cutoff` leakage bound.
- Train a **model family indexed by issuance hour** — either separate models per
  hour, or one model with hours pooled and `hours_elapsed` / `hours_remaining` as
  features plus **per-hour calibration**. Pooling is preferable at 899 effective
  rows; the hour-specific calibrator is the part that must not be shared.
- Replace the single headline metric with an **accuracy-vs-decision-hour curve**,
  reported per metric. This becomes the primary verification artifact.

**Why this is the real fix and not a band-aid:** it makes the model correct *by
construction* at whatever hour is chosen, turns "when should I decide?" into a
measured question rather than a guess, and subsumes every cutoff-tweaking
proposal. It also directly answers the stated goal — consistent accuracy at
specific timeframes.

**Acceptance gate.** Walk-forward, per decision hour, reporting N per cell:
- CRPS at LST 05:00 (LOW) and LST 09:00 (HIGH) must beat the current
  fixed-cutoff models at those same hours.
- The curve must be **monotone non-increasing** in information (later ≥ earlier
  in skill). A non-monotone curve is a leakage or specification bug — treat it as
  a failure, not a finding.

---

## 4. Real solution 2 — decompose the LOW into observed + undercut risk

**Problem.** At the LOW decision hour the running minimum is *known exactly*, and
the true minimum can only be **at or below** it. `models/intraday.py` ratchets
correctly on that constraint, but it does not quantify the one thing that
actually matters: **how likely is a further drop, and how big?**

The data says this is not a rare edge case — **11.5% of days set their minimum in
the last three hours of the climate day**, long after the decision.

**Fix.** Model the LOW at decision time as an explicit mixture:

```
P(Tmin = m_obs)            = 1 − p_undercut
P(Tmin = m_obs − δ)        = p_undercut · f(δ)
```

- `m_obs` — running minimum at the decision instant (exact, from METAR).
- `p_undercut` — a **classifier**: probability the remaining hours of the climate
  day go below `m_obs`. Conditioned on hours remaining, month/season, forecast
  evening temperatures, onshore/offshore flow, dewpoint, cloud state, and the
  margin between `m_obs` and the forecast evening minimum.
- `f(δ)` — conditional magnitude distribution given an undercut occurs.

**Why this is right.** On the ~84% of days with no undercut, the prediction
collapses to a near-point mass at a *measured* value — a level of accuracy no
quantile regressor over continuous temperature will ever reach. On the remaining
days it is honestly, explicitly uncertain. That is precisely "consistent
accuracy": sharp when the world is knowable, wide when it isn't, and never
confidently wrong.

It also matches the settlement object exactly — the CLI minimum is an integer,
and a point mass maps cleanly onto a 2°F bucket.

**Acceptance gate.**
- `p_undercut` must be **calibrated**: reliability diagram over deciles, on
  held-out days. A classifier that says 10% must undercut ~10% of the time.
- Bucket-level ranked probability score at LST 05:00 must beat the current
  intraday LOW.
- **Report the no-undercut subset separately.** If the model isn't
  near-deterministic on days that don't undercut, the decomposition isn't working.

---

## 5. Real solution 3 — give the HIGH the physics it is missing

**Problem.** At LST 09:00 the high is 1–4 hours in the future and hinges on
marine-layer burn-off. The model's only marine-layer information is *surface
proxies at one point* (`morn_sky`, `morn_dtd`, `morn_onshore`, `morn_rate`). It
has no measurement of the quantity that actually decides the outcome: **how deep
the marine layer is and how hard it is capped.**

This is why the 84+ band fails (bias −2.54, MAE 4.39) and why the 80–84 seam is
warm-biased. The model is guessing at a state variable it cannot see.

Three real state variables, all free, all with decades of history, all
backtestable now:

### 5a. Inversion structure — IGRA radiosonde  ⚠️ PARTIALLY TESTED (see §0a)
**Status: the weak form FAILED, the strong form is untested.** 850 hPa
*temperature* alone from IGRA NKX (1,923 days, 95% coverage) was a **wash** on top
of the shipped morning features — it is redundant with `morn_rate`/`morn_dtd`.
Do **not** re-test 850 hPa temperature. What remains genuinely untested is the
*structural* form: **inversion base height** (marine-layer depth) and **inversion
strength** (ΔT across the cap), which require parsing the full sounding profile,
not one pressure level. Those are different quantities and may still carry signal
the surface proxies cannot.

The archive route is confirmed: **IGRA v2.2** (NOAA NCEI), updated daily, plain
HTTP, no key — `access/data-por/USM00072293-data.txt.zip` (92 MB, decades).
Fixed-width: pressure at cols 10–15 (Pa), temp at 23–27 (°C×10), `-9999`/`-8888`
are sentinels; header line carries the nominal hour (use `12` = LST 04:00).

Relevant sounding: **NKX (San Diego/Miramar)**, launched **12Z = LST 04:00** —
comfortably before every decision hour. `VBG` (Vandenberg) as a second.

Derive per sounding: **inversion base height** (marine layer depth), **inversion
strength** (ΔT across the cap), temperature at inversion top, 850 hPa
temperature, and moist-layer depth.

Shallow layer + weak cap → burns off → 80s. Deep layer + strong cap → holds →
low 70s. This is the actual generating mechanism.

*Caveat:* NKX is ~120 mi south. Marine layer depth is regionally coherent along
the SoCal bight but the basin and San Diego decouple on some days. It is a proxy
— and it is the proxy operational forecasters use.

### 5b. Offshore pressure gradient — the Santa Ana index
The **LAX − Daggett (KDAG)** sea-level pressure gradient is the standard
operational offshore-flow index for the LA basin. Offshore gradient → downslope
compression → the 90–102°F days the model currently cannot reach.

Both stations report SLP/altimeter in ASOS. Free from IEM, decades of history,
same fetcher pattern as existing METAR ingestion. **This is the single most
direct predictor of the hot tail that is currently absent**, and it is cheap.

### 5c. Offshore sea-surface temperature
**NOAA OISST v2.1** — daily, free, decades, no key. SST anchors the temperature
of the marine air mass and therefore the floor of the cool mode. Slowly varying,
so it costs almost nothing in degrees of freedom, and it explains why "cool mode"
is 71°F in one week and 76°F in another.

**Acceptance gate (all three).** Feature-ablation backtest, added one group at a
time. Require: 80–84 band bias toward zero **and** 84+ bias improvement, with no
CRPS regression below 80. Report N per band — at N=15–40 a 0.3°F move is noise.

---

## 6. Real solution 4 — separate the two dynamical regimes on physical inputs

**Problem.** LAX summer highs are produced by two different physical systems:

| Regime | Mechanism | What predicts it |
|---|---|---|
| **Onshore / marine** | Marine layer depth + burn-off timing | Inversion base, SST, onshore flow |
| **Offshore / Santa Ana** | Downslope compression, no marine layer | Pressure gradient, 850 hPa temp |

Under offshore flow the marine-layer features are **meaningless**; under onshore
flow the compression features are. A single model must learn this interaction
from ~76 hot days. It will not.

**Fix.** Classify the physical regime from *inputs* (§5 variables), fit
regime-conditional models, and mix by P(regime).

**This is not the thing that already failed.** The 2026-07-25 regime-conditional
calibration conditioned on the **predicted temperature level** — an *output* —
and applied a monotone transform to a quantile set. That is circular, and a
monotone map cannot create probability mass where the base model never predicts.
Its failure was correctly diagnosed in `HANDOFF.md` as a warm-*median* problem.

Conditioning on **observed physical state** is a different and well-posed
problem. The project's own spec listed it as *"Method 3 — marine-layer-score
conditioning"* and deferred it as a non-goal. That was the right idea shelved for
the wrong reason: it was blocked on 850 hPa data that §5a now unblocks.

**Acceptance gate.** The regime classifier must be calibrated on held-out days
first — a mixture built on a miscalibrated gate is worse than no mixture. Only
then evaluate the mixture's bucket-level RPS against the single model.

---

## 7. Real solution 5 — fix the structural extrapolation failure  ✅ TESTED & SHIPPED (hybrid)

**RESULT (2026-07-25).** The diagnosis was right and the naive fix was wrong; the
hybrid is what shipped. Walk-forward, N=1,664, paired bootstrap vs the direct model:

| variant | overall ΔMAE (95% CI) | <80F (N=1552) | 84+ (N=37) | true hot ≥88F (N=20) |
|---|---|---|---|---|
| direct (was live) | — | **1.48** | 4.40 (bias −2.37) | **6.99** (bias **−6.99**) |
| pure residual | **+0.169** [+0.109, +0.228] ❌ | 1.69 | **3.07** (bias +0.22) | **3.66** |
| **hybrid (shipped)** | **−0.021** [−0.044, −0.0004] ✅ | **1.48** (unchanged) | 3.42 | **4.68** (bias −4.14) |

- **The mechanism is confirmed**: regressing `y − fc_anchor` eliminates the hot-tail
  collapse (bias −2.37 → +0.22 in the 84+ band).
- **The naive swap is a net loss**: it degrades the 93% of days below 80F
  (1.48 → 1.69), so overall MAE gets *significantly worse*. Do not ship it alone.
- **The hybrid keeps both**: blend weight `w = clip((direct_median − 82)/6, 0, 1)`,
  so below 82F the output is bit-identical to the old model and the anchor only
  takes over where a GBM provably cannot reach. Hot-day error falls **33%**.

**Shipped as** `models.dayahead.HybridDayAheadModel`, `features.build_features`
now emits `fc_anchor` (fixed coalesce `mos_mex → gfs → fc_mean`, 100% coverage),
config `model.hybrid_anchor: true` (set false to revert).

**Honest caveats.** The overall CI barely clears zero (upper bound −0.0004) — the
gain is *concentrated*, not broad, because only ~2–4% of days are affected. N=20
true-hot days is small. The 82/88 thresholds come from the earlier seam analysis
rather than cross-validation; they are physically motivated defaults, **not
optimised parameters**, and should be re-checked when more hot days accumulate.

**Still untested from this section:** the monotonic constraint on the anchor, and
the linear-quantile-regression residual baseline. Both remain worthwhile.

---

### 7a. Original rationale (retained)

Retained from the previous draft because it is architectural, not cosmetic.

`models/dayahead.py` fits LightGBM quantile regressors on **`y` directly**.
Gradient-boosted trees are piecewise-constant: **predictions are bounded by the
training targets in their leaves.** Days ≥88°F carry ~10 recency-weighted
rows-equivalent across the entire feature space, so no leaf can represent the hot
regime and hot days are dragged toward the dense 72–78°F mass. That is the
−2.54°F bias, mechanically.

**Fix.** Regress the **residual** `y − fc_anchor` and add the anchor back at
prediction time. The target becomes a small, near-stationary, densely-populated
correction; the *level* comes from the forecast, which reaches 100°F without
difficulty. This is the MOS formulation and the standard remedy.

**Anchor must be a fixed coalesce chain**, because model coverage is very uneven
(`mos_mex` 2021-01, `gfs` 2021-03, `ecmwf` 2024-02, `nbm` 2024-10, `aifs`
2025-02). Using `fc_mean` would let the anchor's *composition* drift over time —
a silent non-stationarity.

- HIGH: `mos_mex → gfs → fc_mean`
- LOW: `gfs → nbm → fc_mean`

Record the anchor used per row so the backtest can verify the switch isn't itself
creating a seam.

**Also test, in the same experiment:**
- **Monotonic constraint** on the anchor — a hotter forecast must never produce a
  cooler prediction. Physics you know to be true, so it costs nothing in bias and
  regularizes hard at 899 rows.
- **Linear quantile regression on the residual as a baseline.** Linear models
  extrapolate natively; trees do not. At 22 features and 899 effective rows the
  GBM has not clearly earned its nonlinearity, and this attacks the exact failure.

**Acceptance gate.** 84+ bias toward 0 and 84+ MAE below 4.39; <80 MAE (1.51) and
overall CRPS (1.10) must not regress beyond noise. Test the residual target and
the monotonic constraint **separately** before combining.

---

## 8. What "consistent accuracy" should be measured as

The current headline (MAE/CRPS, pooled over all days at one cutoff) does not
measure the stated goal. Replace it with:

1. **Per-decision-hour verification.** Every metric reported as a curve over
   issuance hour (§3), not a scalar.
2. **Bucket-level ranked probability score.** Settlement is an integer °F inside
   a 2°F bucket. Continuous MAE is a proxy for the real decision object; RPS over
   the actual bucket PMF is not.
3. **Tail error, not just mean error.** Track **p95 absolute error** and
   **worst-case error per regime**. A model with MAE 1.5 that misses by 8°F six
   times a year is *less* consistent than MAE 1.7 with a 3°F ceiling. This is the
   metric that encodes "consistent" over "best average."
4. **Reliability per hour and per regime.** PIT histograms and reliability
   diagrams, sliced by decision hour and by physical regime. Aggregate
   calibration hides conditional miscalibration — that is exactly how the seam
   bias survived a well-calibrated global PIT.
5. **N in every cell.** Non-negotiable at these sample sizes.

---

## 9. Sequence

Ordered by (expected accuracy gain ÷ effort), with dependencies respected:

**Re-ordered 2026-07-25 after the §0a audit.** Effort/uncertainty is now measured,
not guessed, and two items moved because of it.

| # | Item | Depends on | Backtestable now | Why here |
|---|---|---|---|---|
| **1** | **§7 Residual target** (+monotonic, +linear baseline) | — | Yes | Attacks the **largest measured error** (84+ bias −2.54, MAE 4.39). Mechanism is understood, fix is standard (MOS). |
| **2** | **§11 Historical edge backtest** (Kalshi API) | — | Yes | Newly unblocked; answers the *actual* question ("edge?") over ~500 days instead of weeks of waiting. |
| **3** | §0b Hygiene: drop `fc_nws`, kill `mae_nws`, fix DST cutoff | — | Yes | Trivial effort; removes a dead feature and a misleading benchmark. |
| 4 | §4 LOW = observed min + undercut mixture | — | Yes | Largest single *accuracy* gain available; converts ~84% of LOW days to measurement. |
| 5 | §3 Decision-hour indexed models + §8 scoring | — | Yes | Correctness-by-construction, but a large build. |
| 6 | §5b Pressure gradient (Santa Ana index) | — | Yes | Cheapest untested physics; aims at the hot tail. |
| 7 | §5a IGRA **inversion structure** (base height / strength) | — | Yes | 850 hPa *temp* already failed (§0a); only the structural form remains. |
| 8 | §5c SST | — | Yes | Cheap, slow-varying, explains cool-mode level. |
| 9 | §6 Physical regime mixture | §5b, §5a | After 6 & 7 | Needs calibrated inputs first. |

**§7 is done (shipped). §3 is now #1 — and the reason is new.**

**The day-ahead model barely drives any decision.** `signals.live_decision`
prefers the **intraday** prediction whenever one exists, and `auto_cycle` writes
an intraday every wake throughout the climate day. So for the HIGH, the desk's
recommendation is *effectively always* `models/intraday.py` — a **separate class**
that the §7 hybrid does not touch. Observed live 2026-07-25: day-ahead 79.1 °F
(79–80 @ 48%) vs intraday 81.4 °F (81–82 @ 53%); the desk fired **PLACE 81–82**
off the intraday.

**And the decision hour is exactly where the nowcast has no advantage:**

| cutoff | CRPS |
|---|---|
| day-ahead (9am) | 1.13 |
| **intraday 8am** | **1.12** ← the decision window closes ~10:30am PDT |
| noon | 0.60 |
| 2pm | 0.43 |
| 6pm | 0.07 |

The intraday's large gains arrive *after* Julian's window has closed and after the
market has priced the same observations — structurally the same trap as the old
confidence gate. **Therefore: accuracy work that is meant to have trading value
must target the intraday model at/below ~10:30 am PDT.** That is precisely what
§3 (decision-hour-indexed models + per-hour verification) builds, which is why it
returns to #1 despite being the larger job. §7-style tail work on the *day-ahead*
model improves the number on the dashboard, not the recommendation.

**Every new feature must clear the §0a bar:** paired bootstrap vs baseline **and**
a random-noise control, reported with N per band.

**§4 is the largest single accuracy gain available**, because it converts ~84% of
LOW days from a regression problem into a measurement problem.

**§5b is the cheapest real physics win** — one new ASOS fetch, decades of
history, aimed directly at the failure mode nothing else has touched.

---

## 14. ✅ THE LOW MARKET — what actually works (2026-07-26)

The LOW is a **different problem from the HIGH** and every HIGH-derived fix had
to be re-tested for it. Three of four failed.

| change | HIGH | LOW | shipped |
|---|---|---|---|
| desk reads day-ahead in-window | ✅ 1.509 vs 1.620 | ❌ **1.469 vs 0.806** — nowcast wins by miles | metric-aware |
| day-ahead residual/hybrid | ✅ hot tail 6.99→4.68 | ❌ **dMAE +4.72**, cold-band bias +5..+9F | HIGH-only |
| **intraday undercut residual** | ❌ dCRPS +0.050 (worse) | ✅ **MAE 0.806→0.765, CRPS 0.561→0.526** | **LOW-only** |
| morning features (shared) | ✅ significant | not isolated | both |

**Why the LOW differs.** At its decision hour (LST 05:00) the minimum has largely
already occurred, so the nowcast's ratchet on the *observed running minimum*
dominates — the LOW is much closer to a measurement problem than the HIGH. That
is also why regressing the **undercut** (the remaining move below the observed
min) beats regressing the temperature: the target becomes a small, tightly
distributed quantity (0°F on 36% of days, 1°F on 38%).

**And why the HIGH's fixes do not port.** The day-ahead residual assumes a GBM
cannot reach a sparse tail. That is true of the HIGH's hot days (bias −6.99) and
**false for the LOW**, which is already unbiased in every band (+0.14 / +0.13 /
−0.01 / −0.04). Anchoring the LOW on `fc_anchor` also drags it badly, because
`mos_mex` is HIGH-only so the anchor falls back to GFS's weaker low forecast.

**Standing rule this produced:** *a metric-specific win must not be generalised
on faith.* Two live regressions in one session came from exactly that.

---

## 13. ✅ SHIPPED — the desk was reading the wrong model at the decision hour

**The single largest accuracy gain found so far, and it required no new model.**

`signals.live_decision` unconditionally preferred the **intraday nowcast**. Measured
head-to-head at the decision hour (9am local), matched days, paired bootstrap:

| model | MAE | CRPS |
|---|---|---|
| intraday (was used) | 1.620 | 1.178 |
| **day-ahead (hybrid)** | **1.509** | **1.080** |
| Δ (day-ahead − intraday) | **−0.111** [−0.166, −0.054] | **−0.098** [−0.137, −0.058] |
| | N = 1,364 | both CIs clear zero |

For scale: this is ~5× the morning-feature gain (0.05) and the §7 hybrid (0.021)
**combined**. The nowcast only overtakes later in the day (CRPS 0.43 by 2pm), i.e.
*after* the window closes and the market has priced the same observations.

**Shipped:** `live_decision` now resolves the gate first and prefers
`("dayahead", "intraday")` while `within_window` is true, `("intraday",
"dayahead")` afterwards — sharp model while the decision is live, nowcast once
the panel is only informational. Live effect: the recommendation moved from
`PLACE 81–82 @53%` (nowcast, inside the warm-biased seam) to `PLACE 79–80 @48%`
(day-ahead).

**Generalised lesson for this plan:** before improving a model, verify **which
model the decision actually consumes, at the hour it is consumed.** Two of this
session's experiments (§12, and much of §7's value) landed on code paths the desk
never reads at decision time. Ranking by "model quality" is not the same as
ranking by "effect on the bet."

**Open follow-up (cosmetic):** `gating.gate_state` computes its own pick label
independently, so `highcast bet`'s header ("latest pick: 81-82") can disagree
with the DECISION line. Align it with the chosen prediction.

---

## 12. The intraday model is anchored to the WEAKEST forecast — ❌ TESTED, NO EFFECT

**Result (2026-07-25):** swapping `base_p50`/`base_spread` from `BaselineBlend`
(MAE 1.75) to out-of-fold ML day-ahead anchors changed nothing at the decision
hour: MAE 1.620 → 1.642, dMAE **+0.023 [95% CI −0.032, +0.079]** — not
significant (N=1,364).

**Why:** the ML day-ahead's advantage over the baseline comes from its
morning-observation features (`morn_tmpf`, `morn_rate`), and the intraday model
**already consumes those directly** (`cur_tmpf`, `cur_dwpf`, running extremum).
The better anchor carried no information the nowcast did not already have — the
same redundancy that killed the §5a inversion test. **Do not retry.**

The original reasoning is kept below because the leakage note still applies to
`BaselineBlend().fit(day_df)`, which is fit on the whole frame including future
days and should be made out-of-fold regardless.

**This is the single most direct lever on Kalshi accuracy, because the intraday
model *is* the recommendation** (§9: `live_decision` always prefers it).

`models/intraday.py` gets all of its forecast information through two features:

```
INTRADAY_FEATURES = [hour, run_lo, run_hi, run_width, cur_tmpf, cur_dwpf,
                     cur_onshore, cur_sky, base_p50, base_spread, ...]
```

`base_p50`/`base_spread` come from **`BaselineBlend`** — the *fallback* model
(HIGH MAE **1.75**) — not the ML day-ahead model (**1.53**, now hybrid). At the
decision hour the maximum is still 2–5 h in the future, so this anchor carries
most of the signal, and it is the worst forecast available.

**Fix.** Anchor the intraday frame on the ML day-ahead prediction instead.
**Leakage discipline is mandatory**: the anchor must be an *out-of-fold*
walk-forward prediction (the day-ahead model must never have seen the day it
anchors), and the current code's `BaselineBlend().fit(day_df)` — fit on the whole
frame including future days — should be fixed to the same standard.

**Gate.** Walk-forward the intraday model at the **decision hour only** (LST
08:00/09:00), baseline-anchored vs ML-anchored, paired bootstrap on MAE and CRPS.
Nothing else about the intraday model changes, so the comparison is clean.

**Why this outranks more day-ahead work:** §7-style improvements land on a model
the desk almost never reads. This one lands directly on the number that becomes
the bet.

---

## 11. Historical edge backtest — ✅ RUN. **RESULT: NO SIGNIFICANT EDGE.**

**The headline finding of the project so far, and it is a negative one.**

| HIGH, decision hour (8–9am PDT), N=63 days (2026-05-23 → 07-24) | Brier ↓ |
|---|---|
| model (walk-forward hybrid, leakage-free) | 0.5717 |
| **market (de-vigged mid)** | **0.5266** |
| difference (model − market) | **+0.0451** [95% CI −0.0487, +0.1400] |

All 63 candidate days scored; **0 skipped** for missing markets, prices, or
winners — the data is clean. The CI straddles zero, so no edge is demonstrated;
the *point estimate leans toward the market*.

**This was pre-registered** in the previous draft ("most likely no significant
edge, because the market trades the same public forecasts the model consumes").
Recording it as a result rather than reinterpreting it.

**Sample-size correction.** An earlier note claimed ~500 usable days. That is
true only of settlement **results**; **prices** vanish when a market ages past
`historical/cutoff` (verified: 6/6 buckets priced on 2026-05-23, 0/6 on
2026-05-15). The usable window is therefore **~63 days and rolling** — which
also means *today's prices are a depreciating asset*: capture them daily or lose
them.

**What follows.**
1. **Do not fund a real bankroll on this evidence.** Keep the shadow/manual
   ledgers accruing; they are the forward-looking version of this same test.
2. **Start capturing candlesticks daily** so the sample grows instead of rolling
   off. At ~1 day/day, a season's worth of *price* history is the only way this
   CI narrows.
3. Accuracy work is still worth doing — it is necessary but demonstrably **not
   sufficient**. A model can be excellent and still have no edge if the market
   already knows what it knows.

---

### 11a. Method (retained for reuse)

**Why this belongs in an *accuracy* plan.** Accuracy and edge are different
questions, and the project has only ever been able to measure the first. With
~500 days of settled Kalshi markets now reachable, the second becomes measurable
too — and it is the one that decides whether any accuracy gain is worth trading.

**Build.**
1. For each settled day D (≈2025-03 → now): `historical/markets?event_ticker=
   KXHIGHLAX-{D}` → the 6 buckets, their strikes, and `result` (the truth).
2. For each bucket, `candlesticks` at the **decision hour** (LST 08:00/09:00 —
   §0a clock note) → `yes_bid`/`yes_ask`/`close` in `*_dollars`. Use the mid when
   both sides exist; skip the day if the book is empty (record how many are
   skipped — no silent truncation).
3. **De-vig**: normalise the 6 bucket prices to sum to 1 (raw books overround by
   ~7–9%, measured on the 2026-07-25 CSVs).
4. Pair with the model's **walk-forward** prediction for D (leakage-free by
   construction — the model never saw D).
5. Score both with the **multi-category Brier** (and log-loss) against `result`;
   bootstrap the per-day difference.

**Gate.** Edge exists only if the model's Brier is lower **and** the 95% CI on the
difference clears zero. Report N, skipped days, and a per-band split (seam vs
below-80) — edge on easy days is worth nothing because the market prices them too.

**Reuse:** `signals.edge_report` already implements exactly this scoring
(de-vig → Brier → bootstrap CI) for forward data; this section feeds it history.

**Expected result — state it in advance to avoid post-hoc rationalising:** most
likely **no significant edge**, because the market trades the same public
forecasts the model consumes. That is a valid and valuable outcome. The purpose is
to find out cheaply, not to confirm a hope.

---

## 10. Honest limits

- **The HIGH has an irreducible floor at LST 09:00.** 86% of maxima have not yet
  occurred, and burn-off timing genuinely bifurcates outcomes by 10–15°F. Better
  physics narrows the distribution; it will not make marine-seam days sharp. On
  those days the correct output is a wide distribution, and a wide distribution
  is the *accurate* answer, not a failure.
- **The LOW has a hard ~16% residual** at any decision hour in the window. §4
  makes that risk explicit and calibrated; it cannot remove it.
- **Sample size bounds everything.** 899 effective rows, ~76 days ≥84°F. Several
  proposals here will fail their gates. That is the intended outcome of having
  gates — record the failures in `HANDOFF.md` as the regime-calibration work was
  correctly recorded.
- **NKX is a proxy for LAX**, and pressure-gradient indices are coarse summaries
  of a complex flow. Expect partial signal, not resolution.

---

## Sources

- [IGRA v2.2 — NOAA NCEI](https://www.ncei.noaa.gov/products/weather-balloon/integrated-global-radiosonde-archive)
- [NOAA OISST v2.1 — NCEI](https://www.ncei.noaa.gov/products/optimum-interpolation-sst)
- [Iowa Environmental Mesonet — ASOS/AWOS archive](https://mesonet.agron.iastate.edu/ASOS/)
- Internal: `HANDOFF.md`; `docs/superpowers/specs/2026-07-25-regime-conditional-calibration-design.md`;
  `docs/superpowers/specs/2026-07-25-ensemble-uncertainty-features-design.md`
- §1 timing distributions computed directly from `data/highcast.db` — 2,031
  climate days with ≥20 METAR observations, 2021-01-01 → 2026-07-24, binned in LST.
