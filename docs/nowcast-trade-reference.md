# nowcast.trade (NowCast Pro) — feature reference and mechanism

**What this file is:** a plain reference to what NowCast does and how it works.
Companion to `competitor-nowcast-trade.md`, which is the *evaluation* (what to
copy, what not to, and why). This one just describes the product.

**Sources:** pasted page text and screenshots from Julian's logged-in trial,
2026-07-29 and 2026-07-30, plus their own FAQ and docs. Operator: BuildMyApp.me.
Cannot be fetched anonymously — `WebFetch` gets the marketing shell, not the
product.

**Confidence markers used throughout:**

| marker | meaning |
|---|---|
| **[STATED]** | they say it in their own docs/FAQ |
| **[OBSERVED]** | seen directly on the product surface |
| **[VERIFIED]** | checked against our own data and it holds |
| **[REFUTED]** | checked against our own data and it does not hold |
| **[UNVERIFIED]** | read from a screenshot, never confirmed |

---

## 1. How NowCast actually works

The product is **not primarily a forecaster**. Its core loop is latency:

> **[STATED]** "A Lock occurs when NowCast CONFIRMS the daily minimum or maximum
> temperature reading — the same value the market settles on. Once locked, that
> temperature cannot go lower (LOW) or higher (HIGH) for the day."

> **[STATED]** "The buy window is typically 20–40 minutes after lock before the
> market reprices."

So the mechanism is:

1. Watch `api.weather.gov` observations for a given station.
2. Detect that the day's extremum has **already happened** and can no longer move.
3. Alert the user, who buys the now-near-certain bracket **before Kalshi reprices**.

That is a speed game against the order book, and it explains their headline win
rate completely. It is a legitimate strategy. It is not prediction, and its edge
decays as the market gets faster.

**Lock times [STATED]:** LOW markets lock **6:53 AM EST** for NYC, Miami,
Chicago, Austin, Denver, Philadelphia; **7:53 AM EST** for Los Angeles. (Their
"EST" in July is almost certainly EDT — see §8 on their timezone handling.)

**The one genuinely forecasting component** is Morning Edge, which posts
predictions ~6:45–7:45 AM EDT and is graded against settled NWS CLI values.

---

## 2. The model stack

> **[STATED]** "HRRR is the primary forecast — it's the 3km short-range model
> that updates hourly, built specifically for same-day predictions. GFS, ECMWF,
> and Best Match are confirmation models."

Four channels, HRRR-primary. "Best Match" is Open-Meteo's own blend name, so they
are on the same data source we are — with **four** channels where we run nine
(`ecmwf`, `gfs`, `icon`, `hrrr`, `nws`, `mos_mex`, `nbm`, `nam`, `aifs`) plus the
NBS bulletin channel.

Choosing HRRR as primary for a same-day nowcast is defensible: 3 km resolution,
hourly refresh. We carry HRRR as one of nine with no special weighting.

**Their confidence rule [STATED]:**

| model spread | their label |
|---|---|
| 0–2 °F | HIGH CONFIDENCE |
| 3–5 °F | MODERATE |
| 6 °F+ | DIVERGENT — skip or size down |

**[REFUTED] as a confidence signal.** On 2026-07-29 all eight of our guidance
channels agreed inside a 2.3 °F band (65.8–68.1) and were collectively 2 °F
wrong. Agreement is not correctness. They ship this as a buy signal.

---

## 3. Settlement method

> **[STATED]** "computed directly from NWS observations (api.weather.gov) using
> the **6-hour aggregate method** — decimal precision, no rounding until the
> final step"

**[VERIFIED]** — this is the same method `latentmax` uses, and their settled
values match ours exactly. KMIA 2026-07-28 MAX = **91** on both systems, on the
day an ad-hoc `MAX(tmpf_precise)` query in this project returned 89. Two
independent operators, same truth source, same answer.

> **[STATED]** "the RUNNING LOW/MIN reflects the raw trough reading, including
> one-off dips. **Settlement uses the rolling 5-minute average**, so a single dip
> will average back up and may NOT be the settlement value."

> **[STATED]** "A lone reading that jumps 1.5+ away from both its neighbors gets
> a 'spike?' tag and is left out of the day's extremes."

Consistent with what we already do rather than contradicting it: ASOS computes
5-minute average temperatures internally and the `1sTTT`/`2sTTT` six-hour groups
report the extremes **of those averages**. Since `latentmax` reads
`six_max_c`/`six_min_c`, we already take the averaged value.

**Which group to read [STATED], and [VERIFIED] against our own rule:**

> "**LOW markets — use the morning Min, not the overnight one.** … Do NOT use the
> overnight Min row (e.g. ~1–2 AM) — it covers the wrong window and will give you
> the wrong number."

> "**HIGH markets — wait for the afternoon Max.** Eastern ~7:50–7:55 PM, Central
> ~6:50–6:55 PM, Denver ~5:53 PM, Phoenix ~4:51 PM, Pacific ~4:50–5:00 PM."

Our first-principles admissibility rule reproduces this on both stations,
*including which report to reject*:

| station | climate day | admissible groups (local file time) | rejected |
|---|---|---|---|
| KLAX | 08Z→08Z | 18Z **10:53**, 00Z **16:53**, 06Z **22:53** | **12Z** (files 04:53 — their "overnight Min") |
| KMIA | 05Z→05Z | 12Z **07:53**, 18Z **13:53**, 00Z **19:53** | **06Z** (files 01:53 — their "~1–2 AM" warning) |

KLAX's 16:53 matches their "Pacific ~4:50–5:00 PM"; KMIA's 19:53 matches their
"Eastern ~7:50–7:55 PM". The rejected report **differs by station**, exactly as
the differing climate-day offsets require.

---

## 4. Feature-by-feature reference

### 4.1 Market boards

| feature | what it does |
|---|---|
| **LOW Markets / HIGH Markets** | Per-city cards: a headline temperature, a "lock confirmed" state, the leader and contender brackets with bid/ask in cents, and an action badge. Grouped into LOCK DETECTED — EXPENSIVE, LOCK DETECTED — MISMATCH, and WATCHING — NO LOCK YET. |
| **Air Floor / Running HIGH** | Claims to be the running extremum since midnight. **[REFUTED] — see §7.** |
| **Lock Tracker** | Which markets have locked and when. |
| **2-Bracket** | Two-bracket view of a market. |
| **Station Table** | Per-city raw state: wind direction, sky, pressure trend, and an **advection** grade. See §4.4. |
| **Station Map** | Geographic view of the same. |
| **Intraday Edge** | Intraday opportunity view. |
| **LOW / HIGH Time Series** | Tabular observation history with running extremum columns. **[UNVERIFIED] boundary defect — see §7.** |
| **Forecast** | Today's predicted HIGH and LOW for all cities across their four models. |

### 4.2 Order-flow features — a real signal we cannot replicate

| feature | what it does |
|---|---|
| **Whale Watch** | Large resting orders in each market's favourite bracket, with **standing time** — which they correctly call "the real tell". A wall standing 30–50+ min is conviction; one vanishing in under a minute was likely spoofing. They explicitly warn "whales can bluff". |
| **Order Book Watch** | Real dollar depth YES vs NO on the favourite bracket, refreshed ~3 min. Flags "NO-heavy" books where money doubts the price. |
| **Smart Money** | Contract volume by city and bracket. |

Their own honest note: *"A NO-heavy book is NOT a reliable signal on its own — it
has already been wrong in practice."* Credit for saying so.

This needs live book depth and continuous scanning. We have nothing here and
probably should not chase it.

### 4.3 Physics filters

| feature | what it does |
|---|---|
| **⛈ STORM RISK** (LOW) | "Rain cools the air toward the dewpoint… a single storm can undercut the bracket." Shipped as a hard veto. |
| **🔥 RE-HEAT RISK** (HIGH) | Forecast says the temperature can still climb past the bracket. |
| **Cloud-cover read** | "🌙 Clear skies — heat escapes, the low can keep dropping toward ~1 AM. ☁️ Cloudy — cooling is trapped, the low is holding near this floor." |

### 4.4 Advection grade — the one concept we lack entirely

Grades every city using **wind direction + pressure tendency + sky, jointly**:

| grade | meaning |
|---|---|
| GREEN | no advection, stable temperature → safe to trade |
| PARTIAL | advection developing → caution |
| FULL | active warm/cold air mass moving in → **AVOID**, temperature still moving |

**[OBSERVED] 2026-07-29:** they rated Los Angeles wind 180°, OVC/BKN, pressure
falling YES, **WARM ADV = FULL → "AVOID — FULL ADVECTION ACTIVE"** — on the exact
day our LOW committed 67–68 at 81% and lost to a temperature that would not fall.

Every input is now in our `obs` table: `drct`, `skyc`, and (as of 2026-07-30)
`slp_hpa` / `pres_tend_hpa` / `pres_tend_code`.

### 4.5 Weather Guard (preflight)

Grades all 18 cities on **how predictable the weather is**, and gates everything
else on it. **[OBSERVED] 2026-07-29 LOW rankings:**

| tier | cities |
|---|---|
| A+ Trade Candidate (#1–10) | DC, Chicago, Minneapolis, Houston, OKC, Phoenix, Las Vegas, Philadelphia, New Orleans, Denver |
| A− Watch (#11–13) | Austin, **Miami** ("overnight showers"), Atlanta |
| B Caution (#14–15) | New York, Boston ("cloud contamination") |
| **C− REJECT (#16–18)** | **Los Angeles**, San Francisco, Seattle — "Cloud contamination, MARINE LAYER RISK" |

Labelled "Beta — grades are new", with no published validation. It is a **static
per-city climatological grade**, so it returns the same answer every day.

### 4.6 Pick of the Day — their only falsifiable feature

> **[STATED]** "Every pick is graded against official settlement the next morning
> — wins AND losses, on this page, forever. **If the graded record doesn't prove
> an edge, we pull the feature.**"

> **[STATED]** "Some days show **NO PICK**: that means nothing met the bar, and
> skipping is part of the edge."

**[OBSERVED] record:** LOW 3W–0L, HIGH 2W–0L over 4 days, **average entry 71.6¢**.
Up to one pick per metric per day; the bar itself is **not disclosed**.

⚠️ **Their entry price does not transfer to us.** 71.6¢ requires a **73.0%** true
hit rate to break even after fees. Our best commit-hour hit is KLAX LOW at 62.3%.
Buying at their average entry loses ~10.6¢/contract on our best market and
~20.6¢ on KLAX HIGH.

### 4.7 Calculator

Shows entry price, contracts, gross P&L, entry fee, net P&L and breakeven.

**[VERIFIED]** it reproduces Kalshi's published fee schedule exactly:
`fees = round up(0.07 × C × P × (1−P))`. Their worked example — 10 contracts at
65¢ → **$0.16** — matches `0.07 × 10 × 0.65 × 0.35 = 0.159`, rounded up.

Confirmed independently: rounding is **up, per order**; the max per-contract fee
is **1.75¢ at P = 0.50**; maker fees are **25% of taker**; and there is **no
settlement fee** ("holding a winner to resolution costs $0").

### 4.8 NWS Settlement Reports

Settled max/min per city from `api.weather.gov`. **[VERIFIED]** matches our
`cli_reports` exactly where they overlap.

---

## 5. What they disclose about method

**Withheld [STATED]:**

> "Morning Edge is a **proprietary** early-prediction system… the prediction
> logic, scoring, and data integration are not available anywhere else… **The
> methodology is exclusive to NowCast and not disclosed.**"

So there is no mechanism to copy — only the shape. No ML is mentioned anywhere on
the product surface, and given how heavily it leans on order flow, it is plausible
their real edge is market microstructure rather than meteorology.

---

## 6. Their headline number, and why it is not comparable

**757W · 130L · 85.3% all-time** over 887 graded signals since 2026-06-18.

Graded only on published **STRONG SIGNALS**, which their own docs define as:

> "The market has locked onto this bracket — it's the clear favorite and **every
> other bracket has collapsed**."

> "SETTLED badge — the winning bracket stands alone while **every other bracket
> has collapsed to 1¢** (or the leader is ~95¢+)."

They admit the consequence: *"it may already be priced high — this signals the
winning bracket, not necessarily a cheap entry."*

**The arithmetic:** buying at 95¢ and winning 85% of the time is
`0.85 − 0.95 = −$0.10 per contract`. Breakeven at 95¢ needs **95.3%** after fees.
Their flagship number loses money at the prices their own docs describe.

**Their flag filter separates nothing:** CLEAN 110W–22L (83.3%) vs FLAGGED
44W–10L (81.5%) — 1.8 points apart. And 110+22+44+10 = 186 signals against 887
claimed, so 79% of the record is unexplained.

---

## 7. Verified defects

### 7.1 "Air Floor" is the current temperature, not the running minimum — **[REFUTED]**

Their LOW cards show a headline number labelled "Air Floor: X°F since midnight ✅
lock confirmed". Checked against our own observations at the same instant
(2026-07-30 00:24Z):

| city | their "Air Floor since midnight" | our latest obs | our actual running min |
|---|---|---|---|
| KLAX | 75°F | **75.9°F** | **69.08°F** |
| KMIA | 88°F | **89.1°F** | **78.98°F** |

It tracks the **current temperature**. On the face of it this held across all
twelve cities on screen — "LOW · Air Floor 101°F" for Phoenix and 98°F for OKC
cannot be overnight minima.

Their *market* brackets were simultaneously correct (LA 69+, Miami 79–80, matching
our 69.08 and 78.98), so the market feed is fine and the headline is the broken
part. This lands on the **Lock**, the single mechanism the whole business model
rests on.

**Consequence:** their "MISMATCH · Do not act" badge can no longer be read as a
judgment about the market — it may simply be their own bug tripping. I could not
determine the badge logic from screenshots and do not claim to.

### 7.2 Non-monotone running extremum — **[UNVERIFIED]**

Their NYC HIGH time series, 2026-07-30 00:54 ET, read `6HR MAX: 72.0F | RUNNING
HIGH: 78.1F` with the table showing:

```
11:51 am  temp 75.0   RUNNING HIGH 78.1
 9:51 am  temp 70.0   RUNNING HIGH 75.0   <- decreased
 8:51 am  temp 73.9   RUNNING HIGH 73.9   <- decreased again
 8:43 am  temp 69.1   RUNNING HIGH 78.1   <- back up
```

A running maximum can only rise, and their own FAQ says "the high only rises". The
x-axis also spanned ~28 hours, so the 78.1 appears to be **yesterday's** high.
Read from a screenshot, not their source — hence UNVERIFIED. For contrast, our
`latentmax` was swept over 2,880 (station, day, as-of) states across 240 days on
both stations: **0 monotonicity violations, 0 min-above-max violations.**

### 7.3 Data-quality problems — **[OBSERVED]**

- **Duplicate city rows with different records.** The all-time table lists
  uppercase and mixed-case variants as separate rows: `LOS ANGELES LOW 4W-0L`
  beside `Los Angeles LOW 13W-0L`; `MIAMI LOW 11W-4L` beside `Miami LOW 11W-2L`.
  Either two pipelines write the same city under different keys, or the grader is
  case-sensitive on a human-entered field. Either way the totals are suspect.
- **"YESTERDAY · Monday, July 27, 2020"** — wrong year on the results header.
- **"Last 7 days: 0W · 0L"** while claiming 887 all-time — grading appears stalled.
- **Stale bot:** "Bot scan: 2026-07-29 12:41:29 (725 min ago — bot may be offline
  — data may be stale)" on the live Smart Money page.
- **City count contradicts itself:** FAQ says "7 LOW and 18 HIGH — 25 total";
  Whale Watch says "all 36 Kalshi weather markets (18 LOW + 18 HIGH)"; Forecast
  says "16 Kalshi cities".
- **Pick of the Day dated "Tuesday, July 28, 2026"** on a page refreshed 12:49 PM
  ET on Jul 29.
- **Mislabelled settlement source:** their NYC card shows "Official Forecast Low:
  68°F (NowCast settlement source)". Settlement is the observed CLI value, not a
  forecast.

### 7.4 EST/EDT confusion — **[OBSERVED]**

Their sync board is timestamped "12:47:32 **EST**" in late July, and stated lock
times are given in "EST". In a product whose entire edge is a 20–40 minute window,
an hour of drift matters.

---

## 8. Summary: what they have that we do not, and vice versa

**They have:**

- Order-flow visibility (Whale Watch, Order Book, Smart Money) — a real signal
  needing live book depth we do not collect.
- An **advection grade** combining wind direction, pressure tendency and sky.
- Storm-risk and re-heat vetoes shipped as badges.
- 18 cities to our 2.
- A **falsifiable Pick of the Day** with a public graded record and a stated
  promise to pull the feature if it cannot prove an edge.

**We have:**

- Measured **per-hour accuracy curves** and **still-future** curves — nothing on
  their surface shows they know which hour their signal is most informative at.
- **Calibration measurement** (stated vs realised); they publish win rate only.
- **Edge vs market recorded at commit** (`alert_log.kalshi_pct`); they show an
  edge in cents but never grade whether it realised.
- An **immutable committed-bet ledger** with anti-cheat rules in code.
- Written **negative results** (pooling, parametric, convection, rate-of-change,
  and the ≥80 °F seam all measured and rejected).
- Verified extremum invariants and derived, station-specific six-hour-group
  admissibility.
