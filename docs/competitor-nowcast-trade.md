# Competitor analysis — nowcast.trade (NowCast Pro)

Captured 2026-07-29/30 from Julian's logged-in trial. Source: pasted page text +
screenshots. Operator: BuildMyApp.me. 18 cities, 36 markets (18 LOW + 18 HIGH).

---

## 1. THE HEADLINE FINDING — their 85% is NOT our 57%

They advertise **757W · 130L · 85.3% all-time** over 887 graded signals since
2026-06-18. That number is not comparable to anything we measure, and the reason
is in their own documentation:

> **STRONG SIGNAL (HIGH)** — "The market has locked onto this bracket — it's the
> clear favorite and **every other bracket has collapsed**."

> "**Counts only published STRONG SIGNALS**"

> "SETTLED badge — the winning bracket stands alone while **every other bracket
> has collapsed to 1¢** (or the leader is ~95¢+)."

**They grade themselves on signals fired AFTER the market has already decided.**
That is transcription, not prediction — the exact distinction this project spent
2026-07-26 learning (see HANDOFF: the LOW is 84.7% "accurate" at hour 11 because
71% of lows have already happened by then).

They admit the consequence themselves:

> "it may already be priced high — this signals the winning bracket, **not
> necessarily a cheap entry**."

**THE ARITHMETIC THAT MATTERS.** Buying a bracket at 95¢ and winning 85% of the
time is a LOSING strategy:

    EV = 0.85 x $1.00 - $0.95 = -$0.10 per contract

To break even at 95¢ you need 95%. Their own flagship number, taken at face
value on the prices their own docs describe, loses money. A high win rate on
near-certain contracts is the easiest metric in this business to produce and the
least informative.

**Ours measures the opposite thing:** bucket hit at a pre-registered commit hour
with 76-95% of the outcome still unknown. KMIA HIGH 56.8% at 95% still-ahead is
a far harder number than 85% on a collapsed book.

---

## 2. THE FLAG STORY UNDERMINES ITSELF

They present this as a feature — signals they warned you off:

    CLEAN signals  110W-22L  83.3%   <- "trade these"
    FLAGGED        44W-10L   81.5%   <- "we warned you off"

**The flags move the win rate by 1.8 points.** A filter that separates 83.3% from
81.5% is not separating anything. If the warnings had real content the flagged
group would be visibly worse.

Also note 110+22+44+10 = **186 signals**, against a claimed 887 graded. The flag
breakdown covers 21% of the record; the other 79% is unexplained.

---

## 3. DATA-QUALITY PROBLEMS VISIBLE ON THE PAGE

These matter because they bear on whether the 85% can be trusted at all.

**Duplicate city rows with different records.** The all-time table lists both
uppercase and mixed-case variants as SEPARATE rows:

    LOS ANGELES LOW    4W  0L  100%
    Los Angeles LOW   13W  0L  100%
    ATLANTA LOW       15W  0L  100%
    Atlanta LOW       11W  2L   85%
    MIAMI LOW         11W  4L   73%
    Miami LOW         11W  2L   85%

Same market, two rows, different records. Either two pipelines are writing the
same city under different keys, or the grader is case-sensitive on a
human-entered field. Either way the totals are suspect.

**Wrong year on the results header:** "YESTERDAY · Monday, **July 27, 2020**".

**"Last 7 days: 0W · 0L"** — no graded results in a week, while claiming 887
all-time. Grading appears to have stalled.

**Stale bot:** "Bot scan: 2026-07-29 12:41:29 (**725 min ago — bot may be
offline — data may be stale**)" shown on the live Smart Money page.

---

## 4. WHAT THEY DO THAT WE DO NOT — genuinely worth noting

### Order-book / market microstructure (we have NOTHING here)
- **Whale Watch** — large resting orders in each market's favourite bracket.
  Reports **standing time**, which they correctly call "the real tell": a wall
  standing 30-50+ min is conviction; one vanishing in under a minute was likely
  spoofing. They explicitly warn "whales can bluff".
- **Order Book Watch** — real dollar depth YES vs NO on the favourite bracket,
  refreshed ~3 min. Flags "NO-heavy" books where money doubts the price.
- **Smart Money** — contract volume by city/bracket.

Honest note in their own docs: *"A NO-heavy book is NOT a reliable signal on its
own — it has already been wrong in practice."* Credit for saying so.

### Physics filters we identified but have not built
- **⛈ STORM RISK** on a LOW: "Rain cools the air toward the dewpoint... a single
  storm can undercut the bracket." This is EXACTLY the convection gap measured on
  2026-07-29 (see the convection section) — they ship it as a hard veto.
- **🔥 RE-HEAT RISK** on a HIGH: forecast says temperature can still climb past
  the bracket.
- **Cloud-cover read**: "🌙 Clear skies — heat escapes, the low can keep dropping
  toward ~1 AM. ☁️ Cloudy — cooling is trapped, the low is holding near this
  floor." **This is the four-factor rule we researched, shipped as a UI badge.**

### Scope
18 cities vs our 2. **Weather Guard** grades each city A+/CAUTION/REJECT/AVOID
and gates everything else on it.

---

## 5. WHAT THEY DISCLOSE ABOUT METHOD (very little)

**Explicitly withheld:**
> "Morning Edge is a **proprietary** early-prediction system... the prediction
> logic, scoring, and data integration are not available anywhere else... **The
> methodology is exclusive to NowCast and not disclosed.**"

**What IS disclosed, and it matches ours:**
> "computed directly from NWS observations (**api.weather.gov**) using the
> **6-hour aggregate method** — decimal precision, no rounding until the final
> step"

They use the 6-hour remark groups — the same thing `latentmax` does and the same
trap that caught us on 2026-07-29 with an ad-hoc `MAX(tmpf_precise)` query.
Independent confirmation that this is the correct settlement-truth method.

**No NWP model is ever named.** No GFS, ECMWF, HRRR, NBM, MOS, no ML mention
anywhere in the product surface. Combined with how heavily the product leans on
order flow, **it is plausible their edge is market microstructure rather than
meteorology** — reading the book, not forecasting the temperature.

**Morning Edge timing:** "Predictions post ~6:45-7:45 AM EDT and are graded
against official NWS CLI settled temperatures." That is a genuine forecast
window; it is also the ONLY part of the product that forecasts anything.

---

## 6. WHAT WE HAVE THAT THEY APPEAR NOT TO

- A trained model with **measured per-hour accuracy curves** and measured
  **still-future** curves. Nothing on their surface shows they know which hour
  their signal is most informative at.
- **Calibration measurement** (stated vs realised). They publish win rate only.
- **Edge vs market recorded at commit** (`alert_log.kalshi_pct`). They show a
  "⚡ edge" in cents but do not grade whether their edges realised.
- An **immutable committed-bet ledger** with anti-cheat rules in code.
- Honest **negative results** written down (pooling, parametric, convection,
  rate-of-change all measured and rejected).

---

## 7. WHAT TO ACTUALLY TAKE FROM THIS

1. **Build the storm-risk / cloud-cover veto.** They ship it, we researched it,
   and it is the documented cause of our KMIA 92-93 miss. Deterministic, no
   model change, works as a display flag under the freeze.
2. **Do NOT chase their 85%.** It is measured on collapsed books. Chasing it
   would mean abandoning the predict-before-the-extremum goal, which is the
   whole point of this project.
3. **Order flow is a real signal we cannot replicate** and probably should not
   try to. It needs live book depth and continuous scanning.
4. **Their duplicate-row bug is a warning about our own audit.** We already
   group by `model_fingerprint` for exactly this reason — two pipelines writing
   the same market under different keys is how a track record becomes fiction.

---

## 8. RAW EVIDENCE — LIVE MARKET SNAPSHOT (2026-07-29, ~12:41 ET)

Their board on the day our KLAX LOW bet was losing:

    Los Angeles LOW · leader 69+ at 87c · contender 67-68 at 17c
    Los Angeles LOW (Lock Loaded) · PENDING · WATCH · cushion -0.1 · 69-above 87c
    LA LOW (Smart Money) · 69deg or above · 100 contracts · 85c

Our app committed **67-68 at 81%**. The market had it at **17%**. Running
minimum was 68.98 -> rounds to 69. Independent confirmation that the guidance-
drift failure was real and that the market saw it and we did not.

Their own LA card read: **69°F [68-69], Now: 73.9°F, FLOOR IN (provisional)**.

---
---

# ROUND 2 — the FAQ answers everything (2026-07-30 screenshots)

## 9. THE MODEL, FINALLY NAMED

> "The Forecast tab shows today's predicted HIGH and LOW temperatures for all 16
> Kalshi cities using **3 weather models: GFS, ECMWF, and Best Match**."

"Best Match" is Open-Meteo's own blend name — so **they are on the same data
source we are**, with THREE models where we run nine (ecmwf, gfs, icon, hrrr,
nws, mos_mex, nbm, nam, aifs) plus the NBS bulletin channel.

Their consensus rule is eyeball-simple:
> "A tight cluster means all models agree — strong signal. Spread dots mean
> divergence — skip that market or wait."

**That is exactly the rule that failed us on 2026-07-29** — all eight models
agreed at 65.8-68.1 and were collectively 2F wrong. Agreement is not correctness.
We have that written up as a negative finding; they ship it as a buy signal.

## 10. THE BUSINESS MODEL IS LATENCY, NOT FORECASTING

The FAQ states it outright:

> "**A Lock occurs when NowCast CONFIRMS the daily minimum or maximum temperature
> reading** — the same value the market settles on. Once locked, that temperature
> cannot go lower (LOW) or higher (HIGH) for the day."

> "When do LOW markets lock? **6:53 AM EST** for NYC, Miami, Chicago, Austin,
> Denver, Philadelphia. **7:53 AM EST** for Los Angeles."

> "**The buy window is typically 20-40 minutes after lock before the market
> reprices.**"

So the product is: **detect that the extremum has ALREADY happened, then buy in
the 20-40 minute window before Kalshi catches up.** That is a latency/speed
game against the order book. It is a legitimate strategy and it is NOT
forecasting — it explains the 85% win rate completely, and it is why that number
cannot be compared to our 57% at a commit hour with 76-95% still unknown.

It also means their edge decays as the market gets faster, and it requires the
user to be awake and act inside a 20-40 minute window. From Manila that window
is roughly 19:53-20:33 (LOW) and mid-afternoon (HIGH).

## 11. ⭐ ADVECTION — THE ONE CONCEPT WE ARE ENTIRELY MISSING

Their Station Table grades every city on **advection** (horizontal transport of a
different air mass) using wind direction + pressure trend + sky:

    GREEN    no advection, stable temp        -> safe to trade
    PARTIAL  advection developing             -> caution
    FULL     active warm/cold air moving in   -> AVOID, temp still moving

**On 2026-07-29 their board rated Los Angeles: wind 180 deg, OVC/BKN, pressure
falling YES, WARM ADV = 🔥 FULL -> "AVOID — FULL ADVECTION ACTIVE".**

That is precisely the day our LOW model committed 67-68 at 81% and lost to a
temperature that would not fall. **Warm advection was pushing KLAX up while our
model followed cool guidance down.** We have no feature for it and no veto.

This is a better-specified version of the four-factor rule already researched:
it uses wind DIRECTION + PRESSURE TENDENCY + SKY jointly rather than each alone,
and it produces a tradeable/avoid verdict rather than a number. Every input is
already in our `obs` table (drct, skyc) except pressure tendency, which METAR
carries in the 3-hour pressure group.

**This is now the single most promising external idea, ahead of `guidance_gap`**
-- it targets the same failure with a physical mechanism rather than a
statistical proxy, and there is a live example of it working on the exact day we
lost.

## 12. SETTLEMENT DETAIL WORTH CHECKING (probably we are already right)

> "the RUNNING LOW/MIN reflects the raw trough reading, including one-off dips.
> **Settlement uses the rolling 5-minute average**, so a single dip (a lone 41
> among 42s) will average back up and may NOT be the settlement value."

> "A lone reading that jumps 1.5+ away from both its neighbors gets a 'spike?'
> tag and is left out of the day's extremes."

⚠️ VERIFY BEFORE ACTING ON THIS. It is consistent with what we already do rather
than contradicting it: ASOS computes 5-minute average temperatures internally and
the 1sTTT/2sTTT six-hour groups report the extremes OF THOSE AVERAGES. Since
`latentmax` reads `six_max_c`/`six_min_c`, we are already taking the averaged
value, not a raw spike. The warning applies to anyone reading 1-minute or spot
data. Worth a one-off check that no spike has ever set one of our extremes.

## 13. THE ONE FEATURE THAT IS HONESTLY MEASURED

**Pick of the Day** is the only page with a falsifiable promise:

> "Every pick is graded against official settlement the next morning — wins AND
> losses, on this page, forever. **If the graded record doesn't prove an edge, we
> pull the feature.** Nothing stays on NowCast that can't prove itself."

> "Some days show **NO PICK**: that means nothing met the bar, and skipping is
> part of the edge."

Record: **LOW 3W-0L, HIGH 2W-0L, avg entry 71.6c, 4 days.** n=5, so it proves
nothing yet — but the DESIGN is right, and the 71.6c average entry is the number
that matters: unlike the 85%-on-collapsed-books headline, 71.6c leaves room to
profit. The day captured showed "NO PICK TODAY — nothing playable today, no pick
is the honest pick."

**Adopt the discipline, not the signal:** one pick a day, a price ceiling, and
publishing NO PICK when nothing qualifies. That is closer to our edge gate than
anything else on their site.

## 14. MORE DATA-QUALITY PROBLEMS

- City count contradicts itself: FAQ says "**7 LOW markets and 18 HIGH markets
  — 25 total**"; Whale Watch says "**all 36 Kalshi weather markets (18 LOW + 18
  HIGH)**"; Forecast tab says "**16 Kalshi cities**".
- Pick of the Day header dated "Tuesday, **July 28**, 2026" while the page was
  refreshed 12:49 PM ET on Jul 29.
- Sync board timestamped "12:47:32 **EST**" in late July (should be EDT) — the
  same EST/EDT confusion appears across several pages, and their stated lock
  times ("6:53 AM EST") are almost certainly EDT in summer. **An hour of drift in
  a product whose whole edge is a 20-40 minute window.**

## 15. LIVE CROSS-CHECK ON OUR OWN LOSING BET

Their board, same moment:

    Lock Loaded   Los Angeles LOW  PENDING  ⚠ WATCH  cushion -0.1  69-above 87c
    Whale Watch   LAX LOW  🔒⚠️ MISMATCH  "Do not act"
                  leader 69+ 86c/87c · contender 67-68 12c/17c
    Race          LA LOW  favorite 69+ 87%  ·  room -1  NEEDS -1° COLDER
    Sync          LA LOW  BOTTOMED  pred 69F  held 460m  bracket 69+  ask 87c

Four independent surfaces agreeing: 69+, and **their own system flagged it "Do
not act"**. Our app was at 81% on 67-68. The "held 460m" figure is the one to
note — the low had not moved for 7.7 hours, which is the stall our model had no
feature to see.

---

# 16. ⚠️ WEATHER GUARD RATES **LAX LOW AS C-, "REJECT"** — RANK 16 OF 18

Their preflight page ranks all 18 cities by *how predictable the weather is*,
not by temperature. 2026-07-29 LOW rankings:

    #1-10   A+  Trade Candidate   DC, Chicago, Minneapolis, Houston, OKC,
                                  Phoenix, Las Vegas, Philadelphia,
                                  New Orleans, Denver
    #11-13  A-  Watch             Austin, MIAMI ("overnight showers"), Atlanta
    #14-15  B   Caution           New York, Boston ("cloud contamination")
    #16-18  C-  REJECT            LOS ANGELES, San Francisco, Seattle
                                  -- "Cloud contamination, MARINE LAYER RISK"

**The market this entire project is built around is in their bottom tier**, and
the stated reason is exactly the mechanism that beat us on 2026-07-29. All three
rejected cities are Pacific marine-layer stations.

**This is an independent operator, monitoring 18 cities daily, concluding that
LAX's LOW is among the three hardest markets on the board.** We reached the
opposite conclusion in `cities.py` (LAX low ranked #2 most predictable at 40.6%
parity-corrected). BOTH CAN BE TRUE and the tension is informative:

- Our survey measures **guidance bucket-hit accuracy** -- LAX's low has small
  day-to-day variance (persistence 1.91F, the lowest of any market surveyed), so
  guidance lands close.
- Theirs measures **risk of a weather surprise** -- the marine layer is exactly
  the thing that makes "close" and "in the right 2F bucket" different.

A low-variance market where a fog deck can hold the temperature 2F above every
model is precisely the profile that produces a good MAE and a coin-flip bucket
hit. That is our measured reality: MAE 0.885 at the commit hour, bucket hit 62%.

**Do not treat this as proof we picked the wrong city.** It is one operator's
heuristic grade with no published validation ("Beta — grades are new"). But it is
the second independent line of evidence pointing at the marine layer, and it
lands on the same day we lost to it.

Miami LOW graded **A- "Overnight showers"** -- their convection concern, matching
our own measured finding that Miami's low is worse than LAX's.

## 17. OPERATIONAL DETAIL WORTH STEALING — WHICH 6-HOUR GROUP TO READ

> "**LOW markets — use the morning Min, not the overnight one.** The daily low
> posts on the morning 6Hr Min row. Eastern/Central after ~7:30 AM local;
> Mountain/Pacific later. **Do NOT use the overnight Min row (e.g. ~1-2 AM) — it
> covers the wrong window and will give you the wrong number.**"

> "**HIGH markets — wait for the afternoon Max.** Verified posting times (station
> local): Eastern ~7:50-7:55 PM, Central ~6:50-6:55 PM, Denver ~5:53 PM,
> Phoenix ~4:51 PM, Pacific ~4:50-5:00 PM."

This is the same class of trap `latentmax` already handles (the 12Z report
straddles the climate-day start and must be rejected), but their per-timezone
posting times are more specific than anything we have written down. Worth
checking our own group-selection against these for KLAX and KMIA.

## 18. REVISED "WHAT TO TAKE FROM THIS"

Superseding section 7, in priority order:

1. **ADVECTION VETO** (section 11) -- wind direction + pressure tendency + sky,
   producing GREEN/PARTIAL/FULL. They rated LAX "🔥 WARM ADV FULL — AVOID" on the
   day we lost. Physical mechanism, all inputs already in `obs` except pressure
   tendency. **Now ranked above `guidance_gap`.**
2. **Pick-of-the-Day discipline** (section 13) -- one pick, price ceiling, and
   publish NO PICK when nothing qualifies. Closest thing on their site to an
   honest edge gate.
3. **Storm-risk / cloud-cover veto** (section 4) -- still worth it.
4. **Do NOT chase the 85%** -- it is a latency play on collapsed books
   (section 10), not forecasting.

---

# 19. ⚠️ SUSPECTED CLIMATE-DAY BOUNDARY BUG IN THEIR TIME SERIES

Their NYC HIGH Time Series header, 2026-07-30 00:54 ET:

    6HR MAX: 72.0F  |  RUNNING HIGH: 78.1F        Current: 75.0

Two things are wrong with that pair, and the table underneath shows why.

**(a) The x-axis spans ~28 HOURS**, not one climate day:

    8:05 am -> 3:51 pm -> 9:28 pm -> 12:08 am -> 2:51 am -> 6:32 am -> 9:42 am

The oldest rows are YESTERDAY morning. The 78.1F peak sits at the BOTTOM of the
table at "10:51 am" -- i.e. **yesterday's 10:51 am**. Today's highest reading in
the list is 75.0F at 11:51 am.

**So "RUNNING HIGH 78.1F" for today appears to include yesterday's high.** That
is a climate-day boundary error: exactly the class of bug `latentmax` and
`station.std_utc_offset` exist to prevent here (the climate day is midnight-to-
midnight LOCAL STANDARD time, and the 12Z report straddles the boundary and must
be rejected -- see HANDOFF).

**(b) Their RUNNING HIGH column is NOT MONOTONE**, which is impossible for a
running maximum:

    11:51 am  temp 75.0   RUNNING HIGH 78.1
     9:51 am  temp 70.0   RUNNING HIGH 75.0   <-- decreased
     8:51 am  temp 73.9   RUNNING HIGH 73.9   <-- decreased again
     8:43 am  temp 69.1   RUNNING HIGH 78.1   <-- back up

A running high can only rise. Either the column is mislabelled (it may be a
rolling 6-hour max rather than a running one) or the series is being computed
across an inconsistent window. Their own FAQ says "the high only rises" -- the
data contradicts the documentation.

**Why this matters to us:** it is independent evidence that the boundary
handling is genuinely hard and easy to get wrong, and that the care taken here
(`std_utc_offset`, the 12Z rejection, `latentmax`) is load-bearing rather than
pedantic. It is ALSO a reason not to trust their 85% record: if the running
extremes are computed over the wrong window, the grading is downstream of it.

⚠️ NOT FULLY VERIFIED. Read from a screenshot, not their source. The alternative
reading is that the page deliberately shows a rolling 24h context window and the
"6HR MAX 72.0" is the true climate-day figure. Either way the two headline
numbers contradict each other on screen.

# 20. THE MODEL STACK, FULLY NAMED AT LAST

> "**HRRR is the primary forecast** — it's the 3km short-range model that updates
> hourly, built specifically for same-day predictions. **GFS, ECMWF, and Best
> Match are confirmation models.**"

Four models, HRRR-primary. Choosing HRRR as primary for a same-day nowcast is
defensible -- 3km resolution, hourly refresh. We carry HRRR as one of nine
channels (`gfs_hrrr` via Open-Meteo) with no special weighting.

Their confidence rule is spread-based:

    SPREAD 0-2 deg  = HIGH CONFIDENCE
    SPREAD 3-5 deg  = MODERATE
    SPREAD 6+  deg  = DIVERGENT -- skip or size down

**This is the rule we have measured to be actively misleading.** On 2026-07-29
all eight of our guidance models agreed inside a 2.3F band (65.8-68.1) and were
collectively 2F wrong -- their rule would have graded that HIGH CONFIDENCE.
Agreement is not correctness; see section 9 and the guidance-reliability spec.

**Their own table shows the failure mode.** The forecast grid has a column
labelled MEAN H / MEAN L that is plainly carrying SPREAD values (DC shows
"86 68 1 0" -- 1 and 0 are degree spreads, not means), and the consensus cards
disagree with the grid for the same city (NYC grid mean high 78, card says 79).

# 21. A FALSIFIABLE PREDICTION TO CHECK TOMORROW

Their 2026-07-30 forecast, rated "✅ HIGH CONFIDENCE — models agree":

    LA   HIGH 89F   LOW 67F

KLAX has settled 79, 80, 79, 79 across 2026-07-25..28. **An 89F high would be
~10F above the recent run.** Either a genuine heat event is coming, or their
forecast layer has a problem on the exact city they rate C- REJECT.

Worth checking against `cli_reports` for KLAX 2026-07-30 once it settles. This
is the cheapest available test of whether their forecast layer is trustworthy,
and it costs nothing but a query.

---

# 22. ✅ THEIR SETTLEMENTS MATCH OURS EXACTLY — independent validation

Their NWS Settlement Reports page, coverage JUL 28 2026, cross-checked against
our `cli_reports` table:

    city          theirs (max/min)    ours (high_f/low_f)
    Miami KMIA        91 / 78             91 / 78     ✅
    Los Angeles       -- (below fold)     79 / 69
    New York KNYC     79 / 69             (not tracked)
    Chicago KMDW      79 / 68             (not tracked)

**KMIA 2026-07-28 MAX = 91 on both systems.** This is the day an ad-hoc
`MAX(tmpf_precise)` query in this project returned **89** and was reported to
Julian as contradicting him. He was right, `latentmax` was right, the six-hour
remark group (32.8C = 91.04F) was right, and now a completely independent
operator reading `api.weather.gov` agrees. Two systems, same truth source, same
answer -- see the project notes for the standing rule this produced.

# 23. ⚠️ WE DO NOT MODEL KALSHI'S FEES — AND THE GATE IS COMPUTED GROSS

Their Calculator page shows a worked example that exposes a real gap here:

    entry 65c, 10 contracts
    gross P&L        +$3.50
    ENTRY FEE        -$0.16
    net P&L          +$3.34
    breakeven win     65%

$0.16 reproduces Kalshi's published fee exactly: **0.07 x contracts x P x (1-P)**.
Verified: 0.07 x 10 x 0.65 x 0.35 = $0.159.

**The fee peaks at P = 0.50 -- precisely where a genuine edge trade lives:**

    price   fee as % of stake   gross return   net of fee
    0.50          3.50%           +100.0%        +96.5%
    0.62          2.72%            +61.3%        +58.6%
    0.80          1.40%            +25.0%        +23.6%
    0.87          0.91%            +14.9%        +14.1%

**Two concrete defects in our code:**

1. `staking.evaluate` compares `p_model - price >= min_edge` on a GROSS basis.
   At mid-range prices a "+5% edge" is really **~+3.3% net**. A third of the gate
   is silently eaten, and it is worst exactly where you would want to bet.
2. `appbet.py`'s `payout_if_win = stake / price` has no fee term, so every
   recorded payout in the ledger is optimistic by 1.0-3.5%.

Fix is small and verifiable: a fee function, and compare `min_edge` against net.
This is a GATING/DISPLAY defect, not a model change -- it alters when the desk
says "place", so treat it as decision-affecting and do it deliberately, not
mid-week.

---

# ⭐ EXECUTIVE SUMMARY — what to actually do

Ranked, after 23 sections of evidence:

**1. ADVECTION VETO** (§11). Wind direction + pressure tendency + sky -> GREEN /
PARTIAL / FULL. They rated LAX "WARM ADV FULL — AVOID" on 2026-07-29, the exact
day our LOW committed 67-68 at 81% and lost to a temperature that would not fall.
Physical mechanism, inputs already in `obs` except pressure tendency. **Now the
top model-side candidate, above `guidance_gap`.**

**2. KALSHI FEE MODEL** (§23). Smallest, most certain win. Our edge gate is gross;
fees peak at 3.5% right where edge trades live.

**3. PICK-OF-THE-DAY DISCIPLINE** (§13). One pick, price ceiling, publish NO PICK
when nothing qualifies. Their only honestly-measured feature and the closest
thing on their site to our edge gate.

**4. STORM-RISK / CLOUD-COVER VETO** (§4). Already researched here; they ship it.

**DO NOT COPY:**
- Their **85% win rate** (§1, §10) -- measured on collapsed books where every
  other bracket is at 1c. It is a 20-40 minute latency play against the order
  book, not forecasting. Buying at 95c and winning 85% LOSES 10c a contract.
- Their **spread=confidence rule** (§20) -- we have measured it to be actively
  misleading. Eight models agreed inside 2.3F and were 2F wrong.
- Their **order-flow features** -- real signal, but needs live book depth we do
  not have and probably should not chase.

**WHAT WE DO BETTER:** measured per-hour accuracy AND still-future curves,
calibration measurement, an immutable anti-cheat ledger, edge-vs-market recorded
at commit, and written negative results. Their surface is 20+ signal types and
one aggregate win rate.

**OPEN QUESTIONS:**
- §19 suspected climate-day boundary bug in their time series (NOT verified --
  read from a screenshot).
- §21 they forecast **LA HIGH 89F for 2026-07-30** at "HIGH CONFIDENCE" when KLAX
  has settled 79/80/79/79. Check `cli_reports` once it lands -- cheapest possible
  test of their forecast layer.
- §16 they rate LAX LOW **C- REJECT, rank 16/18** for marine layer, where our own
  survey ranked it #2 most predictable. Both may be true (low variance, high
  surprise risk) and the tension is worth resolving.
