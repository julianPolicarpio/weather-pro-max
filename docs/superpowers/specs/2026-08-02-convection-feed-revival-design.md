# Convection feed revival — "a storm is here" → "a storm is coming"

Written 2026-08-02. Scope: **KMIA only** (`forecast_conv` holds no other station).

---

## 0. What this is for

Every storm signal we currently have fires **at or after** the storm. None gives
lead time, so none can change a decision that was committed hours earlier.

| signal | lead time | state |
|---|---|---|
| `forecast_conv` (pop/qpf/cape) | **hours ahead** | **DEAD since 2026-07-29** |
| `obs.sky_cb` (cumulonimbus building) | minutes | live |
| `obs.wx_thunder` (thunder observed) | zero — it is happening | live |
| NWS storm alert | ~zero for our purposes | 0 rows, cannot backfill |

`forecast_conv` is the only one that could warn Julian **before** the commit
hour. It has 50,904 rows and stopped on 2026-07-29.

---

## 1. The defect

`scripts/fetch_convection.py` pulls hourly `precipitation_probability`, `qpf`
and `cape` from Open-Meteo's previous-runs API (leak-free by construction: the
`*_previous_day1` fields are what that model predicted a day earlier, so a
backtest cannot see the future). It works. **It is a manual script and was never
added to the wake cycle**, so the feed stopped the last time somebody ran it by
hand.

This is the SAME BUG `nbm_nbs` had. That one was cured with
`backfill._nbs_fetch_due` (src/highcast/backfill.py:116) and **the cure was never
generalised** — so the next feed to be added by hand died the same way, silently,
and nobody noticed for four days. `nbm_nbs` going quiet for five days drove the
day-ahead LOW 5–7 °F cold; this feed going quiet cost nothing only because
nothing consumes it yet.

---

## 2. Why it is worth reviving — measured, not assumed

KMIA lows, 2,383 climate days, split by whether thunder was observed. "Late
drop" = how much further the low fell **after** the ~12:00Z commit:

| | days | mean late drop | **fell ≥2 °F after commit** |
|---|---|---|---|
| thunder day | 666 | 1.37 °F | **24.2%** |
| quiet day | 1,717 | 0.44 °F | **6.3%** |

A thunderstorm day is ~**4× more likely** to move the low by a full 2 °F bucket
after we have committed. Note the median is 0.00 °F on both: this is a TAIL risk,
not a steady penalty. That shape is exactly what a warning is for and exactly
what a model feature would struggle with.

⚠️ That measurement uses OBSERVED thunder (`obs.wx_thunder`), which is available
only at or after the fact. The whole question this spec exists to answer is
whether the FORECAST (`forecast_conv`) identifies the same days in advance.
**It may not.** Convection is the hardest thing in short-range forecasting.

---

## 3. TASK 0 — the gate, run BEFORE anything is built

`HANDOFF-2026-08-03.md` §4.1: the bound-regime plan put its gate at Task 5, it
failed there, and three tasks of work were wasted. "It belongs at Task 0.
Running it early cost an hour and saved three tasks." This spec obeys that.

**Pre-registered question:** on days `forecast_conv` predicted convection, does
the low actually drop late more often than on days it did not?

**Arms**, on the 2024-07-29 → 2026-07-29 range the feed already covers:
- `pop` above a threshold during the station's afternoon
- `cape` above a threshold
- a shuffled control (the same labels, randomly reassigned to days)

**Score:** rate of "fell ≥2 °F after commit", forecast-positive days vs
forecast-negative, with a paired bootstrap CI.

**The bar:** the real arm must beat the shuffled control, and its separation must
be a meaningful fraction of the 24.2% vs 6.3% that OBSERVED thunder achieves. If
the forecast only recovers a sliver of what the observation shows, then
convection is not predictable at this lead time for this station, and the honest
deliverable is a null result plus keeping the observed-thunder warning.

**A null result here is a valid and expected outcome.** Do not tune thresholds
until the numbers look good — the sample is ~730 days and a swept threshold on a
rare event fits noise trivially.

`scripts/convection_eval.py` already exists and cannot run until the feed is
alive; check whether it already answers this before writing anything new.

---

## 4. The cure

### 4.1 Wire the fetch into the wake cycle

Add a `_conv_fetch_due(conn, cfg)` alongside `_nbs_fetch_due`, keyed on
`max(fetched_at)` from `forecast_conv` **per station**, failing OPEN on an
unparseable stamp (one extra HTTP call is cheaper than a dead feed).

**GENERALISE, do not copy.** Two near-identical due-checks is how the third one
gets written wrong. This project has been bitten four times this week by
one-truth-two-copies: `orderbot.CONFIDENCE_FLOOR` shadowing `gating`'s,
`fc_anchor` re-entering the blend it was derived from, a settlement predicate
that guarded fetching but not grading, and 13 scripts reading a config field that
had become per-metric. Extract one `_feed_fetch_due(conn, table, station,
model=None, max_age_hours=…)` and have both callers use it.

### 4.2 KMIA only

`forecast_conv` has no KLAX rows. The due-check must be per-station, and a
station with no configured convection source must be a no-op — never a silent
fallback to another station's data.

---

## 5. The staleness guard — the part that outlives this fix

Reviving one feed fixes one feed. **The class of bug is "a feed dies and nothing
says so."** It has now happened twice (`nbm_nbs`, `forecast_conv`).

Add a check that reports, per feed, how long since its last successful fetch,
and surfaces anything past its expected cadence. Cheap options: a line in the
existing wake-cycle output, or a Discord warning reusing the `alert_stamps`
exactly-once pattern from `notify.notify_unstable`.

**This is worth more than the convection data itself.** Without it, feed #3 dies
the same way.

---

## 6. What this unblocks

1. **"A storm is coming"** — `notify.notify_unstable` currently fires on NWS
   alerts, the worst of the four signals (no lead time, no history). With a live
   convection forecast it can warn BEFORE the commit hour, which is the only
   version Julian can act on.
2. `scripts/convection_eval.py`, already written, currently unrunnable.
3. The handoff's own #1 next item, and the one lead on the board that could
   **create** an edge rather than remove an error — convection is information a
   temperature model structurally lacks (it has NO precipitation input) and the
   ladder may underprice it.

---

## 7. Non-goals

- **Do NOT add convection as a feature to the temperature model** on the strength
  of this. That is a separate change with its own gate. The bound-regime layer
  looked strong in-sample and failed out of sample; a rare-event feature is at
  least as easy to fool yourself with.
- Do not extend to KLAX. 22 thunder days in six years — there is nothing to learn.
- Do not wire convection into `decision_gates` as an automatic refusal. We cannot
  justify auto-skipping: near-boundary days still hit 72%, and skipping is only
  correct if the market has not already discounted them, which needs price history
  we do not have (9 priced days). **Warn, do not block.**

---

## 8. Verification

- `.venv/bin/python -m pytest -q` — the suite must stay green.
- A test that `_feed_fetch_due` (or its generalisation) returns True on a stale
  feed and False on a fresh one, per station.
- A test that a station with no convection source is a no-op rather than
  borrowing another station's rows.
- After one wake cycle, `max(fetched_at)` in `forecast_conv` must be recent —
  the check that would have caught the original defect.
- Any measurement runs against a **copy** of the DB (`db.connect` executes DDL
  and the wake cycle writes every 5 minutes).
