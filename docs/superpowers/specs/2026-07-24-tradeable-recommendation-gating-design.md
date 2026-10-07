# Tradeable Recommendation Gating & Scoring — Design

**Date:** 2026-07-24
**Status:** DESIGN — awaiting user review before an implementation plan is written.
**Author:** brainstormed with Julian (main loop), grounded in a July 22 post-mortem.

## Problem — the July 22 post-mortem

Julian reported that "all the predictions were wrong for July 22": the dashboard
showed a confident-looking recommendation (~"83–84 @ 51%" early, "~81–82 @ 60%"
later) while the day actually settled cool.

Investigation of `data/highcast.db` shows the **model was not badly broken** — it
converged correctly. The failures were in *how the pick is presented and judged*:

- **Ground truth, July 22:** CLI settled high = **77°F**; the raw METAR peak was
  also exactly **77.0°F** (the sensor never reached 79–80). So every number in
  the 79–84 range that was on screen was above the truth.
- **What the model actually stored:** the intraday pick **oscillated** between
  adjacent 2°F buckets all morning/midday (`77–78` → `79–80` → `75–76` → `77–78`)
  at **40–62 % confidence**, then correctly ratcheted to `77–78 @ 94 %` by LA
  evening. It ended right; it spent the whole *useful* part of the day unsure.
- **Why the on-screen numbers looked hot:** a hot forecast regime inflated the
  early distribution (the ECMWF member for July 22 ran **86–88°F** vs. a 77°F
  actual), and the dashboard's "today" resolves to the **LA date** — at Julian's
  Manila midday the board is showing the *previous* LA day (July 20/21 were
  genuinely hotter: actuals 83 and 81, with picks in the low-80s). The remembered
  "83–84 / 81–82" match those hotter days' picks, not July 22's.

**Root cause (two defects, both in presentation/judgment, not the forecaster):**

1. **False precision.** The dashboard surfaces a single 2°F "pick + %" *at all
   times*, including the long 40–62 % oscillating phase before the value has
   formed. A pick shown at 51 % reads as advice but is a coin flip.
2. **No notion of "too late."** The pick only becomes trustworthy (**≥80 %,
   stable**) *after peak heating* — LA ~15:00–18:00 = **Manila ~06:00–09:00 next
   morning** — by which point Julian is asleep *and* the running-max ratchet has
   already revealed the answer to everyone. A "correct" call there wins nothing,
   yet today's scorecard would still credit it as a hit.

## Goal

Change the recommendation from *"always show the argmax bucket"* to *"only show a
pick once it is trustworthy, and judge the model on whether that pick arrived in
time to be **tradeable** — not on whether it was eventually correct."*

Concretely:

1. **Gate the display** — suppress the 2°F pick until it is *confident* and
   *stable*; otherwise show the raw distribution, a **countdown** to the expected
   confident time, and Julian's personal **cut-off marker**.
2. **Score for tradeability** — a settled day is a **WIN** only if the model
   produced a confident, correct pick **on time** (before the tradeable window
   closes). A correct pick that arrives after the window is scored **LATE
   (loss)** — "right but worthless."

This is not a request to retrain the forecaster. The scoreboard is expected, at
first, to show a low win-rate on July-22-style days — that is the system
correctly reporting that the model is not yet *tradeable on Julian's schedule*.
"Get to ≥80 %-stable earlier (from morning obs, before the afternoon ratchet)"
becomes the explicit, measurable improvement target that this scoreboard exposes.

## Confirmed facts (from the investigation)

- Predictions are stored per `(station, target_date, issued_at_utc, kind, metric)`
  with a per-degree `buckets` pmf (JSON) — see `db.py`. The 5-min `wake` cycle
  writes an intraday row each run through the climate day.
- The dashboard's temperature "pick" is `_model_pick` / `_pmf_pick` in
  `web/app.py`: the **argmax odd-based 2°F bucket** (`buckets.bucket_pair`) and
  its summed mass. `scoring.score_day` computes a *near-identical* pick a second
  way (mode → `bucket_pair`). **These two pick definitions must be unified.**
- Confidence empirically crosses ~80 % only post-peak: the `clock.SHARPNESS`
  table (CRPS 1.13 day-ahead → 0.60 noon → 0.33 4pm → 0.07 6pm) and the July
  20/22 stored picks agree — ≥85 % appears at LA ~15:00–18:00.
- `latentmax.visible_running_max(conn, cfg, day, asof_utc=…)` already returns the
  observed running-max interval as of any instant — the exact primitive needed to
  compute when the answer became "publicly visible."
- **Manila 01:30 ≈ LA 10:30 of the same climate day** (Manila is 15 h ahead of
  PDT). This is *before* peak heating — so Julian's cut-off is the binding
  deadline, always earlier than the thermometer-reveal ceiling in summer.
- Config is a frozen `AppConfig`/`StationConfig` loaded from `config/klax.yaml`
  (`config.py`). `display_tz` is already `Asia/Manila`.

## Key definitions

Let `THRESH` = confidence threshold (default **0.80**), `N` = stability count
(default **3** readings ≈ 15 min), `CUTOFF` = personal Manila wall time (default
**01:30**), `REVEAL_MARGIN` = °F (default **1.0**).

- **Pick(p)** — for a stored prediction `p`, the argmax odd-based 2°F bucket
  `(label, lo, hi)` of its pmf and that bucket's summed probability `prob`.
  (Single shared definition; see Component 1.)
- **Confident(p)** — `Pick(p).prob ≥ THRESH`.
- **Stable(p, history)** — `Pick(p).label` is unchanged across `p` and the `N-1`
  immediately-preceding intraday predictions for the same `(target_date)`.
  Fewer than `N` readings exist ⇒ not stable.
- **Committed(p, history)** — `Confident(p) AND Stable(p, history)`. This is the
  gate: a pick is only shown / counted once it is *committed*.
- **Personal-cut-off instant** `t_cut(D)` — the unique UTC instant whose
  `display_tz` wall-clock equals `CUTOFF` and which falls inside climate day `D`
  (`[start, end)` from `timeutil.climate_day_bounds`). For KLAX/Manila that is
  Manila `01:30` of calendar date `D+1` ≈ LA `10:30` of `D`.
- **Reveal instant** `t_reveal(D)` — (scoring only; needs the settled high) the
  earliest instant `t` in climate day `D` at which
  `round_half_up(visible_running_max(…, asof=t).hi) ≥ actual_high − REVEAL_MARGIN`.
  Before this, the crowd can't yet read the answer off public obs; after it, the
  edge is gone. `None` (no reveal reached) if the high is a hidden spike.
- **Deadline** `t_deadline(D)` = `min(t_cut(D), t_reveal(D))` (drop `None`s).
- **The call** — the *last* intraday prediction with `issued_at ≤ t_deadline(D)`
  that is `Committed`. `None` ⇒ the model never committed in the tradeable window.

## Day classification (scoring)

For each **settled** target date `D` (has `cli_reports.high_f`), `classify_day`
returns a three-valued `outcome` plus a boolean `late` flag:

| `outcome`  | Condition |
|------------|-----------|
| **WIN**    | a call exists **and** its bucket contains `actual_high` |
| **WRONG**  | a call exists **and** its bucket does **not** contain `actual_high` |
| **NO_CALL**| no `Committed` pick at/before `t_deadline(D)` |

`late` (only meaningful on `NO_CALL` days) = the model *did* become
`Committed`-and-correct, but only *after* `t_deadline(D)` (its first correct
committed pick has `issued_at > t_deadline`). It flags the "right but too late"
pile — days the model *could* have helped had it been faster (or had Julian
stayed up). A `NO_CALL` with `late=false` means never-confident-and-correct at
all. `WIN`/`WRONG` never set `late` (a call already existed on time).

Reported metrics: **win-rate over all settled days** (`WIN / all`), **win-rate
over days-with-a-call** (`WIN / (WIN+WRONG)` — the model's accuracy *when* it
commits in time), **tradeable coverage** (`(WIN+WRONG) / all`), and the **LATE
count** (`NO_CALL` days with `late=true`).

## Design

### Component 1 — one shared "argmax 2°F bucket" helper (`buckets.py`)

Add `argmax_pair(pmf) -> {label, lo, hi, prob}` and delete the two divergent
inline pick computations (`web.app._pmf_pick`, and the mode-based pick inside
`scoring.score_day`). Everything downstream — display gate, scoring — reads the
same definition, so the number on the dashboard and the number the scoreboard
judges can never disagree. `score_day`'s existing `tail_floor`/`tail_label`
fields stay.

### Component 2 — the commit gate (`scoring.py`)

- `pick_series(conn, cfg, target_date) -> [ {issued_at, label, prob, committed} ]`
  — the chronological intraday picks for a day with `Confident`/`Stable` flags
  computed over the rolling window (`THRESH`, `N` from config).
- `committed_at(pick_series, at_instant) -> pick | None` — last committed pick
  with `issued_at ≤ at_instant`.
- `classify_day(conn, cfg, target_date) -> {outcome, call, actual, t_cut,
  t_reveal, t_deadline}` — implements the table above.
- `tradeable_scorecard(conn, cfg) -> {records, summary}` — `classify_day` over
  every settled date + the rollup metrics. This is the tradeable counterpart to
  the existing `scorecard`/`summarize` (which stay, unchanged, as the raw
  eventual-accuracy view).

### Component 3 — reveal & deadline helpers

- `latentmax.reveal_instant(conn, cfg, day, actual_high, margin)` — scans obs in
  climate-day order, returns the first instant the visible running-max hi is
  within `margin` of `actual_high` (reuses `visible_running_max`'s candidate
  logic; `None` if never reached).
- `clock.personal_cutoff_instant(cfg, day)` — `t_cut(D)` as defined above, plus a
  formatter for the Manila/LA cut-off strings the dashboard shows.

### Component 4 — display gating (`web/app.py`, `/api/timing` + `/api/today`)

Replace the always-on `_model_pick` with a **gated recommendation** object:

```
recommendation = {
  "committed": bool,                 # is there a shown pick right now?
  "pick": {label, prob} | null,      # present ONLY when committed
  "best_so_far": {label, prob},      # current argmax even when not committed
  "status": "committed" | "forming" | "no_obs",
  "expected_confident": {            # the countdown target (see below)
     "la": "...", "manila": "...", "eta_minutes": int | null },
  "personal_cutoff": {               # Julian's binding deadline
     "la": "10:30", "manila": "01:30", "passed": bool },
  "past_cutoff_note": str | null,    # shown once now > cutoff and still forming
}
```

- **Committed** ⇒ show `pick` as the recommendation (unchanged look).
- **Forming** ⇒ **no pick**; show `best_so_far` clearly marked provisional, the
  raw distribution (already rendered), and the **countdown** to
  `expected_confident`.
- `expected_confident` = the historical **median time-of-day** at which
  `Committed` is first reached, computed from settled days **when ≥ `min_days`
  (default 20) are available**, then projected onto today's climate day for the
  countdown; otherwise a config fallback (`expected_confident_la_hour`, default
  15:00 LA). Always labeled "typical," never a promise.
- New endpoint `GET /api/tradeable` → `tradeable_scorecard(...)` for the
  scoreboard panel (WIN/WRONG/NO_CALL/LATE per day + the four summary metrics).

### Component 5 — config (`config.py`, `config/klax.yaml`)

New frozen `RecommendationConfig` on `AppConfig`, loaded from an optional
`recommendation:` block (all fields defaulted, so existing configs keep working):

```yaml
recommendation:
  confidence_threshold: 0.80      # THRESH — top 2°F bucket mass to be "confident"
  stability_readings: 3           # N — consecutive readings the pick must hold
  personal_cutoff: "01:30"        # CUTOFF — last actionable time, in display_tz
  reveal_margin_f: 1.0            # REVEAL_MARGIN — running max within this = revealed
  expected_confident_la_hour: 15  # countdown fallback when history is thin
  expected_confident_min_days: 20 # settled days needed to use the historical median
```

### Component 6 — templates (`web/templates/index.html`)

- Recommendation card: render the committed pick, or the "forming — not confident
  yet" state with the countdown and the `01:30` cut-off marker + `best_so_far`.
- New "Tradeable scoreboard" panel bound to `/api/tradeable`: per-day
  WIN/WRONG/NO_CALL/LATE chips and the summary line (win-rate, win-rate-when-
  called, coverage, LATE count).

## Data flow

```
obs ──► latentmax.visible_running_max(asof) ──► reveal_instant ─┐
predictions(intraday) ──► argmax_pair ──► pick_series ──► committed_at ─┤
                                                                        ▼
cli_reports.high_f (actual) ─────────────────────────► classify_day ──► outcome
                                                                        │
config.recommendation (THRESH,N,CUTOFF,MARGIN) ─────────────────────────┘
                                                                        │
                              ┌──────────────────────────┬─────────────┘
                       /api/tradeable (scoreboard)   /api/timing+today (gated live rec)
```

## Error handling & edge cases

- **No obs / no predictions for a day** ⇒ `NO_CALL`; `status:"no_obs"` live.
- **Mac asleep → gaps** (observed July 22, Manila ~01:00–02:27): `committed_at`
  uses the last prediction *before* the deadline; if none is committed, `NO_CALL`.
  Stability is over *consecutive stored* rows, so a gap simply resets the run.
- **Hidden spike** (`actual_high` above every observation) ⇒ `t_reveal = None`;
  deadline falls back to `t_cut`. `latentmax.hidden_spike_pmf` already quantifies
  how often this happens.
- **Fast-heating day** where `t_reveal < t_cut` ⇒ deadline = `t_reveal` (edge gone
  before the cut-off). Rare in summer (peak is Manila 03:00–07:00, after 01:30)
  but handled by the `min`.
- **DST:** Manila has none; LA does. `t_cut` is derived by converting the Manila
  wall time for the correct calendar date and asserting membership in
  `[start,end)` (climate day is 01:00→01:00 local during DST), so it stays correct
  year-round.
- **Config back-compat:** missing `recommendation:` block ⇒ all defaults; the low
  path (`metric.py`) is untouched — gating is metric-agnostic and reads the same
  block.

## Testing

- `tests/test_buckets.py` — `argmax_pair` (ties, even/odd keys, tails).
- `tests/test_scoring.py` — `Confident`/`Stable`/`committed_at`; `classify_day`
  for each outcome using **the July 22 fixture** (must classify as `NO_CALL`
  with `LATE=true` under CUTOFF 01:30: committed-correct only at LA ~15–17 =
  after the deadline) and a `WIN` fixture (a day that commits before 10:30 LA);
  deadline = `min(cut, reveal)` incl. the `reveal < cut` branch and the
  `reveal=None` branch.
- `tests/test_latentmax.py` — `reveal_instant` (reached / never-reached / exact
  margin boundary).
- `tests/test_web.py` — `/api/timing` returns a gated rec (no `pick` when
  forming, `pick` present when committed) and `/api/tradeable` shape. Hermetic
  fixtures only — no live Kalshi/obs fetches.
- Full suite must stay green; the existing `scorecard`/`/api/scorecard` behavior
  is unchanged (raw-accuracy view retained alongside the new tradeable view).

## Non-goals (YAGNI)

- **No market-price edge model.** Kalshi prices aren't stored historically, so
  "edge" is proxied by the thermometer-reveal, not by real mispricing. (A future
  upgrade could log market prices and compare, but not here.)
- **No auto-trading, sizing, or financial advice** — display + scoreboard only.
- **No forecaster retraining** in this change. This work *measures* the
  earliness gap; closing it (confident-by-morning model) is a separate project.
- **No new metric.** Gating/scoring apply to the existing `high` path and will
  carry to `low` for free via the shared, metric-agnostic helpers.

## Open items for review

1. Defaults sane? (`THRESH 0.80`, `N 3`, `CUTOFF 01:30`, `REVEAL_MARGIN 1.0`.)
2. Is **WRONG-when-called** (model committed on time but the 2°F bucket missed)
   worth surfacing as its own chip, or folded into "not a win"? (Spec keeps it
   separate.)
3. Keep the old always-on `/api/scorecard` raw-accuracy view, or replace it
   entirely with the tradeable one? (Spec keeps both.)
