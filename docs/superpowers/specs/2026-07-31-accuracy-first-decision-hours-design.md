# Accuracy-first decision hours and the gates a bot will stand on

**Date:** 2026-07-31
**Status:** design, approved in brainstorming; not implemented
**Scope:** phases 0 and 1 of a six-phase programme (see *Sequencing*)

---

## 1. Why

Julian's instruction: *prioritise accuracy and predictions instead of edge*. Two
bots are wanted downstream — an autonomous entry bot for the two HIGH markets and
a semi-automatic exit bot — but neither should be built on the current decision
hours, because those hours were chosen under a different objective and the model
is measurably overconfident at them.

This spec covers **only** the accuracy foundation. It ships nothing that can
spend money.

## 2. The finding this rests on

`MIN_DECISION_HOUR` currently commits at KLAX LOW h4, KMIA LOW h5, both HIGHs
h11. Those hours were deliberately moved EARLY (2026-07-26) to escape hours that
were "transcription, not forecasting". The correction over-shot: it landed on
hours where very little of the extremum has happened AND the model is dishonest
about it.

Measured 2026-07-31 (walk-forward, refit 45, RECENT ≥ 2025-01-01, n≈570/arm):

| arm | bucket hit | overall shortfall (overconfidence) |
|---|---|---|
| KLAX HIGH h11 | 53.4 % | **+4.8 %** |
| KLAX HIGH **h13** | 69.5 % | **−0.6 %** |
| KMIA HIGH h11 | 56.1 % | **+3.1 %** |
| KMIA HIGH **h14** | 79.7 % | **−1.9 %** |

**Moving the hour does not merely raise accuracy; it removes the
overconfidence.** Both HIGHs go from overstating themselves to slightly
understating themselves.

The proposed hours are not arbitrary. They are each station's **median
peak-heat hour**, measured from the CLI reports' own `high_time`
(KMIA n=2,366, median 13:58; KLAX n=1,959, median 13:05). The bot commits at the
moment the high is a coin flip between "already set" and "still to come" — about
half the extremum still ahead, so it remains a forecast rather than a
transcription.

### Proposed hours

| market | now | proposed | still ahead | Manila (summer) |
|---|---|---|---|---|
| KMIA LOW | h5 | **h7** | 41.7 % | 19:00 |
| KLAX LOW | h4 | **h6** | 49.9 % | 21:00 |
| KMIA HIGH | h11 | **h14** | 49.7 % | 02:00 |
| KLAX HIGH | h11 | **h13** | 51.7 % | 04:00 |

Julian takes both LOWs manually; the bot takes both HIGHs. Manila times shift one
hour later in winter, so the scheduler MUST compute from the station timezone and
never from a fixed Manila clock.

## 3. Phase 0 — move the decision hours

`MIN_DECISION_HOUR` is a shipped constant gating real decisions, so it gets a
pre-registered gate. **Written before running:**

- **Candidates:** the four hours in the table above.
- **Primary outcome:** parity-averaged bucket hit at the candidate hour minus the
  incumbent, walk-forward (`calib_eval`, refit 30), RECENT window.
- **Ship threshold, all must hold:**
  1. Δhit > 0,
  2. the CI on Δhit excludes zero at Bonferroni ×4,
  3. ≥ 40 % of the extremum still ahead at the candidate hour.
- **Each market is decided independently.** A market that fails keeps its hour.

**Expected failure mode, stated up front:** the walk-forward estimator reads more
pessimistic than the single-holdout `hour_curve` numbers above. It is entirely
possible a market fails. That is the gate working.

### What changes if a market passes

- `gating.MIN_DECISION_HOUR[station][metric]`
- the `HOUR_ACCURACY` commentary block that quotes the knee's hit rate
- `gating.BAND_CALIB` for that market, **re-derived at the new hour** (the
  current bands were fitted at h11 and do not transfer)

`staking.personal_cutoff` / `low_personal_cutoff` need **no change**: the
invariant `test_the_window_admits_the_decision_hour` requires
`cutoff_local_hour > knee`, and the existing cutoffs (KLAX LA 18:00/13:00, KMIA
Miami 21:00/16:00) already sit after every candidate knee. Verified 2026-07-31.

## 4. Phase 1 — the refusal gates

### Gate A — HIGH preflight: DOES NOT EXIST, and will not be built

A preflight grade was a declared fire condition for the bot. It cannot be built
honestly. `scripts/preflight_high_eval.py` screened five admissible factors —
`bound_gap`, `stall_h`, `override` (|model point − base_p50|), `guid_churn`
(|guidance at lead-1 − lead-2|), plus `seam` for KLAX and `cape` for KMIA —
across both markets at both the current and proposed hours, Bonferroni-corrected
to 99.55 % over 11 tests.

Candidates were restricted by the same rule the shipped preflight used: a factor
is admissible only if the model **cannot already see it**. `cur_tmpf`,
`cur_dwpf`, `cur_onshore`, `cur_sky`, `base_p50`, `base_spread`, the ratchet
(`run_lo`/`run_hi`/`run_width`) and `heat_alert` are all inside the feature set
and inadmissible alone.

**Result: exactly one factor cleared** — KMIA HIGH `guid_churn` at h11, Δ+8.8 %,
CI [+0.6 %, +16.6 %]. **It evaporates at h14** (Δ−1.2 %, CI [−7.9 %, +5.4 %]),
the hour we are moving to, so it cannot serve the bot.

The reason nothing separates at the later hours is §2: a preflight catches days
when the model stays confident while being wrong, and at h13/h14 that behaviour
has largely stopped. **The hour move is the preflight.** Gate A is dropped.

`cape` was **untestable**, not negative — though the reason stated here was
itself wrong, and was corrected 2026-08-01. It is not that `forecast_conv`'s rows
"carry recent `fetched_at`": it is that `forecast_conv`'s PRIMARY KEY EXCLUDES
`fetched_at`, so INSERT OR REPLACE **destroys** the original write time on every
re-fetch. As-of reconstruction is impossible on that table by construction, not
by coincidence. (`forecast_hourly` and `forecast_intraday` DO key on
`fetched_at`, which is exactly why they can answer freshness questions and the
older tables cannot.) Convection remains a live
hypothesis for KMIA and needs capture-forward data like `forecast_hourly`.

### Gate B — calibrated-confidence floor

Machinery exists (`gating.BAND_CALIB`, `band_calib()`,
`staking.evaluate(calib=…)`), including the behaviour that already blocked all
three losing bets: above the highest measured band `band_calib()` returns `None`,
which `staking.evaluate` turns into a refusal.

**New work:** re-derive the bands at the new hours, and fix the floor by a rule
declared here rather than tuned later.

**Floor rule (policy target, stated not fitted):** the bot may only act when it
is measurably right **4 days in 5**. So per market the floor is the *lowest*
stated-probability threshold whose realized hit rate has a **95 % lower bound
≥ 0.80**. A market with no such threshold gets **no floor and the bot never
fires there**.

Applied to the measured table (verified 2026-07-31):

| market | admissible floor | fires on | realized hit |
|---|---|---|---|
| KLAX HIGH h13 | **none** — every floor's lower bound is 63.9–66.1 % | never | — |
| KMIA HIGH h14 | **0.70** (lower bound 81.9 %) | 75.3 % of days | 85.5 % |

⚠️ **KLAX HIGH is excluded by its own numbers**, independently of the price
evidence below — which reaches the same conclusion by a different route. Two
independent lines of evidence agreeing that the bot should not trade KLAX HIGH is
the strongest result in this spec. It is the expected outcome, not an obstacle to
work around: do not lower the target to admit it.

Evidence (n≈570/arm):

| arm | floor | fires on | realized hit | max justified price |
|---|---|---|---|---|
| KLAX HIGH h13 | p ≥ 0.70 | 47.2 % of days | 71.4 % | 69¢ |
| KMIA HIGH h14 | p ≥ 0.80 | 44.9 % of days | **93.4 %** [89.6, 95.8] | **91¢** |

### Gate C — data freshness

New: `gating.guidance_lead_ready(conn, cfg, target_date)`, mirroring the existing
`nbs_guidance_ready`. Refuses when the newest guidance for the target is lead-2.

⚠️ **CORRECTED 2026-08-01 — this gate's justification was withdrawn.** The
lead-2 evidence below came from a query using `fetched_at`, which `db.upsert`
(INSERT OR REPLACE, PK excluding that column) overwrites on every re-fetch. The
cold bias is real but its cause is the all-NaN `morn_*` regime that
`morning_obs_ready` already detects (+0.31 °F with morning obs, n=18; −5.35 °F
without, n=13). The gate is kept as a precautionary fail-closed check only.

Original rationale, superseded: the evening `dayahead_tomorrow`
run fires at 17:00 local, increasingly sees only two-day-old guidance, and reads
**−4.86 °F** at KLAX HIGH (MAE 5.40) versus +0.06 °F at the 16Z run. On
2026-07-30 it printed 70.2 °F when its own coldest guidance member said 74.7 °F.
No unattended bot may run without this gate.

**Bundled here:** invert `kalshi.resolve_price` to prefer a live two-sided book
over `last_price`. 16.1 % of stored quotes sit outside their book. The deviation
is small (median 1¢) and was shown NOT to explain the losing bets, but a bot must
never act on a stale print. Re-check `staking.max_spread` at the same time — the
KMIA ≥81 book was 0.03/0.11, 8¢ wide.

### The price rail (carried into phase 4, recorded here)

The bot must refuse when `price ≥ calibrated_p − fee`. This is a
loss-prevention floor, not edge-hunting: it does not reward large gaps.

Live prices at the proposed hours show why it matters, and what the bot will
actually do:

| market | favored bucket price | model justifies | outcome |
|---|---|---|---|
| KLAX HIGH h13 (n=6) | 0.75–0.96, median ~0.90 | ~69¢ | bot sits out nearly always |
| KMIA HIGH h14 (n=3) | 0.50 / 0.79 / 0.82 | 91¢ | genuine room |

⚠️ **n=3 and n=6.** This is the weakest evidence in the programme. The bot is
scoped to both HIGHs and lets the rail decide, rather than hardcoding today's
sample.

## 5. Composition and error handling

One aggregator, `gating.decision_gates(...)`, returning a structured verdict —
`ok` plus a list of named refusals — **not** scattered `if` statements. Same
reasoning as `board.evidence()` being assembled in Python: the dashboard, the
terminal and later the bot must read one verdict or they drift and produce advice
that disagrees with itself.

**Every gate fails CLOSED.** Missing data, an exception, or an unmeasured market
all produce a refusal, matching `nbs_guidance_ready`'s existing contract. For an
unattended trader, "I could not tell" and "no" must be the same answer.

## 6. Testing

- One regression test per gate asserting **refusal in the bad regime**: the
  lead-2 day, the stale-price day, the ≥0.80 unmeasured band. These are real
  2026-07-31 bugs turned into tests.
- A test that the aggregator is the only path — no gate can be bypassed.
- `test_the_window_admits_the_decision_hour` must still pass at the new knees.
- Freeze tests: `bankroll` stays 0, `app_bets` stays paper, nothing reads
  `forecast_hourly`.
- `scripts/preflight_high_eval.py` must keep choosing its feature set via
  `features_for` (guarded by `tests/test_eval_scripts.py`).

## 7. Out of scope

All trading. No authentication, no order placement, no portfolio reads, no bot.
`bankroll` stays 0 and `app_bets` stays paper for this entire phase, so the
app-vs-Julian scorecard keeps measuring the model rather than a filter.

The later phases, recorded so the sequencing is not lost:

| phase | what |
|---|---|
| 2 | Kalshi trading client: request signing, portfolio read, order placement, dry-run default |
| 3 | Exit bot (semi-auto): ratchet-kill trigger first, then ratchet-adjacent behind its own gate |
| 4 | Entry bot for both HIGHs: $1 and one order per day total, maker orders preferred |
| 5 | Intrinsic model work: `gfs`/`hrrr` duplicate-channel gate, evening-minimum feature |

## 8. Decisions taken in brainstorming

- All three accuracy levers, staged, each behind its own pre-registered gate.
- Julian trades both LOWs manually (19:00 / 21:00 Manila); the bot is **built
  for** both HIGHs (02:00 / 04:00 Manila) and lets the gates decide which it
  actually trades. On current measurement that is **KMIA only** — KLAX HIGH is
  excluded by both the floor rule (§4 Gate B) and the price rail (§4). Building
  for both avoids hardcoding an n=6 price sample, and costs no extra code.
- Entry bot: **$1 and one order per day in total** across both HIGHs — it picks
  the stronger, and fires at most once.
- Fire conditions: calibrated-confidence floor + freshness gates + the
  never-negative-EV price rail. (Preflight was chosen too, but §4 Gate A shows it
  cannot be built; the hour move replaces it.)
- Exit bot: semi-automatic — it stages the order, Julian confirms, it submits.
  Triggers are **ratchet-kill** (certainty: the observed extremum has already
  passed the bucket edge, so it cannot settle there) and **ratchet-adjacent**
  (alive but the path is against it — needs its own measured gate). Explicitly
  NOT model-flip or market-move triggers.
- Position source: Julian's real Kalshi portfolio, via the signed API that
  semi-auto already requires.

## 9. Boundary

Claude writes this code. Claude does not run the live-trading path, does not
handle or enter Julian's API credentials, and does not place trades. Julian
installs his own credentials and arms the bot himself.

## 10. Honest limits

- Every hit rate in §2 is walk-forward on ≈570 days, but the **price** evidence in
  §4 is n=3 and n=6. Accuracy is measured; edge is not.
- The `guid_churn` signal that cleared at KMIA h11 is 1 of 18 tests at a CI lower
  bound of +0.6 %. It is recorded, not shipped, and it does not reproduce at h14.
- Phase 0 may fail for one or more markets, which would change the bot's hours
  and could remove KMIA HIGH's measured room along with them.
- This project is 3-for-9 on confident pre-measurement hypotheses.
