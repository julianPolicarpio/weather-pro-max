# One-off measurement scripts

Analysis scripts kept because the numbers they produced are quoted in
`HANDOFF-2026-07-26.md` and `ACCURACY_IMPROVEMENTS.md`. **A claim nobody can
re-run is a claim nobody can check** — the LOW decision-hour figures in this
project were once produced by scratch scripts that no longer existed, and that is
why `highcast verify-hour` is now a real command.

Anything here that earns a permanent place should graduate into `src/highcast/`
with tests, the way `verify-hour` did.

Run from the project root: `.venv/bin/python scripts/<name>.py`

| script | question it answers |
|---|---|
| `hour_sweep.py` | Does training the nowcast **at the hour it predicts** beat the fixed `CHECKPOINT_HOURS` default? PROD vs MATCH for (low,3/6/20) and (high,3/9), single 2025-01-01 holdout, paired bootstrap. **Answered: yes — shipped.** Samples only 5 hand-picked hours; use `hour_curve.py` for the unbiased version. |
| `hour_curve.py [low\|high\|both] [n_boot]` | **At which hour is the shipped nowcast's answer most accurate?** All 24 hours, MATCH regime only, scored on MAE/CRPS **and modal-bucket hit rate** (what a bucket bettor actually consumes) + calibration. Source of `gating.HOUR_ACCURACY` and `MIN_DECISION_HOUR`. Buckets are **even**-based to match the live Kalshi ladder — `buckets.argmax_pair` is odd-based and does **not** line up; do not swap it in. |
| `tomorrow_regime.py <metric> [refit]` | How bad is the no-morning-observations regime the tomorrow card ships? WITH obs / WITHOUT-but-trained / **NAIVE** (train WITH, predict WITHOUT = production). |
| `recheck_agg.py` | Re-verifies the 6-hour-min ratchet claim **through the shipped `visible_running_extremum`** rather than a reimplementation. Reports days moved / fixed / broke. Edit `"min"`→`"max"` and `low_f`→`high_f` for the HIGH. |
| `late_undercut.py` | How much does the settled CLI low undercut the running min as of 02:00Z (~6h of climate day left)? Overall, July-only, JJA, and warm-night subsets. |
| `score_live_low.py` | Scores the LOW predictions the system **actually recorded live** at each decision cutoff against settled CLI values — a forward record, not a re-simulation. Thin until the live LOW history accumulates. |

## Caveats that apply to all of them

- `hour_sweep.py`, `hour_curve.py` and `tomorrow_regime.py` use a **single train/test holdout**,
  not walk-forward. Valid for the arm-vs-arm deltas they report; their absolute
  MAE/CRPS are **not** comparable to the walk-forward numbers in the handoffs.
- `build_checkpoint_frame` fits its `BaselineBlend` on the whole frame including
  future days (a known leakage smell, open item 4). It inflates absolute figures
  equally across arms, so comparisons stand and absolutes are optimistic.
- These read `data/highcast.db` directly and write nothing.
