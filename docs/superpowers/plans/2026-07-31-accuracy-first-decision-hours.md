# Accuracy-First Decision Hours and Refusal Gates — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move each market's commit hour to its measured accuracy-honest hour, and add the two refusal gates a future bot will stand on — behind one fail-closed aggregator.

**Architecture:** Task 1 is a pre-registered measurement that decides which markets move; Task 2 applies it. Tasks 3–6 build the gates as pure, independently testable functions. Task 7 composes them into `gating.decision_gates`, the single verdict every surface reads. Task 8 wires that verdict into the display and locks the freeze contracts.

**Tech Stack:** Python 3.12, LightGBM, pandas/numpy, SQLite, pytest, Flask (dashboard).

**Spec:** `docs/superpowers/specs/2026-07-31-accuracy-first-decision-hours-design.md`

## Global Constraints

- **THIS PROJECT IS NOT A GIT REPOSITORY. Never run `git` commands and never offer to commit.** The writing-plans template ends each task with a commit step; that step is replaced here by a full-suite verification step. This is a standing project rule and it overrides the skill.
- **No trading code in this plan.** No authentication, no order placement, no portfolio reads. `staking.bankroll` stays `0.0` in both configs and `app_bets` stays paper.
- Run everything with `PYTHONPATH=src .venv/bin/python`.
- Full suite baseline before starting: **614 passed, 12 deselected**. Any task that ends with a different pass count must explain why.
- **Never derive a daily high/low from raw `obs` snapshots.** Use `latentmax.visible_running_extremum` / `visible_running_max`, or include `six_max_c` / `six_min_c` in the SQL. (`CLAUDE.md`)
- **Two stations share one SQLite file.** Anything keyed by `metric:date` alone is a latent cross-station bug — always key by station too.
- Kalshi's 2 °F ladder parity is a coin flip per day. Never hardcode even or odd; score with `buckets.expected_bucket_hit`.
- Any script that fits an `IntradayModel` MUST choose its feature set via `models.intraday.features_for` (guarded by `tests/test_eval_scripts.py`).
- Every gate **fails closed**: missing data, an exception, or an unmeasured market all produce a refusal.

---

### Task 1: Pre-registered decision-hour move gate

The spec's §3 gate. This is a **measurement that produces a decision**, not a shipped constant. It must be written and read before Task 2 changes anything.

**Files:**
- Create: `scripts/decision_hour_move_eval.py`
- Test: `tests/test_eval_scripts.py` (existing guard must still pass — no edit expected)

**Interfaces:**
- Consumes: `models.intraday.features_for`, `build_checkpoint_frame`, `IntradayModel`; `buckets.expected_bucket_hit`; `models.baseline.BaselineBlend`; `features.build_features`, `uses_bias_correction`.
- Produces: printed per-market verdict `MOVE` / `KEEP` for the four candidate hours, consumed by Task 2 as a human decision. Writes no constant.

- [ ] **Step 1: Write the script**

Create `scripts/decision_hour_move_eval.py`:

```python
"""Should each market's commit hour move later? PRE-REGISTERED GATE.

WHY. `MIN_DECISION_HOUR` was moved EARLY on 2026-07-26 to escape hours that were
transcription rather than forecasting. That correction over-shot: it landed on
hours where very little of the extremum has happened AND the model is dishonest
about it. Measured 2026-07-31, overall shortfall (stated minus realised) is
+4.8% for KLAX HIGH at h11 and -0.6% at h13; +3.1% for KMIA HIGH at h11 and
-1.9% at h14. Moving the hour does not merely raise accuracy, it removes the
overconfidence.

The candidate hours are each station's MEDIAN PEAK-EXTREMUM hour, so roughly
half the extremum is still ahead and the answer remains a forecast:

    KMIA LOW  h5 -> h7   (41.7% still ahead)
    KLAX LOW  h4 -> h6   (49.9% still ahead)
    KMIA HIGH h11 -> h14 (49.7% still ahead)
    KLAX HIGH h11 -> h13 (51.7% still ahead)

PRE-REGISTERED, WRITTEN BEFORE RUNNING:

  Primary outcome: parity-averaged bucket hit at the candidate hour MINUS the
  incumbent hour, walk-forward, RECENT window, PAIRED on matched target dates.
  Pairing matters: the two arms score the same days, so an unpaired interval
  would throw away the pairing and read wider than the truth.

  SHIP THRESHOLD, all three must hold:
    1. Dhit > 0
    2. the CI on Dhit excludes zero at Bonferroni x4 (4 markets)
    3. >= 40% of the extremum still ahead at the candidate hour

  Each market is decided INDEPENDENTLY. A market that fails keeps its hour.

EXPECTED FAILURE MODE, stated up front: the walk-forward estimator reads more
pessimistic than the single-holdout `hour_curve.py` numbers that motivated this.
A market may well fail. That is the gate working, not a setback. This project is
3-for-9 on confident pre-measurement hypotheses.

Writes NO constant. Its output is read by a human who then edits
`gating.MIN_DECISION_HOUR`.

Usage: PYTHONPATH=src .venv/bin/python scripts/decision_hour_move_eval.py
"""
import numpy as np

from highcast.buckets import bucket_probs, expected_bucket_hit
from highcast.config import load_config
from highcast.db import connect
from highcast.features import build_features, uses_bias_correction
from highcast.gating import EXTREMUM_STILL_FUTURE
from highcast.metric import get_metric
from highcast.models.baseline import BaselineBlend
from highcast.models.intraday import (IntradayModel, build_checkpoint_frame,
                                      features_for)

REFIT = 30
RECENT = "2025-01-01"
MIN_TRAIN = 400
N_BOOT = 2000
N_MARKETS = 4
ALPHA = 0.05 / N_MARKETS
PCTS = [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)]
MIN_STILL_AHEAD = 0.40
rng = np.random.default_rng(20260731)

_SHIPPED_BASE = ["hour", "run_lo", "run_hi", "run_width", "cur_tmpf",
                 "cur_dwpf", "cur_onshore", "cur_sky", "base_p50",
                 "base_spread", "doy_sin", "doy_cos", "year_index",
                 "heat_alert"]

# (config, metric, incumbent hour, candidate hour)
ARMS = [("config/kmia.yaml", "low", 5, 7),
        ("config/klax.yaml", "low", 4, 6),
        ("config/kmia.yaml", "high", 11, 14),
        ("config/klax.yaml", "high", 11, 13)]


def hits_at(cfg_path: str, metric_key: str, hour: int) -> dict[str, float]:
    """{target_date: parity-averaged bucket hit} for the walk-forward test set."""
    cfg = load_config(cfg_path)
    icao = cfg.station.icao
    conn = connect(cfg.db_path)
    spec = get_metric(cfg, metric_key)
    resid = bool(getattr(cfg.model, f"intraday_residual_{metric_key}", False))
    k = int(getattr(getattr(cfg, "model", None), "intraday_hour_bandwidth", 0) or 0)
    tr_hours = tuple(sorted({(hour + d) % 24 for d in range(-k, k + 1)}))
    feats = features_for(icao, metric_key)

    dates = [r["date"] for r in conn.execute(
        "SELECT date FROM cli_reports WHERE station=? ORDER BY date", (icao,))]
    day_df = build_features(conn, cfg, spec, dates, require_target=True)
    base = BaselineBlend(
        prefer_corrected=uses_bias_correction(spec.key)).fit(day_df)
    fr = build_checkpoint_frame(conn, cfg, dates, hours=tr_hours, metric=spec,
                                baseline=base)

    te = fr[(fr["hour"] == float(hour)) & fr["y"].notna()].sort_values("target_date")
    te = te[te["target_date"] >= RECENT]
    td = te["target_date"].to_numpy()

    out: dict[str, float] = {}
    for b0 in range(0, len(td), REFIT):
        cut = td[b0]
        tr = fr[(fr["target_date"] < cut) & fr["y"].notna()]
        if len(tr) < MIN_TRAIN:
            continue
        blk = te.iloc[b0:b0 + REFIT]
        if blk.empty:
            continue
        m = IntradayModel(extremum=spec.extremum, residual=resid,
                          extra_features=[c for c in feats
                                          if c not in _SHIPPED_BASE]).fit(tr)
        for (_, row), q in zip(blk.iterrows(), m.predict_quantiles(blk)):
            out[row["target_date"]] = expected_bucket_hit(
                bucket_probs(q), int(row["y"]))
    return out


print(__doc__.split("Usage:")[0])
print(f"Paired bootstrap, Bonferroni x{N_MARKETS} "
      f"-> {100 * (1 - ALPHA):.2f}% CI\n")

for cfg_path, metric_key, h_now, h_new in ARMS:
    cfg = load_config(cfg_path)
    icao = cfg.station.icao
    a = hits_at(cfg_path, metric_key, h_now)
    b = hits_at(cfg_path, metric_key, h_new)
    common = sorted(set(a) & set(b))
    if len(common) < 100:
        print(f"{icao} {metric_key.upper()} h{h_now}->h{h_new}: "
              f"only {len(common)} matched days — SKIP")
        continue
    old = np.array([a[d] for d in common])
    new = np.array([b[d] for d in common])
    diff = new - old
    idx = rng.integers(0, len(diff), size=(N_BOOT, len(diff)))
    lo, hi = np.percentile(diff[idx].mean(axis=1), PCTS)
    ahead = EXTREMUM_STILL_FUTURE.get(icao, {}).get(metric_key, {}).get(h_new)

    c1 = diff.mean() > 0
    c2 = (lo > 0) == (hi > 0)
    c3 = ahead is not None and ahead >= MIN_STILL_AHEAD
    verdict = "MOVE" if (c1 and c2 and c3) else "KEEP"

    print(f"### {icao} {metric_key.upper()}  h{h_now} -> h{h_new}   "
          f"n={len(common)} matched days")
    print(f"    hit {old.mean():.1%} -> {new.mean():.1%}   "
          f"Dhit {diff.mean():+.1%}  CI [{lo:+.1%},{hi:+.1%}]")
    print(f"    still ahead at h{h_new}: "
          f"{ahead:.1%}" if ahead is not None else "    still ahead: UNKNOWN")
    print(f"    (1) Dhit>0 {c1}   (2) CI excludes 0 {c2}   "
          f"(3) >={MIN_STILL_AHEAD:.0%} ahead {c3}")
    print(f"    -> {verdict}\n")

print("MOVE markets get their MIN_DECISION_HOUR edited by hand (Task 2).")
print("This script writes nothing.")
```

- [ ] **Step 2: Verify the eval-script guard still passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_eval_scripts.py -q`
Expected: `12 passed` — the new script uses `features_for` and `extra_features`, so it is not flagged.

- [ ] **Step 3: Run the gate**

Run: `PYTHONPATH=src .venv/bin/python scripts/decision_hour_move_eval.py`
Expected: four `### STATION METRIC h_now -> h_new` blocks, each ending in `-> MOVE` or `-> KEEP`. Runtime ~20–40 min (8 walk-forward fits).

- [ ] **Step 4: Record the verdicts**

Paste the full output into the task's report. **Do not proceed to Task 2 for any market whose verdict is `KEEP`.** The verdicts are the input to Task 2 — Task 2 is not authorised to move a market the gate rejected.

- [ ] **Step 5: Verify nothing shipped**

Run: `PYTHONPATH=src .venv/bin/python -m pytest -q`
Expected: `614 passed, 12 deselected` — unchanged, because this task ships no constant.

---

### Task 2: Apply the hour move for markets the gate passed

**Files:**
- Modify: `src/highcast/gating.py` (`MIN_DECISION_HOUR` dict and the comment block above it)
- Test: `tests/test_decision_hour.py`

**Interfaces:**
- Consumes: Task 1's printed `MOVE` / `KEEP` verdict per market.
- Produces: `gating.MIN_DECISION_HOUR[station][metric]` at the new hours. Tasks 3 and 6 key their tables on these hours.

⚠️ **Only edit markets whose Task 1 verdict was `MOVE`.** If a market says `KEEP`, leave its hour alone and note it in the report.

- [ ] **Step 1: Confirm the existing invariant test passes before touching anything**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_decision_hour.py -q`
Expected: all pass. This is the baseline for `test_the_window_admits_the_decision_hour`, which asserts `cutoff_local_hour > knee`.

- [ ] **Step 2: Edit `MIN_DECISION_HOUR`**

In `src/highcast/gating.py`, find `MIN_DECISION_HOUR` and set each **passing** market to its candidate hour:

```python
MIN_DECISION_HOUR: dict[str, dict[str, int]] = {
    # Moved later 2026-07-31 by `scripts/decision_hour_move_eval.py`, a
    # pre-registered paired gate (Dhit CI excluding zero at Bonferroni x4, and
    # >=40% of the extremum still ahead so the answer stays a FORECAST).
    #
    # These hours are each station's MEDIAN PEAK-EXTREMUM hour, measured from
    # the CLI reports' own high_time/low_time (KMIA n=2366, median 13:58;
    # KLAX n=1959, median 13:05). Committing there means the extremum is a coin
    # flip between "already set" and "still to come".
    #
    # The move was made for HONESTY as much as accuracy: overall shortfall
    # (stated minus realised) went +4.8% -> -0.6% for KLAX HIGH and
    # +3.1% -> -1.9% for KMIA HIGH. That is also why no HIGH preflight exists —
    # there is no longer a confidently-wrong regime for one to catch.
    # See docs/superpowers/specs/2026-07-31-accuracy-first-decision-hours-design.md
    "KLAX": {"low": 6, "high": 13},
    "KMIA": {"low": 7, "high": 14},
}
```

- [ ] **Step 3: Run the invariant test**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_decision_hour.py -q`
Expected: PASS. The cutoffs (KLAX LA 18:00/13:00, KMIA Miami 21:00/16:00) already sit after every candidate knee, so no config change is needed. **If this fails, the cutoff is now before the knee — stop and report; do not "fix" it by moving the cutoff without a gate.**

- [ ] **Step 4: Add a regression test pinning the new hours**

Append to `tests/test_decision_hour.py`:

```python
def test_decision_hours_are_the_measured_peak_extremum_hours():
    """Pins the 2026-07-31 move so a future edit is deliberate, not drift.

    These hours came from `scripts/decision_hour_move_eval.py`, a pre-registered
    paired gate. They are each station's MEDIAN peak-extremum hour, which is why
    roughly half the extremum is still ahead at commit — the property that keeps
    the answer a forecast rather than a transcription.
    """
    assert gating.MIN_DECISION_HOUR["KLAX"] == {"low": 6, "high": 13}
    assert gating.MIN_DECISION_HOUR["KMIA"] == {"low": 7, "high": 14}


def test_every_decision_hour_still_leaves_the_extremum_mostly_unknown():
    """THE reason these hours were chosen. If a future edit pushes a commit hour
    past the point where the extremum has usually happened, the desk is
    transcribing a published number, not forecasting — and a bucket the market
    has already read carries no edge either."""
    for icao, per_metric in gating.MIN_DECISION_HOUR.items():
        for metric, hour in per_metric.items():
            ahead = gating.EXTREMUM_STILL_FUTURE[icao][metric][hour]
            assert ahead >= 0.40, (
                f"{icao} {metric} h{hour}: only {ahead:.1%} of the extremum is "
                f"still ahead — that is scorekeeping, not forecasting")
```

- [ ] **Step 5: Run the new tests**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_decision_hour.py -q`
Expected: PASS, 2 more tests than the baseline.

- [ ] **Step 6: Run the full suite and fix stale literals**

Run: `PYTHONPATH=src .venv/bin/python -m pytest -q`

Expected: some failures in `tests/test_board.py`, `tests/test_notify.py`, and any test asserting an hour-dependent display string — the same class fixed on 2026-07-31 when `HOUR_ACCURACY` was refreshed. For each failure, read the assertion, confirm the new value is the one `HOUR_ACCURACY[station][metric][new_hour]` reports, and update the literal with a short comment naming the move. **Do not weaken an assertion to make it pass.**

Re-run until: `616 passed, 12 deselected` (614 + the 2 new tests).

---

### Task 3: Re-derive `BAND_CALIB` at the new hours

`BAND_CALIB` is keyed `(station, metric, hour)`. After Task 2 every key is stale — `band_calib()` returns `None` for an unknown key, which means **refuse everything**. That is safe but useless, so the tables must be re-measured at the new hours.

**Files:**
- Modify: `scripts/band_calib_eval.py` (point it at the new hours), `scripts/calib_eval.py` (same)
- Modify: `src/highcast/gating.py` (`BAND_CALIB`)
- Test: `tests/test_band_calib.py`

**Interfaces:**
- Consumes: `gating.MIN_DECISION_HOUR` (Task 2).
- Produces: `BAND_CALIB[(station, metric, new_hour)] -> list[tuple[float, float]]`, consumed by `gating.band_calib()` and therefore by Task 6 and Task 7.

- [ ] **Step 1: Confirm the tables are stale**

Run:
```bash
PYTHONPATH=src .venv/bin/python -c "
from highcast.gating import BAND_CALIB, MIN_DECISION_HOUR, band_calib
for st, per in MIN_DECISION_HOUR.items():
    for m, h in per.items():
        print(st, m, h, 'table:', (st,m,h) in BAND_CALIB,
              'band_calib(0.75) ->', band_calib(st, m, h, 0.75))
"
```
Expected: `table: False` and `-> None` for every market — i.e. everything currently refuses.

- [ ] **Step 2: Re-run the calibration measurement at the new hours**

`scripts/calib_eval.py` reads its hour from `MIN_DECISION_HOUR[icao][metric]`, so Task 2 already redirected it. Confirm, then run:

```bash
PYTHONPATH=src .venv/bin/python -c "
import re; s=open('scripts/calib_eval.py').read()
print('reads MIN_DECISION_HOUR:', 'MIN_DECISION_HOUR[' in s)"
```
Expected: `True`. If `False`, stop and report — the script pins its hour some other way and would measure the wrong one.

Then run: `PYTHONPATH=src .venv/bin/python scripts/calib_eval.py 30 2025-01-01`
Expected: four `FULL` / `RECENT` blocks and a fresh `data/calib_records-*.csv`. Runtime ~30–60 min.

⚠️ Verify the dump is not duplicated before using it — `calib_eval.py` shipped an `"a"`-mode append bug on 2026-07-30 that silently stacked three models' records into one file:

```bash
PYTHONPATH=src .venv/bin/python -c "
import csv,collections,sys
p=sys.argv[1]; rows=list(csv.DictReader(open(p)))
k=lambda r:(r['station'],r['metric'],r['hour'],r['target_date'])
c=collections.Counter(k(r) for r in rows)
print('rows',len(rows),'distinct',len(c),'dupes',sum(1 for v in c.values() if v>1))
" data/calib_records-<TODAY>.csv
```
Expected: `dupes 0` and `rows == distinct`.

- [ ] **Step 3: Re-derive the band table**

Run: `PYTHONPATH=src .venv/bin/python scripts/band_calib_eval.py`
Expected: per-market band offsets at the new hours.

- [ ] **Step 4: Update `BAND_CALIB`**

Replace the dict in `src/highcast/gating.py` with the new keys, keeping the existing structure `(upper_bound, offset_to_subtract)` and preserving the explanatory comment. Add:

```python
    # ⚠️ RE-DERIVED 2026-07-31 AT THE NEW COMMIT HOURS. The previous rows were
    # keyed on h11/h4/h5 and did not transfer: `band_calib()` returns None for an
    # unknown key, so between the hour move and this re-derivation every market
    # refused. That is the correct failure direction, and it is why this task
    # cannot be skipped.
```

- [ ] **Step 5: Update `tests/test_band_calib.py` keys**

Every test referencing `("KLAX", "high", 11)` etc. must use the new hour. Read each test first: the ones asserting *behaviour* (None above the top band, clamping to (0,1)) keep their assertions and only change the key; the ones asserting a specific offset literal need the new measured value.

- [ ] **Step 6: Run the band tests**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_band_calib.py -q`
Expected: PASS.

- [ ] **Step 7: Run the full suite**

Run: `PYTHONPATH=src .venv/bin/python -m pytest -q`
Expected: `616 passed, 12 deselected`.

---

### Task 4: Gate C-1 — `gating.guidance_lead_ready`

Refuse when the newest guidance for the target is two days old. This is the fix for a bug measured live on 2026-07-31.

**Files:**
- Modify: `src/highcast/gating.py`
- Test: `tests/test_guidance_lead.py` (create)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `gating.guidance_lead_ready(conn, cfg, target: str) -> bool`, consumed by Task 7.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_guidance_lead.py`:

```python
"""The evening day-ahead runs on stale guidance and drifts cold.

CAUGHT LIVE 2026-07-31. `should_predict_tomorrow` fires after 17:00 local, and
increasingly the newest guidance that has landed by then is TWO days old. The
day-ahead model is trained on lead-1 and extrapolates badly on lead-2:

    issue 00Z (17:00 local, evening)  n=12  mean error -4.86F  MAE 5.40
    issue 16Z (09:00 local, cutoff)   n=12  mean error +0.06F  MAE 1.31

Every lead-2 evening run was badly cold (-4.8, -6.8, -7.2, -9.6, -7.8). On
2026-07-30 it printed 70.2F when its own COLDEST guidance member said 74.7F and
it settled 78 — the model went below its own guidance floor.

Same class as the NBS bug: a model silently consuming out-of-regime input and
drifting one way instead of failing loudly. `nbs_guidance_ready` guards the
missing-feed case; this guards the stale-lead case.
"""
import yaml

from highcast import db as hdb, gating
from highcast.config import load_config


def _cfg(tmp_path, src="config/klax.yaml"):
    raw = yaml.safe_load(open(src).read())
    raw["db_path"] = str(tmp_path / "t.db")
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(raw))
    return load_config(str(p))


def _fc(station, target, run_date, model="gfs", metric="high", val=80.0):
    return {"station": station, "target_date": target, "model": model,
            "run_date": run_date, "max_temp_f": val, "metric": metric,
            "fetched_at": f"{run_date}T12:00:00+00:00"}


def test_lead_1_guidance_is_ready(tmp_path):
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    hdb.upsert(conn, "forecasts", [_fc("KLAX", "2026-08-02", "2026-08-01")])
    assert gating.guidance_lead_ready(conn, cfg, "2026-08-02") is True


def test_lead_2_guidance_is_refused(tmp_path):
    """THE bug. Two-day-old guidance is a regime the model was not trained on."""
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    hdb.upsert(conn, "forecasts", [_fc("KLAX", "2026-08-02", "2026-07-31")])
    assert gating.guidance_lead_ready(conn, cfg, "2026-08-02") is False


def test_newest_run_wins_when_both_leads_exist(tmp_path):
    """A stale row alongside a fresh one must not veto — only the NEWEST matters."""
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    hdb.upsert(conn, "forecasts", [
        _fc("KLAX", "2026-08-02", "2026-07-31"),
        _fc("KLAX", "2026-08-02", "2026-08-01"),
    ])
    assert gating.guidance_lead_ready(conn, cfg, "2026-08-02") is True


def test_no_guidance_at_all_fails_closed(tmp_path):
    """Missing data and 'no' must be the same answer for an unattended caller."""
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    assert gating.guidance_lead_ready(conn, cfg, "2026-08-02") is False


def test_the_other_station_cannot_satisfy_this_one(tmp_path):
    """Both stations share one SQLite file — anything not keyed by station is a
    latent cross-station bug."""
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    hdb.upsert(conn, "forecasts", [_fc("KMIA", "2026-08-02", "2026-08-01")])
    assert gating.guidance_lead_ready(conn, cfg, "2026-08-02") is False
```

- [ ] **Step 2: Run to verify they fail**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_guidance_lead.py -q`
Expected: FAIL with `AttributeError: module 'highcast.gating' has no attribute 'guidance_lead_ready'`.

- [ ] **Step 3: Implement**

Add to `src/highcast/gating.py`, directly after `nbs_guidance_ready`:

```python
def guidance_lead_ready(conn, cfg, target: str) -> bool:
    """Is the newest guidance for `target` from YESTERDAY's run, not the day before?

    CAUGHT LIVE 2026-07-31. `should_predict_tomorrow` fires after 17:00 local and
    increasingly sees only two-day-old guidance. The day-ahead model is trained on
    lead-1 and extrapolates badly on lead-2 -- KLAX HIGH ran -4.86F mean error
    (MAE 5.40) at the evening run versus +0.06F (MAE 1.31) at the 09:00 one, and
    on 2026-07-30 it printed 70.2F when its own coldest guidance member said 74.7F
    and it settled 78. It went BELOW ITS OWN GUIDANCE FLOOR.

    Unlike `nbs_guidance_ready` this checks FRESHNESS, not presence: an NBS row a
    few hours old is ordinary, but guidance a whole extra run cycle old is a
    different regime. Both fail closed.
    """
    row = conn.execute(
        """SELECT MAX(run_date) FROM forecasts
           WHERE station=? AND target_date=?""",
        (cfg.station.icao, target)).fetchone()
    newest = row[0] if row else None
    if newest is None:
        return False
    from datetime import date
    try:
        lead = (date.fromisoformat(target) - date.fromisoformat(newest)).days
    except (TypeError, ValueError):
        return False
    return lead <= 1
```

- [ ] **Step 4: Run the tests**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_guidance_lead.py -q`
Expected: `5 passed`.

- [ ] **Step 5: Run the full suite**

Run: `PYTHONPATH=src .venv/bin/python -m pytest -q`
Expected: `621 passed, 12 deselected`.

---

### Task 5: Gate C-2 — `resolve_price` prefers a live book

**Files:**
- Modify: `src/highcast/fetchers/kalshi.py:117-127`
- Test: `tests/test_kalshi.py` (append; create if absent)

**Interfaces:**
- Consumes: nothing.
- Produces: `kalshi.resolve_price(r: dict) -> float | None` with inverted precedence. Consumed by `signals.py:120` and `web/app.py:219` unchanged.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_kalshi.py`:

```python
from highcast.fetchers import kalshi


def test_a_live_two_sided_book_beats_a_stale_last_trade():
    """MEASURED 2026-07-31: 16.1% of stored quotes (4,251 of 26,460) sat OUTSIDE
    their own bid/ask because `last_price` was preferred unconditionally. The
    median deviation is only 1c and it did NOT explain the losing bets — but the
    worst was 52c, and an unattended bot must never act on a stale print."""
    r = {"last_price": 0.63, "yes_bid": 0.10, "yes_ask": 0.11}
    assert kalshi.resolve_price(r) == 0.105


def test_last_price_is_used_when_there_is_no_book():
    r = {"last_price": 0.42, "yes_bid": None, "yes_ask": None}
    assert kalshi.resolve_price(r) == 0.42


def test_a_crossed_book_is_rejected_in_favour_of_the_last_trade():
    """ask < bid is nonsense; trusting its midpoint would invent a price."""
    r = {"last_price": 0.50, "yes_bid": 0.60, "yes_ask": 0.40}
    assert kalshi.resolve_price(r) == 0.50


def test_a_single_sided_book_is_used_when_nothing_else_exists():
    assert kalshi.resolve_price(
        {"last_price": None, "yes_bid": 0.33, "yes_ask": None}) == 0.33
    assert kalshi.resolve_price(
        {"last_price": None, "yes_bid": None, "yes_ask": 0.37}) == 0.37


def test_no_information_at_all_returns_none():
    assert kalshi.resolve_price(
        {"last_price": None, "yes_bid": None, "yes_ask": None}) is None
```

- [ ] **Step 2: Run to verify they fail**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_kalshi.py -q`
Expected: FAIL — `test_a_live_two_sided_book_beats_a_stale_last_trade` returns `0.63`.

- [ ] **Step 3: Implement**

Replace `resolve_price` in `src/highcast/fetchers/kalshi.py`:

```python
def resolve_price(r: dict) -> float | None:
    """market_price for a parsed range: a LIVE two-sided book wins, else the last
    trade, else whichever single side exists, else None.

    PRECEDENCE INVERTED 2026-07-31. This used to return `last_price` whenever it
    was present, so a trade from hours ago overrode a live bid/ask that said
    otherwise. Measured across 26,460 stored quotes, 16.1% sat OUTSIDE their own
    book -- median deviation 1c, 78.9% within 2c, but a 52c worst case.

    The small median is why this was NOT the cause of the losing bets (they were
    priced within 0-2c of their book at commit) -- and the 52c tail is why an
    unattended caller must not inherit the old behaviour anyway.
    """
    bid, ask = r.get("yes_bid"), r.get("yes_ask")
    if bid is not None and ask is not None and ask >= bid:
        return round((bid + ask) / 2, 4)
    lp = r.get("last_price")
    if lp is not None:
        return lp
    return bid if bid is not None else ask
```

- [ ] **Step 4: Run the tests**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_kalshi.py -q`
Expected: PASS.

- [ ] **Step 5: Re-check `max_spread` against a real wide book**

The KMIA `≥81` book was `0.03 / 0.11` — 8 c wide, inside the 10 c `max_spread`, so it passes the liquidity gate while being nearly untradeable. Confirm current behaviour and report it (do **not** change the constant in this task — that is a decision-affecting change needing its own gate):

```bash
PYTHONPATH=src .venv/bin/python -c "
from highcast.config import load_config
cfg = load_config('config/kmia.yaml')
print('max_spread =', cfg.staking.max_spread)
print('0.03/0.11 spread =', round(0.11-0.03, 4),
      '-> passes:', (0.11-0.03) <= cfg.staking.max_spread)
"
```
Expected: `max_spread = 0.1`, `spread = 0.08`, `passes: True`. Record this in the report as a known gap.

- [ ] **Step 6: Run the full suite**

Run: `PYTHONPATH=src .venv/bin/python -m pytest -q`
Expected: `626 passed, 12 deselected`.

---

### Task 6: Gate B — the calibrated-confidence floor

**Files:**
- Modify: `src/highcast/gating.py`
- Test: `tests/test_confidence_floor.py` (create)

**Interfaces:**
- Consumes: `gating.band_calib` (Task 3), `gating.MIN_DECISION_HOUR` (Task 2).
- Produces: `gating.CONFIDENCE_FLOOR: dict[tuple[str, str, int], float]` and `gating.confidence_floor_ok(station: str, metric: str, hour: int, p_hat: float) -> bool`, consumed by Task 7.

The floor rule from the spec, stated not fitted: **act only when measurably right 4 days in 5** — the lowest stated-probability threshold whose realised hit rate has a 95 % lower bound ≥ 0.80. Verified against the measured table on 2026-07-31:

| market | floors tried (hit / lower bound) | admissible floor |
|---|---|---|
| KLAX HIGH h13 | .50 69.5/65.6 · .60 70.3/66.1 · .70 71.4/65.8 · .80 75.0/63.9 | **none** |
| KMIA HIGH h14 | .50 80.5/76.9 · .60 82.1/78.6 · **.70 85.5/81.9** · .80 93.4/89.6 | **0.70** |

- [ ] **Step 1: Write the failing tests**

Create `tests/test_confidence_floor.py`:

```python
"""The bot may act only when the model is measurably right 4 days in 5.

THE FLOOR IS A POLICY TARGET, NOT A FITTED PARAMETER. Per market it is the
LOWEST stated-probability threshold whose realised hit rate has a 95% lower
bound >= 0.80. A market with no such threshold gets NO floor and never fires.

Measured 2026-07-31, n~570/arm, walk-forward:

    KLAX HIGH h13  .50 69.5/65.6  .60 70.3/66.1  .70 71.4/65.8  .80 75.0/63.9
                   -> no floor clears; the bot never trades KLAX HIGH
    KMIA HIGH h14  .50 80.5/76.9  .60 82.1/78.6  .70 85.5/81.9  .80 93.4/89.6
                   -> floor 0.70, fires on 75.3% of days at 85.5%

KLAX HIGH is excluded by its own calibration AND, independently, by price: at
h13 the favoured bucket traded 0.75-0.96 (median ~0.90) while the model can
justify only ~69c, and the market's favoured bucket was right 5 of 6. Two
independent lines of evidence agreeing is the strongest result in the spec.
DO NOT lower the target to admit KLAX HIGH.
"""
import pytest

from highcast import gating


def test_kmia_high_has_the_measured_floor():
    assert gating.CONFIDENCE_FLOOR[("KMIA", "high", 14)] == 0.70


def test_klax_high_has_no_floor_and_therefore_never_fires():
    """Not an oversight. No threshold reached a 0.80 lower bound."""
    assert ("KLAX", "high", 13) not in gating.CONFIDENCE_FLOOR
    assert gating.confidence_floor_ok("KLAX", "high", 13, 0.99) is False


def test_a_probability_at_or_above_the_floor_passes():
    assert gating.confidence_floor_ok("KMIA", "high", 14, 0.70) is True
    assert gating.confidence_floor_ok("KMIA", "high", 14, 0.93) is True


def test_a_probability_below_the_floor_is_refused():
    assert gating.confidence_floor_ok("KMIA", "high", 14, 0.699) is False


def test_an_unmeasured_market_fails_closed():
    assert gating.confidence_floor_ok("KMIA", "low", 7, 0.99) is False
    assert gating.confidence_floor_ok("NOPE", "high", 14, 0.99) is False


def test_the_wrong_hour_fails_closed():
    """The floor is keyed by hour. A caller at the old h11 must not inherit the
    h14 floor — the two hours have completely different accuracy."""
    assert gating.confidence_floor_ok("KMIA", "high", 11, 0.99) is False


@pytest.mark.parametrize("bad", [None, float("nan")])
def test_a_missing_probability_fails_closed(bad):
    assert gating.confidence_floor_ok("KMIA", "high", 14, bad) is False
```

- [ ] **Step 2: Run to verify they fail**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_confidence_floor.py -q`
Expected: FAIL with `AttributeError: ... has no attribute 'CONFIDENCE_FLOOR'`.

- [ ] **Step 3: Implement**

Add to `src/highcast/gating.py`, after `BAND_CALIB` and `band_calib`:

```python
# The CALIBRATED probability a market must clear before it may be acted on.
#
# THIS IS A POLICY TARGET, NOT A FITTED PARAMETER: act only when measurably right
# 4 days in 5. Per market the floor is the LOWEST stated-probability threshold
# whose realised hit rate has a 95% lower bound >= 0.80. A market with no such
# threshold is ABSENT here, and an absent market never fires.
#
# Measured 2026-07-31, walk-forward, n~570/arm:
#     KLAX HIGH h13  .50 69.5/65.6  .60 70.3/66.1  .70 71.4/65.8  .80 75.0/63.9
#     KMIA HIGH h14  .50 80.5/76.9  .60 82.1/78.6  .70 85.5/81.9  .80 93.4/89.6
#
# ⚠️ KLAX HIGH IS DELIBERATELY ABSENT. No threshold reached a 0.80 lower bound.
# It is independently excluded by price too: at h13 its favoured bucket traded
# 0.75-0.96 (median ~0.90) against a model that can justify ~69c, and the
# market's favoured bucket was right 5 of 6. Do NOT lower the target to admit it.
#
# The LOWs are absent because Julian trades them by hand; this floor exists for
# the unattended path only.
CONFIDENCE_FLOOR: dict[tuple[str, str, int], float] = {
    ("KMIA", "high", 14): 0.70,
}


def confidence_floor_ok(station: str, metric: str, hour: int,
                        p_hat: float) -> bool:
    """Does the CALIBRATED probability clear this market's measured floor?

    Fails closed on every unknown: an unmeasured market, the wrong hour, a
    missing or NaN probability. For an unattended caller "I could not tell" and
    "no" must be the same answer.

    `p_hat` must already be calibrated (`band_calib`), never the raw p_model --
    the raw number is what was 6.7 points overconfident at KLAX HIGH.
    """
    floor = CONFIDENCE_FLOOR.get((station, metric, hour))
    if floor is None:
        return False
    if p_hat is None:
        return False
    try:
        p = float(p_hat)
    except (TypeError, ValueError):
        return False
    if p != p:            # NaN
        return False
    return p >= floor
```

- [ ] **Step 4: Run the tests**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_confidence_floor.py -q`
Expected: `8 passed`.

- [ ] **Step 5: Run the full suite**

Run: `PYTHONPATH=src .venv/bin/python -m pytest -q`
Expected: `634 passed, 12 deselected`.

---

### Task 7: `gating.decision_gates` — the single fail-closed verdict

**Files:**
- Modify: `src/highcast/gating.py`
- Test: `tests/test_decision_gates.py` (create)

**Interfaces:**
- Consumes: `nbs_guidance_ready`, `metric_uses_nbs`, `guidance_lead_ready` (Task 4), `band_calib` (Task 3), `confidence_floor_ok` (Task 6), `fees.taker_fee`.
- Produces:
  - `gating.GateVerdict` — frozen dataclass with `ok: bool` and `refusals: tuple[str, ...]`
  - `gating.decision_gates(conn, cfg, *, metric: str, target: str, hour: int, p_model: float, price: float | None = None) -> GateVerdict`

  Consumed by Task 8 and, later, by the phase-4 bot.

The price rail is implemented here as an **optional** check: when `price is None` it is skipped. Nothing in this plan passes a price — the rail is exercised by tests only, so the aggregator does not need a second edit in phase 4.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_decision_gates.py`:

```python
"""ONE verdict, read by every surface, and it fails closed.

WHY AN AGGREGATOR AND NOT SCATTERED `if`s. The dashboard, the terminal and
(later) the bot must reach the same answer or they drift and the desk gives
advice that disagrees with itself. `board.evidence()` is assembled in Python for
exactly this reason.

WHY FAIL-CLOSED. For an unattended caller, "I could not tell" and "no" must be
the same answer. Every branch here returns a refusal on missing data or an
exception -- matching `nbs_guidance_ready`'s existing contract.
"""
import yaml

from highcast import db as hdb, gating
from highcast.config import load_config


def _cfg(tmp_path, src="config/kmia.yaml"):
    raw = yaml.safe_load(open(src).read())
    raw["db_path"] = str(tmp_path / "t.db")
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(raw))
    return load_config(str(p))


def _ready(conn, station="KMIA", target="2026-08-02"):
    """Populate everything the gates need so a test can knock ONE thing out."""
    hdb.upsert(conn, "forecasts", [
        {"station": station, "target_date": target, "model": "gfs",
         "run_date": "2026-08-01", "max_temp_f": 94.0, "metric": "high",
         "fetched_at": "2026-08-01T12:00:00+00:00"},
        {"station": station, "target_date": target, "model": "nbm_nbs",
         "run_date": "2026-08-01", "max_temp_f": 94.0, "metric": "high",
         "fetched_at": "2026-08-01T12:00:00+00:00"},
    ])


def test_all_gates_green_is_ok(tmp_path):
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    _ready(conn)
    v = gating.decision_gates(conn, cfg, metric="high", target="2026-08-02",
                              hour=14, p_model=0.90)
    assert v.ok is True
    assert v.refusals == ()


def test_stale_lead_2_guidance_refuses(tmp_path):
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    hdb.upsert(conn, "forecasts", [
        {"station": "KMIA", "target_date": "2026-08-02", "model": "gfs",
         "run_date": "2026-07-31", "max_temp_f": 94.0, "metric": "high",
         "fetched_at": "2026-07-31T12:00:00+00:00"}])
    v = gating.decision_gates(conn, cfg, metric="high", target="2026-08-02",
                              hour=14, p_model=0.90)
    assert v.ok is False
    assert "guidance_lead" in v.refusals


def test_an_unmeasured_confidence_band_refuses(tmp_path):
    """`band_calib` returns None above the highest measured band. All three
    losing bets on 2026-07-29/30 were stated 81% — from exactly that region."""
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    _ready(conn)
    v = gating.decision_gates(conn, cfg, metric="high", target="2026-08-02",
                              hour=14, p_model=0.999)
    assert v.ok is False
    assert "unmeasured_band" in v.refusals


def test_below_the_confidence_floor_refuses(tmp_path):
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    _ready(conn)
    v = gating.decision_gates(conn, cfg, metric="high", target="2026-08-02",
                              hour=14, p_model=0.55)
    assert v.ok is False
    assert "confidence_floor" in v.refusals


def test_klax_high_always_refuses_on_the_floor(tmp_path):
    """It has no admissible floor by construction. Two independent lines of
    evidence say the bot should not trade it."""
    cfg = _cfg(tmp_path, src="config/klax.yaml")
    conn = hdb.connect(cfg.db_path)
    _ready(conn, station="KLAX")
    v = gating.decision_gates(conn, cfg, metric="high", target="2026-08-02",
                              hour=13, p_model=0.95)
    assert v.ok is False
    assert "confidence_floor" in v.refusals


def test_the_price_rail_refuses_a_knowingly_negative_ev_buy(tmp_path):
    """Not edge-hunting — a loss floor. Paying 0.98 for a bucket calibrated near
    0.85 is a guaranteed loss after the 2c fee."""
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    _ready(conn)
    v = gating.decision_gates(conn, cfg, metric="high", target="2026-08-02",
                              hour=14, p_model=0.90, price=0.98)
    assert v.ok is False
    assert "price_rail" in v.refusals


def test_the_price_rail_is_skipped_when_no_price_is_supplied(tmp_path):
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    _ready(conn)
    v = gating.decision_gates(conn, cfg, metric="high", target="2026-08-02",
                              hour=14, p_model=0.90, price=None)
    assert "price_rail" not in v.refusals


def test_a_broken_connection_fails_closed(tmp_path):
    """An exception must become a refusal, never an exception reaching a caller
    that might treat a crash as 'no opinion' and proceed."""
    cfg = _cfg(tmp_path)

    class Boom:
        def execute(self, *a, **k):
            raise RuntimeError("db is gone")

    v = gating.decision_gates(Boom(), cfg, metric="high", target="2026-08-02",
                              hour=14, p_model=0.90)
    assert v.ok is False
    assert v.refusals != ()


def test_refusals_accumulate_rather_than_short_circuit(tmp_path):
    """The desk should see EVERY reason, not just the first one."""
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    v = gating.decision_gates(conn, cfg, metric="high", target="2026-08-02",
                              hour=14, p_model=0.55, price=0.99)
    assert v.ok is False
    assert len(v.refusals) >= 2
```

- [ ] **Step 2: Run to verify they fail**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_decision_gates.py -q`
Expected: FAIL with `AttributeError: ... has no attribute 'decision_gates'`.

- [ ] **Step 3: Implement**

Add to `src/highcast/gating.py` (top-level `from dataclasses import dataclass` if not already imported):

```python
@dataclass(frozen=True)
class GateVerdict:
    """Why the desk may or may not act. `refusals` names EVERY failing gate, not
    just the first: a desk shown one reason fixes it and is surprised by the
    next."""
    ok: bool
    refusals: tuple[str, ...]


def decision_gates(conn, cfg, *, metric: str, target: str, hour: int,
                   p_model: float, price: float | None = None) -> GateVerdict:
    """THE verdict. Every surface -- terminal, dashboard, and later the bot --
    must read this one function.

    WHY ONE FUNCTION. Scattered `if`s drift apart and the desk ends up giving
    advice that disagrees with itself. `board.evidence()` is assembled in Python
    for the same reason.

    EVERY GATE FAILS CLOSED. Missing data, an unmeasured market and an exception
    all produce a refusal. For an unattended caller "I could not tell" and "no"
    must be the same answer.

    `price` is OPTIONAL and the rail is skipped when it is None. Nothing in
    phase 1 passes a price; the rail is here so the aggregator does not need a
    second edit when the bot arrives.
    """
    from highcast import fees

    icao = cfg.station.icao
    refusals: list[str] = []

    # --- Gate C: data freshness -------------------------------------------
    try:
        if metric_uses_nbs(metric) and not nbs_guidance_ready(conn, cfg, target):
            refusals.append("nbs_missing")
    except Exception:
        refusals.append("nbs_missing")

    try:
        if not guidance_lead_ready(conn, cfg, target):
            refusals.append("guidance_lead")
    except Exception:
        refusals.append("guidance_lead")

    # --- Gate B: calibrated confidence ------------------------------------
    p_hat = None
    try:
        p_hat = band_calib(icao, metric, hour, p_model)
    except Exception:
        p_hat = None
    if p_hat is None:
        # None means REFUSE, never "use the raw number" -- it is returned for an
        # unmeasured market/hour or for a probability above the highest measured
        # band, and all three losing bets were stated 81%, i.e. from there.
        refusals.append("unmeasured_band")
    elif not confidence_floor_ok(icao, metric, hour, p_hat):
        refusals.append("confidence_floor")

    # --- The price rail (optional) ----------------------------------------
    # A loss floor, NOT edge-hunting: never knowingly pay more than the bucket is
    # worth. At 1 contract Kalshi's taker fee rounds up to 2c at every price, so
    # the fee is a real part of the comparison.
    if price is not None:
        try:
            if p_hat is None:
                refusals.append("price_rail")
            else:
                fee = fees.taker_fee(1, float(price))
                if float(price) >= (p_hat - fee):
                    refusals.append("price_rail")
        except Exception:
            refusals.append("price_rail")

    return GateVerdict(ok=not refusals, refusals=tuple(refusals))
```

- [ ] **Step 4: Run the tests**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_decision_gates.py -q`
Expected: `9 passed`.

⚠️ If `test_all_gates_green_is_ok` fails on `unmeasured_band`, the Task 3 band table does not cover `p_model=0.90` at `("KMIA","high",14)`. That is a **real finding, not a test bug**: it means the top band is unmeasured and the market genuinely refuses there. Lower the test's `p_model` to a value inside a measured band and record the discovery in the report.

- [ ] **Step 5: Run the full suite**

Run: `PYTHONPATH=src .venv/bin/python -m pytest -q`
Expected: `643 passed, 12 deselected`.

---

### Task 8: Surface the verdict, and lock the freeze contracts

**Files:**
- Modify: `src/highcast/board.py` (render the refusals in the terminal board)
- Test: `tests/test_freeze_contracts.py` (create)

**Interfaces:**
- Consumes: `gating.decision_gates`, `gating.GateVerdict` (Task 7).
- Produces: no new API — the verdict appears on the board, and the freeze tests pin phase 1's boundary.

- [ ] **Step 1: Write the freeze tests**

Create `tests/test_freeze_contracts.py`:

```python
"""Phase 1 ships NO trading capability. These tests are the boundary.

The spec's scope is deliberate: the accuracy foundation lands and accumulates
live evidence BEFORE anything can spend money. If a later change quietly adds
authentication or order placement, these fail.
"""
import pathlib

import yaml

SRC = pathlib.Path("src/highcast")


def test_no_trading_authentication_anywhere_in_src():
    """The app is read-only on Kalshi: the public markets endpoint, no auth.
    Signing keys or a portfolio read would mean phase 2 landed early."""
    banned = ("private_key", "rsa", "sign_request", "/portfolio",
              "/trade-api/v2/portfolio", "create_order")
    hits = []
    for f in SRC.rglob("*.py"):
        low = f.read_text().lower()
        for token in banned:
            if token in low:
                hits.append(f"{f}: {token}")
    assert not hits, f"trading capability appeared in phase 1: {hits}"


def test_bankroll_is_zero_in_both_configs():
    """Paper mode. At bankroll 0 the ledger records the model's genuine opinion
    with no sizing constraint, so app-vs-Julian measures the MODEL."""
    for p in ("config/klax.yaml", "config/kmia.yaml"):
        raw = yaml.safe_load(open(p).read())
        assert raw["staking"]["bankroll"] == 0.0, f"{p} left paper mode"


def test_nothing_reads_the_hourly_capture_yet():
    """`forecast_hourly` is capture-only. A feature built on it is a MODEL change
    and needs its own pre-registered gate."""
    readers = []
    for f in SRC.rglob("*.py"):
        if f.name in ("db.py", "backfill.py", "forecasts.py"):
            continue
        if "forecast_hourly" in f.read_text():
            readers.append(str(f))
    assert not readers, f"forecast_hourly is being read by {readers}"


def test_the_decision_gates_aggregator_is_the_only_composed_verdict():
    """No surface may re-implement the gate logic. If a file calls the
    individual gates directly instead of `decision_gates`, the surfaces will
    drift -- which is the whole reason the aggregator exists."""
    allowed = {"gating.py"}
    offenders = []
    for f in SRC.rglob("*.py"):
        if f.name in allowed:
            continue
        text = f.read_text()
        if "confidence_floor_ok" in text and "decision_gates" not in text:
            offenders.append(str(f))
    assert not offenders, f"gate logic re-implemented in {offenders}"
```

- [ ] **Step 2: Run to verify the current state**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_freeze_contracts.py -q`
Expected: `4 passed`. **If `test_no_trading_authentication_anywhere_in_src` fails, stop** — something in `src/` already reaches a trading endpoint and that contradicts the spec's scope.

- [ ] **Step 3: Render the verdict on the board**

In `src/highcast/board.py`, inside the function that renders a market's row (search for where `hour_accuracy` is read, around line 341), add the refusal line. Read the surrounding formatting first and match it:

```python
        # ONE verdict, from `gating.decision_gates`, so the terminal and the
        # dashboard can never disagree about whether today is actionable.
        verdict = gating.decision_gates(
            conn, cfg, metric=metric, target=target_date, hour=hour,
            p_model=p_model)
        if not verdict.ok:
            lines.append(f"   GATES: REFUSE ({', '.join(verdict.refusals)})")
        else:
            lines.append("   GATES: clear")
```

Add `from highcast import gating` to the imports if `gating` is not already imported as a module (it currently imports names individually at line 38 — extend that import rather than adding a second style).

- [ ] **Step 4: Add a board test**

Append to `tests/test_board.py`:

```python
def test_the_board_shows_the_gate_refusals(tmp_path, monkeypatch):
    """The desk must see WHY it may not act, not merely that it may not."""
    from highcast import board, gating

    monkeypatch.setattr(
        gating, "decision_gates",
        lambda *a, **k: gating.GateVerdict(
            ok=False, refusals=("guidance_lead", "confidence_floor")))
    text = board.render(*_board_args(tmp_path))
    assert "GATES: REFUSE" in text
    assert "guidance_lead" in text
    assert "confidence_floor" in text
```

⚠️ `_board_args` is a placeholder for whatever fixture the existing `tests/test_board.py` uses to build a board — read the file's existing tests and reuse their setup verbatim rather than inventing one.

- [ ] **Step 5: Run the board tests**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_board.py -q`
Expected: PASS.

- [ ] **Step 6: Run the full suite**

Run: `PYTHONPATH=src .venv/bin/python -m pytest -q`
Expected: `648 passed, 12 deselected`.

- [ ] **Step 7: Restart the dashboard so it serves the new gates**

```bash
launchctl kickstart -k gui/$(id -u)/com.julian.highcast.serve
```

Then verify it is up: `curl -s -o /dev/null -w "HTTP %{http_code}\n" http://127.0.0.1:8712/`
Expected: `HTTP 200`.

---

## Self-Review

**Spec coverage:**

| spec section | task |
|---|---|
| §3 pre-registered hour-move gate | Task 1 |
| §3 apply the move, cutoffs unchanged | Task 2 |
| §4 Gate A — not built, deleted | no task, by design (§4 documents why) |
| §4 Gate B — floor rule + re-derived bands | Tasks 3, 6 |
| §4 Gate C — guidance freshness | Task 4 |
| §4 Gate C — `resolve_price` + `max_spread` recheck | Task 5 |
| §4 price rail | Task 7 (optional arg, tests only) |
| §5 fail-closed aggregator, single verdict | Tasks 7, 8 |
| §6 testing — refusal in the bad regime | Tasks 4, 5, 6, 7 |
| §6 freeze tests | Task 8 |
| §7 no trading code | Task 8 `test_no_trading_authentication_anywhere_in_src` |

No spec requirement is unassigned. Gate A is the one section with no task, which is the spec's own conclusion rather than a gap.

**Placeholder scan:** one deliberate marker remains — `_board_args` in Task 8 Step 4, flagged inline with an instruction to reuse the existing fixture. Every other step carries runnable code or an exact command.

**Type consistency:** `GateVerdict(ok: bool, refusals: tuple[str, ...])` is constructed in Task 7 and consumed in Task 8's monkeypatch with the same field names. `confidence_floor_ok(station, metric, hour, p_hat)` and `guidance_lead_ready(conn, cfg, target)` are called in Task 7 exactly as defined in Tasks 6 and 4. `band_calib(station, metric, hour, p_model)` matches the existing signature. Refusal strings (`nbs_missing`, `guidance_lead`, `unmeasured_band`, `confidence_floor`, `price_rail`) are identical between the implementation and every test.

**Running test counts — CORRECTED 2026-08-01 after execution.** The predicted
sequence was 614 → 616 (T2) → 616 (T3) → 621 → 626 → 634 → 643 → 648.

**T2's prediction was WRONG and the plan contradicted itself.** `BAND_CALIB` and
`HOUR_CALIB` are keyed `(station, metric, hour)`, so moving the hours orphaned
every entry the instant T2 landed: `band_calib()` returned None for all four
markets, `p_hat` went None downstream, and the suite dropped to **26 failed /
590 passed**. That is the fail-closed contract working — every market correctly
refusing — but the plan's Known Risks section named that inert window while its
task text still predicted 616. **Tasks 2 and 3 are an ATOMIC PAIR; do not split
them across a stopping point.**

Actual: 614 → *(26 failed)* → 617 (T3) → 627 (T4+T5) → 644 (T6+T7) → 650 (T8 and
the out-of-plan fixes).

---

## Known risks

- **Task 1 may return `KEEP`** for one or more markets. If KMIA HIGH keeps h11, Task 6's floor table is wrong (0.70 was measured at h14) and Task 6 must be re-measured at whatever hour survives. Do not carry an h14 floor to h11.
- **Task 3 is the longest step** (~30–60 min of walk-forward fitting) and is a hard dependency of Tasks 6 and 7. Between Task 2 and Task 3 every market refuses, because `band_calib` returns `None` for the new keys. That window is safe but the app is inert; do not stop there.
- **Task 8 Step 3 edits a render function this plan has not read in full.** Match the surrounding formatting rather than the illustrative snippet.
