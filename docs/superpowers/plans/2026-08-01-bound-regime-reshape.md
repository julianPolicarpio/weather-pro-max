# Bound-Regime pmf Reshape Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reshape the KMIA LOW ladder pmf at h7 using the empirical drop-magnitude distribution for the day's (bound level × guidance disagreement) regime, so the model stops putting 37 % on the bucket adjacent to an already-observed minimum.

**Architecture:** A pure post-hoc layer between the stored prediction and the bucket pmf. The frozen model is not retrained. A new module `boundregime.py` computes the regime key and the reshaped pmf; a fitted table in `gating.py` holds the per-cell drop histograms and blend weights; a script runs the walk-forward and the pre-registered gate. The layer is a no-op (`w = 1.0`, input returned unchanged) on every unknown, and is wired into the live path **only if the gate passes**.

**Tech Stack:** Python 3.14, sqlite3, pytest. `pandas` is available but not needed. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-01-bound-regime-reshape-design.md`

## Global Constraints

- **This project is NOT a git repository.** Never run `git` and never offer to commit. Where a plan would normally commit, run the full suite instead.
- **Run tests with the venv:** `.venv/bin/python -m pytest`. Bare `python` is not on PATH.
- **Scope is `("KMIA", "low", 7)` and months `(6, 7, 8, 9)`.** Any other station, metric, hour or month must return the input pmf unchanged.
- **`MIN_CELL_N = 40`** — pre-registered in spec §3.3. May not be lowered.
- **The bound is evaluated with `asof` = the decision hour, never later.** The bound is still falling between h7 and h8 on 40.2 % of days; a later `asof` is look-ahead.
- **Guidance uses `run_date < target_date` (strict).** `forecasts.run_date` is a DATE, so a same-day row may be an 18Z run unavailable at the 11Z decision.
- **`fetched_at` must never appear in an availability filter.** `db.upsert` is `INSERT OR REPLACE` and the `forecasts` primary key excludes `fetched_at`, so it is overwritten on every re-fetch.
- **Never derive a daily extreme from raw `obs` snapshots.** Use `latentmax.visible_running_extremum`, which folds in the `six_min_c` remark groups (project CLAUDE.md).

## Existing interfaces this plan consumes

```python
latentmax.visible_running_extremum(conn, cfg, day: date, extremum: str,
                                   asof_utc: str | None = None)
    # -> (lo, hi) interval of true temperatures, or None

buckets.pmf_from_buckets(buckets: dict | None) -> dict[int, float] | None
    # predictions.buckets is JSON {"81": 0.06, ...}; keys become ints
buckets.pmf_mass(pmf: dict | None, lo: int | None, hi: int | None) -> float | None

gating.min_decision_hour(station: str, metric: str) -> int | None
```

`gating.BAND_CALIB` is the structural precedent for the fitted table: a
module-level dict keyed by `(station, metric, hour)`, read through an accessor
that returns `None` on anything unmeasured.

## File Structure

| File | Responsibility |
|---|---|
| `src/highcast/boundregime.py` (create) | Everything: regime inputs, key, fit, reshape, blend, and the single `apply()` entry point. Pure — no network, no DB writes. |
| `src/highcast/gating.py` (modify) | Holds the fitted `BOUND_REGIME` table only. No logic. |
| `scripts/bound_regime_eval.py` (create) | Walk-forward fit, pre-registered gate, negative control. Emits the table. |
| `tests/test_boundregime.py` (create) | Fail-closed rails, look-ahead rails, scope containment. |

One module rather than three: the pieces are ~200 lines total and are only ever
used together, and this codebase already keeps cohesive units in one file
(`gating.py`, `notify.py`). Splitting would create interfaces with one caller each.

---

### Task 1: Regime inputs and key

**Files:**
- Create: `src/highcast/boundregime.py`
- Test: `tests/test_boundregime.py`

**Interfaces:**
- Consumes: `latentmax.visible_running_extremum`
- Produces:
  - `SCOPE: tuple[str, str, int]` = `("KMIA", "low", 7)`
  - `SEASON_MONTHS: tuple[int, ...]` = `(6, 7, 8, 9)`
  - `MIN_CELL_N: int` = `40`
  - `LEVEL_BANDS`, `GAP_BANDS`: `tuple[tuple[float | None, float | None], ...]`
  - `bound_for(conn, cfg, day: date, hour: int) -> float | None`
  - `guidance_median(conn, station: str, day: date) -> float | None`
  - `regime_key(bound: float | None, gap: float | None) -> tuple[int, int] | None`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_boundregime.py
"""The regime layer. Every test here is about it REFUSING to act.

The layer's job is to reshape a pmf; its safety property is that it returns the
input UNCHANGED on every unknown. A reshape that fires on a cell it has no data
for is worse than no layer, because it launders a guess as a measurement.
"""
import datetime as dt

import pytest
import yaml

from highcast import boundregime as br
from highcast import db as hdb
from highcast.config import load_config


def _cfg(tmp_path, src="config/kmia.yaml"):
    raw = yaml.safe_load(open(src).read())
    raw["db_path"] = str(tmp_path / "t.db")
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(raw))
    return load_config(str(p))


def test_the_scope_is_exactly_kmia_low_h7():
    """Widening scope must be a deliberate edit, not a side effect. The spec
    gates one market; the HIGHs have the opposite shape (the extremum lands
    late, so the bound is uninformative for most of the day)."""
    assert br.SCOPE == ("KMIA", "low", 7)
    assert br.SEASON_MONTHS == (6, 7, 8, 9)


def test_min_cell_n_is_pinned_at_the_pre_registered_value():
    """Pre-registered in spec 3.3. Lowering it after seeing results is the
    classic way a gate gets talked into passing, so it must be a visible edit
    to a test rather than a quiet constant change."""
    assert br.MIN_CELL_N == 40


@pytest.mark.parametrize("bound,gap,expected", [
    (78.0, 0.2, (0, 0)),
    (80.0, 1.0, (1, 1)),
    (82.14, 2.1, (2, 2)),      # 2026-08-01, the worked example in the spec
    (84.0, 3.0, (3, 3)),
    (81.0, 1.5, (2, 2)),       # band edges are inclusive-low, exclusive-high
    (83.0, 2.5, (3, 3)),
])
def test_regime_key_bands(bound, gap, expected):
    assert br.regime_key(bound, gap) == expected


@pytest.mark.parametrize("bound,gap", [
    (None, 1.0), (82.0, None), (None, None),
    (float("nan"), 1.0), (82.0, float("nan")),
])
def test_regime_key_fails_closed_on_every_unknown(bound, gap):
    """For an unattended caller, "I could not tell" and "no" are the same
    answer -- the same rule gating.confidence_floor_ok follows."""
    assert br.regime_key(bound, gap) is None


def test_guidance_median_uses_only_runs_from_BEFORE_the_target_date(tmp_path):
    """forecasts.run_date is a DATE, so a same-day row may be an 18Z run that
    did not exist at the 11Z decision. Strict inequality is the only safe
    filter, and it costs nothing: correlation with the drop is +0.281 either
    way (spec 2b)."""
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    day = dt.date(2026, 8, 1)
    conn.executemany(
        """INSERT INTO forecasts (station, target_date, model, run_date,
                                  max_temp_f, metric)
           VALUES (?,?,?,?,?,?)""",
        [("KMIA", "2026-08-01", "gfs",   "2026-07-31", 77.0, "low"),
         ("KMIA", "2026-08-01", "ecmwf", "2026-07-31", 79.0, "low"),
         # SAME-DAY run: must be excluded, it may be an 18Z run.
         ("KMIA", "2026-08-01", "nam",   "2026-08-01", 60.0, "low")])
    conn.commit()
    assert br.guidance_median(conn, "KMIA", day) == 78.0


def test_guidance_median_is_none_when_nothing_is_available(tmp_path):
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    assert br.guidance_median(conn, "KMIA", dt.date(2026, 8, 1)) is None


def test_guidance_median_ignores_the_other_metric(tmp_path):
    """`max_temp_f` holds the LOW when metric='low' -- a column name that has
    caused this exact confusion before."""
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    conn.executemany(
        """INSERT INTO forecasts (station, target_date, model, run_date,
                                  max_temp_f, metric) VALUES (?,?,?,?,?,?)""",
        [("KMIA", "2026-08-01", "gfs", "2026-07-31", 77.0, "low"),
         ("KMIA", "2026-08-01", "gfs", "2026-07-31", 95.0, "high")])
    conn.commit()
    assert br.guidance_median(conn, "KMIA", dt.date(2026, 8, 1)) == 77.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_boundregime.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'highcast.boundregime'`

- [ ] **Step 3: Write the implementation**

```python
# src/highcast/boundregime.py
"""Regime-conditional pmf reshape for the KMIA LOW at h7.

Design: `docs/superpowers/specs/2026-08-01-bound-regime-reshape-design.md`

WHAT THIS FIXES. The intraday model already consumes the observed running
minimum as `run_lo`/`run_hi`, but nothing tells it WHICH of its two information
sources to trust today. `models/intraday.py` measured the same blind spot from
the other side and left the fix unshipped: guidance error is flat across
regimes while the bound's DOUBLES, and the model weights them identically.

On 2026-08-01 that ran in reverse -- the minimum had already landed (82.04F,
flat three hours, sun up), which is the regime where the bound is the reliable
side, and the model still emitted 80.0 tracking guidance 2-5F below an observed
fact. It put 37.4% on the bucket adjacent to the observed minimum, where the
regime-conditional rate is 22%.

WHY A RESHAPE AND NOT A SHIFT. The drop magnitude D = bound - settled is
zero-inflated: 65% of days never go below the bound at all, and the MEDIAN D is
0.16F in every single regime cell. All the signal is in the tail probability and
none of it in the centre, so a point forecast cannot carry it and a smooth
unimodal pmf cannot represent it.

PURE MODULE. No network, no DB writes, no model. It reads obs and forecasts and
returns a dict. That is what makes the money-free parts testable without either.
"""
from __future__ import annotations

import datetime as dt
import statistics
from zoneinfo import ZoneInfo

from highcast import latentmax

# The one (station, metric, hour) this layer may touch. Widening it is a
# deliberate edit guarded by a test -- the HIGH markets have the opposite shape
# (the extremum lands late, so the bound is uninformative most of the day) and
# would need their own gate.
SCOPE = ("KMIA", "low", 7)
SEASON_MONTHS = (6, 7, 8, 9)

# Pre-registered in spec 3.3. May NOT be lowered after seeing results.
MIN_CELL_N = 40

# Bound level bands. REQUIRED as a key dimension: D is not translation-
# invariant. Measured at h7 over 789 Jun-Sep days, P(D >= 1.5F) runs
# 22.2 / 34.9 / 40.4 / 36.6 % across these bands, so pooling would import
# cool-morning behaviour into a warm-morning forecast.
LEVEL_BANDS = ((None, 79.0), (79.0, 81.0), (81.0, 83.0), (83.0, None))

# Guidance-disagreement bands. bound - guidance_median; positive means guidance
# is COLDER than what has already been observed. P(D >= 1.5F) runs
# 19.6 / 36.0 / 40.1 / 44.1 % across these -- disagreement doubles the drop rate.
GAP_BANDS = ((None, 0.5), (0.5, 1.5), (1.5, 2.5), (2.5, None))


def _band(value: float, bands) -> int | None:
    """Index of the band containing `value`. Inclusive low, exclusive high."""
    for i, (lo, hi) in enumerate(bands):
        if (lo is None or value >= lo) and (hi is None or value < hi):
            return i
    return None


def regime_key(bound: float | None, gap: float | None) -> tuple[int, int] | None:
    """(level_index, gap_index), or None meaning "fail closed".

    None on any missing or NaN input. For an unattended caller "I could not
    tell" and "no" must be the same answer -- the rule `confidence_floor_ok`
    already follows.
    """
    if bound is None or gap is None:
        return None
    if bound != bound or gap != gap:          # NaN
        return None
    lv, gp = _band(bound, LEVEL_BANDS), _band(gap, GAP_BANDS)
    if lv is None or gp is None:
        return None
    return (lv, gp)


def bound_for(conn, cfg, day: dt.date, hour: int) -> float | None:
    """Upper end of the running-MIN interval as of `hour` LOCAL. Never later.

    Upper end, not lower: it is the tightest value consistent with everything
    observed, and the settled low cannot exceed it.

    THE HOUR IS LOAD-BEARING. The bound is still falling between h7 and h8 on
    40.2% of days, so evaluating one hour late inflated the measured skill by
    ~7 points on the worked example. Callers pass the DECISION hour.
    """
    asof = dt.datetime(day.year, day.month, day.day, hour,
                       tzinfo=ZoneInfo(cfg.station.tz)).astimezone(dt.timezone.utc)
    iv = latentmax.visible_running_extremum(conn, cfg, day, "min",
                                            asof_utc=asof.isoformat())
    return None if iv is None else iv[1]


def guidance_median(conn, station: str, day: dt.date) -> float | None:
    """Unweighted median guidance low from runs STRICTLY BEFORE the target date.

    `max_temp_f` holds the LOW when metric='low' -- the column name is a
    historical wart, not a bug.

    NO MODEL ALLOW-LIST. Naming a subset would need its own justification and
    the set has changed twice this month. Note `gfs` and `hrrr` are a known
    duplicated channel, so one model effectively votes twice here; that is
    deliberate and recorded in spec 6, because fixing it inside this change
    would confound the gate.

    `run_date < target_date` IS STRICT, and `fetched_at` IS NOT USED. `run_date`
    is a DATE, so a same-day row may be an 18Z run that did not exist at the 11Z
    decision. And `fetched_at` cannot stand in: `db.upsert` is INSERT OR REPLACE
    and the `forecasts` primary key excludes it, so it is overwritten on every
    re-fetch -- the artifact found 2026-07-31.
    """
    iso = day.isoformat()
    rows = conn.execute(
        """SELECT max_temp_f FROM forecasts
           WHERE station=? AND metric='low' AND target_date=?
             AND run_date < ? AND max_temp_f IS NOT NULL""",
        (station, iso, iso)).fetchall()
    vals = [float(r[0]) for r in rows]
    return statistics.median(vals) if vals else None
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_boundregime.py -q`
Expected: PASS — 11 passed

- [ ] **Step 5: Run the full suite (this project has no git; the suite is the checkpoint)**

Run: `.venv/bin/python -m pytest -q`
Expected: `751 passed` (740 existing + 11 new), 12 deselected. If any pre-existing test fails, stop and report — do not proceed.

---

### Task 2: Fit the cell table from history

**Files:**
- Modify: `src/highcast/boundregime.py`
- Test: `tests/test_boundregime.py`

**Interfaces:**
- Consumes: `bound_for`, `guidance_median`, `regime_key` from Task 1
- Produces:
  - `observations(conn, cfg, days: list[date], hour: int) -> list[dict]` — each `{"day": date, "bound": float, "gap": float, "delta": float, "cell": tuple[int,int]}`
  - `fit(observations: list[dict]) -> dict[tuple[int,int], dict]` — cell → `{"n": int, "hist": dict[float, int]}`

- [ ] **Step 1: Write the failing tests**

```python
def test_fit_groups_deltas_by_cell_and_counts_them():
    """The estimator does not MODEL zero-inflation, it COUNTS it. That is the
    whole reason this is an empirical table rather than a fitted distribution:
    a spike at zero plus a decaying tail has no convenient parametric form, and
    every smooth family we could fit would reintroduce the defect."""
    obs = [
        {"day": None, "bound": 82.0, "gap": 2.0, "delta": 0.0, "cell": (2, 2)},
        {"day": None, "bound": 82.0, "gap": 2.0, "delta": 0.0, "cell": (2, 2)},
        {"day": None, "bound": 82.0, "gap": 2.0, "delta": 1.8, "cell": (2, 2)},
        {"day": None, "bound": 80.0, "gap": 1.0, "delta": 3.6, "cell": (1, 1)},
    ]
    table = br.fit(obs)
    assert table[(2, 2)]["n"] == 3
    assert table[(2, 2)]["hist"] == {0.0: 2, 1.8: 1}
    assert table[(1, 1)]["n"] == 1


def test_fit_rounds_deltas_to_a_tenth_so_the_table_stays_compact():
    """Temperatures arrive as tenths of a degree C via the METAR T-group, so
    finer resolution stores noise, not signal."""
    obs = [{"day": None, "bound": 82.0, "gap": 2.0, "delta": d, "cell": (2, 2)}
           for d in (1.7999, 1.8001, 1.84)]
    assert br.fit(obs)[(2, 2)]["hist"] == {1.8: 3}


def test_observations_skips_days_outside_the_season(tmp_path):
    """v1 is Jun-Sep. A January day has a different mechanism entirely and
    would pollute the cells with behaviour that cannot recur in August."""
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    out = br.observations(conn, cfg, [dt.date(2026, 1, 15)], hour=7)
    assert out == []


def test_observations_skips_days_with_no_bound_or_no_guidance(tmp_path):
    """An empty DB yields no bound and no guidance. Both must drop the day
    silently rather than raise -- `fit` runs over years of history and one bad
    day must not abort it."""
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    assert br.observations(conn, cfg, [dt.date(2026, 8, 1)], hour=7) == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_boundregime.py -q -k "fit or observations"`
Expected: FAIL with `AttributeError: module 'highcast.boundregime' has no attribute 'fit'`

- [ ] **Step 3: Write the implementation (append to `boundregime.py`)**

```python
def observations(conn, cfg, days: list[dt.date], hour: int) -> list[dict]:
    """One record per usable day: bound, gap, realised drop, regime cell.

    Silently drops a day that lacks any input. `fit` runs over years of history
    and a single missing METAR must not abort it -- the same reason
    `load_credentials` never raises.

    The realised drop D = bound - settled low comes from `cli_reports`, the
    settlement source. NOT from obs: the settled low is what Kalshi pays on.
    """
    out = []
    for day in days:
        if day.month not in SEASON_MONTHS:
            continue
        row = conn.execute(
            "SELECT low_f FROM cli_reports WHERE station=? AND date=?",
            (cfg.station.icao, day.isoformat())).fetchone()
        if row is None or row[0] is None:
            continue
        bound = bound_for(conn, cfg, day, hour)
        gap_ref = guidance_median(conn, cfg.station.icao, day)
        if bound is None or gap_ref is None:
            continue
        gap = bound - gap_ref
        cell = regime_key(bound, gap)
        if cell is None:
            continue
        out.append({"day": day, "bound": bound, "gap": gap,
                    "delta": bound - float(row[0]), "cell": cell})
    return out


def fit(observations: list[dict]) -> dict[tuple[int, int], dict]:
    """cell -> {"n": count, "hist": {delta_rounded: count}}.

    A histogram rather than a raw list so the shipped table stays small, and
    rounded to a tenth of a degree because that is the resolution temperatures
    actually arrive at (the METAR T-group reports 0.1C).
    """
    table: dict[tuple[int, int], dict] = {}
    for o in observations:
        cell = table.setdefault(o["cell"], {"n": 0, "hist": {}})
        d = round(float(o["delta"]), 1)
        cell["n"] += 1
        cell["hist"][d] = cell["hist"].get(d, 0) + 1
    return table
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_boundregime.py -q`
Expected: PASS — 15 passed

- [ ] **Step 5: Verify against real data (this is the step that catches a wrong join)**

Run:
```bash
.venv/bin/python -c "
import datetime as dt
from highcast import boundregime as br, db
from highcast.config import load_config
cfg = load_config('config/kmia.yaml'); conn = db.connect(cfg.db_path)
days = [dt.date.fromisoformat(r[0]) for r in conn.execute(
    \"SELECT date FROM cli_reports WHERE station='KMIA' AND low_f IS NOT NULL ORDER BY date\")]
obs = br.observations(conn, cfg, days, hour=7)
t = br.fit(obs)
print('observations:', len(obs))
for cell in sorted(t): print(' ', cell, 'n =', t[cell]['n'])
print('cells >= MIN_CELL_N:', sum(1 for c in t if t[c]['n'] >= br.MIN_CELL_N))
"
```
Expected, matching spec §3.1 exactly:
```
observations: 789
  (0, 0) n = 215     (0, 1) n = 37     (0, 2) n = 13     (0, 3) n = 1
  (1, 0) n = 74      (1, 1) n = 87     (1, 2) n = 49     (1, 3) n = 19
  (2, 0) n = 10      (2, 1) n = 33     (2, 2) n = 95     (2, 3) n = 55
  (3, 0) n = 2       (3, 1) n = 4      (3, 2) n = 25     (3, 3) n = 70
cells >= MIN_CELL_N: 7
```
**If `observations` is not 789 or the cell counts differ, stop.** The spec's
numbers came from this exact definition; a mismatch means the join or the
`asof` differs from what was measured, and every downstream number would be
wrong in a way the gate cannot detect.

- [ ] **Step 6: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: `755 passed`

---

### Task 3: Reshape and blend

**Files:**
- Modify: `src/highcast/boundregime.py`
- Test: `tests/test_boundregime.py`

**Interfaces:**
- Consumes: `fit` output from Task 2
- Produces:
  - `reshape(hist: dict[float, int], bound: float) -> dict[int, float] | None` — pmf over integer °F
  - `blend(model_pmf, layer_pmf, w: float) -> dict[int, float] | None`

- [ ] **Step 1: Write the failing tests**

```python
def test_reshape_applies_the_drop_histogram_to_todays_bound():
    """4 of 4 days dropped 0, 0, 1.8 and 3.6 from a bound of 82.14 -> settles
    82, 82, 80, 78."""
    pmf = br.reshape({0.0: 2, 1.8: 1, 3.6: 1}, bound=82.14)
    assert pmf == {82: 0.5, 80: 0.25, 78: 0.25}


def test_reshape_produces_a_pmf_that_sums_to_one():
    pmf = br.reshape({0.0: 7, 1.8: 3, 5.4: 1}, bound=81.4)
    assert abs(sum(pmf.values()) - 1.0) < 1e-9


def test_reshape_returns_none_on_an_empty_histogram_or_missing_bound():
    assert br.reshape({}, bound=82.0) is None
    assert br.reshape({0.0: 5}, bound=None) is None


def test_blend_at_weight_one_returns_the_MODEL_PMF_ITSELF():
    """Identity, asserted on object identity rather than on rounded values.
    w=1.0 is the default on every unknown, so if it is only approximately the
    input then the layer silently perturbs every prediction it declines to
    touch -- a no-op that is not a no-op."""
    model = {81: 0.4, 80: 0.6}
    assert br.blend(model, {79: 1.0}, w=1.0) is model


def test_blend_mixes_both_distributions_at_an_intermediate_weight():
    out = br.blend({81: 1.0}, {79: 1.0}, w=0.25)
    assert out == {81: 0.25, 79: 0.75}


def test_blend_falls_back_to_the_model_when_the_layer_is_missing():
    model = {81: 1.0}
    assert br.blend(model, None, w=0.5) is model


def test_blend_refuses_a_result_that_is_not_a_probability_distribution():
    """A pmf that does not sum to 1 means an upstream bug, and shipping it
    would silently corrupt every downstream edge calculation. Fail closed to
    the model rather than normalise -- normalising would HIDE the bug."""
    model = {81: 1.0}
    assert br.blend(model, {79: 0.3}, w=0.5) is model
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_boundregime.py -q -k "reshape or blend"`
Expected: FAIL with `AttributeError: ... has no attribute 'reshape'`

- [ ] **Step 3: Write the implementation (append to `boundregime.py`)**

```python
def reshape(hist: dict[float, int], bound: float | None) -> dict[int, float] | None:
    """Empirical drop histogram + today's bound -> pmf over integer degrees F.

    Integer degrees because that is what `cli_reports` settles at and what
    `buckets.pmf_mass` consumes, so the result drops straight into the existing
    ladder path with no adapter.
    """
    if not hist or bound is None:
        return None
    total = sum(hist.values())
    if total <= 0:
        return None
    pmf: dict[int, float] = {}
    for delta, count in hist.items():
        t = int(round(bound - delta))
        pmf[t] = pmf.get(t, 0.0) + count / total
    return pmf


def blend(model_pmf: dict | None, layer_pmf: dict | None,
          w: float) -> dict | None:
    """w * model + (1-w) * layer.

    w = 1.0 returns THE MODEL OBJECT ITSELF, not a copy that happens to be
    equal. w=1.0 is the default on every unknown, so an "identity" that only
    approximately reproduced its input would silently perturb every prediction
    the layer declined to touch.
    """
    if model_pmf is None:
        return None
    if w >= 1.0 or layer_pmf is None:
        return model_pmf
    out = {k: w * model_pmf.get(k, 0.0) + (1.0 - w) * layer_pmf.get(k, 0.0)
           for k in set(model_pmf) | set(layer_pmf)}
    if abs(sum(out.values()) - 1.0) > 1e-6:
        # Upstream bug. Normalising here would hide it; the model pmf is the
        # known-good value and the layer is optional by construction.
        return model_pmf
    return out
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_boundregime.py -q`
Expected: PASS — 22 passed

- [ ] **Step 5: Reproduce the spec's worked example**

Run:
```bash
.venv/bin/python -c "
import datetime as dt
from highcast import boundregime as br, db
from highcast.config import load_config
cfg = load_config('config/kmia.yaml'); conn = db.connect(cfg.db_path)
days = [dt.date.fromisoformat(r[0]) for r in conn.execute(
    \"SELECT date FROM cli_reports WHERE station='KMIA' AND low_f IS NOT NULL ORDER BY date\")]
t = br.fit(br.observations(conn, cfg, days, hour=7))
pmf = br.reshape(t[(2, 2)]['hist'], bound=82.14)
from highcast.buckets import pmf_mass
for lab, lo, hi in [('>=81',81,None),('79-80',79,80),('77-78',77,78),
                    ('75-76',75,76),('73-74',73,74),('<=72',None,72)]:
    print(f'  {lab:>6} {pmf_mass(pmf, lo, hi):6.1%}')
"
```
Expected, matching spec §1b:
```
    >=81  58.9%
   79-80  22.1%
   77-78  15.8%
   75-76   2.1%
   73-74   1.1%
   <=72   0.0%
```
**If these differ, stop and report.** This is the number the whole design rests on.

- [ ] **Step 6: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: `762 passed`

---

### Task 4: The `apply` entry point and scope containment

**Files:**
- Modify: `src/highcast/boundregime.py`
- Modify: `src/highcast/gating.py` (add the empty `BOUND_REGIME` table after `CONFIDENCE_FLOOR`, near line 475)
- Test: `tests/test_boundregime.py`

**Interfaces:**
- Consumes: `regime_key`, `reshape`, `blend`, `bound_for`, `guidance_median`
- Produces: `apply(conn, cfg, *, station, metric, hour, day, model_pmf, table=None) -> dict | None`
- Produces: `gating.BOUND_REGIME: dict[tuple[int, int], dict]` — cell → `{"n": int, "w": float, "hist": dict[float, int]}`

- [ ] **Step 1: Write the failing tests**

```python
def _table(n=50, w=0.5):
    return {(2, 2): {"n": n, "w": w, "hist": {0.0: 30, 1.8: 12, 3.6: 8}},
            (2,): {"n": n, "w": w, "hist": {0.0: 40, 1.8: 10}}}


@pytest.mark.parametrize("station,metric,hour", [
    ("KLAX", "low", 7), ("KMIA", "high", 7), ("KMIA", "low", 6),
    ("KMIA", "low", 14),
])
def test_apply_is_a_no_op_outside_its_scope(tmp_path, station, metric, hour):
    """Freeze-contract style. The HIGHs have the opposite shape -- the extremum
    lands late, so the bound is uninformative for most of the day -- and KLAX
    is a different mechanism (marine layer, not convection). Each needs its own
    gate, so widening scope must require deleting an assertion."""
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    model = {81: 0.4, 80: 0.6}
    out = br.apply(conn, cfg, station=station, metric=metric, hour=hour,
                   day=dt.date(2026, 8, 1), model_pmf=model, table=_table())
    assert out is model


def test_apply_is_a_no_op_outside_the_season(tmp_path):
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    model = {81: 0.4, 80: 0.6}
    out = br.apply(conn, cfg, station="KMIA", metric="low", hour=7,
                   day=dt.date(2026, 1, 15), model_pmf=model, table=_table())
    assert out is model


def test_apply_is_a_no_op_when_the_cell_is_below_MIN_CELL_N(tmp_path):
    """9 of 16 cells never reach n=40, so backoff is the normal path rather
    than a corner case."""
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    model = {81: 0.4, 80: 0.6}
    out = br.apply(conn, cfg, station="KMIA", metric="low", hour=7,
                   day=dt.date(2026, 8, 1), model_pmf=model,
                   table=_table(n=br.MIN_CELL_N - 1))
    assert out is model


def test_apply_is_a_no_op_on_an_empty_database(tmp_path):
    """No obs means no bound; no forecasts means no gap. Either one must leave
    the prediction untouched rather than raise inside the wake cycle."""
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    model = {81: 0.4, 80: 0.6}
    out = br.apply(conn, cfg, station="KMIA", metric="low", hour=7,
                   day=dt.date(2026, 8, 1), model_pmf=model, table=_table())
    assert out is model


def test_apply_never_raises_whatever_the_database_contains(tmp_path):
    """This runs inside the wake cycle that also fetches observations and
    issues predictions. A layer bug must not cost the weather pipeline -- the
    same rule botrunner.run_once follows."""
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)

    class Exploding:
        def execute(self, *a, **k):
            raise RuntimeError("boom")

    model = {81: 1.0}
    assert br.apply(Exploding(), cfg, station="KMIA", metric="low", hour=7,
                    day=dt.date(2026, 8, 1), model_pmf=model,
                    table=_table()) is model


def test_the_shipped_table_starts_empty_so_the_layer_ships_disabled():
    """The table is populated by scripts/bound_regime_eval.py only AFTER the
    gate passes. An empty table means every cell backs off to the pure model,
    which is the correct state until there is evidence."""
    from highcast import gating
    assert gating.BOUND_REGIME == {}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_boundregime.py -q -k "apply or shipped_table"`
Expected: FAIL — `has no attribute 'apply'`

- [ ] **Step 3: Add the empty table to `gating.py`**

Insert immediately after the `CONFIDENCE_FLOOR` dict (which ends near line 478):

```python
# ── BOUND-REGIME RESHAPE ────────────────────────────────────────────────────
#
# cell (level_index, gap_index) -> {"n": days, "w": blend weight, "hist": {D: count}}
#
# Populated by `scripts/bound_regime_eval.py` ONLY IF the pre-registered gate in
# `docs/superpowers/specs/2026-08-01-bound-regime-reshape-design.md` §4 passes on
# the held-out 2025-26 seasons. EMPTY MEANS DISABLED, and empty is the correct
# state until there is evidence: every cell backs off to the frozen model.
#
# `w` is the weight on the MODEL, so w = 1.0 disables a cell individually. If the
# fit returns 1.0 everywhere, the honest reading is "this did not help".
BOUND_REGIME: dict[tuple[int, ...], dict] = {}
```

- [ ] **Step 4: Write `apply` (append to `boundregime.py`)**

```python
def apply(conn, cfg, *, station: str, metric: str, hour: int, day: dt.date,
          model_pmf: dict | None, table: dict | None = None) -> dict | None:
    """THE entry point. Returns `model_pmf` UNCHANGED on every unknown.

    NEVER RAISES. This runs inside the wake cycle that also fetches
    observations and issues predictions; a layer bug must not cost the weather
    pipeline. Same rule `botrunner.run_once` follows.

    Backoff ladder (spec 3.3): full cell -> level-only cell -> pure model.
    """
    if table is None:
        from highcast import gating
        table = gating.BOUND_REGIME
    try:
        if (station, metric, hour) != SCOPE:
            return model_pmf
        if day.month not in SEASON_MONTHS:
            return model_pmf
        if model_pmf is None or not table:
            return model_pmf

        bound = bound_for(conn, cfg, day, hour)
        gap_ref = guidance_median(conn, station, day)
        cell = regime_key(bound, None if gap_ref is None else bound - gap_ref)
        if cell is None:
            return model_pmf

        for key in (cell, (cell[0],)):          # full cell, then level-only
            entry = table.get(key)
            if entry and entry.get("n", 0) >= MIN_CELL_N:
                return blend(model_pmf, reshape(entry.get("hist"), bound),
                             float(entry.get("w", 1.0)))
        return model_pmf
    except Exception:  # noqa: BLE001 - the weather pipeline must survive
        return model_pmf
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_boundregime.py -q`
Expected: PASS — 31 passed

- [ ] **Step 6: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: `771 passed`

---

### Task 5: The walk-forward gate

**Files:**
- Create: `scripts/bound_regime_eval.py`

**Interfaces:**
- Consumes: `boundregime.observations`, `fit`, `reshape`, `blend`
- Produces: a printed verdict and, on PASS, the `BOUND_REGIME` literal to paste into `gating.py`

**This task produces evidence, not a shipped change.** Nothing is wired in here.

- [ ] **Step 1: Write the script**

```python
# scripts/bound_regime_eval.py
"""PRE-REGISTERED GATE: does the bound-regime reshape beat the frozen model?

Pre-registration (written before this ran):
`docs/superpowers/specs/2026-08-01-bound-regime-reshape-design.md` §4

THE DISCOVERY-BIAS PROBLEM, and why the split is what it is. This effect was
FOUND in this data on 2026-08-01. An expanding-window walk-forward over the same
history is therefore contaminated -- the answer is already known. So:

    FIT on   Jun-Sep 2020-2024
    SCORE on Jun-Sep 2025-2026   (untouched during discovery)

PRIMARY: mean log score of the six-bucket ladder pmf, layer vs frozen model,
paired bootstrap over held-out days, p < 0.05. One hypothesis, one market, so no
Bonferroni. A proper scoring rule, because the change is a distribution reshape
and bucket hit rate is too coarse to detect "moved mass out of the middle".

GUARDRAILS, both must hold:
    bucket hit rate not worse by more than 1.0 pt
    top-bucket mean |calibration error| not worse by more than 2.0 pt
                   (10-pt bins, bins with < 10 days excluded)

NEGATIVE CONTROL: the identical pipeline with the regime key SHUFFLED across
days. If the shuffle also passes, the gate is measuring overfitting and the real
result is void.
"""
import datetime as dt
import json
import random
import sys

sys.path.insert(0, "src")

from highcast import boundregime as br
from highcast import db
from highcast.buckets import pmf_mass
from highcast.config import load_config

HOUR = 7
FIT_YEARS = range(2020, 2025)        # 2020-2024 inclusive
SCORE_YEARS = (2025, 2026)
WEIGHT_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
LADDER = [(81, None), (79, 80), (77, 78), (75, 76), (73, 74), (None, 72)]
EPS = 1e-6


def ladder_probs(pmf):
    return [max(pmf_mass(pmf, lo, hi) or 0.0, EPS) for lo, hi in LADDER]


def settled_index(low):
    for i, (lo, hi) in enumerate(LADDER):
        if (lo is None or low >= lo) and (hi is None or low <= hi):
            return i
    return None


def log_score(pmf, low):
    import math
    i = settled_index(low)
    if i is None:
        return None
    p = ladder_probs(pmf)
    return math.log(p[i] / sum(p))


def model_pmfs(conn, cfg, dates):
    """{target_date_iso: pmf} for the FROZEN model, walk-forward.

    ⚠️ DO NOT READ THE `predictions` TABLE. It is the live log, not an
    evaluation source: it holds 5 distinct KMIA LOW target dates (2026-07-28
    onward), so a held-out comparison against it is impossible. Every other gate
    in this project regenerates predictions with `walk_forward_at_hour`, which
    is expanding-window — the model scoring day i was fitted only on days < i —
    and that is what makes thousands of genuinely out-of-sample days available.
    """
    from highcast import backtest
    from highcast.buckets import bucket_probs
    from highcast.metric import get_metric
    recs, _frame = backtest.walk_forward_at_hour(
        conn, cfg, get_metric(cfg, "low"), dates, HOUR, "intraday",
        min_train=365, refit_every=30)
    return {r["target_date"]: bucket_probs(r["quantiles"]) for r in recs}


def fit_weights(table, fit_obs, pmfs):
    """Per cell, the weight on the MODEL that maximises log score IN-FIT.

    Fitted on the 2020-2024 window only, so the weight itself never sees the
    held-out seasons. Defaults to 1.0 (layer disabled for that cell) when the
    cell is thin or when nothing beats the model.
    """
    out = {}
    for cell, entry in table.items():
        if entry["n"] < br.MIN_CELL_N:
            out[cell] = 1.0
            continue
        days = [o for o in fit_obs if o["cell"] == cell]
        best_w, best_s = 1.0, None
        for w in WEIGHT_GRID:
            tot, k = 0.0, 0
            for o in days:
                mp = pmfs.get(o["day"].isoformat())
                if mp is None:
                    continue
                s = log_score(
                    br.blend(mp, br.reshape(entry["hist"], o["bound"]), w),
                    round(o["bound"] - o["delta"]))
                if s is not None:
                    tot += s; k += 1
            if k and (best_s is None or tot / k > best_s):
                best_s, best_w = tot / k, w
        out[cell] = best_w
    return out


def evaluate(table, weights, score_obs, pmfs):
    """Paired per-day log scores: (model, layer). Days with no walk-forward
    prediction are dropped from BOTH arms so the pairing stays honest."""
    pairs = []
    for o in score_obs:
        mp = pmfs.get(o["day"].isoformat())
        if mp is None:
            continue
        low = round(o["bound"] - o["delta"])
        entry = table.get(o["cell"]) or table.get((o["cell"][0],))
        w = weights.get(o["cell"], 1.0) if entry else 1.0
        lp = br.blend(mp, br.reshape(entry["hist"], o["bound"]) if entry else None, w)
        a, b = log_score(mp, low), log_score(lp, low)
        if a is not None and b is not None:
            pairs.append((a, b, mp, lp, low))
    return pairs


def bootstrap_p(diffs, n=10000, seed=20260801):
    rnd = random.Random(seed)
    k = len(diffs)
    if k == 0:
        return 1.0
    worse = 0
    for _ in range(n):
        s = sum(diffs[rnd.randrange(k)] for _ in range(k)) / k
        if s <= 0:
            worse += 1
    return worse / n


def hit_rate(pairs, which):
    ok = 0
    for a, b, mp, lp, low in pairs:
        pmf = mp if which == "model" else lp
        p = ladder_probs(pmf)
        if p.index(max(p)) == settled_index(low):
            ok += 1
    return ok / len(pairs) if pairs else 0.0


def calib_error(pairs, which):
    bins = {}
    for a, b, mp, lp, low in pairs:
        pmf = mp if which == "model" else lp
        p = ladder_probs(pmf)
        top = max(p) / sum(p)
        bins.setdefault(int(top * 10), []).append(
            (top, 1.0 if p.index(max(p)) == settled_index(low) else 0.0))
    num = den = 0.0
    for vals in bins.values():
        if len(vals) < 10:
            continue
        stated = sum(v[0] for v in vals) / len(vals)
        obs = sum(v[1] for v in vals) / len(vals)
        num += abs(obs - stated) * len(vals); den += len(vals)
    return num / den if den else 0.0


def run(shuffle=False, label="REAL"):
    cfg = load_config("config/kmia.yaml")
    conn = db.connect(cfg.db_path)
    days = [dt.date.fromisoformat(r[0]) for r in conn.execute(
        "SELECT date FROM cli_reports WHERE station='KMIA' AND low_f IS NOT NULL "
        "ORDER BY date")]
    obs = br.observations(conn, cfg, days, hour=HOUR)
    if shuffle:
        rnd = random.Random(999)
        cells = [o["cell"] for o in obs]
        rnd.shuffle(cells)
        obs = [{**o, "cell": c} for o, c in zip(obs, cells)]

    # Regenerate the frozen model's predictions walk-forward over ALL days, not
    # just the seasonal subset — the model trains on the full year and slicing
    # to Jun-Sep first would starve it.
    pmfs = model_pmfs(conn, cfg, [d.isoformat() for d in days])

    fit_obs = [o for o in obs if o["day"].year in FIT_YEARS]
    score_obs = [o for o in obs if o["day"].year in SCORE_YEARS]
    table = br.fit(fit_obs)
    weights = fit_weights(table, fit_obs, pmfs)
    pairs = evaluate(table, weights, score_obs, pmfs)
    diffs = [b - a for a, b, *_ in pairs]

    print(f"\n===== {label} =====")
    print(f"  fit days   {len(fit_obs)}   score days {len(score_obs)}   "
          f"scored pairs {len(pairs)}")
    if not pairs:
        print("  NO PAIRS -- no stored model predictions in the held-out years.")
        print("  VERDICT: CANNOT EVALUATE. Report this; do not ship.")
        return
    mean_d = sum(diffs) / len(diffs)
    p = bootstrap_p(diffs)
    hm, hl = hit_rate(pairs, "model"), hit_rate(pairs, "layer")
    cm, cl = calib_error(pairs, "model"), calib_error(pairs, "layer")
    print(f"  mean log-score delta  {mean_d:+.4f}   bootstrap p = {p:.4f}")
    print(f"  hit rate    model {hm:.1%}  layer {hl:.1%}  "
          f"delta {(hl-hm)*100:+.1f} pt")
    print(f"  calib err   model {cm:.3f}  layer {cl:.3f}  "
          f"delta {(cl-cm)*100:+.1f} pt")
    print(f"  weights fitted: {weights}")
    if len(pairs) < 60:
        print(f"  ⚠️  held-out n = {len(pairs)} < 60. Result is INDICATIVE only; "
              f"spec §4 says the layer stays disabled.")
    ok = (mean_d > 0 and p < 0.05 and (hl - hm) * 100 >= -1.0
          and (cl - cm) * 100 <= 2.0 and len(pairs) >= 60)
    print(f"  VERDICT: {'PASS' if ok else 'FAIL'}")
    if ok and label == "REAL":
        print("\n  Paste into gating.py (only if the negative control FAILED):")
        print("  BOUND_REGIME = " + repr(
            {c: {"n": table[c]["n"], "w": weights.get(c, 1.0),
                 "hist": table[c]["hist"]}
             for c in table if table[c]["n"] >= br.MIN_CELL_N}))


if __name__ == "__main__":
    run(shuffle=False, label="REAL")
    run(shuffle=True, label="NEGATIVE CONTROL (shuffled regime key)")
```

- [ ] **Step 2: Run the gate**

Run: `.venv/bin/python scripts/bound_regime_eval.py`

Record the full output verbatim in the report. There are three possible outcomes and **all three are acceptable results**:

| outcome | meaning | action |
|---|---|---|
| REAL PASS, control FAIL | the effect is real | proceed to Task 6 |
| REAL FAIL | the layer does not help out of sample | **stop.** Report it. Do not proceed. |
| REAL PASS **and** control PASS | the gate measures overfitting | **stop.** The result is void per spec §4. |

- [ ] **Step 3: Sanity-check the held-out sample size before trusting the verdict**

The walk-forward runs the real `IntradayModel` over years of history and takes
**several minutes**. That is expected; do not interrupt it.

Confirm from the printed header that `scored pairs` is at least 60 (spec §4).
Jun–Sep 2025 alone is ~120 days, so a much smaller number means the walk-forward
dropped days — most likely `min_train=365` biting, or `build_hour_frame`
having no checkpoint row at h7 for those dates. Investigate before reading the
verdict; a gate that passes on 12 days has told you nothing.

**Do not, under any circumstance, switch to an in-sample evaluation to produce a
number.** If the held-out sample is genuinely too small, report that — it is a
finding, and the layer staying disabled is a complete outcome.

---

### Task 6: Wire into the live path — CONDITIONAL

**Do not start this task unless Task 5 printed `VERDICT: PASS` for REAL and
`VERDICT: FAIL` for the negative control.** If it did not, this task does not
happen and the layer ships disabled with an empty `BOUND_REGIME`, which is a
complete and correct outcome.

**Files:**
- Modify: `src/highcast/gating.py` (paste the fitted `BOUND_REGIME` literal)
- Modify: `src/highcast/signals.py` (apply the layer where the ladder pmf is built)
- Test: `tests/test_boundregime.py`

**Interfaces:**
- Consumes: `boundregime.apply`

- [ ] **Step 1: Paste the fitted table into `gating.py`**

Replace `BOUND_REGIME: dict[tuple[int, ...], dict] = {}` with the literal the
script printed. Keep the comment block above it unchanged, and append one line
recording the gate result, e.g.:

```python
# Gate passed 2026-08-XX: mean log-score delta +0.0XXX, bootstrap p = 0.0XX,
# held-out n = XXX (Jun-Sep 2025-26). Negative control FAILED as required.
```

- [ ] **Step 2: Write the failing integration test**

```python
def test_the_live_path_applies_the_layer_for_kmia_low_at_h7(tmp_path):
    """End to end through the real accessor, with the shipped table."""
    from highcast import gating
    if not gating.BOUND_REGIME:
        pytest.skip("layer ships disabled — the gate has not passed")
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    # Empty DB -> no bound -> must be an exact no-op even with a live table.
    model = {81: 0.4, 80: 0.6}
    assert br.apply(conn, cfg, station="KMIA", metric="low", hour=7,
                    day=dt.date(2026, 8, 1), model_pmf=model) is model
```

- [ ] **Step 3: Find the ladder-construction site**

Run: `grep -n "pmf_from_buckets" src/highcast/signals.py`

Apply `boundregime.apply` to the pmf immediately after it is built from the
stored prediction and **before** any bucket mass is computed, so every consumer
(`p_model`, edge, staking) sees one consistent distribution. Pass the real
`station`, `metric`, the prediction's local hour, and the target date.

- [ ] **Step 4: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass. Any failure in `test_signals.py` means the layer changed a
number a test pinned — investigate whether the pin or the layer is right; do not
update the pin reflexively.

- [ ] **Step 5: Verify live**

Run: `.venv/bin/highcast --config config/kmia.yaml wake`
Then: `.venv/bin/highcast --config config/kmia.yaml board`
Confirm the KMIA LOW ladder differs from the pre-change values in the direction
the spec predicts (mass moved off the bucket adjacent to the observed minimum).

---

## Self-Review

**Spec coverage:**

| spec section | task |
|---|---|
| §3.0 inputs (bound, guidance median, gap) | Task 1 |
| §3.1 regime key, bands, occupancy | Tasks 1, 2 |
| §3.2 estimator | Tasks 2, 3 |
| §3.3 blend, backoff, fail-closed, MIN_CELL_N | Tasks 3, 4 |
| §3.4 components | all |
| §4 pre-registered gate, discovery-bias split, negative control | Task 5 |
| §5 testing — fail-closed, w=1 identity, total probability, no look-ahead, scope containment, MIN_CELL_N pinned, cell occupancy | Tasks 1–4 (occupancy is Task 2 Step 5) |
| §6 honest limits | scope tests in Task 4 |

**Placeholder scan:** no TBD/TODO; every code step contains complete code; every
command has expected output.

**Type consistency:** `regime_key` returns `tuple[int, int]`; `fit` keys on that
same tuple; `apply`'s backoff uses `(cell[0],)`, a 1-tuple, which is why
`gating.BOUND_REGIME` is annotated `dict[tuple[int, ...], dict]` rather than
`dict[tuple[int, int], dict]`. `reshape` takes `hist` and returns
`dict[int, float]`, which is what `blend` and `buckets.pmf_mass` both consume.

**One error found and fixed during self-review, recorded because it would have
wasted the whole of Task 5.** The first draft read held-out model pmfs from the
`predictions` table. That table is the **live log**, not an evaluation source —
it holds 5 distinct KMIA LOW target dates (2026-07-28 onward), so the gate would
have printed `NO PAIRS` and the plan would have concluded the effect was
unmeasurable. Every other gate in this project regenerates predictions with
`backtest.walk_forward_at_hour`, which is expanding-window and yields thousands
of genuinely out-of-sample days. Task 5 now uses it.

The near-miss is worth stating plainly: the wrong source would have produced a
confident "cannot evaluate", which reads like a careful negative result and is
actually just a bug.

**Task 6 remains conditional by design.** It runs only on `REAL PASS` +
`CONTROL FAIL`. Any other outcome ships the layer disabled with an empty
`BOUND_REGIME`, which is a complete and correct result rather than a failure.
