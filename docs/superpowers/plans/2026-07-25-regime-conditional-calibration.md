# Regime-Conditional Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a togglable regime-conditional calibration layer to the HIGH day-ahead model that widens (and optionally shifts) the predictive distribution in the marine-layer seam (~80–84°F), plus a backtest that decides on evidence which mode (if any) to ship.

**Architecture:** A new `RegimeCalibrator` transforms the post-PIT quantile set as a smooth function of the predicted median (spread-multiplier `s`, mean-shift `m`), learned from out-of-fold holdout residuals and shrunk toward the null where data is sparse. It slots into `DayAheadModel.predict_quantiles` after the existing global `PITCalibrator`, gated by a config `calibration.mode`. A backtest harness runs all three modes seam-conditionally so the default is chosen from data.

**Tech Stack:** Python 3, NumPy, pandas, LightGBM (unchanged), SQLite; pytest.

## Global Constraints

- **Not a git repo.** No commits. Each task ends with `.venv/bin/python -m pytest -q`. Reviewers read files directly.
- **`off` mode is the default and MUST be a provable no-op** — live behavior is byte-identical (per-level values) to today until a mode is chosen on backtest evidence.
- **YAML gotcha:** unquoted `off` parses as boolean `False`. The `klax.yaml` value MUST be quoted: `mode: "off"`.
- **Layer order:** the regime correction is applied AFTER the global PIT remap, on the quantile set. `s > 0` (floored at 0.05) keeps quantiles ordered — the existing non-crossing sort still runs upstream.
- **Fit on out-of-fold holdout residuals**, reusing the existing PIT holdout machinery (train-only model predicting the `CAL_HOLDOUT_DAYS` window); shrink toward null (`m=0, s=1`) where a predicted-level region is data-sparse.
- **Ship decision is the controller's, from the Task-3 backtest** — do NOT hardcode a non-`off` default. The plan ships with `mode: "off"`.
- **Scope:** HIGH only. Condition on predicted level only. NO changes to betting/staking/gates/dashboard/LOW.
- Run tests with `.venv/bin/python -m pytest ...`.

---

### Task 1: `RegimeCalibrator`

**Files:**
- Modify: `src/highcast/models/calibrate.py` (add `RegimeCalibrator` beside `PITCalibrator`)
- Test: `tests/test_regime_calibrator.py` (create)

**Interfaces:**
- Produces:
  - `RegimeCalibrator(bandwidth=2.5, min_eff_n=8.0)`
  - `.fit(level, resid, pred_spread_sd) -> RegimeCalibrator` (arrays of OOF triples; `resid = y − predicted_median`, `pred_spread_sd` = predicted interdecile/2.5631)
  - `.mean_shift(x) -> float`, `.spread_mult(x) -> float`
  - `.apply(quantiles: dict, mode: str) -> dict` — `mode ∈ {"off","width","center_width"}`

- [ ] **Step 1: Write the failing test**

Create `tests/test_regime_calibrator.py`:

```python
import numpy as np

from highcast.models.calibrate import RegimeCalibrator


def _synth():
    # below 80: well-calibrated (mean resid 0, MAD 1.0 -> realized sd ~1.48 ~ pred 1.5)
    below_lvl = np.linspace(72.0, 79.0, 400)
    below_res = np.tile([-1.0, 1.0], 200)
    # seam 80-84: warm overshoot (y-m0 ~ -1.5) AND under-dispersed (wide resid)
    seam_lvl = np.linspace(80.0, 84.0, 160)
    seam_res = -1.5 + np.tile([-3.0, -1.0, 1.0, 3.0], 40)
    level = np.concatenate([below_lvl, seam_lvl])
    resid = np.concatenate([below_res, seam_res])
    pspread = np.full(level.size, 1.5)          # model claims sd 1.5 everywhere
    return level, resid, pspread


def test_learns_seam_shift_and_widen():
    rc = RegimeCalibrator().fit(*_synth())
    assert rc.mean_shift(81.0) < -0.5           # pull center down at the seam
    assert abs(rc.mean_shift(75.0)) < 0.4       # ~no shift below
    assert rc.spread_mult(81.0) > 1.2           # widen at the seam
    assert abs(rc.spread_mult(75.0) - 1.0) < 0.35


def test_off_and_null_are_noops():
    rc = RegimeCalibrator().fit(*_synth())
    q = {0.1: 78.0, 0.5: 81.0, 0.9: 84.0}
    assert rc.apply(q, "off") == q
    null = RegimeCalibrator().fit([], [], [])   # below the 30-sample floor
    assert null.apply(q, "center_width") == q


def test_width_keeps_median_center_shifts_both_ordered():
    rc = RegimeCalibrator().fit(*_synth())
    q = {0.1: 78.0, 0.5: 81.0, 0.9: 84.0}
    w = rc.apply(q, "width")
    assert abs(w[0.5] - 81.0) < 1e-9            # width keeps the median
    assert w[0.1] < w[0.5] < w[0.9]            # still ordered
    assert (w[0.9] - w[0.1]) > (q[0.9] - q[0.1])  # wider
    cw = rc.apply(q, "center_width")
    assert cw[0.5] < 81.0                       # center shifted down at seam
    assert cw[0.1] < cw[0.5] < cw[0.9]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_regime_calibrator.py -q`
Expected: FAIL — `ImportError: cannot import name 'RegimeCalibrator'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/highcast/models/calibrate.py` (the file already imports `numpy as np`):

```python
class RegimeCalibrator:
    """Smooth, regime-conditional correction of a quantile set, keyed on the
    predicted median. Learns a mean-shift m(level) and spread-multiplier
    s(level) from out-of-fold (median, actual, pred_spread_sd) triples,
    kernel-smoothed over predicted level and shrunk toward the null
    (m=0, s=1) where data is sparse. mode 'off' is a no-op."""

    def __init__(self, bandwidth: float = 2.5, min_eff_n: float = 8.0):
        self.bw = bandwidth
        self.min_eff_n = min_eff_n
        self._fitted = False

    def fit(self, level, resid, pred_spread_sd) -> "RegimeCalibrator":
        self._level = np.asarray(level, dtype=float)
        self._resid = np.asarray(resid, dtype=float)
        self._pspread = np.asarray(pred_spread_sd, dtype=float)
        self._fitted = self._level.size >= 30
        return self

    def _local(self, x: float):
        d = (self._level - x) / self.bw
        w = np.exp(-0.5 * d * d)
        return w, float(w.sum())

    def mean_shift(self, x: float) -> float:
        if not self._fitted:
            return 0.0
        w, eff = self._local(x)
        if eff <= 0:
            return 0.0
        raw = float((w * self._resid).sum() / eff)
        return raw * (eff / (eff + self.min_eff_n))          # shrink toward 0

    def spread_mult(self, x: float) -> float:
        if not self._fitted:
            return 1.0
        w, eff = self._local(x)
        if eff <= 0:
            return 1.0
        med = float((w * self._resid).sum() / eff)
        realized = 1.4826 * float((w * np.abs(self._resid - med)).sum() / eff)
        predicted = float((w * self._pspread).sum() / eff)
        if predicted <= 1e-6:
            return 1.0
        raw = realized / predicted
        s = 1.0 + (raw - 1.0) * (eff / (eff + self.min_eff_n))  # shrink toward 1
        return float(max(s, 0.05))

    def apply(self, quantiles: dict, mode: str) -> dict:
        if mode == "off" or not self._fitted:
            return dict(quantiles)
        m0 = quantiles[0.5]
        s = self.spread_mult(m0)
        shift = self.mean_shift(m0) if mode == "center_width" else 0.0
        return {l: float(m0 + shift + s * (v - m0)) for l, v in quantiles.items()}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_regime_calibrator.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Suite checkpoint (no commit — not a git repo)**

Run: `.venv/bin/python -m pytest -q`
Expected: baseline (222 passed, 12 deselected) + 3 new, no regressions.

---

### Task 2: Integrate into `DayAheadModel` + config toggle

**Files:**
- Modify: `src/highcast/models/dayahead.py` (import, `__init__`, extract `_apply_pit`, fit the regime corrector, apply it in `predict_quantiles`)
- Modify: `src/highcast/config.py` (`CalibrationConfig`, `AppConfig` field, loader)
- Modify: `config/klax.yaml` (add `calibration:` block)
- Modify: `src/highcast/cli.py:194` (live predict path passes the configured mode)
- Test: `tests/test_dayahead_regime.py` (create)

**Interfaces:**
- Consumes: `RegimeCalibrator` (Task 1).
- Produces:
  - `DayAheadModel(cal_mode: str = "off")`; new method `_apply_pit(raw: dict) -> dict`; `self.regime` (a fitted `RegimeCalibrator`).
  - `config.CalibrationConfig(mode="off")`; `AppConfig.calibration`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_dayahead_regime.py`:

```python
import numpy as np
import pandas as pd

from highcast.features import FEATURES
from highcast.models.baseline import LEVELS
from highcast.models.calibrate import RegimeCalibrator
from highcast.models.dayahead import DayAheadModel


def _frame(n=600, seed=3):
    rng = np.random.default_rng(seed)
    y = 75 + 10 * np.sin(np.arange(n) * 2 * np.pi / 365) + rng.normal(0, 2, n)
    df = pd.DataFrame({c: np.nan for c in FEATURES}, index=range(n))
    df["target_date"] = pd.date_range("2023-01-01", periods=n).strftime("%Y-%m-%d")
    df["fc_ecmwf"] = y + rng.normal(0, 1, n)
    df["fc_gfs"] = y + rng.normal(0, 2, n)
    df["doy_sin"] = np.sin(2 * np.pi * np.arange(n) / 365)
    df["doy_cos"] = np.cos(2 * np.pi * np.arange(n) / 365)
    df["year_index"] = 24.0
    df["heat_alert"] = 0.0
    df["y"] = np.round(y)
    return df


def test_off_mode_is_pit_passthrough():
    df = _frame()
    m = DayAheadModel(cal_mode="off").fit(df.iloc[:500])
    rows = df.iloc[500:510]
    out = m.predict_quantiles(rows)
    ref = [m._apply_pit(raw) for raw in m._raw_quantiles(m.models, rows)]
    assert out == ref                          # regime is a no-op in off mode


def test_regime_layer_wired_and_widens():
    df = _frame()
    m = DayAheadModel(cal_mode="width").fit(df.iloc[:500])
    rows = df.iloc[500:510]
    pit_only = [m._apply_pit(raw) for raw in m._raw_quantiles(m.models, rows)]
    # inject a corrector that widens everywhere (realized sd >> claimed 1.5)
    m.regime = RegimeCalibrator().fit(
        np.linspace(60.0, 95.0, 300),
        np.tile([-3.0, 3.0], 150),
        np.full(300, 1.5))
    widened = m.predict_quantiles(rows)
    for pit_q, wide_q in zip(pit_only, widened):
        assert abs(wide_q[0.5] - pit_q[0.5]) < 1e-6                    # median kept
        assert (wide_q[0.9] - wide_q[0.1]) > 1.3 * (pit_q[0.9] - pit_q[0.1])
        vals = [wide_q[l] for l in LEVELS]
        assert vals == sorted(vals)                                    # ordered


def test_calibration_config_defaults_off():
    from highcast.config import CalibrationConfig, load_config
    assert CalibrationConfig().mode == "off"
    assert load_config().calibration.mode == "off"    # klax.yaml default
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_dayahead_regime.py -q`
Expected: FAIL — `TypeError: DayAheadModel() takes no arguments` (and no `CalibrationConfig`).

- [ ] **Step 3a: Config (`config.py` + `klax.yaml`)**

In `src/highcast/config.py`, add the dataclass (after `StakingConfig`, before `AppConfig`):

```python
@dataclass(frozen=True)
class CalibrationConfig:
    """Regime-conditional seam calibration. `off` = global PIT only (default,
    live behavior unchanged). `width` widens the distribution at the marine-layer
    seam; `center_width` also shifts its center. Default stays `off` until the
    walk-forward backtest picks a winner."""
    mode: str = "off"          # off | width | center_width
```

Add the field to `AppConfig`:

```python
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
```

And in `load_config`, add to the `AppConfig(...)` construction:

```python
        calibration=CalibrationConfig(**(raw.get("calibration") or {})),
```

In `config/klax.yaml`, add a top-level block (value MUST be quoted so YAML doesn't read `off` as boolean `False`):

```yaml
calibration:
  mode: "off"   # off | width | center_width — regime seam calibration (default off; set by backtest)
```

- [ ] **Step 3b: `DayAheadModel` (`models/dayahead.py`)**

Add the import (top of file, with the other `models` imports):

```python
from highcast.models.calibrate import PITCalibrator, RegimeCalibrator
```

Add an `__init__` at the top of the class body (before `recency_weights`):

```python
    def __init__(self, cal_mode: str = "off"):
        self.cal_mode = cal_mode
        self.regime = RegimeCalibrator().fit([], [], [])   # null until fit()
```

Replace the current `predict_quantiles` method with an extracted `_apply_pit` + a thin `predict_quantiles`:

```python
    def _apply_pit(self, raw: dict) -> dict:
        cdf_levels = sorted(raw)                 # nominal levels
        vals = [raw[l] for l in cdf_levels]
        adj = {}
        for l in LEVELS:
            u = self.cal.level(l)
            adj[l] = float(np.interp(u, cdf_levels, vals))
        return dict(sorted(adj.items()))

    def predict_quantiles(self, df: pd.DataFrame) -> list[dict[float, float]]:
        return [self.regime.apply(self._apply_pit(raw), self.cal_mode)
                for raw in self._raw_quantiles(self.models, df)]
```

In `fit`, add the regime corrector. Replace the current body from the `self.cal = PITCalibrator().fit([])` line through the calibration `if` block with:

```python
        self.cal = PITCalibrator().fit([])
        self.regime = RegimeCalibrator().fit([], [], [])
        if len(train) >= 200 and len(hold) >= 30:
            tmp = self._fit_raw(train)
            raw_hold = self._raw_quantiles(tmp, hold)
            pits = [cdf_from_quantiles(qs)(y) for qs, y in
                    zip(raw_hold, hold["y"])]
            self.cal = PITCalibrator().fit(pits)
            # regime corrector: out-of-fold post-PIT quantiles on the holdout
            levels, resids, spreads = [], [], []
            for qs, y in zip(raw_hold, hold["y"].to_numpy()):
                adj = self._apply_pit(qs)
                levels.append(adj[0.5])
                resids.append(float(y) - adj[0.5])
                spreads.append((adj[0.9] - adj[0.1]) / 2.5631)
            self.regime = RegimeCalibrator().fit(levels, resids, spreads)
```

(The `self.models = self._fit_raw(df)` line and `return self` stay unchanged after this block.)

- [ ] **Step 3c: Live predict path (`cli.py:194`)**

Change:

```python
        model = DayAheadModel().fit(train)
```
to:
```python
        model = DayAheadModel(cal_mode=cfg.calibration.mode).fit(train)
```

(`cfg` is already in scope in that function.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_dayahead_regime.py tests/test_dayahead.py -q`
Expected: PASS (the 3 new + the 2 existing `test_dayahead.py` tests — the existing ones prove `off` default is byte-compatible).

- [ ] **Step 5: Suite checkpoint**

Run: `.venv/bin/python -m pytest -q`
Expected: full suite green (default `off` → no behavior change anywhere), no regressions.

---

### Task 3: Seam-conditional backtest harness + CLI

**Files:**
- Modify: `src/highcast/backtest.py` (`slice_metrics`, `compare_calibration_modes`)
- Modify: `src/highcast/cli.py` (`cmd_calibration_backtest` + subparser)
- Test: `tests/test_calibration_backtest.py` (create)

**Interfaces:**
- Consumes: `walk_forward`, `summarize`, `_coverage` (existing); `DayAheadModel(cal_mode=...)` (Task 2).
- Produces:
  - `backtest.slice_metrics(records, lo, hi) -> dict` (`n, mae, crps, cov_50, cov_90` over records with `lo <= point < hi`)
  - `backtest.compare_calibration_modes(df, modes=("off","width","center_width"), min_train=365, refit_every=30) -> dict`
  - CLI subcommand `calibration-backtest`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_calibration_backtest.py`:

```python
import numpy as np

from highcast.backtest import slice_metrics


def _recs():
    return [
        {"point": 78.0, "y": 79, "abs_err": 1.0, "crps": 0.5,
         "quantiles": {0.5: 78.0, 0.9: 82.0}},
        {"point": 81.0, "y": 80, "abs_err": 1.0, "crps": 0.7,
         "quantiles": {0.5: 81.0, 0.9: 83.0}},
        {"point": 82.5, "y": 84, "abs_err": 1.5, "crps": 0.9,
         "quantiles": {0.5: 82.5, 0.9: 85.0}},
    ]


def test_slice_metrics_filters_by_point():
    seam = slice_metrics(_recs(), 80.0, 84.0)
    assert seam["n"] == 2
    assert abs(seam["mae"] - 1.25) < 1e-9
    below = slice_metrics(_recs(), -999.0, 80.0)
    assert below["n"] == 1
    empty = slice_metrics(_recs(), 90.0, 95.0)
    assert empty["n"] == 0 and np.isnan(empty["mae"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_calibration_backtest.py -q`
Expected: FAIL — `ImportError: cannot import name 'slice_metrics'`.

- [ ] **Step 3: Write minimal implementation**

In `src/highcast/backtest.py`, add after `summarize`:

```python
def slice_metrics(records: list[dict], lo: float, hi: float) -> dict:
    """Metrics over records whose predicted point falls in [lo, hi)."""
    sub = [r for r in records if lo <= r["point"] < hi]
    if not sub:
        return {"n": 0, "mae": float("nan"), "crps": float("nan"),
                "cov_50": float("nan"), "cov_90": float("nan")}
    return {"n": len(sub),
            "mae": float(np.mean([r["abs_err"] for r in sub])),
            "crps": float(np.mean([r["crps"] for r in sub])),
            "cov_50": _coverage(sub, 0.5), "cov_90": _coverage(sub, 0.9)}


def compare_calibration_modes(df, modes=("off", "width", "center_width"),
                              min_train: int = 365, refit_every: int = 30) -> dict:
    """Walk-forward each calibration mode; report overall + seam(80-84) + <80."""
    from highcast.models.dayahead import DayAheadModel
    out = {}
    for mode in modes:
        recs = walk_forward(df, lambda m=mode: DayAheadModel(cal_mode=m),
                            min_train=min_train, refit_every=refit_every)
        out[mode] = {
            "overall": summarize(recs, df),
            "seam_80_84": slice_metrics(recs, 80.0, 84.0),
            "below_80": slice_metrics(recs, -999.0, 80.0),
        }
    return out
```

In `src/highcast/cli.py`, add the command function (near `cmd_backtest`):

```python
def cmd_calibration_backtest(args) -> None:
    import json

    from highcast import backtest as bt, features
    from highcast.metric import get_metric

    cfg = load_config(args.config)
    conn = db.connect(cfg.db_path)
    spec = get_metric(cfg, "high")
    dates = [r["date"] for r in conn.execute(
        "SELECT date FROM cli_reports WHERE station=? ORDER BY date",
        (cfg.station.icao,))]
    df = features.build_features(conn, cfg, spec, dates)
    result = bt.compare_calibration_modes(df, refit_every=args.refit_every)
    print(json.dumps(result, indent=2, default=float))
```

And register the subparser (with the other `sub.add_parser(...)` calls in the arg-parser setup):

```python
    cb = sub.add_parser("calibration-backtest")
    cb.add_argument("--refit-every", type=int, default=30)
    cb.set_defaults(func=cmd_calibration_backtest)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_calibration_backtest.py -q`
Expected: PASS (1 test).

- [ ] **Step 5: Suite checkpoint**

Run: `.venv/bin/python -m pytest -q`
Expected: full suite green.

- [ ] **Step 6: Run the acceptance-gate backtest (controller step — slow, run in background)**

This is the decision gate, NOT a code change. Run the three-mode comparison over the real history:

Run: `.venv/bin/highcast calibration-backtest --refit-every 60 > /tmp/calib_compare.json 2>&1`
(Expect a few minutes × 3 modes; run detached / in the background.)

Then read `/tmp/calib_compare.json` and compare, per the spec's ship rule:
- adopt the mode whose **seam_80_84** `cov_50`/`cov_90` move toward nominal (0.50 / 0.90) and whose `crps` drops, **AND** whose **below_80** `mae`/`crps` do not regress vs `off`;
- if neither `width` nor `center_width` beats `off`, **leave `mode: "off"`** and record that predicted-level conditioning alone doesn't fix the seam (→ Method 3 / ensemble later).

Set `config/klax.yaml` `calibration.mode` to the winner (or leave `"off"`). This edit is the only place a non-`off` default is chosen, and only on evidence.

---

## Self-Review

**Spec coverage (against `2026-07-25-regime-conditional-calibration-design.md`):**
- Component 1 (`RegimeCalibrator`: `s`/`m` smooth, shrink-to-null, `apply` modes) → Task 1. ✓
- Component 2 (fit on OOF holdout residuals) → Task 2 Step 3b (reuses `raw_hold` train-only predictions). ✓
- Component 3 (integrate after global PIT in `predict_quantiles`) → Task 2 Step 3b. ✓
- Component 4 (config toggle, default `off`) → Task 2 Step 3a + 3c. ✓
- Component 5 (3-mode seam-conditional backtest gate; ship rule) → Task 3. ✓
- `off` provable no-op → `test_off_mode_is_pit_passthrough` + existing `test_dayahead.py` still green. ✓
- Ordering / `s>0` floor → Task 1 `spread_mult` `max(s, 0.05)` + `test_..._ordered`. ✓
- Ship decision from data, default stays `off` → Task 3 Step 6, Global Constraints. ✓

**Placeholder scan:** No TBD/TODO; every code step is complete; every run step has an exact command + expected result. The Step-6 decision is controller analysis, not a code placeholder. ✓

**Type consistency:** `RegimeCalibrator(bandwidth, min_eff_n)` / `.fit(level, resid, pred_spread_sd)` / `.apply(quantiles, mode)` defined Task 1, consumed Task 2 (`self.regime.apply(..., self.cal_mode)`, fit with `levels/resids/spreads`). `DayAheadModel(cal_mode=...)` defined Task 2, used by `compare_calibration_modes` (Task 3) and `cli.py:194`. `_apply_pit(raw)->dict` defined + used within Task 2. `slice_metrics(records, lo, hi)` defined Task 3, used in `compare_calibration_modes`. `CalibrationConfig(mode="off")` / `AppConfig.calibration` defined Task 2, read at `cli.py:194`. ✓

**Note for the controller:** after Task 3, the model behavior is unchanged (`mode: "off"`) until Step 6's backtest picks a mode. If a mode is adopted, the live launchd `wake` re-reads `klax.yaml` each run, so it takes effect on the next prediction with no restart (the `serve` dashboard doesn't compute predictions, so it needs no restart either).
