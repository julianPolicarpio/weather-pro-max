# Regime-Conditional Calibration (Approach B, HIGH) — Design

Updated 2026-07-25. Make the HIGH day-ahead model's predictive distribution
**honest in the marine-layer seam (~80–84°F)** by adding a calibration layer
that conditions on the predicted level — widening (and optionally shifting) the
distribution where history says the model is miscalibrated, while leaving the
well-behaved <80 regime alone. **Backtestable now** on the full 4-year history
(unlike ensemble features). Goal is calibration for betting, not point accuracy.

## Problem — the global PIT can't fix a conditional error

The HIGH model overshoots at the marine-layer seam (2026-07-25 walk-forward,
1,663 days):

| predicted | N | bias (P−A) | MAE | overshoot % |
|---|---|---|---|---|
| < 80 | 1549 | −0.21 | 1.51 | 47% |
| **80–82** | 35 | **+0.43** | 2.05 | **63%** |
| **82–84** | 39 | **+0.48** | 2.01 | **62%** |
| 84+ | 40 | −2.54 | 4.39 | 40% |

The model's calibrator (`models/calibrate.py` `PITCalibrator`) is **global**: it
collects PIT values `F_pred(y)` across all held-out days and applies one
recalibration curve everywhere (`DayAheadModel.predict_quantiles`). That fixes
*average* coverage but cannot widen only the seam — at 80–84 the distribution is
too narrow and warm-centered (it collapses to a false single peak; observed
2026-07-24), while <80 is already well-calibrated. A single global curve can't
serve both.

## Goal

On a seam day the distribution should be **honestly wide** (and where the data
supports it, cooler-centered) so the edge-vs-market gate stops seeing phantom
edge on the trough bucket. Concretely: **seam-conditional coverage moves toward
nominal and CRPS improves, without regressing the <80 regime.** Which correction
strength wins is decided by backtest, not asserted.

## Confirmed facts (from the codebase)

- `models/calibrate.py`: `PITCalibrator` — global PIT remap; `cal.level(nominal)`
  reads the raw quantile curve at the empirically-correct level.
- `models/dayahead.py`: `DayAheadModel` fits LightGBM quantile regressors at
  `LEVELS`, then `PITCalibrator` on a tail held-out window (`CAL_HOLDOUT_DAYS`);
  `predict_quantiles` applies the global PIT to each raw quantile set.
- `backtest.py`: `walk_forward(df, factory)` yields per-day records with
  `point`, `quantiles`, `crps`; `summarize` computes MAE/CRPS/coverage.
  `_coverage` already exists.
- The correction operates on a **quantile set** (dict level→°F); `bucket_probs`
  downstream turns it into the Kalshi PMF — so a monotonic quantile transform is
  all that's needed; nothing downstream changes.

## Key definitions

- **Predicted level `m0`** — the (post-global-PIT) predicted median quantile, the
  variable the measured bias is indexed by.
- **`s(m0)`** — spread multiplier: rescales the quantile set's dispersion around
  its median. `>1` = wider.
- **`m(m0)`** — mean shift: added to the whole quantile set (center+width mode).
- **Shrink-to-null** — where predicted level is data-sparse, `s→1`, `m→0` (no
  correction). The thin-data safeguard.
- **Seam** — predicted 80–84°F.

## Design

### Component 1 — `RegimeCalibrator` (`models/calibrate.py`)
A new class beside `PITCalibrator`.
- `fit(level, resid, pred_spread)` — from **out-of-fold** `(predicted median,
  actual)` pairs: learns two smooth, kernel-weighted local functions of `m0`:
  - `m(·)` = local mean of the residual `y − m0`;
  - `s(·)` = local ratio of realized dispersion (e.g. local MAD of `y − m0`) to
    the model's predicted spread at that level;
  both **shrunk toward the null (`m=0`, `s=1`)** with weight ∝ local sample
  count (so sparse regions get ~no correction). Exact kernel/bandwidth is an
  Open Item; start with a wide Gaussian kernel + a min-effective-N shrinkage.
- `apply(quantiles: dict, mode: str) -> dict` — with `m0 = quantiles[0.5]`:
  - `off`: return `quantiles` unchanged (no-op).
  - `width`: `new[l] = m0 + s(m0)·(quantiles[l] − m0)`.
  - `center_width`: `new[l] = m0 + m(m0) + s(m0)·(quantiles[l] − m0)`.
  `s>0` keeps quantiles ordered (no crossing).

### Component 2 — fit the corrector on out-of-fold residuals
To have enough *seam* days, the `RegimeCalibrator` is fit on **out-of-fold
residuals across the training set** (not only the small tail held-out window the
PIT uses). Concretely, within `DayAheadModel.fit`: after fitting the raw models +
global PIT, produce out-of-fold `(m0, y, pred_spread)` triples (reusing the
existing holdout mechanism, extended to K folds or an expanding-window OOF pass),
and fit the `RegimeCalibrator` on them. This is the design's key enabler for a
conditional correction on thin data; the OOF scheme's exact form is an Open Item.

### Component 3 — integrate into `predict_quantiles`
After the existing global-PIT remap, call
`self.regime.apply(adj, self.cal_mode)` and return the result. `cal_mode == "off"`
makes `predict_quantiles` **byte-identical to today**.

### Component 4 — config toggle
`config.py` gains `calibration.mode: str = "off"` (`off | width | center_width`);
`config/klax.yaml` documents it. `DayAheadModel` reads it (constructor arg or
from cfg). **Default `off`** until the backtest picks a winner.

### Component 5 — backtest harness (the acceptance gate)
Extend the walk-forward (or a thin wrapper) to run **all three modes** over the
full history and report, each **with N**:
- **Seam-conditional coverage** (50/90%) on predicted 80–84, vs nominal;
- **CRPS** overall and seam-conditional;
- **<80 non-regression** (MAE + CRPS on the <80 slice must not degrade).
- **Ship rule:** set the `klax.yaml` default `calibration.mode` to whichever mode
  wins; if neither beats `off`, keep `off` — recording that predicted-level
  conditioning alone can't fix the seam (→ Method 3 / ensemble later).

## Data flow

```
LGBM raw quantiles → global PITCalibrator (unchanged)
  → RegimeCalibrator.apply(quantiles, mode):
       m0 = median; width: scale spread by s(m0); center_width: also shift by m(m0)
       (mode=off → passthrough, identical to today)
  → bucket_probs → Kalshi PMF → edge gate (UNCHANGED)
Fit path: DayAheadModel.fit → OOF (m0, y, pred_spread) → RegimeCalibrator.fit (shrink-to-null)
```

## Error handling & edge cases

- **Sparse predicted-level region:** shrink `s→1`, `m→0` — no correction, never a
  wild extrapolation.
- **Too little training data** (below the existing PIT thresholds): `RegimeCalibrator`
  fits to the null identity → `apply` is a no-op regardless of mode (fail-safe).
- **Ordering:** `s>0` enforced (clip to a small positive floor) so the transform
  never produces crossing quantiles; the existing non-crossing sort still runs.
- **`off` mode:** provably a no-op (return input) — the default, so live behavior
  is unchanged until a mode is chosen on evidence.

## Testing

- **Unit (`RegimeCalibrator`):**
  - On synthetic OOF data where the 80–84 band is systematically narrow + warm
    and <80 is well-calibrated: `fit` then assert `s(81) > 1` and `m(81) < 0`,
    while `s(75) ≈ 1` and `m(75) ≈ 0`.
  - `apply(..., "off")` returns the input unchanged.
  - `apply(..., "width")` preserves the median and scales spread; `center_width`
    also shifts; both keep quantiles strictly ordered.
  - Shrink-to-null: a level with < min-effective-N support gets `s≈1, m≈0`.
- **Integration:** `DayAheadModel(cal_mode="off")` produces byte-identical
  `predict_quantiles` output to the current model on a fixture.
- **Empirical gate:** the Component 5 backtest, three modes, seam-conditional
  coverage + CRPS + <80 non-regression.

## Non-goals (YAGNI)

- **Marine-layer-score conditioning** (Method 3 — a residual model on
  dewpoint-depression/onshore/fc_spread/ensemble-std). Deferred; revisit only if
  predicted-level conditioning proves out but leaves signal.
- **Hard regime bins** (Method 1) — rejected for edge discontinuities + thin-bin
  overfit.
- **LOW** metric.
- **Ensemble-std** as a conditioning input — the features aren't in the model yet
  (still accumulating).
- Any change to **betting/staking/gates/dashboard** — the model just emits
  better-calibrated probabilities.

## Open items for review

- **Kernel/bandwidth + shrinkage** form for `s(·)`/`m(·)` — start wide + strongly
  shrunk; the backtest will show if more flexibility helps.
- **OOF scheme** — K-fold vs expanding-window out-of-fold for the corrector's
  training residuals; must stay leakage-free.
- **Spread definition** for `s` — interdecile (`q90−q10`) vs IQR; pick one, keep
  it consistent between fit and apply.
- **Default mode** — set by the Component 5 backtest, not in advance.
