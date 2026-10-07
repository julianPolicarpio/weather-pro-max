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
    k = cfg.model.intraday_bandwidth_for(metric_key)
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
