"""PRE-REGISTERED GATE: does fixing the gfs/hrrr duplicate channel help?

Pre-registration (written before any arm ran):
`docs/superpowers/specs/2026-08-01-gfs-hrrr-duplicate-channel-gate.md`

THE DEFECT. `MODELS` maps "gfs" -> gfs_seamless, and over CONUS the *seamless*
product IS HRRR for the first 48 hours. So `fc_gfs` and `fc_hrrr` hold the same
series at every lead the models consume -- identical on 3,733 of 3,735 stored
lead-0/1 pairs, and 240/240 on a historical probe. `BaselineBlend` weights by
inverse MSE and therefore counts one channel twice, giving HRRR-derived guidance
19-26% of the blend while it appears to be two independent sources.

THE ARMS. Varied by DATA, not by code: each arm runs against its own COPY of the
database, so the real production path is exercised unmodified. Monkeypatching
`FC_MODELS` would measure a code path nobody runs -- the mistake this project
already made once with `IntradayModel`'s feature set.

    A  STATUS QUO   pristine copy. gfs (=HRRR) and hrrr both present.
    B  DEDUPE       `hrrr` rows deleted. Isolates whether double-weighting
                    itself costs anything, using only pre-existing data.
    C  REPLACE      `gfs` rows overwritten with real `gfs_global` values,
                    keeping the column NAME. This is exactly the proposed
                    production change: `fc_gfs` starts meaning global GFS.

PRIMARY OUTCOME. Parity-averaged bucket hit at each market's decision hour,
walk-forward (refit 30, RECENT >= 2025-01-01), PAIRED on matched target dates
against arm A. Both ladder parities scored and averaged -- Kalshi's parity is a
coin flip per day.

SHIP THRESHOLD, per market, all three:
    1. Dhit > 0
    2. paired bootstrap CI excludes zero at Bonferroni x4 (98.75%)
    3. MAE does not worsen by more than 0.05F

EXPECTED OUTCOME, on record: B ~= A and C a small improvement at best. Inverse-MSE
weighting is fairly robust to a duplicated column -- the duplicate splits the
weight it would otherwise earn alone, so the COMBINED influence is close to what
one good channel would get anyway. The double-count is worth fixing for honesty
and for feature-importance readings; it is not obviously an accuracy bug.

⚠️ KNOWN DATA HAZARD FOR ARM C: raw `gfs_global` LOW runs +7.9F (KLAX) / +6.6F
(KMIA) warmer than the incumbent, while its HIGH is only ~+1.5F off. That
asymmetry says the coarse GFS cell is far more maritime than the station -- water
does not cool overnight. The `fcc_*` bias-corrected columns exist to absorb
exactly this, so arm C is a fair test of the CORRECTED channel; but if C fails on
the LOWs, this is the first thing to look at.

Usage: PYTHONPATH=src .venv/bin/python scripts/gfs_hrrr_gate_eval.py
"""
import os
import shutil
import sqlite3
import tempfile

import numpy as np

from highcast.buckets import bucket_probs, expected_bucket_hit
from highcast.config import load_config
from highcast.db import connect
from highcast.features import build_features, uses_bias_correction
from highcast.gating import MIN_DECISION_HOUR
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
MAX_MAE_REGRESSION = 0.05
rng = np.random.default_rng(20260801)

_SHIPPED_BASE = ["hour", "run_lo", "run_hi", "run_width", "cur_tmpf",
                 "cur_dwpf", "cur_onshore", "cur_sky", "base_p50",
                 "base_spread", "doy_sin", "doy_cos", "year_index",
                 "heat_alert"]

ARMS = ("A_status_quo", "B_dedupe", "C_replace")
MARKETS = [("config/kmia.yaml", "high"), ("config/klax.yaml", "high"),
           ("config/kmia.yaml", "low"), ("config/klax.yaml", "low")]


def make_arm_db(src: str, arm: str, workdir: str) -> str:
    """A private copy of the database, mutated to express one arm."""
    dst = os.path.join(workdir, f"{arm}.db")
    shutil.copy(src, dst)
    if arm == "A_status_quo":
        return dst
    conn = sqlite3.connect(dst)
    if arm == "B_dedupe":
        n = conn.execute("DELETE FROM forecasts WHERE model='hrrr'").rowcount
        print(f"    [B] deleted {n} hrrr rows", flush=True)
    elif arm == "C_replace":
        # Overwrite fc_gfs's VALUES with real global GFS, keeping the name so the
        # feature list is untouched -- this is precisely the production change.
        n = conn.execute("""UPDATE forecasts SET max_temp_f = (
                 SELECT g.max_temp_f FROM forecasts g
                 WHERE g.station=forecasts.station AND g.target_date=forecasts.target_date
                   AND g.run_date=forecasts.run_date AND g.metric=forecasts.metric
                   AND g.model='gfs_global')
               WHERE model='gfs' AND EXISTS (
                 SELECT 1 FROM forecasts g
                 WHERE g.station=forecasts.station AND g.target_date=forecasts.target_date
                   AND g.run_date=forecasts.run_date AND g.metric=forecasts.metric
                   AND g.model='gfs_global')""").rowcount
        print(f"    [C] rewrote {n} gfs rows with gfs_global values", flush=True)
    # gfs_global must not ALSO enter as its own channel in any arm -- it is not
    # in FC_MODELS, but delete it so the frame cannot pick it up by accident.
    conn.execute("DELETE FROM forecasts WHERE model='gfs_global'")
    conn.commit()
    conn.close()
    return dst


def score(cfg_path: str, metric_key: str, db_path: str) -> dict[str, tuple]:
    """{target_date: (hit, abs_error)} at the decision hour, walk-forward."""
    cfg = load_config(cfg_path)
    icao = cfg.station.icao
    conn = connect(db_path)
    spec = get_metric(cfg, metric_key)
    hour = MIN_DECISION_HOUR[icao][metric_key]
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

    out: dict[str, tuple] = {}
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
            y = int(row["y"])
            out[row["target_date"]] = (
                expected_bucket_hit(bucket_probs(q), y),
                abs(float(q[0.5]) - row["y"]))
    return out


print(__doc__.split("Usage:")[0])
work = tempfile.mkdtemp(prefix="gfsgate_")
src = load_config("config/klax.yaml").db_path
print(f"workdir {work}\n")
arm_dbs = {a: make_arm_db(str(src), a, work) for a in ARMS}

for cfg_path, metric_key in MARKETS:
    icao = load_config(cfg_path).station.icao
    hour = MIN_DECISION_HOUR[icao][metric_key]
    print(f"\n### {icao} {metric_key.upper()} h{hour}", flush=True)
    res = {a: score(cfg_path, metric_key, arm_dbs[a]) for a in ARMS}
    ref = res["A_status_quo"]
    print(f"    arm            n     hit      MAE  |   dhit   CI(Bonferroni x4)   dMAE  verdict")
    for a in ARMS:
        common = sorted(set(ref) & set(res[a]))
        h_a = np.array([res[a][d][0] for d in common])
        e_a = np.array([res[a][d][1] for d in common])
        h_r = np.array([ref[d][0] for d in common])
        e_r = np.array([ref[d][1] for d in common])
        if a == "A_status_quo":
            print(f"    {a:14s}{len(common):5d}  {h_a.mean():6.1%}  {e_a.mean():6.3f}  |  (reference)")
            continue
        dh = h_a - h_r
        idx = rng.integers(0, len(dh), size=(N_BOOT, len(dh)))
        lo, hi = np.percentile(dh[idx].mean(axis=1), PCTS)
        dmae = e_a.mean() - e_r.mean()
        ship = (dh.mean() > 0) and ((lo > 0) == (hi > 0)) and (dmae <= MAX_MAE_REGRESSION)
        print(f"    {a:14s}{len(common):5d}  {h_a.mean():6.1%}  {e_a.mean():6.3f}  |"
              f" {dh.mean():+6.1%}  [{lo:+.1%},{hi:+.1%}] {dmae:+7.3f}  "
              f"{'** SHIP **' if ship else 'no'}")

print(f"\nB isolates the double-weighting; C is the intended fix (real global GFS).")
print("Nothing here writes a constant.")
shutil.rmtree(work, ignore_errors=True)
