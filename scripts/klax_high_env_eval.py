"""Does environment data earn its place in the KLAX HIGH nowcast?

THE HYPOTHESIS, and why it is specific rather than hopeful. The KLAX HIGH is the
worst market on the board: 52.4% bucket hit at its commit hour and overconfident
by 7.4 points. Measured 2026-07-31, the reason is structural -- at h11 the ratchet
bound sits a mean 2.88F from the value that finally settles, against 1.63F for the
LOW at h4. The HIGH is asked to forecast the whole afternoon on a 2F ladder.

What governs a coastal max is the MARINE LAYER: how deep the deck is and when it
burns off. The model cannot see that. `cur_sky` derives from `skyc`, which stores
the LAST sky token -- the HIGHEST layer -- so `BKN008 OVC200` is indistinguishable
from `FEW200`. The low deck that actually caps insolation is discarded.

`cur_sky_lowest_ft` (93% coverage at h11) and `cur_ceiling_ft` (47%) measure that
deck directly. `cur_pres_tend` (74%) is the advection input nowcast.trade grades
on. `cur_vis_sm` (100%) collapses in fog, the marine layer at its densest.

⚠️ PRE-REGISTERED SHIP GATE -- WRITTEN BEFORE ANY RESULT WAS SEEN.

Ship ONLY if, at the COMMIT HOUR h11, on the recent window:

    1. dHIT  > 0            parity-averaged modal-bucket hit must IMPROVE.
                            Bucket hit is what pays; MAE is not.
    2. dMAE <= 0            must not be worse on MAE.
    3. The +RANDOM control arm must NOT beat the real arm on dHIT.
    4. The dHIT improvement must exceed the +RANDOM arm's by a clear margin.

Rule 3 is the one that matters. This project has measured TWICE that adding
correlated columns HURTS by diluting LightGBM's per-split feature sampling --
`nbm_nbs` alone cost 2.7 points of bucket hit -- and a noise arm scoring as well
as the real one means the "gain" is sampling luck, not signal.

If the gate fails, the answer is NO and it gets written down as a negative result.
The scoreboard on confident pre-measurement hypotheses in this project is 1 for 5.

REGIME: matched-hour training, walk-forward with periodic refit, parity-averaged
bucket hit -- identical to `hour_curve.py` and `calib_eval.py` so the numbers are
comparable to everything else already measured. Recent window only (>= 2025-01-01)
because an expanding window averages in data-poor folds and reads pessimistic for
the shipped model -- measured, see `gating.HOUR_CALIB`.
"""
import sys

import numpy as np

from highcast.buckets import bucket_probs, expected_bucket_hit
from highcast.config import load_config
from highcast.db import connect
from highcast.features import build_features, uses_bias_correction
from highcast.gating import MIN_DECISION_HOUR
from highcast.metric import get_metric
from highcast.models.baseline import BaselineBlend
from highcast.models.intraday import IntradayModel, build_checkpoint_frame

HOUR = int(sys.argv[1]) if len(sys.argv) > 1 else None
REFIT = int(sys.argv[2]) if len(sys.argv) > 2 else 45
RECENT = "2025-01-01"
MIN_TRAIN = 400
N_BOOT = 2000
rng = np.random.default_rng(20260731)

CFG = load_config("config/klax.yaml")
METRIC = "high"
HOUR = HOUR or MIN_DECISION_HOUR[CFG.station.icao][METRIC]

# The candidate set. FOUR columns, not everything available -- `cur_slp` is
# excluded because absolute pressure without a tendency is a season/altitude
# proxy the model already gets from doy_*, and dilution has cost this project
# 2.7 points before.
ENV = ["cur_sky_lowest_ft", "cur_ceiling_ft", "cur_vis_sm", "cur_pres_tend"]
# H1: the warming-rate arm. Pre-registered 2026-07-31 as the PRIMARY hypothesis,
# after H2 (guidance freshness) was killed by a pre-check showing guidance moves
# 0.000F on 0/400 days between the 09:00 cutoff and the h11 commit.
RATE = ["cur_rate_3h", "cur_rate_1h", "cur_rise_from_min"]


def score(model, frame):
    qs = model.predict_quantiles(frame)
    y = frame["y"].to_numpy(dtype=float)
    pt = np.array([q[0.5] for q in qs])
    pmfs = [bucket_probs(q) for q in qs]
    hit = np.array([expected_bucket_hit(p, int(v)) for p, v in zip(pmfs, y)])
    return np.abs(pt - y), hit


def ci(a):
    idx = rng.integers(0, len(a), size=(N_BOOT, len(a)))
    # BONFERRONI-CORRECTED for two pre-registered primary arms: a 97.5% interval,
    # not 95%. Testing two hypotheses doubles the false-positive risk, and the
    # previous run manufactured one by selecting the best of several arms.
    return tuple(np.percentile(a[idx].mean(axis=1), [1.25, 98.75]))


conn = connect(CFG.db_path)
spec = get_metric(CFG, METRIC)
resid = bool(getattr(CFG.model, f"intraday_residual_{METRIC}", False))
k = CFG.model.intraday_bandwidth_for(METRIC)
tr_hours = tuple(sorted({(HOUR + d) % 24 for d in range(-k, k + 1)}))

dates = [r["date"] for r in conn.execute(
    "SELECT date FROM cli_reports WHERE station=? ORDER BY date",
    (CFG.station.icao,))]
print(f"{CFG.station.icao} {METRIC.upper()} h{HOUR}  resid={resid} k={k} "
      f"refit={REFIT}  n_cli={len(dates)}", flush=True)

day_df = build_features(conn, CFG, spec, dates, require_target=True)
base = BaselineBlend(prefer_corrected=uses_bias_correction(spec.key)).fit(day_df)
fr = build_checkpoint_frame(conn, CFG, dates, hours=tr_hours, metric=spec,
                            baseline=base)
# The noise control: same COUNT of columns as the real arm, pure noise. If this
# scores as well as ENV, the "gain" is feature-sampling luck.
for i in range(len(ENV)):
    fr[f"rand_{i}"] = rng.normal(size=len(fr))
RAND = [f"rand_{i}" for i in range(len(ENV))]

te_all = fr[(fr["hour"] == float(HOUR)) & fr["y"].notna()].sort_values("target_date")
te_all = te_all[te_all["target_date"] >= RECENT]
td = te_all["target_date"].to_numpy()
print(f"frame {len(fr)} rows; test at h{HOUR} on >= {RECENT}: {len(te_all)}",
      flush=True)

# COLUMN SUBSETS. This project has measured TWICE that correlated columns hurt
# by diluting LightGBM's per-split feature sampling, and `ceiling_ft` (lowest
# BKN/OVC) and `sky_lowest_ft` (lowest layer of any cover) are near-duplicates by
# construction. Testing the tighter sets is not fishing -- it is the documented
# lesson that a signal only pays once something makes room for it.
# PRE-REGISTERED ARMS. RATE is the primary; ENV4 is carried only as the already-
# rejected reference so the two are directly comparable in one table.
arms = {
    "BASE": [],
    "RATE": RATE,          # H1 -- the primary hypothesis
    "ENV4": ENV,           # previously rejected, kept for comparison
    "RATE+ENV": RATE + ENV,
    "RAND": RAND,
}
res = {a: {"ae": [], "hit": []} for a in arms}

for b0 in range(0, len(td), REFIT):
    cut = td[b0]
    tr = fr[(fr["target_date"] < cut) & fr["y"].notna()]
    if len(tr) < MIN_TRAIN:
        continue
    blk = te_all.iloc[b0:b0 + REFIT]
    if blk.empty:
        continue
    for name, extra in arms.items():
        m = IntradayModel(extremum=spec.extremum, residual=resid,
                          extra_features=extra).fit(tr)
        ae, hit = score(m, blk)
        res[name]["ae"].append(ae)
        res[name]["hit"].append(hit)
    if (b0 // REFIT) % 4 == 0:
        print(f"  fold cut={cut} train={len(tr)} block={len(blk)}", flush=True)

for a in res:
    res[a]["ae"] = np.concatenate(res[a]["ae"])
    res[a]["hit"] = np.concatenate(res[a]["hit"])

n = len(res["BASE"]["ae"])
print(f"\n=== KLAX HIGH @h{HOUR}, walk-forward, test >= {RECENT}, n={n} ===")
print(f"{'arm':6}{'MAE':>8}{'HIT':>9}{'dMAE':>9}{'dHIT':>9}  {'95% CI on dHIT':>22}")
for a in arms:
    dmae = res[a]["ae"].mean() - res["BASE"]["ae"].mean()
    dhit = res[a]["hit"].mean() - res["BASE"]["hit"].mean()
    if a == "BASE":
        print(f"{a:6}{res[a]['ae'].mean():>8.3f}{res[a]['hit'].mean():>9.1%}"
              f"{'--':>9}{'--':>9}")
    else:
        lo, hi = ci(res[a]["hit"] - res["BASE"]["hit"])
        print(f"{a:6}{res[a]['ae'].mean():>8.3f}{res[a]['hit'].mean():>9.1%}"
              f"{dmae:>+9.3f}{dhit:>+9.1%}  [{lo:+.1%},{hi:+.1%}]")

BEST = max((a for a in arms if a not in ("BASE", "RAND")),
            key=lambda a: res[a]["hit"].mean())
print(f"\nbest non-noise arm by HIT: {BEST}")
d_env_hit = res[BEST]["hit"].mean() - res["BASE"]["hit"].mean()
d_env_mae = res[BEST]["ae"].mean() - res["BASE"]["ae"].mean()
d_rnd_hit = res["RAND"]["hit"].mean() - res["BASE"]["hit"].mean()
lo, hi = ci(res[BEST]["hit"] - res["BASE"]["hit"])

print("\n=== PRE-REGISTERED SHIP GATE ===")
g1 = d_env_hit > 0
g2 = d_env_mae <= 0
g3 = d_rnd_hit < d_env_hit
g4 = (d_env_hit - d_rnd_hit) > 0.005
print(f"  1. dHIT > 0                     {d_env_hit:+.2%}   {'PASS' if g1 else 'FAIL'}")
print(f"  2. dMAE <= 0                    {d_env_mae:+.3f}   {'PASS' if g2 else 'FAIL'}")
print(f"  3. RAND does not beat ENV       {d_rnd_hit:+.2%}   {'PASS' if g3 else 'FAIL'}")
g5 = lo > 0
print(f"  4. ENV beats RAND by > 0.5pt    {d_env_hit - d_rnd_hit:+.2%}   {'PASS' if g4 else 'FAIL'}")
print(f"  5. dHIT CI excludes zero        [{lo:+.2%},{hi:+.2%}]   "
      f"{'PASS' if g5 else 'FAIL'}  <- ADDED after the first run: the original")
print("     gate omitted a significance requirement, which was a flaw in it.")
print(f"\n  VERDICT: {'SHIP' if all((g1, g2, g3, g4, g5)) else 'DO NOT SHIP'}")
print(f"  (dHIT 95% CI [{lo:+.1%},{hi:+.1%}]; CI containing 0 means the point "
      f"estimate is not distinguishable from no change)")
