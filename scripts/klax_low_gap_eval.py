"""Does the bound-vs-guidance disagreement earn its place in the KLAX LOW nowcast?

⚠️ OFFLINE MEASUREMENT ONLY. The KLAX LOW is FROZEN. Running this changes
nothing; only SHIPPING would need the freeze lifted, and that is Julian's call
AFTER seeing the number.

THE PROBLEM, measured 2026-07-31. The LOW commits at h4 anchored on the ratchet
bound -- the coldest reading so far. That bound is only informative once the
minimum has ACTUALLY HAPPENED, and it has not on **36% of days**:

    when the minimum lands      share    bound MAE   guidance MAE
    at or before h4 (commit)    64.0%      1.22         1.10
    h5-h7  (dawn continuing)    25.1%      2.35         1.24
    h19-h23 (evening cooling)   10.8%      2.45         1.14

Guidance error is FLAT at ~1.1-1.2F regardless of when the low lands. The bound's
error DOUBLES. The model weights the bound identically in both regimes because
nothing in its inputs tells it which regime it is in.

TWO LIVE CASES, one each way:

    2026-07-30 LOST   bound 69.88  guidance 67.52  gap +2.36  actual 67
                      Guidance was within 0.5F. The model output 68.76 -- split
                      the difference, landed the wrong side of a bucket line.
    2026-07-29 WON    bound 68.98  guidance 66.96  gap +2.02  actual 68
                      Here the blend at 68.08 BEAT guidance. So "always trust
                      guidance" is wrong; the bound genuinely helps most days.

THE HYPOTHESIS. `cur_bound_gap = bound - base_p50` is the regime signal. It is
large exactly when the bound is misleading. Crucially A TREE CANNOT DERIVE IT:
both inputs are already features, but splits threshold one feature at a time, so
it can never form their difference. This is the identical argument that justified
the `fcc_*` bias-corrected columns, which measured as a significant win.

THIS IS NOT A BAND-AID. A band-aid would be a hardcoded override ("if gap > 2,
use guidance"). This supplies a variable the model provably cannot compute and
lets it learn the weighting -- including learning to ignore it, if the gate says
the signal is not there.

⚠️ PRE-REGISTERED SHIP GATE -- WRITTEN BEFORE ANY RESULT WAS SEEN.

Two pre-registered arms, so the significance threshold is BONFERRONI-CORRECTED to
a 97.5% CI (the [1.25, 98.75] percentiles), not 95%.

    PRIMARY    GAP1 = [cur_bound_gap]                  the pure test
    SECONDARY  GAP3 = GAP1 + cur_rise_from_min + cur_rate_3h

An arm ships only if ALL of:
    1. dHIT  > 0
    2. dMAE <= 0
    3. RAND does not beat it
    4. it beats RAND by > 0.5pt
    5. its 97.5% CI on dHIT excludes zero

`cur_rate_1h` is EXCLUDED by pre-check: zero on 52% of days at h4, and an
inert feature cannot help while still costing feature-sampling dilution.

If both fail, that is the answer and it is recorded as a negative result. This
project is 2-for-7 on confident pre-measurement hypotheses.
"""
import numpy as np

from highcast.buckets import bucket_probs, expected_bucket_hit
from highcast.config import load_config
from highcast.db import connect
from highcast.features import build_features, uses_bias_correction
from highcast.gating import MIN_DECISION_HOUR
from highcast.metric import get_metric
from highcast.models.baseline import BaselineBlend
from highcast.models.intraday import IntradayModel, build_checkpoint_frame

REFIT = 45
RECENT = "2025-01-01"
MIN_TRAIN = 400
N_BOOT = 2000
rng = np.random.default_rng(20260731)

CFG = load_config("config/klax.yaml")
METRIC = "low"
HOUR = MIN_DECISION_HOUR[CFG.station.icao][METRIC]

GAP1 = ["cur_bound_gap"]
GAP3 = ["cur_bound_gap", "cur_rise_from_min", "cur_rate_3h"]


def score(model, frame):
    qs = model.predict_quantiles(frame)
    y = frame["y"].to_numpy(dtype=float)
    pt = np.array([q[0.5] for q in qs])
    hit = np.array([expected_bucket_hit(bucket_probs(q), int(v))
                    for q, v in zip(qs, y)])
    return np.abs(pt - y), hit


def ci(a):
    idx = rng.integers(0, len(a), size=(N_BOOT, len(a)))
    # Bonferroni for two pre-registered arms: a 97.5% interval, not 95%.
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
for i in range(len(GAP3)):
    fr[f"rand_{i}"] = rng.normal(size=len(fr))
RAND = [f"rand_{i}" for i in range(len(GAP3))]

te_all = fr[(fr["hour"] == float(HOUR)) & fr["y"].notna()].sort_values("target_date")
te_all = te_all[te_all["target_date"] >= RECENT]
td = te_all["target_date"].to_numpy()
print(f"test at h{HOUR} on >= {RECENT}: {len(te_all)}", flush=True)

arms = {"BASE": [], "GAP1": GAP1, "GAP3": GAP3, "RAND": RAND}
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
        print(f"  fold cut={cut} train={len(tr)}", flush=True)

for a in res:
    res[a]["ae"] = np.concatenate(res[a]["ae"])
    res[a]["hit"] = np.concatenate(res[a]["hit"])

n = len(res["BASE"]["ae"])
print(f"\n=== KLAX LOW @h{HOUR}, walk-forward, test >= {RECENT}, n={n} ===")
print(f"{'arm':6}{'MAE':>8}{'HIT':>9}{'dMAE':>9}{'dHIT':>9}  {'97.5% CI on dHIT':>22}")
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

d_rnd = res["RAND"]["hit"].mean() - res["BASE"]["hit"].mean()
print("\n=== PRE-REGISTERED SHIP GATE (per arm) ===")
for a in ("GAP1", "GAP3"):
    dh = res[a]["hit"].mean() - res["BASE"]["hit"].mean()
    dm = res[a]["ae"].mean() - res["BASE"]["ae"].mean()
    lo, hi = ci(res[a]["hit"] - res["BASE"]["hit"])
    g = (dh > 0, dm <= 0, d_rnd < dh, (dh - d_rnd) > 0.005, lo > 0)
    print(f"  {a}: dHIT {dh:+.2%}  dMAE {dm:+.3f}  CI [{lo:+.2%},{hi:+.2%}]  "
          f"-> {'SHIP' if all(g) else 'DO NOT SHIP'}")
    if not all(g):
        names = ["dHIT>0", "dMAE<=0", "beats RAND", "margin>0.5pt", "CI excl 0"]
        print(f"       failed: {[nm for nm, ok in zip(names, g) if not ok]}")
