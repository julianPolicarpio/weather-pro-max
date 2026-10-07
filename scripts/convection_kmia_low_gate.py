"""DOES CONVECTION HELP THE KMIA **LOW** AT ITS COMMIT HOUR? The untested gap.

WHY THIS EXISTS, and why it is not a re-run of a dead idea. Convection features
were measured and rejected in `scripts/convection_eval.py` — but at KMIA **HIGH**
h13. QUEUE item 3 records the gap in its own words: *"The LOW's commit hour is
not scored at all by the eval — its predictive-hour filter needs >=60% of the
extremum still ahead and h7 has 42%, so h7 is excluded by construction."*

The loss decomposition (2026-08-28) says that is exactly the wrong market to have
skipped. Of 29 settled losses, 4 are >=4F out and **every one of them is KMIA
LOW**, driven by rain-cooled downdrafts:

    2026-08-01  bet >=81   actual 74   7F out   low at 4:34 PM  0.58"
    2026-08-03  bet 77-78  actual 73   4F out   low at 1:46 PM  1.20"
    2026-08-19  bet 80-81  actual 76   4F out   low at 6:05 PM  0.25"
    2026-08-26  bet 81-82  actual 77   4F out   low at 2:48 PM  0.02"

A daily minimum stamped in the afternoon cannot be diurnal. Across 2,391 CLI
days, 10.0% of KMIA days have a low timed 09:00-19:00 and 84% of those had
>=0.10" precip against a 28% base rate — so the mechanism is established at
scale, not inferred from four days.

────────────────────────────────────────────────────────────────────────────
WHAT IS CHANGED FROM THE SHIPPED EVAL: ONE THING.
────────────────────────────────────────────────────────────────────────────
The hour filter. `PREDICTIVE_MIN = 0.60` is replaced by naming the hours
directly, so h7 is scored. **`PEAK = (11, 17)` is DELIBERATELY UNCHANGED** even
though the observed storm-lows run to 18:05 — widening it would be a free
parameter tuned on the very failures that motivated the test, and this project
has lost four ideas to exactly that move.

ADDED: a NOISE CONTROL the original lacks. `klax_high_env_eval` established that
perturbing LightGBM's column set moves scores on noise alone, and ARM B adds
FOUR columns. ARM C adds four random ones so "helps" can be told from "jiggled".

────────────────────────────────────────────────────────────────────────────
PRE-REGISTERED, WRITTEN BEFORE THE RUN
────────────────────────────────────────────────────────────────────────────
PRIMARY: KMIA LOW at **h7**, its commit hour. h5 and h6 are reported for shape
and are NOT the decision.

SHIP only if all three hold at h7:
    1. dHIT > 0                       (bucket hit is what pays; MAE is not)
    2. paired bootstrap CI on dHIT excludes 0 at 95%
    3. ARM B beats ARM C (noise) on dHIT

⚠️ EXPECTED OUTCOME: FAILURE. The shipped eval's verdict was *"by the commit
hour the ratchet and current temperature have already absorbed what convection
would have told us"*, and at h7 the ratchet already sees 58% of the low. The
honest reason to run it anyway is that h7 is the only hour where the damage
actually lands and it has never been scored — untested is not the same as
rejected. A negative result is a real result and gets written down.

⚠️ SINGLE HOLDOUT, not walk-forward — inherited from the shipped harness so the
numbers stay comparable to the rejected HIGH arm. Read the CI accordingly.

Usage: PYTHONPATH=src .venv/bin/python scripts/convection_kmia_low_gate.py
"""
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from highcast.buckets import bucket_probs, expected_bucket_hit
from highcast.config import load_config
from highcast.db import connect
from highcast.metric import get_metric
from highcast.models.intraday import (INTRADAY_FEATURES, IntradayModel,
                                      build_checkpoint_frame, features_for)

CONFIG, METRIC = "config/kmia.yaml", "low"
CONV_START, SPLIT = "2024-10-09", "2026-01-01"
PEAK = (11, 17)                 # UNCHANGED from the shipped eval, on purpose
HOURS = (5, 6, 7)               # h7 is the commit hour and the PRIMARY test
PRIMARY = 7
N_BOOT = 4000
CONV = ["conv_pop_max", "conv_cape_max", "conv_qpf", "conv_pop_hour"]
rng = np.random.default_rng(20260828)

print(__doc__.split("Usage:")[0])
cfg = load_config(CONFIG)
conn = connect(cfg.db_path, migrate=False)
tz = ZoneInfo(cfg.station.tz)
spec = get_metric(cfg, METRIC)
resid = bool(getattr(cfg.model, f"intraday_residual_{METRIC}", False))
k = cfg.model.intraday_bandwidth_for(METRIC)
# ⚠️ Measure the model that SHIPS — `features_for` may carry extras beyond
# INTRADAY_FEATURES. Same rule the eval-script guard enforces elsewhere.
BASE_EXTRA = [c for c in features_for(cfg.station.icao, METRIC)
              if c not in INTRADAY_FEATURES]

per_day: dict[str, dict] = {}
for r in conn.execute("""SELECT valid_utc, pop, qpf, cape FROM forecast_conv
                         WHERE station=? ORDER BY valid_utc""",
                      (cfg.station.icao,)):
    loc = datetime.fromisoformat(r["valid_utc"]).replace(
        tzinfo=ZoneInfo("UTC")).astimezone(tz)
    if not (PEAK[0] <= loc.hour <= PEAK[1]):
        continue
    b = per_day.setdefault(loc.date().isoformat(),
                           {"pop": [], "qpf": [], "cape": [], "pop_h": []})
    if r["pop"] is not None:
        b["pop"].append(r["pop"]); b["pop_h"].append((r["pop"], loc.hour))
    if r["qpf"] is not None:
        b["qpf"].append(r["qpf"])
    if r["cape"] is not None:
        b["cape"].append(r["cape"])
conv = pd.DataFrame([{
    "target_date": d,
    "conv_pop_max": max(v["pop"]) if v["pop"] else np.nan,
    "conv_cape_max": max(v["cape"]) if v["cape"] else np.nan,
    "conv_qpf": float(np.mean(v["qpf"])) if v["qpf"] else np.nan,
    "conv_pop_hour": (max(v["pop_h"])[1] if v["pop_h"] else np.nan),
} for d, v in per_day.items()])
print(f"conv rows {len(conv)}  {conv['target_date'].min()} .. "
      f"{conv['target_date'].max()}", flush=True)

dates = [r["date"] for r in conn.execute(
    "SELECT date FROM cli_reports WHERE station=? AND date>=? ORDER BY date",
    (cfg.station.icao, CONV_START))]

print(f"\n{'h':>3}{'n':>6} | {'A mae':>7}{'A hit':>7} | {'B mae':>7}{'B hit':>7} | "
      f"{'C(rand) hit':>12} | {'dHIT':>7}  {'CI (95%)':>18}")
verdict = {}
for h in HOURS:
    tr_hours = tuple(sorted({(h + d) % 24 for d in range(-k, k + 1)}))
    fr = build_checkpoint_frame(conn, cfg, dates, hours=tr_hours, metric=spec)
    fr = fr[fr["y"].notna()].merge(conv, on="target_date", how="left")
    for i in range(len(CONV)):
        fr[f"rand_{i}"] = rng.normal(size=len(fr))
    RAND = [f"rand_{i}" for i in range(len(CONV))]
    tr = fr[fr["target_date"] < SPLIT]
    te = fr[(fr["target_date"] >= SPLIT) & (fr["hour"] == float(h))]
    if len(te) < 40 or len(tr) < 200:
        print(f"{h:>3}{len(te):>6} | skipped (train {len(tr)})"); continue
    y = te["y"].to_numpy(dtype=float)
    hit = {}
    mae = {}
    for arm, extra in (("A", []), ("B", CONV), ("C", RAND)):
        m = IntradayModel(extremum=spec.extremum, residual=resid,
                          extra_features=BASE_EXTRA + extra).fit(tr)
        qs = m.predict_quantiles(te)
        pt = np.array([q[0.5] for q in qs])
        hit[arm] = np.array([expected_bucket_hit(bucket_probs(q), int(v))
                             for q, v in zip(qs, y)])
        mae[arm] = float(np.abs(pt - y).mean())
    d = hit["B"] - hit["A"]
    idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
    lo, hi = np.percentile(d[idx].mean(axis=1), [2.5, 97.5])
    dc = (hit["C"] - hit["A"]).mean()
    star = "  <-- PRIMARY" if h == PRIMARY else ""
    print(f"{h:>3}{len(te):>6} | {mae['A']:>7.3f}{hit['A'].mean():>6.1%} | "
          f"{mae['B']:>7.3f}{hit['B'].mean():>6.1%} | {hit['C'].mean():>11.1%} | "
          f"{d.mean()*100:>+6.1f}  [{lo*100:+.1f},{hi*100:+.1f}]{star}")
    verdict[h] = (d.mean() > 0, lo > 0, d.mean() > dc, d.mean(), lo, hi, dc)

print()
if PRIMARY in verdict:
    c1, c2, c3, dm, lo, hi, dc = verdict[PRIMARY]
    print(f"PRIMARY h{PRIMARY}:  (1) dHIT>0 {c1}   (2) CI excludes 0 {c2}   "
          f"(3) beats noise {c3}  (noise arm {dc*100:+.1f} pts)")
    print(f"  ==> {'SHIP' if (c1 and c2 and c3) else 'DO NOT SHIP'}")
    if not (c1 and c2 and c3):
        print("      Write it down as a negative result. The KMIA LOW storm tail")
        print("      is then blindspot B — unforecastable at the commit hour —")
        print("      and the answer is to SIZE for it, not to predict it.")
else:
    print("PRIMARY hour not scored — insufficient data.")
