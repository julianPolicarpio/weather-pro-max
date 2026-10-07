"""Should the nowcast's `heat_alert` be evaluated AS OF the checkpoint?
PRE-REGISTERED GATE — written 2026-08-19 before any result was seen.

THE DEFECT, found while trying to answer "does an active advisory predict a
bad forecast?" (2026-08-19). `models/intraday.build_checkpoint_frame` sets

    "heat_alert": day_row["heat_alert"]

i.e. the nowcast INHERITS the day-ahead row's value, computed at the DAY-AHEAD
cutoff. That is right for a day-ahead forecast and wrong for a nowcast running
hours later. Miami's Heat Advisories are issued the same morning (onset
typically 15:00-16:00Z), after the day-ahead cutoff and well before h13:

    market      inherited (shipped)   as-of the checkpoint
    KMIA high        1 / 593                46 / 593
    KLAX high       17 / 593                25 / 593

KMIA's single inherited hit is 2026-02-23 — a RED FLAG WARNING. The feature has
never once fired on a Miami heat day. KLAX's 17 are January/March Santa Ana
fire-weather products, so at that station `heat_alert` is in practice a WIND
flag. `HEAT_EVENTS` includes Red Flag Warning, which is why.

⚠️ THIS IS A LIVE MODEL FEATURE ON ALL FOUR MARKETS. `features._alert_flag`
warns in its own docstring that changing it silently would be "a model change
disguised as a data change, with no gate and no way to notice". Hence this gate.
`heat_alert_asof` is written to the frame as a SEPARATE column; no shipped model
reads it. This script swaps the COLUMN CONTENTS, so the arms differ in exactly
the way shipping would and in no other way.

PRE-REGISTERED SHIP GATE. For each market independently, at its commit hour, on
the recent window, walk-forward with periodic refit:

    1. dHIT > 0                 parity-averaged modal-bucket hit must IMPROVE.
                                Bucket hit is what pays; MAE is not.
    2. CI on dHIT excludes 0    at Bonferroni x4 (four markets) => 98.75% interval
    3. dMAE <= 0                must not be worse on the point forecast
    4. the RANDOM control must NOT match the real arm on dHIT

Rule 4 is the one that matters. This project has measured TWICE that perturbing
columns shifts LightGBM's per-split sampling and moves scores on noise alone. The
control replaces `heat_alert` with a random 0/1 flag at the SAME marginal rate as
`heat_alert_asof`, so it reproduces the perturbation without the information.

⚠️ EXPECTED OUTCOME, STATED UP FRONT: **null, and probably underpowered.** 46
flagged days in 593 is 7.8%; at KLAX the flag is a wind product and there is no
reason it should predict a temperature error at all. A correctness fix is not
automatically an accuracy win, and "it is more principled" is precisely the
argument that lost four times here. If the gate fails, the honest outcome is:
leave the shipped column alone, keep `heat_alert_asof` in the frame as a
measured-inert column, and record the negative result.

⚠️ A FAILURE HERE DOES NOT MEAN THE FEATURE IS FINE. It means the repair does not
pay in bucket hit. The DESCRIPTIVE fact — the nowcast cannot see Miami heat
advisories — stands either way and belongs in the handoff regardless.

Usage: PYTHONPATH=src .venv/bin/python scripts/heat_alert_asof_gate.py [refit]
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
from highcast.models.intraday import (INTRADAY_FEATURES, IntradayModel,
                                      build_checkpoint_frame, features_for)

REFIT = int(sys.argv[1]) if len(sys.argv) > 1 else 45
RECENT = "2025-01-01"
MIN_TRAIN = 400
N_BOOT = 2000
N_MARKETS = 4
rng = np.random.default_rng(20260819)


def score(model, frame):
    qs = model.predict_quantiles(frame)
    y = frame["y"].to_numpy(dtype=float)
    pt = np.array([q[0.5] for q in qs])
    hit = np.array([expected_bucket_hit(bucket_probs(q), int(v))
                    for q, v in zip(qs, y)])
    return np.abs(pt - y), hit


def ci(a):
    """Paired bootstrap, Bonferroni x4 => 98.75% two-sided."""
    idx = rng.integers(0, len(a), size=(N_BOOT, len(a)))
    return tuple(np.percentile(a[idx].mean(axis=1), [0.625, 99.375]))


print(__doc__.split("Usage:")[0])
verdicts = {}
for cfgp in ("config/klax.yaml", "config/kmia.yaml"):
    for mkey in ("high", "low"):
        cfg = load_config(cfgp)
        icao = cfg.station.icao
        conn = connect(cfg.db_path, migrate=False)
        spec = get_metric(cfg, mkey)
        hour = MIN_DECISION_HOUR[icao][mkey]
        resid = bool(getattr(cfg.model, f"intraday_residual_{mkey}", False))
        k = cfg.model.intraday_bandwidth_for(mkey)
        tr_hours = tuple(sorted({(hour + d) % 24 for d in range(-k, k + 1)}))
        # ⚠️ MEASURE THE MODEL THAT ACTUALLY SHIPS. Fitting with the default
        # feature set pins this to `INTRADAY_FEATURES`, but KLAX HIGH also ships
        # `RATE_FEATURES` -- so the first version of this gate measured a model
        # nobody runs on that market. Caught by
        # `tests/test_eval_scripts.py::test_live_eval_scripts_choose_their_feature_set_explicitly`,
        # which exists because `calib_eval.py` made this exact mistake.
        extra = [c for c in features_for(icao, mkey)
                 if c not in INTRADAY_FEATURES]
        dates = [r["date"] for r in conn.execute(
            "SELECT date FROM cli_reports WHERE station=? ORDER BY date", (icao,))]
        day_df = build_features(conn, cfg, spec, dates, require_target=True)
        base = BaselineBlend(
            prefer_corrected=uses_bias_correction(spec.key)).fit(day_df)
        fr = build_checkpoint_frame(conn, cfg, dates, hours=tr_hours,
                                    metric=spec, baseline=base)
        rate = float(fr["heat_alert_asof"].mean())
        print(f"\n{'='*74}\n{icao} {mkey.upper()} @h{hour}  refit={REFIT}  "
              f"frame={len(fr)}  inherited={fr['heat_alert'].sum():.0f} rows, "
              f"as-of={fr['heat_alert_asof'].sum():.0f} rows\n{'='*74}", flush=True)

        # THE SWAP. Identical frames but for what `heat_alert` CONTAINS.
        arms = {"BASE": fr}
        f2 = fr.copy(); f2["heat_alert"] = f2["heat_alert_asof"]; arms["ASOF"] = f2
        f3 = fr.copy()
        f3["heat_alert"] = (rng.random(len(f3)) < rate).astype(float)
        arms["RAND"] = f3

        te_idx = fr[(fr["hour"] == float(hour)) & fr["y"].notna()] \
            .sort_values("target_date")
        te_idx = te_idx[te_idx["target_date"] >= RECENT]
        td = te_idx["target_date"].to_numpy()
        res = {a: {"ae": [], "hit": []} for a in arms}
        for b0 in range(0, len(td), REFIT):
            cut = td[b0]
            blk_dates = set(td[b0:b0 + REFIT])
            for name, F in arms.items():
                tr = F[(F["target_date"] < cut) & F["y"].notna()]
                if len(tr) < MIN_TRAIN:
                    break
                blk = F[(F["hour"] == float(hour)) & F["y"].notna()
                        & F["target_date"].isin(blk_dates)].sort_values("target_date")
                if blk.empty:
                    break
                m = IntradayModel(extremum=spec.extremum, residual=resid,
                                  extra_features=extra).fit(tr)
                ae, hit = score(m, blk)
                res[name]["ae"].append(ae)
                res[name]["hit"].append(hit)
        if not res["BASE"]["ae"]:
            print("  no folds — skip"); continue
        for a in res:
            res[a]["ae"] = np.concatenate(res[a]["ae"])
            res[a]["hit"] = np.concatenate(res[a]["hit"])
        n = len(res["BASE"]["ae"])
        print(f"  n={n}")
        print(f"  {'arm':<6}{'MAE':>8}{'HIT':>9}{'dMAE':>9}{'dHIT':>9}"
              f"{'dHIT CI (Bonf x4)':>26}")
        out = {}
        for a in ("BASE", "ASOF", "RAND"):
            mae, hit = res[a]["ae"].mean(), res[a]["hit"].mean()
            if a == "BASE":
                print(f"  {a:<6}{mae:>8.3f}{hit:>9.1%}")
                continue
            dae = res[a]["ae"] - res["BASE"]["ae"]
            dh = res[a]["hit"] - res["BASE"]["hit"]
            lo, hi = ci(dh)
            out[a] = (dh.mean(), lo, hi, dae.mean())
            print(f"  {a:<6}{mae:>8.3f}{hit:>9.1%}{dae.mean():>+9.3f}"
                  f"{dh.mean():>+9.1%}   [{lo:+.1%}, {hi:+.1%}]")
        dh, lo, hi, dae = out["ASOF"]
        ok = (dh > 0 and lo > 0 and dae <= 0 and dh > out["RAND"][0])
        verdicts[f"{icao} {mkey}"] = ok
        print(f"  -> {'SHIP' if ok else 'DO NOT SHIP'}"
              f"   (dHIT>0 {dh>0}, CI excl 0 {lo>0}, dMAE<=0 {dae<=0}, "
              f"beats RAND {dh>out['RAND'][0]})")

print(f"\n{'='*74}\nVERDICT PER MARKET\n{'='*74}")
for k_, v in verdicts.items():
    print(f"  {k_:<12} {'SHIP' if v else 'DO NOT SHIP'}")
print("\n  Markets decide independently. A market that fails keeps the inherited")
print("  column. The DESCRIPTIVE defect (the nowcast cannot see Miami heat")
print("  advisories) is true regardless of this table.")
