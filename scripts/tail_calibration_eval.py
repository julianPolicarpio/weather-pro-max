"""Is the model's TAIL calibrated? The whole NO-side idea rests on this.

THE QUESTION. Every calibration measurement this project has made
(`HOUR_CALIB`, `BAND_CALIB`, `hour_curve`) scores ONE bucket per day: the modal
one. Buying a NO contract is a bet on a bucket the model says will NOT happen,
so it is priced off the tail -- a region nothing here has ever measured.

If the model says a bucket is 8% and it hits 8% of the time, NO on it is fairly
priced and there is no trade. If it says 8% and it hits 3%, NO carries 5 points
of edge, against a fee hurdle of roughly 1 point at the 80-90c prices Julian
wants to trade.

WHAT IS MEASURED. Walk-forward at each market's decision hour, exactly the
harness `decision_hour_move_eval.py` uses. For EVERY 2F bucket on the ladder --
not just the modal one -- record (model probability, did it settle there). Then
bin by model probability and compare the mean stated probability in each bin
against the realised hit rate.

    reliability gap = mean(stated) - mean(hit)
    gap > 0  => the model OVERSTATES that bucket => its NO is UNDERPRICED => edge
    gap < 0  => the model UNDERSTATES it        => its NO is OVERPRICED  => no trade

PARITY IS AVERAGED, as everywhere else in this project. Kalshi centres each
day's ladder on the consensus, so the parity is a coin flip; scoring one fixed
ladder is right about half the time and silently wrong the rest. Each day is
scored on BOTH ladders and the two are averaged, matching
`buckets.expected_bucket_hit`.

⚠️ THIS MEASURES THE MODEL, NOT THE TRADE. A positive gap is necessary for the
NO idea but not sufficient: the market also has to be pricing that bucket ABOVE
the model's corrected number, and this script sees no prices. It answers only
"is there anything in the tail to trade", which is the question that has to come
first.

⚠️ EXPECTED OUTCOME: the tail is probably UNDER-stated, not over-stated. Quantile
models trained on a squared/pinball loss habitually under-disperse, and
`models/intraday.py` already records measured under-dispersion in its tails
(0.10 -> 0.125, 0.90 -> 0.833). Under-dispersion means the model puts too LITTLE
mass in the tail, which makes its NO contracts OVERpriced and kills the idea.
Being wrong about that is the point of running this.

Usage: PYTHONPATH=src .venv/bin/python scripts/tail_calibration_eval.py
"""
import collections
import math

import numpy as np

from highcast.buckets import bucket_probs, ladder_lo
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
# The probability bins. The 0.05-0.25 rows are the ones the NO idea lives in:
# a bucket the model calls 5-25% trades its NO at 75-95c.
BINS = [(0.00, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 0.15),
        (0.15, 0.20), (0.20, 0.30), (0.30, 0.50), (0.50, 1.01)]

_SHIPPED_BASE = ["hour", "run_lo", "run_hi", "run_width", "cur_tmpf",
                 "cur_dwpf", "cur_onshore", "cur_sky", "base_p50",
                 "base_spread", "doy_sin", "doy_cos", "year_index",
                 "heat_alert"]

ARMS = [("config/kmia.yaml", "high"), ("config/klax.yaml", "high"),
        ("config/kmia.yaml", "low"), ("config/klax.yaml", "low")]


def wilson(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def bucket_series(pmf: dict[int, float], parity: int) -> list[tuple[int, float]]:
    """Every 2F bucket on this parity's ladder, as (lo_degree, probability).

    A bucket is the DEGREE PAIR (lo, lo+1); `ladder_lo` maps a degree to its
    bucket's lower edge for the given parity, so summing the pmf by that key
    yields the ladder the market actually quotes.
    """
    agg: dict[int, float] = collections.defaultdict(float)
    for deg, p in pmf.items():
        agg[ladder_lo(int(deg), parity)] += p
    return sorted(agg.items())


def collect(cfg_path: str, metric_key: str) -> list[tuple[float, float]]:
    """(stated probability, hit) for EVERY bucket on EVERY test day, both parities."""
    cfg = load_config(cfg_path)
    icao = cfg.station.icao
    conn = connect(cfg.db_path)
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
    print(f"\n### {icao} {metric_key.upper()} h{hour}   test days={len(te)}", flush=True)

    out: list[tuple[float, float]] = []
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
            pmf = bucket_probs(q)
            y = int(row["y"])
            for parity in (0, 1):
                ywin = ladder_lo(y, parity)
                for lo, p in bucket_series(pmf, parity):
                    # weight 0.5 each so a day contributes 1.0 total per bucket
                    out.append((p, 1.0 if lo == ywin else 0.0))
    return out


print(__doc__.split("Usage:")[0])
pooled: list[tuple[float, float]] = []
for cfg_path, metric_key in ARMS:
    recs = collect(cfg_path, metric_key)
    pooled += recs
    arr = np.array(recs)
    print(f"    {'model says':>14}{'n':>8}{'stated':>9}{'actual':>9}"
          f"{'gap':>9}{'95% CI on actual':>20}")
    for lo, hi in BINS:
        sel = arr[(arr[:, 0] >= lo) & (arr[:, 0] < hi)]
        if len(sel) < 30:
            continue
        stated, hit = sel[:, 0].mean(), sel[:, 1].mean()
        cl, cu = wilson(sel[:, 1].sum(), len(sel))
        flag = ""
        if 0.02 <= lo <= 0.20:
            flag = "  <== NO zone" if stated - hit > 0 else "  <== NO zone (wrong way)"
        print(f"    {lo:>6.0%}-{hi:<7.0%}{len(sel):>8}{stated:>9.1%}{hit:>9.1%}"
              f"{stated - hit:>+9.1%}   [{cl:>5.1%},{cu:>5.1%}]{flag}")

print("\n=== POOLED (all four markets) ===")
arr = np.array(pooled)
print(f"    {'model says':>14}{'n':>8}{'stated':>9}{'actual':>9}{'gap':>9}"
      f"{'95% CI on actual':>20}")
for lo, hi in BINS:
    sel = arr[(arr[:, 0] >= lo) & (arr[:, 0] < hi)]
    if len(sel) < 30:
        continue
    stated, hit = sel[:, 0].mean(), sel[:, 1].mean()
    cl, cu = wilson(sel[:, 1].sum(), len(sel))
    print(f"    {lo:>6.0%}-{hi:<7.0%}{len(sel):>8}{stated:>9.1%}{hit:>9.1%}"
          f"{stated - hit:>+9.1%}   [{cl:>5.1%},{cu:>5.1%}]")

print("\ngap > 0 => model OVERSTATES that bucket => its NO is UNDERPRICED => tradeable")
print("gap < 0 => model UNDERSTATES it => its NO is OVERPRICED => no trade there")
print("\nThis measures the MODEL only. A positive gap still has to survive the")
print("market's own price before it is an edge.")
