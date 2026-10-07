"""Is there ANY preflight signal for the two HIGH markets — at the hours we are
proposing to move to?

WHY THIS EXISTS. `preflight_eval.py` shipped a CLEAR/FAIR/AVOID grade for KLAX
LOW and found nothing for either HIGH: on `bound_gap` KLAX HIGH's effect
REVERSES (-1.9%, CI [-9.0,+5.2]) and KMIA HIGH is +3.0% [-3.8,+10.1]. A bot is
being designed that trades ONLY the two HIGHs, and one of its declared fire
conditions is "preflight says CLEAR/FAIR". That condition currently cannot be
built. This script is the attempt to build it honestly, or to establish that it
cannot be and let the bot ship without it.

TWO THINGS ARE DIFFERENT FROM THE FIRST RUN.

1. **The hour.** Both HIGH measurements above were taken at h11. The proposal is
   to commit at h13 (KLAX) / h14 (KMIA) — each station's MEDIAN peak-heat hour,
   measured from 2,366 KMIA and 1,959 KLAX CLI reports. At h11 roughly 90% of
   the high is still ahead and there is barely any observed trace to stratify
   ON; by h13/h14 half of it has happened. A factor that separates nothing at
   h11 may separate a great deal once there is a partial answer to compare
   against. Both hours are run so the comparison is explicit.

2. **More candidates**, chosen under the SAME rule the first run used: a factor
   is only admissible if the model CANNOT ALREADY SEE IT. Stratifying on a model
   input mostly re-derives p_model and dresses it up as new information.

   The shipped feature set is `_SHIPPED_BASE` (+ RATE_FEATURES for KLAX HIGH),
   so cur_tmpf, cur_dwpf, cur_onshore, cur_sky, base_p50, base_spread, the
   ratchet (run_lo/run_hi/run_width) and heat_alert are all INSIDE and therefore
   inadmissible on their own.

CANDIDATES — DECLARED IN FULL BEFORE ANY RESULT WAS SEEN:

    bound_gap   |ratchet bound - base_p50|. Carried over as PRIMARY for
                continuity with the shipped KLAX LOW gate.
    stall_h     hours the running extremum has sat unchanged. Strongest
                exploratory signal in the first run (+11.7% KLAX, +12.0% KMIA,
                both CIs crossing zero).
    override    |model point - base_p50|. NEW. base_p50 is an input but the
                MAGNITUDE BY WHICH THE NOWCAST OVERRIDES IT is not, and it has a
                documented motivating case: on 2026-07-30 guidance had the LOW
                nearly exact (67.52 vs a settled 67), the model overrode it, and
                the bet lost.
    guid_churn  |mean guidance at lead-1 - mean guidance at lead-2|. NEW. The
                model sees one blended number and its cross-model spread, but
                never how much the guidance MOVED between runs. A day the models
                are revising hard is a day to distrust. Also directly relevant:
                the evening day-ahead bug found 2026-07-31 was a lead-staleness
                failure.
    seam        KLAX HIGH ONLY: model point inside the 80-84F marine-layer seam,
                where the model is documented to overshoot (+1.85F live, n=4).
                An OUTPUT-regime split, not an input split — the same thing
                BAND_CALIB does for probability, asked of the point.
    cape        KMIA HIGH ONLY: max forecast CAPE over the target day. NEW.
                Miami's high is convection-limited; a storm caps the afternoon.
                Genuinely outside every feature set. Coverage starts 2024-07-29
                so n is smaller than the others — reported, not hidden.

MULTIPLICITY. This is a SCREENING run over 5-6 factors x 2 markets. Reporting an
uncorrected 95% CI across ~11 tests would manufacture a winner by construction.
Every CI here is Bonferroni-corrected to 99.55% (0.05/11). A factor that clears
that bar has EARNED a confirmatory pre-registered run — it has not shipped.

SPLIT is the within-market median, fixed before any result was seen, exactly as
in `preflight_eval.py`. `seam` is the one exception and is a natural split (in
band / out of band), declared as such here rather than chosen after the fact.

SHIP GATE (all must hold, same as the shipped one):
    1. shortfall is HIGHER in the flagged stratum
    2. the flagged stratum's shortfall is materially positive (> 3 points)
    3. the Bonferroni CI on the difference excludes zero

EXPECTED OUTCOME: nothing clears. This project is 3-for-9 on confident
pre-measurement hypotheses, and the first preflight run already looked here and
found nothing. A null result is publishable and is the answer the bot design
needs: it would mean the HIGH bot ships WITHOUT a preflight condition rather
than with a borrowed or manufactured one.

Usage: PYTHONPATH=src .venv/bin/python scripts/preflight_high_eval.py
"""
import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np

from highcast import board
from highcast.buckets import bucket_probs, expected_bucket_hit, modal_bucket
from highcast.config import load_config
from highcast.db import connect
from highcast.features import build_features, uses_bias_correction
from highcast.metric import get_metric
from highcast.models.baseline import BaselineBlend
from highcast.models.intraday import (IntradayModel, build_checkpoint_frame,
                                      features_for)

REFIT = 45
RECENT = "2025-01-01"
MIN_TRAIN = 400
N_BOOT = 2000
N_TESTS = 11                      # 5 factors x 2 markets, +1 for KLAX seam
ALPHA = 0.05 / N_TESTS
PCTS = [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)]
rng = np.random.default_rng(20260731)

_SHIPPED_BASE = ["hour", "run_lo", "run_hi", "run_width", "cur_tmpf",
                 "cur_dwpf", "cur_onshore", "cur_sky", "base_p50",
                 "base_spread", "doy_sin", "doy_cos", "year_index",
                 "heat_alert"]

# (config, metric, [hours to evaluate])  -- current hour first, proposal second
ARMS = [("config/klax.yaml", "high", [11, 13]),
        ("config/kmia.yaml", "high", [11, 14])]


def p_model(pmf: dict) -> float:
    return float(np.mean([modal_bucket(pmf, q)[1] for q in (0, 1)]))


def boot_diff(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    ia = rng.integers(0, len(a), size=(N_BOOT, len(a)))
    ib = rng.integers(0, len(b), size=(N_BOOT, len(b)))
    d = a[ia].mean(axis=1) - b[ib].mean(axis=1)
    return tuple(np.percentile(d, PCTS))


def guid_churn(conn, icao: str, metric: str, target: str) -> float | None:
    """|mean guidance at lead-1 - mean guidance at lead-2| for this target.

    Uses the plain cross-model mean at each run rather than the fitted blend:
    the blend's weights are themselves fitted on history, and re-deriving them
    per day here would make the factor depend on the model it is trying to
    police."""
    out = {}
    for lead in (1, 2):
        r = conn.execute(
            """SELECT AVG(max_temp_f) FROM forecasts
               WHERE station=? AND metric=? AND target_date=?
                 AND run_date=date(?, ?)""",
            (icao, metric, target, target, f"-{lead} day")).fetchone()
        out[lead] = r[0] if r and r[0] is not None else None
    if out[1] is None or out[2] is None:
        return None
    return abs(out[1] - out[2])


def cape_max(conn, icao: str, target: str, asof: str) -> float | None:
    """Highest forecast CAPE anywhere in the target day, as known at `asof`.

    `fetched_at <= asof` keeps the factor honest: a preflight may only use what
    was on the desk at commit time."""
    r = conn.execute(
        """SELECT MAX(cape) FROM forecast_conv
           WHERE station=? AND cape IS NOT NULL
             AND valid_utc >= ? AND valid_utc < date(?, '+1 day')
             AND fetched_at <= ?""",
        (icao, target, target, asof)).fetchone()
    return None if r is None or r[0] is None else float(r[0])


def collect(cfg_path: str, metric_key: str, hour: int) -> dict:
    cfg = load_config(cfg_path)
    icao = cfg.station.icao
    conn = connect(cfg.db_path)
    spec = get_metric(cfg, metric_key)
    resid = bool(getattr(cfg.model, f"intraday_residual_{metric_key}", False))
    # PER-METRIC as of 2026-08-02 (`metric_key` is this run's metric, not a
    # station-wide setting): KLAX ships k=1 on the LOW and k=0 on the HIGH, so a
    # station-wide read scores a model nobody runs.
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
    print(f"\n### {icao} {metric_key.upper()} h{hour}   test n={len(te)}", flush=True)

    recs = []
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
            bg = row.get("cur_bound_gap")
            bp = row.get("base_p50")
            recs.append({
                "date": row["target_date"],
                "p": p_model(pmf),
                "hit": expected_bucket_hit(pmf, int(row["y"])),
                "point": float(q[0.5]),
                "bound_gap": abs(bg) if bg is not None and not np.isnan(bg) else np.nan,
                "override": (abs(float(q[0.5]) - float(bp))
                             if bp is not None and not np.isnan(bp) else np.nan),
            })

    tz = ZoneInfo(cfg.station.tz)
    for r in recs:
        d = dt.date.fromisoformat(r["date"])
        asof = dt.datetime(d.year, d.month, d.day, hour,
                           tzinfo=tz).astimezone(dt.timezone.utc)
        r["stall_h"] = board._stall_hours(conn, cfg, d, spec.extremum, asof)
        r["guid_churn"] = guid_churn(conn, icao, metric_key, r["date"])
        r["seam"] = 1.0 if 80.0 <= r["point"] <= 84.0 else 0.0
        r["cape"] = cape_max(conn, icao, r["date"], asof.isoformat())
    return {"icao": icao, "metric": metric_key, "hour": hour, "recs": recs}


def report(res: dict, factor: str, natural_split: float | None = None) -> None:
    recs = [r for r in res["recs"]
            if r.get(factor) is not None and not (
                isinstance(r[factor], float) and np.isnan(r[factor]))]
    if len(recs) < 40:
        print(f"  {factor:11s} n={len(recs)} — too few to test")
        return
    f = np.array([float(r[factor]) for r in recs])
    short = np.array([r["p"] - r["hit"] for r in recs])
    cut = natural_split if natural_split is not None else float(np.median(f))
    hi, lo = short[f > cut], short[f <= cut]
    if len(hi) < 20 or len(lo) < 20:
        print(f"  {factor:11s} degenerate split at {cut:.2f} "
              f"(n_hi={len(hi)} n_lo={len(lo)})")
        return
    d = hi.mean() - lo.mean()
    l, u = boot_diff(hi, lo)
    ship = (d > 0) and (hi.mean() > 0.03) and ((l > 0) == (u > 0))
    kind = "natural" if natural_split is not None else "median"
    print(f"  {factor:11s} {kind} split {cut:.2f}   n={len(recs)}")
    print(f"      flagged n={len(hi):4d}  shortfall {hi.mean():+.1%}   "
          f"clean n={len(lo):4d}  shortfall {lo.mean():+.1%}")
    print(f"      Dshortfall {d:+.1%}  CI [{l:+.1%},{u:+.1%}] "
          f"(Bonferroni x{N_TESTS})   {'** CLEARS **' if ship else 'no'}")


print(__doc__.split("Usage:")[0])
print(f"Bonferroni-corrected CI level: {100*(1-ALPHA):.2f}%  "
      f"({N_TESTS} tests)\n")

# Per-day records are dumped so the "is there still an edge at the later hour?"
# question can be answered WITHOUT refitting 13 folds x 4 arms again. Written
# with a header and truncated ("w"), never appended -- `calib_eval.py` shipped an
# "a"-mode dump on 2026-07-30 that silently accumulated three models' records
# into one file and was invisible until someone counted distinct keys.
CSV_OUT = "data/preflight_high_records-20260731.csv"
with open(CSV_OUT, "w") as fh:
    fh.write("station,metric,hour,target_date,stated,hit,point,"
             "bound_gap,override,guid_churn,stall_h\n")
    for cfg_path, metric_key, hours in ARMS:
        for hour in hours:
            res = collect(cfg_path, metric_key, hour)
            for fac in ("bound_gap", "stall_h", "override", "guid_churn"):
                report(res, fac)
            if res["icao"] == "KLAX":
                report(res, "seam", natural_split=0.5)
            else:
                report(res, "cape")
            for r in res["recs"]:
                def _n(v):
                    return "" if v is None or (isinstance(v, float)
                                               and np.isnan(v)) else f"{v:.4f}"
                fh.write(f"{res['icao']},{res['metric']},{res['hour']},"
                         f"{r['date']},{r['p']:.6f},{r['hit']:.6f},"
                         f"{r['point']:.4f},{_n(r['bound_gap'])},"
                         f"{_n(r['override'])},{_n(r['guid_churn'])},"
                         f"{_n(r['stall_h'])}\n")
print(f"\nper-day records -> {CSV_OUT}")

print("\nA factor that CLEARS has earned a confirmatory pre-registered run.")
print("It has NOT shipped. Nothing here writes a constant.")
