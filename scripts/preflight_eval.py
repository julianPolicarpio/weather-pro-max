"""Is there a DAILY signal that today's read is shakier than the market's average?

⚠️ OFFLINE MEASUREMENT ONLY. Every market is FROZEN. This script fits nothing
that ships and writes nothing. It answers a display question: may the dashboard
tell Julian "today looks unstable, sit this one out"?

THE QUESTION, AND WHY IT IS NOT THE ONE THE OTHER EVAL SCRIPTS ASK.
`klax_low_gap_eval.py` asked "does this column make the MODEL better" and the
answer was no (GAP1 +1.0%, CI [-1.1%,+3.1%]). That is a different question from
"does this column tell me WHEN TO DISBELIEVE the model", and a factor can fail
the first and pass the second: a tree averaging a weak signal across all days
gains little, while a bettor who SKIPS the worst days keeps his money.

THE OUTCOME BEING MEASURED IS **SHORTFALL**, NOT ACCURACY.

    shortfall = p_model - hit

p_model is the stated modal-bucket probability, parity-averaged over both
ladders exactly as `expected_bucket_hit` averages the outcome. Shortfall is the
quantity that costs money: you pay a price anchored near p_model, and shortfall
is how much of it was fiction. A stratum can be LESS accurate and still fine to
trade if the model already says so -- what must not happen is the model staying
confident while being wrong. Accuracy alone would flag the hard days; shortfall
flags the DISHONEST ones, which is what a preflight is for.

CANDIDATE FACTORS -- CHOSEN BY WHAT THE MODEL CANNOT ALREADY SEE.
`INTRADAY_FEATURES` contains cur_tmpf, cur_dwpf, cur_onshore and cur_sky, so
dewpoint depression, sea breeze and cloud are all INSIDE the model. Stratifying
on those would mostly re-derive p_model and dress it up as new information --
theatre. A well-fit model leaves little residual signal in its own inputs. Only
three observable-at-commit quantities are outside it:

    bound_gap    |bound - base_p50|   ratchet and guidance disagree
    stall_h      hours the running extremum has sat unchanged
    pres_tend    3-hour pressure tendency, the competitor's advection input

PRIMARY: bound_gap, on all four markets, Bonferroni-corrected to a 98.75% CI.
It has the strongest prior -- measured on KLAX LOW, the bound's error DOUBLES
(1.22 -> 2.35F) when the minimum has not happened yet, while guidance error
stays flat, and the gap is large exactly in that regime.

SECONDARY (EXPLORATORY, NOT SHIPPABLE ON THIS RUN): stall_h, pres_tend. Reported
uncorrected and labelled, so a lucky one cannot be quietly promoted. If either
looks strong it earns its own pre-registered run, not a shortcut.

THE SPLIT IS THE WITHIN-MARKET MEDIAN, FIXED BEFORE ANY RESULT WAS SEEN. No
threshold search: hunting the cut that maximises the gap is exactly how a
band-aid gets manufactured. The median is a blunt instrument -- a 50% flag rate
is useless as a "sit out today" rule -- but the significance test must run on a
pre-registered cut. Quartile shortfalls are printed as DESCRIPTIVE context, and
only if the median split clears the gate does the top quartile become a
legitimate operating point.

SHIP GATE (per market, primary factor only), all must hold:
    1. shortfall is higher in the flagged stratum
    2. the flagged stratum's shortfall is materially positive (> 3 points)
    3. the 98.75% bootstrap CI on the difference excludes zero

This project is 2-for-8 on confident pre-measurement hypotheses. The expected
outcome is that nothing clears, and that is a publishable answer: it would mean
the honest preflight is the STATIC per-market reliability table already shipped,
plus the hard facts (groups still to file), and no daily grade at all.
"""
import json
import os

import numpy as np

from highcast import board
from highcast.buckets import bucket_probs, expected_bucket_hit, modal_bucket
from highcast.config import load_config
from highcast.db import connect
from highcast.features import build_features, uses_bias_correction
from highcast.gating import MIN_DECISION_HOUR
from highcast.metric import get_metric
from highcast.models.baseline import BaselineBlend
from highcast.models.intraday import IntradayModel, build_checkpoint_frame, features_for

import datetime as dt
from zoneinfo import ZoneInfo

REFIT = 45
RECENT = "2025-01-01"
MIN_TRAIN = 400
N_BOOT = 2000
# Bonferroni for four pre-registered markets on the primary factor: a 98.75%
# two-sided interval, i.e. the [0.625, 99.375] percentiles.
PCTS = [0.625, 99.375]
MIN_FLAGGED_SHORTFALL = 0.03

MARKETS = [("config/klax.yaml", "low"), ("config/klax.yaml", "high"),
           ("config/kmia.yaml", "low"), ("config/kmia.yaml", "high")]

PRIMARY = "bound_gap"
EXPLORATORY = ["stall_h", "pres_tend"]

rng = np.random.default_rng(20260731)


def p_model(pmf: dict) -> float:
    """Stated modal-bucket probability, averaged over both ladder parities.

    Must match `expected_bucket_hit`'s averaging or shortfall compares a number
    from one ladder against an outcome from both. Kalshi's parity is a coin flip
    per day (see `buckets`), so this is what a bettor actually faces.
    """
    return float(np.mean([modal_bucket(pmf, q)[1] for q in (0, 1)]))


def boot_diff(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """CI on mean(a) - mean(b) for two INDEPENDENT strata (unpaired: the split
    partitions days, so there is no day-to-day pairing to preserve)."""
    ia = rng.integers(0, len(a), size=(N_BOOT, len(a)))
    ib = rng.integers(0, len(b), size=(N_BOOT, len(b)))
    d = a[ia].mean(axis=1) - b[ib].mean(axis=1)
    return tuple(np.percentile(d, PCTS))


def pres_tend_at(conn, icao: str, asof: str) -> float | None:
    r = conn.execute(
        """SELECT pres_tend_hpa FROM obs
           WHERE station=? AND source='metar' AND pres_tend_hpa IS NOT NULL
             AND valid_utc <= ? ORDER BY valid_utc DESC LIMIT 1""",
        (icao, asof)).fetchone()
    return None if r is None else r["pres_tend_hpa"]


def collect(cfg_path: str, metric_key: str, cache_dir: str | None = None) -> dict:
    """Walk-forward per-day records at the commit hour. `cache_dir` memoises the
    result so the robustness pass can re-analyse without refitting 13 folds x 4
    markets; the cache key includes every knob that changes a number."""
    if cache_dir:
        # THE COMMIT HOUR IS PART OF THE KEY. It was not until 2026-08-01, when
        # the decision hours moved (KLAX low h4->h6, KMIA low h5->h7) and this
        # key would have served h4 records for an h6 question -- silently, since
        # the filename looked right. The docstring's promise that the key
        # "includes every knob that changes a number" only became true here.
        _h = MIN_DECISION_HOUR.get(load_config(cfg_path).station.icao,
                                   {}).get(metric_key)
        key = (f"{os.path.basename(cfg_path)}.{metric_key}.h{_h}.{RECENT}"
               f".r{REFIT}.m{MIN_TRAIN}.json")
        path = os.path.join(cache_dir, key)
        if os.path.exists(path):
            with open(path) as fh:
                return json.load(fh)
        res = collect(cfg_path, metric_key)
        os.makedirs(cache_dir, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(res, fh)
        return res
    cfg = load_config(cfg_path)
    icao = cfg.station.icao
    hour = MIN_DECISION_HOUR[icao][metric_key]
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
    print(f"\n### {icao} {metric_key.upper()} h{hour}  test n={len(te)}", flush=True)

    recs = []
    for b0 in range(0, len(td), REFIT):
        cut = td[b0]
        tr = fr[(fr["target_date"] < cut) & fr["y"].notna()]
        if len(tr) < MIN_TRAIN:
            continue
        blk = te.iloc[b0:b0 + REFIT]
        if blk.empty:
            continue
        # The shipped feature set for THIS market -- a preflight measured against
        # a model Julian is not running would be measuring the wrong thing. Only
        # KLAX HIGH has extras today (RATE_FEATURES); the rest resolve to [].
        m = IntradayModel(extremum=spec.extremum, residual=resid,
                          extra_features=[c for c in feats
                                          if c not in _SHIPPED_BASE]).fit(tr)
        qs = m.predict_quantiles(blk)
        for (_, row), q in zip(blk.iterrows(), qs):
            pmf = bucket_probs(q)
            recs.append({
                "date": row["target_date"],
                "p": p_model(pmf),
                "hit": expected_bucket_hit(pmf, int(row["y"])),
                "bound_gap": abs(row["cur_bound_gap"])
                if row.get("cur_bound_gap") is not None
                and not np.isnan(row["cur_bound_gap"]) else np.nan,
            })

    # Factors that live outside the frame, read straight from `obs` at the
    # commit instant. Reuses board._stall_hours rather than reimplementing it:
    # a second copy of the stall clock is how the two would drift apart.
    tz = ZoneInfo(cfg.station.tz)
    for r in recs:
        d = dt.date.fromisoformat(r["date"])
        asof = dt.datetime(d.year, d.month, d.day, hour,
                           tzinfo=tz).astimezone(dt.timezone.utc)
        r["stall_h"] = board._stall_hours(conn, cfg, d, spec.extremum, asof)
        r["pres_tend"] = pres_tend_at(conn, icao, asof.isoformat())
        if r["pres_tend"] is not None:
            r["pres_tend"] = abs(r["pres_tend"])
    return {"icao": icao, "metric": metric_key, "hour": hour, "recs": recs}


_SHIPPED_BASE = ["hour", "run_lo", "run_hi", "run_width", "cur_tmpf",
                 "cur_dwpf", "cur_onshore", "cur_sky", "base_p50",
                 "base_spread", "doy_sin", "doy_cos", "year_index",
                 "heat_alert"]


def report(res: dict, factor: str, primary: bool) -> None:
    recs = [r for r in res["recs"]
            if r.get(factor) is not None and not (
                isinstance(r[factor], float) and np.isnan(r[factor]))]
    tag = "PRIMARY" if primary else "exploratory"
    name = f"{res['icao']} {res['metric']} h{res['hour']}"
    if len(recs) < 60:
        print(f"  {factor:10s} [{tag}]  n={len(recs)} -- too few, skipped")
        return
    f = np.array([r[factor] for r in recs], dtype=float)
    short = np.array([r["p"] - r["hit"] for r in recs], dtype=float)
    hit = np.array([r["hit"] for r in recs], dtype=float)
    p = np.array([r["p"] for r in recs], dtype=float)
    med = float(np.median(f))
    hi, lo = f > med, f <= med
    if hi.sum() < 30 or lo.sum() < 30:
        print(f"  {factor:10s} [{tag}]  degenerate split at median {med:.3f}")
        return
    d_lo, d_hi = boot_diff(short[hi], short[lo])
    dif = short[hi].mean() - short[lo].mean()
    print(f"  {factor:10s} [{tag}]  median split at {med:.2f}   n={len(recs)}")
    for label, m in (("flagged (above)", hi), ("clean   (below)", lo)):
        print(f"      {label}  n={m.sum():3d}  stated {p[m].mean():.1%}  "
              f"hit {hit[m].mean():.1%}  shortfall {short[m].mean():+.1%}")
    print(f"      Dshortfall {dif:+.1%}  CI [{d_lo:+.1%},{d_hi:+.1%}]"
          f"{'  (98.75%, Bonferroni x4)' if primary else '  (95%, uncorrected)'}")
    # Descriptive only -- never a threshold to ship on by itself.
    qs = np.quantile(f, [0.25, 0.5, 0.75])
    edges = [-np.inf, *qs, np.inf]
    cells = []
    for i in range(4):
        m = (f > edges[i]) & (f <= edges[i + 1])
        cells.append("--" if m.sum() < 15 else f"{short[m].mean():+.1%}")
    print(f"      shortfall by quartile (low->high f): {'  '.join(cells)}")
    if primary:
        g = (dif > 0, short[hi].mean() > MIN_FLAGGED_SHORTFALL, d_lo > 0)
        print(f"      GATE -> {'SHIP' if all(g) else 'DO NOT SHIP'}")
        if not all(g):
            nm = ["flagged worse", "flagged shortfall>3pt", "CI excludes 0"]
            print(f"      failed: {[n for n, ok in zip(nm, g) if not ok]}")


if __name__ == "__main__":
    print("PREFLIGHT: does any commit-time factor predict the model being "
          "CONFIDENTLY WRONG?")
    print(f"outcome = shortfall (stated p - realised hit); test >= {RECENT}")
    for cfg_path, metric_key in MARKETS:
        res = collect(cfg_path, metric_key)
        if not res["recs"]:
            print("  no folds")
            continue
        report(res, PRIMARY, primary=True)
        for f in EXPLORATORY:
            report(res, f, primary=False)
