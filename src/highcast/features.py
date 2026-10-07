import math
from bisect import bisect_right
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from highcast.config import AppConfig
from highcast.metric import MetricSpec, get_metric
from highcast.timeutil import climate_day_bounds

# Trailing window (in AVAILABLE climate days) for the guidance bias correction.
# Measured on 2021-2026 guidance vs settled CLI: correcting the ensemble mean by
# its own trailing error reduces the ENSEMBLE's MAE (paired bootstrap, strictly
# past errors only, n~910):
#     LOW   k=3  2.401 -> 1.371   k=14  2.407 -> 1.305   k=30  2.406 -> 1.301
#     HIGH  k=3  2.061 -> 1.868   k=14  2.063 -> 1.771   k=30  2.067 -> 1.764
# All SIG. Flat past ~14, so 14 is the knee: long enough to average out noise,
# short enough to track a drifting offset.
BIAS_WINDOW = 14

# `nbm_nbs` is the NWS NBM STATION BULLETIN from IEM, deliberately distinct from
# `nbm` (Open-Meteo's gridded `ncep_nbm_conus`). Different source, different
# definition, different bias -- and Open-Meteo's archive only starts 2024-10
# while the bulletin archive reaches 2021, so merging them would splice two
# series mid-training and silently change what the column means.
FC_MODELS = ["ecmwf", "gfs", "icon", "hrrr", "nws", "mos_mex",
             "nbm", "nam", "aifs", "nbm_nbs"]
SKY_ORD = {"CLR": 0.0, "SKC": 0.0, "FEW": 1.0, "SCT": 2.0, "BKN": 3.0, "OVC": 4.0}
ONSHORE_DEG = 250.0  # sea-breeze direction at LAX-like coastal stations

# Marine-layer / burn-off predictors from the NBS bulletin (`forecast_nbs`).
# A CURATED FIVE, not everything available -- the table also stores sol_max,
# lcb_min, wsp_mean and dpt_mean, but sol_max/lcb_min are near-duplicates of
# sol_mean/cig_min, and this session measured that extra correlated columns
# actively HURT by diluting LightGBM's per-split feature sampling (adding
# nbm_nbs alone cost 2.7 points of bucket hit). Five physically distinct signals:
#   nbs_sol_mean  how much sun          -> the direct burn-off proxy
#   nbs_sky_mean  how much cloud
#   nbs_cig_min   how low the deck sits (999 = no deck; see the -88 sentinel note)
#   nbs_offshore  onshore sea-breeze vs offshore downslope (subsumes §5b)
#   nbs_tsd_mean  NBM's OWN forecast uncertainty for the day
NBS_AUX_FEATURES = ["nbs_sol_mean", "nbs_sky_mean", "nbs_cig_min",
                    "nbs_offshore", "nbs_tsd_mean"]

# WHY BIAS-CORRECTED GUIDANCE IS A SEPARATE FEATURE AND NOT LEFT TO THE MODEL.
# The raw `fc_*` values carry large station-vs-gridpoint offsets -- KLAX is
# coastal and the marine-layer gradient is steep, so a grid point a few km inland
# runs hot. Measured bias vs settled CLI (day-ahead-legal runs only):
#     HIGH  ecmwf +6.09 (sd 6.00, MAE 6.94!)   mos_mex -0.85   hrrr -1.39
#     LOW   hrrr  -3.30                        ecmwf   -1.34   gfs   +0.23 (sd 4.79)
# A tree CAN learn a CONSTANT offset by shifting its split thresholds. It cannot
# learn a DRIFTING one -- and these drift: the day-to-day autocorrelation of each
# model's error is r=0.86 (gfs low), 0.80 (ecmwf high), 0.51 (ecmwf low). Nor can
# a tree SUBTRACT: given raw `fc_m` and a separate `bias_m` column it cannot form
# `fc_m - bias_m`, because splits threshold one feature at a time. So the
# correction must be applied BEFORE the model sees it, as `fcc_*`.
#
# Raw `fc_*` are kept alongside: they are what the existing `fc_anchor` chain and
# the hybrid model read, and dropping them would be an unmeasured behaviour change.
FEATURES = ([f"fc_{m}" for m in FC_MODELS]
            + [f"fcc_{m}" for m in FC_MODELS]
            + NBS_AUX_FEATURES
            + ["fc_spread", "fc_mean", "fcc_mean", "bias_mean", "bias_abs_mean",
               "bias_n", "doy_sin", "doy_cos", "year_index",
               "yesterday_high", "morn_tmpf", "morn_dwpf", "morn_onshore",
               "morn_sky", "morn_dtd", "morn_rate", "heat_alert"])


BIAS_FEATURES = ([f"fcc_{m}" for m in FC_MODELS]
                 + ["fcc_mean", "bias_mean", "bias_abs_mean", "bias_n"])

# Columns kept in the FRAME (fc_anchor and BaselineBlend still read them) but not
# fed to either day-ahead model. Both are effectively empty:
#   fc_nws  -- 92 HIGH / 27 LOW rows, ~0.4% non-null. This is open item 5.
#   fc_nbm  -- Open-Meteo `ncep_nbm_conus`, all-NaN before 2024-10 and superseded
#              by the `nbm_nbs` bulletin channel, which reaches 2021.
# Dropping them is worth real accuracy, but ONLY together with adding nbm_nbs --
# see `model_features`.
DEAD_FEATURES = ["fc_nws", "fcc_nws", "fc_nbm", "fcc_nbm"]

# NBM station bulletins (IEM), LOW-only. Measured below.
NBS_FEATURES = ["fc_nbm_nbs", "fcc_nbm_nbs"]



def uses_bias_correction(metric_key: str) -> bool:
    """Does this metric consume the trailing guidance-bias correction? LOW only.

    ONE rule for BOTH consumers -- the day-ahead model's feature columns
    (`model_features`) and the nowcast's guidance channel
    (`BaselineBlend(prefer_corrected=...)`). They were measured separately and
    agreed, so they must not be allowed to drift apart:

                      day-ahead dMAE        nowcast dMAE (its best hour)
      LOW    -0.091 [-0.130,-0.055] SIG    -0.002 .. -0.022, small but never harmful
      HIGH   +0.005 [-0.040,+0.049]  ns    +0.050 at hour 9, bucket hit 45.7% -> 43.2%

    The HIGH is a wash-to-slightly-worse on both, so it opts out of both.
    """
    return metric_key == "low"


def model_features(metric_key: str) -> list[str]:
    """The FEATURES columns a metric's day-ahead model should actually consume.

    THE TRAILING-BIAS COLUMNS ARE LOW-ONLY, and that is a measured decision, not
    a guess. Walk-forward, refit 120, test >= 2025-01-01, N=569, paired bootstrap
    (`scripts/bias_feature_eval.py`):

      LOW   MAE 1.291 -> 1.199   dMAE -0.091 [-0.130,-0.055] SIG
            CRPS 0.921 -> 0.865  dCRPS -0.057 [-0.077,-0.037] SIG
            modal-bucket hit 45.9% -> 47.3%
      HIGH  MAE 1.367 -> 1.372   dMAE +0.005 [-0.040,+0.049]  ns
            CRPS 0.996 -> 0.991  dCRPS -0.005 [-0.029,+0.020]  ns
            modal-bucket hit 45.7% -> 44.3%

    The HIGH is a wash, indistinguishable from the +RANDOM noise arm (+0.002),
    so it does not get them: a wash is not an improvement, and 13 inert columns
    dilute LightGBM's feature sampling. Why the metrics differ has a clean
    explanation -- the HIGH's guidance includes `mos_mex` (16,254 rows), and MOS
    is already statistically bias-corrected against the station, so a trailing
    correction is redundant. The LOW has NO MOS channel at all (`mos_field=None`
    in `metric.py`); it runs on raw NWP grid points whose station offsets are both
    large and drifting (hrrr -3.30, error autocorrelation r up to 0.86).

    Re-measure before extending this to the HIGH. If a MOS low channel is ever
    added, re-measure the LOW too -- the reason for the split would change.

    NBM BULLETINS ARE ALSO LOW-ONLY, and the interesting part is that neither
    half of that change pays on its own. Walk-forward, refit 120, N=569, paired
    bootstrap (`scripts/nbs_eval.py`), LOW:

        arm                      MAE     CRPS    bucket hit   dMAE vs baseline
        baseline               1.232    0.846        44.3%    --
        + nbm_nbs              1.198    0.848        46.9%    -0.034 ns
        + nbm_nbs - dead       1.184    0.839        47.6%    -0.048 [-0.084,-0.012] SIG
        - dead only            1.226    0.840        44.8%    -0.006 ns

    Adding the channel alone is not significant, and pruning alone is not
    significant; TOGETHER they are. Adding a 10th correlated guidance column
    dilutes LightGBM's per-split feature sampling -- the same mechanism that
    makes the `+RANDOM` control arm score WORSE, not neutral. A new channel only
    pays once something makes room for it.

    HIGH gets neither: every arm came back ns (47.5% / 46.9% / 46.6% / 47.6%),
    and its 3-hourly-derived NBS max is badly biased anyway (-2.75 F over a
    16-day live check) because a sharp afternoon peak falls between the 3-hourly
    steps. The LOW's overnight trough is broad and flat, so it is captured well
    (-0.75 F, MAE 1.00). It does drop the dead columns, which was ns-or-better.

    THE MARINE-LAYER AUX FEATURES ARE LOW-ONLY TOO -- and not for the reason they
    were built. `scripts/aux_eval.py`, same walk-forward and the same +RANDOM
    control:

        LOW    MAE 1.184 -> 1.148  dMAE -0.036 [-0.058,-0.014] SIG
               CRPS 0.839 -> 0.822   bucket hit 47.6% -> 50.6%
               +RANDOM +0.001 ns (hit 45.2%) -- control clean
        HIGH   MAE 1.362 -> 1.351  dMAE -0.011 [-0.040,+0.019] ns
               bucket hit 47.6% -> 46.0%, indistinguishable from +RANDOM (-0.005)

    They were added to fix the HIGH's marine-layer burn-off problem and they
    fixed the LOW instead. That is physically coherent rather than a fluke: the
    overnight minimum is set by radiational cooling, for which sky cover and wind
    are the textbook predictors, and `nbs_sol_mean`/`nbs_sky_mean` measure
    exactly that. The daily max depends on burn-off TIMING -- a threshold crossed
    at some hour -- which a 09-17 window average smears away. If the HIGH is
    retried, try a timing feature (e.g. the first hour `sol` exceeds a threshold)
    rather than a window mean.
    """
    cols = [c for c in FEATURES if c not in DEAD_FEATURES]
    if uses_bias_correction(metric_key):
        return cols
    return [c for c in cols if c not in BIAS_FEATURES
            and c not in NBS_FEATURES and c not in NBS_AUX_FEATURES]


def _actual_available_utc(cfg: AppConfig, d: date) -> datetime:
    """The instant a climate day's settled CLI value may first be used.

    LEAKAGE GUARD. The climate day ends at 08:00Z (DST-invariant) and the final
    CLI product lands shortly after -- observed `product` stamps in
    `cli_reports.raw` for KLAX: 08:33Z, 08:48Z the following morning. One hour
    past the climate-day end is therefore conservative.

    This is what makes the two metrics' bias windows differ, and it is not
    cosmetic. The HIGH's cutoff is 09:00 local on the target day (16:00Z), so
    yesterday's actual IS available. The LOW's cutoff is 20:00 the PRIOR evening
    (03:00Z on the target day), which is BEFORE 08:00Z, so the most recent usable
    actual is from TWO days back. Deriving it per-date rather than hardcoding a
    lag also keeps it correct across DST, where `_cutoff_utc` shifts an hour but
    `climate_day_bounds` does not (open item 3).
    """
    _, end = climate_day_bounds(d, cfg.station.std_utc_offset)
    return end + timedelta(hours=1)


def _forecast_error_series(conn, cfg: AppConfig, metric: MetricSpec) -> dict:
    """{model: (avail_instants, errors)} -- two parallel lists sorted by the
    instant each day's error became usable.

    The error is (day-ahead-legal forecast - settled actual), so every entry
    could itself have been computed on the day. Sorted-and-paired rather than a
    dict keyed by date because `_bias_features` then bisects instead of scanning:
    a per-row scan is 2030 dates x 9 models x ~2000 entries (~36M) per frame,
    and `build_features` is on the live prediction path.

    Sorting by AVAILABILITY instant (not by date) is what makes the bisect valid
    -- it is the ordering the leakage guard actually cares about, and the two
    orderings coincide only because the availability offset is constant.
    """
    icao, col = cfg.station.icao, metric.target_col
    actual = {r["date"]: r[col] for r in conn.execute(
        f"SELECT date, {col} FROM cli_reports WHERE station=? AND {col} IS NOT NULL",
        (icao,))}
    latest: dict[str, dict[str, float]] = {m: {} for m in metric.fc_models}
    for r in conn.execute(
            """SELECT target_date, model, max_temp_f FROM forecasts
               WHERE station=? AND metric=? AND run_date < target_date
               ORDER BY target_date, model, run_date DESC, fetched_at DESC""",
            (icao, metric.key)):
        m, td = r["model"], r["target_date"]
        if m not in latest or td not in actual or r["max_temp_f"] is None:
            continue
        # ORDER BY puts the latest legal run first; keep only that one.
        latest[m].setdefault(td, float(r["max_temp_f"]) - float(actual[td]))
    out: dict[str, tuple[list, list]] = {}
    for m, series in latest.items():
        items = sorted((_actual_available_utc(cfg, date.fromisoformat(ds)), e)
                       for ds, e in series.items())
        out[m] = ([t for t, _ in items], [e for _, e in items])
    return out


def _bias_features(cfg: AppConfig, metric: MetricSpec, d: date,
                   cutoff: datetime, err: dict, raw: dict) -> dict:
    """Trailing per-model bias, and guidance corrected by it.

    Only days whose settled value was available by `cutoff` contribute, so this
    is computable in real time at the instant the forecast is issued.
    `bias_abs_mean` is an uncertainty signal -- when guidance has been erratic
    lately, trust it less -- and `bias_n` exposes how much history backed the
    correction so the model can discount a thin one.
    """
    out: dict[str, float] = {}
    corrected, biases, n_used = [], [], 0
    for m in metric.fc_models:
        instants, errors = err.get(m, ([], []))
        hi = bisect_right(instants, cutoff)          # strictly available by cutoff
        window = errors[max(0, hi - BIAS_WINDOW):hi]
        fc = raw.get(f"fc_{m}", np.nan)
        if not window or fc is None or (isinstance(fc, float) and np.isnan(fc)):
            out[f"fcc_{m}"] = np.nan
            continue
        b = float(np.mean(window))
        out[f"fcc_{m}"] = float(fc) - b
        corrected.append(out[f"fcc_{m}"])
        biases.append(b)
        n_used = max(n_used, len(window))
    out["fcc_mean"] = float(np.mean(corrected)) if corrected else np.nan
    out["bias_mean"] = float(np.mean(biases)) if biases else np.nan
    out["bias_abs_mean"] = float(np.mean(np.abs(biases))) if biases else np.nan
    out["bias_n"] = float(n_used)
    return out


def _cutoff_utc(cfg: AppConfig, d: date, hour_local: int) -> datetime:
    tz = ZoneInfo(cfg.station.tz)
    return datetime(d.year, d.month, d.day, hour_local, tzinfo=tz).astimezone(
        ZoneInfo("UTC"))


def _forecast_features(conn, cfg, d: date, cutoff: datetime,
                       cutoff_date: date, metric: MetricSpec) -> dict:
    # Leakage guard: a forecast counts only if it was known by `cutoff`. The
    # run_date boundary is the cutoff's LOCAL date (`cutoff_date`), not the
    # target date `d`. For the high they coincide (cutoff is 09:00 on the
    # target day) so this reproduces the old query exactly. For the low the
    # cutoff is the prior evening, so `cutoff_date = d-1`: prior-day runs are
    # bounded by `fetched_at <= cutoff` (a run fetched after the evening cutoff
    # is excluded) and target-day runs never match. Only models in
    # `metric.fc_models` are queried; the rest are left absent so the FEATURES
    # reindex fills their `fc_*` column with NaN (no column is dropped).
    icao = cfg.station.icao
    out: dict[str, float] = {}
    for m in metric.fc_models:
        row = conn.execute(
            """SELECT max_temp_f FROM forecasts
               WHERE station=? AND target_date=? AND model=? AND metric=?
                 AND (run_date < ? OR (run_date = ? AND fetched_at <= ?))
               ORDER BY run_date DESC, fetched_at DESC LIMIT 1""",
            (icao, d.isoformat(), m, metric.key, cutoff_date.isoformat(),
             cutoff_date.isoformat(), cutoff.isoformat())).fetchone()
        out[f"fc_{m}"] = float(row["max_temp_f"]) if row else np.nan
    vals = [v for v in out.values() if not np.isnan(v)]
    out["fc_mean"] = float(np.mean(vals)) if vals else np.nan
    out["fc_spread"] = float(np.max(vals) - np.min(vals)) if len(vals) >= 2 else np.nan
    return out


def metar_point_features(row, prefix: str) -> dict:
    """tmpf_precise-fallback + onshore trig + sky-ordinal features for a
    single METAR-shaped row (tmpf, tmpf_precise, dwpf, sknt, drct, skyc),
    keyed with the given prefix. Shared by the day-ahead morning snapshot
    (`_morning_features`, prefix "morn_") and the intraday as-of snapshot
    (`models.intraday._asof_features`, prefix "cur_") so both funnel through
    identical fallback/derivation logic."""
    if row is None:
        return {f"{prefix}tmpf": np.nan, f"{prefix}dwpf": np.nan,
                f"{prefix}onshore": np.nan, f"{prefix}sky": np.nan}
    tmpf = row["tmpf_precise"] if row["tmpf_precise"] is not None else row["tmpf"]
    onshore = np.nan
    if row["sknt"] is not None and row["drct"] is not None:
        onshore = row["sknt"] * math.cos(math.radians(row["drct"] - ONSHORE_DEG))
    out = {f"{prefix}tmpf": tmpf if tmpf is not None else np.nan,
           f"{prefix}dwpf": row["dwpf"] if row["dwpf"] is not None else np.nan,
           f"{prefix}onshore": onshore,
           f"{prefix}sky": SKY_ORD.get(row["skyc"], np.nan)}
    # ── ENVIRONMENT EXTRAS (2026-07-31) ─────────────────────────────────────
    # Candidate features only. They appear in the FRAME but enter no model
    # unless a caller names them via `IntradayModel(extra_features=[...])`, so
    # adding them here changes no prediction. The four keys above are untouched.
    #
    # WHY THESE, FOR THE KLAX HIGH SPECIFICALLY. `sky` is derived from `skyc`,
    # which stores the LAST sky token, i.e. the HIGHEST layer. LA's marine layer
    # is a LOW deck whose depth governs burn-off timing -- the single mechanism
    # that decides a coastal max -- and the model currently cannot see it at all:
    # `BKN008 OVC200` is indistinguishable from `FEW200`. `sky_lowest_ft` (86% of
    # KLAX reports) and `ceiling_ft` (46%) measure exactly that deck.
    #
    # `pres_tend` is the advection input: nowcast.trade grades wind + pressure
    # tendency + sky jointly and rated LAX "WARM ADV FULL -- AVOID" on the exact
    # day our LOW lost. `vis_sm` collapses in fog, the marine layer's densest form.
    #
    # `_g` because these columns are absent on older rows and on any fixture
    # built before the migration; a missing key must read NaN, not raise.
    def _g(k):
        try:
            v = row[k]
        except (KeyError, IndexError):
            return np.nan
        return np.nan if v is None else float(v)

    out[f"{prefix}ceiling_ft"] = _g("ceiling_ft")
    out[f"{prefix}sky_lowest_ft"] = _g("sky_lowest_ft")
    out[f"{prefix}vis_sm"] = _g("vis_sm")
    out[f"{prefix}pres_tend"] = _g("pres_tend_hpa")
    out[f"{prefix}slp"] = _g("slp_hpa")
    return out


def _morning_features(conn, cfg, d: date, cutoff: datetime,
                      metric: MetricSpec) -> dict:
    # As-of METAR snapshot: latest METAR up to `cutoff`. High (cutoff on the
    # target day) keeps the climate-day start as the lower bound so its output
    # is unchanged; the low's cutoff is the prior evening, so a climate-day
    # start would sit AFTER the cutoff and yield nothing — use a 24h lookback
    # ending at cutoff to capture the evening-before observation instead.
    if metric.cutoff_day_offset == 0:
        start, _ = climate_day_bounds(d, cfg.station.std_utc_offset)
    else:
        start = cutoff - timedelta(hours=24)

    def _latest(upto: datetime):
        return conn.execute(
            """SELECT tmpf, tmpf_precise, dwpf, sknt, drct, skyc,
                      ceiling_ft, sky_lowest_ft, vis_sm, pres_tend_hpa, slp_hpa
               FROM obs
               WHERE station=? AND source='metar' AND valid_utc >= ? AND valid_utc <= ?
               ORDER BY valid_utc DESC LIMIT 1""",
            (cfg.station.icao, start.isoformat(), upto.isoformat())).fetchone()

    feats = metar_point_features(_latest(cutoff), "morn_")
    # dewpoint depression at the cutoff snapshot (drier air => marine layer eroding)
    feats["morn_dtd"] = feats["morn_tmpf"] - feats["morn_dwpf"]
    # temp change over the 3h before the cutoff: for the high this is the
    # 6am->9am warming rate — the marine-layer burn-off signal (backtested win).
    prior = _latest(cutoff - timedelta(hours=3))
    prior_t = np.nan
    if prior is not None:
        prior_t = prior["tmpf_precise"] if prior["tmpf_precise"] is not None \
            else prior["tmpf"]
        prior_t = np.nan if prior_t is None else prior_t
    feats["morn_rate"] = feats["morn_tmpf"] - prior_t
    return feats


def _nbs_features(conn, cfg, d: date, cutoff: datetime, cutoff_date: date,
                  metric: MetricSpec) -> dict:
    """Marine-layer / burn-off predictors from the latest day-ahead-legal NBS run.

    Same leakage rule as `_forecast_features`: a run counts only if it was issued
    before the cutoff DATE, or on it and fetched by the cutoff INSTANT.

    These describe the mechanism that actually governs a coastal max -- whether
    the marine layer burns off -- which pure temperature guidance represents
    poorly. `nbs_offshore` also subsumes the long-open §5b Santa-Ana idea without
    needing a second station's pressure feed.
    """
    row = conn.execute(
        f"""SELECT {', '.join(NBS_AUX_FEATURES)} FROM forecast_nbs
            WHERE station=? AND target_date=? AND metric=?
              AND (run_date < ? OR (run_date = ? AND fetched_at <= ?))
            ORDER BY run_date DESC, fetched_at DESC LIMIT 1""",
        (cfg.station.icao, d.isoformat(), metric.key, cutoff_date.isoformat(),
         cutoff_date.isoformat(), cutoff.isoformat())).fetchone()
    if row is None:
        return {c: np.nan for c in NBS_AUX_FEATURES}
    return {c: (float(row[c]) if row[c] is not None else np.nan)
            for c in NBS_AUX_FEATURES}


def _alert_flag(conn, cfg, d: date, cutoff: datetime) -> float:
    """The `heat_alert` feature: is a HEAT product active by the cutoff?

    ⚠️ THE EVENT FILTER IS LOAD-BEARING AND WAS ADDED 2026-08-02.
    This query had none, because every row in `alerts` was a heat row -- the
    fetcher discarded everything else. On 2026-08-02 the fetcher was widened to
    capture Severe Thunderstorm / Tornado products (a storm cut KMIA 16F in 50
    minutes and destroyed a low the day before). Without this filter, the very
    next storm warning would have set `heat_alert=1` and silently changed a live
    model feature on all four markets -- a model change disguised as a data
    change, with no gate and no way to notice.

    Same shape as the evening/day-ahead kinds sharing one label: once two
    different things live in one table, every READER has to say which it means.
    """
    from highcast.fetchers.alerts import HEAT_EVENTS

    # Leakage guard: the alert must have been issued (onset) by the prediction
    # cutoff — not merely by end of climate day — to count as known.
    start, _ = climate_day_bounds(d, cfg.station.std_utc_offset)
    marks = ",".join("?" * len(HEAT_EVENTS))
    row = conn.execute(
        f"""SELECT 1 FROM alerts WHERE station=? AND onset_utc <= ?
             AND (expires_utc IS NULL OR expires_utc = '' OR expires_utc >= ?)
             AND event IN ({marks})
           LIMIT 1""",
        (cfg.station.icao, cutoff.isoformat(), start.isoformat(),
         *sorted(HEAT_EVENTS))).fetchone()
    return 1.0 if row else 0.0


def _f(v) -> float:
    """float() that treats a missing OR NULL value as NaN.

    A stored None and an absent key mean the same thing to the model -- "no
    value" -- but `float(None)` raises while `float` of a default NaN does not.
    Any lookup against `cli_reports` needs this: the row can exist with a NULL
    column (KMIA has 3 such days), so a `.get(k, nan)` default never fires.
    """
    return float(v) if v is not None else np.nan


def build_features(conn, cfg: AppConfig, metric: MetricSpec, dates: list[str],
                   require_target: bool = True,
                   cutoff_local_hour: int | None = None) -> pd.DataFrame:
    """Metric-general day-ahead feature frame.

    Target `y` and `yesterday_high` are drawn from `metric.target_col`
    (`high_f`/`low_f`); the prediction cutoff is
    `_cutoff_utc(cfg, d + metric.cutoff_day_offset, metric.cutoff_local_hour)`
    (high: 09:00 on the target day; low: 20:00 the prior evening). The FEATURES
    columns and their names are identical for every metric — the low frame
    populates the same-named columns with low-appropriate values, and any
    `fc_*` model not in `metric.fc_models` is filled NaN by the reindex.
    `cutoff_local_hour` overrides `metric.cutoff_local_hour` when supplied.
    """
    icao = cfg.station.icao
    hour = cutoff_local_hour if cutoff_local_hour is not None \
        else metric.cutoff_local_hour
    targets = {r["date"]: r[metric.target_col] for r in conn.execute(
        f"SELECT date, {metric.target_col} FROM cli_reports WHERE station=?",
        (icao,))}
    err_series = _forecast_error_series(conn, cfg, metric)
    rows = []
    for ds in dates:
        d = date.fromisoformat(ds)
        y = targets.get(ds, np.nan)
        if require_target and (y is None or (isinstance(y, float) and np.isnan(y))):
            continue
        cutoff_date = d + timedelta(days=metric.cutoff_day_offset)
        cutoff = _cutoff_utc(cfg, cutoff_date, hour)
        doy = d.timetuple().tm_yday
        rec = {"target_date": ds, "y": float(y) if y is not None else np.nan,
               "doy_sin": math.sin(2 * math.pi * doy / 365.25),
               "doy_cos": math.cos(2 * math.pi * doy / 365.25),
               "year_index": float(d.year - 2000),
               # `.get(k, nan)` is NOT enough: a CLI row can EXIST with a NULL
               # value, so the key is present and the default never fires --
               # `float(None)` then raises. KLAX has no NULL targets and never
               # hit this; KMIA has 3 NULL lows, which is how it surfaced.
               # Same guard as `y` on the line above.
               "yesterday_high": _f(targets.get(
                   (d - timedelta(days=1)).isoformat())),
               "heat_alert": _alert_flag(conn, cfg, d, cutoff)}
        rec.update(_forecast_features(conn, cfg, d, cutoff, cutoff_date, metric))
        rec.update(_bias_features(cfg, metric, d, cutoff, err_series, rec))
        rec.update(_nbs_features(conn, cfg, d, cutoff, cutoff_date, metric))
        rec.update(_morning_features(conn, cfg, d, cutoff, metric))
        rows.append(rec)
    out = pd.DataFrame(rows, columns=["target_date", "y", *FEATURES])
    # Level anchor for the residual(MOS) target — a FIXED coalesce chain, never
    # fc_mean alone: model coverage is very uneven over history (mos_mex 2021-01,
    # gfs 2021-03, ecmwf 2024-02), so a composite anchor would silently change
    # composition over time. Not a model input; consumed by HybridDayAheadModel.
    out["fc_anchor"] = (out["fc_mos_mex"] if "fc_mos_mex" in out else np.nan)
    if "fc_gfs" in out:
        out["fc_anchor"] = out["fc_anchor"].fillna(out["fc_gfs"])
    out["fc_anchor"] = out["fc_anchor"].fillna(out["fc_mean"])
    return out


def build_dayahead(conn, cfg: AppConfig, dates: list[str],
                   cutoff_local_hour: int = 9,
                   require_target: bool = True) -> pd.DataFrame:
    """Thin HIGH wrapper over `build_features`. Behavior is byte-identical to
    the pre-Task-4 builder: the high metric has offset 0 / hour 9, so the
    cutoff date is the target date and the query/snapshot bounds are unchanged.
    A non-default `cutoff_local_hour` (none of the current callers pass one) is
    forwarded as an override."""
    return build_features(conn, cfg, get_metric(cfg, "high"), dates,
                          require_target=require_target,
                          cutoff_local_hour=cutoff_local_hour)
