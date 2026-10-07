from datetime import date

from highcast.config import load_config
from highcast.fetchers import forecasts
from highcast.metric import get_metric

CFG = load_config()

OM_MULTI = {
    "daily": {
        "time": ["2026-07-17", "2026-07-18"],
        "temperature_2m_max_ecmwf_ifs025": [84.2, 86.0],
        "temperature_2m_max_gfs_seamless": [85.1, None],
    }
}

def test_parse_openmeteo_multi_model():
    rows = forecasts.parse_openmeteo(OM_MULTI, "KLAX", "2026-07-17", "t0")
    assert {(r["model"], r["target_date"], r["max_temp_f"]) for r in rows} == {
        ("ecmwf", "2026-07-17", 84.2), ("ecmwf", "2026-07-18", 86.0),
        ("gfs", "2026-07-17", 85.1),
    }
    assert all(r["run_date"] == "2026-07-17" for r in rows)


def test_parse_openmeteo_low_uses_min_var_and_tags_metric():
    from highcast.fetchers import forecasts
    payload = {"daily": {"time": ["2026-07-22"],
               "temperature_2m_min_ecmwf_ifs025": [64.0],
               "temperature_2m_min_gfs_seamless": [61.5]}}
    rows = forecasts.parse_openmeteo(payload, "KLAX", "2026-07-22", "t0",
                                     get_metric(CFG, "low"))
    assert {(r["model"], r["max_temp_f"]) for r in rows} == {("ecmwf",64.0),("gfs",61.5)}
    assert all(r["metric"] == "low" for r in rows)

def test_high_path_unchanged_and_tagged_high():
    from highcast.fetchers import forecasts
    payload = {"daily": {"time": ["2026-07-22"],
               "temperature_2m_max_gfs_seamless": [84.0]}}
    rows = forecasts.parse_openmeteo(payload, "KLAX", "2026-07-22", "t0",
                                     get_metric(CFG, "high"))
    assert rows[0]["metric"] == "high" and rows[0]["max_temp_f"] == 84.0

OM_PREV = {
    "hourly": {
        "time": ["2026-07-17T00:00", "2026-07-17T13:00", "2026-07-17T14:00",
                 "2026-07-18T13:00"],
        "temperature_2m_previous_day1_ecmwf_ifs025": [64.0, 83.5, 84.9, 88.0],
    }
}

def test_parse_openmeteo_lead1_takes_daily_max_per_local_day():
    rows = forecasts.parse_openmeteo_lead1(OM_PREV, "KLAX", "t0")
    by_date = {(r["model"], r["target_date"]): r for r in rows}
    r = by_date[("ecmwf", "2026-07-17")]
    assert r["max_temp_f"] == 84.9
    assert r["run_date"] == "2026-07-16"          # lead-1: issued the day before

NWS_GRID = {
    "properties": {"maxTemperature": {"uom": "wmoUnit:degC", "values": [
        {"validTime": "2026-07-17T14:00:00+00:00/PT19H", "value": 29.0},
        {"validTime": "2026-07-18T14:00:00+00:00/PT19H", "value": 30.5},
    ]}}
}

def test_parse_nws_grid_converts_c_to_f():
    rows = forecasts.parse_nws(NWS_GRID, "KLAX", "2026-07-17", "t0",
                               tz="America/Los_Angeles")
    assert rows[0]["model"] == "nws"
    assert rows[0]["target_date"] == "2026-07-17"
    assert abs(rows[0]["max_temp_f"] - 84.2) < 0.1


# Shape observed live from https://mesonet.agron.iastate.edu/api/1/mos.json
# for station=KLAX&model=MEX: the "n_x" (NWS N/X) column alternates between
# the overnight MIN (ftime valid ~12Z, ~5am local) and the following
# afternoon's MAX (ftime valid ~00Z, ~5pm local) - it is not a daily max
# keyed by the UTC calendar date in `ftime[:10]`.
MOS_PAYLOAD = {
    "data": [
        {"runtime": "2026-07-16 12:00", "ftime": "2026-07-17 12:00", "n_x": 70},
        {"runtime": "2026-07-16 12:00", "ftime": "2026-07-18 00:00", "n_x": 82},
        {"runtime": "2026-07-16 12:00", "ftime": "2026-07-18 12:00", "n_x": 69},
    ]
}

def test_parse_mos_keeps_afternoon_max_for_the_high():
    """The HIGH's mapping must be BYTE-IDENTICAL to the pre-2026-07-26 behaviour
    (which bucketed by local date). Climate-day bucketing agrees with it for every
    afternoon-valid row, and 16,254 stored HIGH rows depend on that."""
    rows = forecasts.parse_mos(MOS_PAYLOAD, "KLAX", "t0", "mos_mex",
                               tz="America/Los_Angeles", extremum="max",
                               std_utc_offset=-8)
    # ftime 2026-07-18 00:00Z = 2026-07-17 17:00 local -> afternoon high (82)
    # ftime 2026-07-17 12:00Z = 2026-07-17 05:00 local -> overnight low, dropped
    # ftime 2026-07-18 12:00Z = 2026-07-18 05:00 local -> overnight low, dropped
    assert len(rows) == 1
    r = rows[0]
    assert r["model"] == "mos_mex"
    assert r["target_date"] == "2026-07-17"
    assert r["max_temp_f"] == 82.0
    assert r["run_date"] == "2026-07-16"           # runtime 12Z = 5am local, same day


def test_parse_mos_keeps_overnight_min_for_the_low():
    """THE FIX (2026-07-26). These morning rows used to be hard-dropped with the
    comment "overnight low, not the daily max we want", and `fetch_mos` claimed
    "IEM MOS has no min-temp field for the low metric". The min is right here in
    the same n_x column. The LOW had ZERO MOS rows as a result, leaving it the
    only metric with no station-calibrated guidance at all."""
    rows = forecasts.parse_mos(MOS_PAYLOAD, "KLAX", "t0", "mos_mex",
                               tz="America/Los_Angeles", extremum="min",
                               std_utc_offset=-8)
    # The two morning-valid rows are now KEPT and the afternoon one dropped —
    # exactly inverted from the high.
    assert len(rows) == 2
    by_target = {r["target_date"]: r["max_temp_f"] for r in rows}
    # ftime 2026-07-17 12:00Z -> climate day 2026-07-17 (12:00Z-8h = 04:00Z 07-17)
    # ftime 2026-07-18 12:00Z -> climate day 2026-07-18
    assert by_target == {"2026-07-17": 70.0, "2026-07-18": 69.0}


def test_min_and_max_never_collide_on_the_same_row():
    """The two extrema partition the rows: every row goes to exactly one metric.
    If they overlapped, one would overwrite the other under the shared
    (station, target_date, model, run_date, metric) primary key."""
    kw = dict(tz="America/Los_Angeles", std_utc_offset=-8)
    hi = forecasts.parse_mos(MOS_PAYLOAD, "KLAX", "t0", "mos_mex",
                             extremum="max", **kw)
    lo = forecasts.parse_mos(MOS_PAYLOAD, "KLAX", "t0", "mos_mex",
                             extremum="min", **kw)
    assert len(hi) + len(lo) == 3          # all three payload rows accounted for
    hi_keys = {(r["target_date"], r["run_date"], r["max_temp_f"]) for r in hi}
    lo_keys = {(r["target_date"], r["run_date"], r["max_temp_f"]) for r in lo}
    assert not (hi_keys & lo_keys)


# Real CSV snippet from https://mesonet.agron.iastate.edu/cgi-bin/request/
# mos.py?station=KLAX&model=MEX&sts=2024-07-01T00:00Z&ets=2024-07-02T00:00Z&
# format=csv (fetched live 2026-07-17). Column set varies slightly by era
# but runtime/ftime/n_x/station are always present.
MOS_CSV = """runtime,ftime,model,n_x,tmp,dpt,cld,wsp,p12,q12,t12_1,t12_2,station,q24,p24,t24,t12
2024-07-01 00:00:00,2024-07-02 00:00:00,MEX,71,67,59,PC,12,2,0.0,1,0,KLAX,,,,1/0
2024-07-01 00:00:00,2024-07-02 12:00:00,MEX,60,61,57,OV,12,0,0.0,0,0,KLAX,,,4.0,0/0
2024-07-01 12:00:00,2024-07-02 00:00:00,MEX,70,67,60,PC,12,2,0.0,0,1,KLAX,0.0,3.0,,0/1
2024-07-01 12:00:00,2024-07-02 12:00:00,MEX,M,62,60,PC,7,1,0.0,1,0,KLAX,,,1.0,1/0
"""


def test_parse_mos_csv_keeps_afternoon_max_and_skips_missing_nx():
    rows = forecasts.parse_mos_csv(MOS_CSV, "KLAX", "t0", "mos_mex",
                                   tz="America/Los_Angeles", extremum="max",
                                   std_utc_offset=-8)
    # row1: runtime 2024-07-01 00:00Z = 2024-06-30 17:00 local -> run_date 06-30
    #       ftime 2024-07-02 00:00Z = 2024-07-01 17:00 local -> afternoon high (71),
    #       target_date 07-01
    # row2: same runtime; ftime 2024-07-02 12:00Z = 2024-07-02 05:00 local ->
    #       overnight low, dropped
    # row3: runtime 2024-07-01 12:00Z = 2024-07-01 05:00 local -> run_date 07-01
    #       ftime 2024-07-02 00:00Z = 2024-07-01 17:00 local -> afternoon high (70),
    #       target_date 07-01 (different run_date than row1, so both survive - the
    #       forecasts PK is station/target_date/model/run_date, i.e. distinct leads)
    # row4: n_x == "M" (missing) -> dropped regardless of ftime hour
    assert len(rows) == 2
    by_run = {r["run_date"]: r for r in rows}
    assert by_run["2024-06-30"]["target_date"] == "2024-07-01"
    assert by_run["2024-06-30"]["max_temp_f"] == 71.0
    assert by_run["2024-06-30"]["model"] == "mos_mex"
    assert by_run["2024-07-01"]["target_date"] == "2024-07-01"
    assert by_run["2024-07-01"]["max_temp_f"] == 70.0


import pytest
from highcast import db as hdb
from highcast.config import load_config

@pytest.mark.live
def test_live_openmeteo_current():
    rows = forecasts.fetch_openmeteo(load_config())
    models = {r["model"] for r in rows}
    assert "ecmwf" in models and "gfs" in models, models
    # If 'hrrr' is missing here, remove it from MODELS and note it in README.

@pytest.mark.live
def test_live_openmeteo_low():
    rows = forecasts.fetch_openmeteo(CFG, get_metric(CFG, "low"))
    models = {r["model"] for r in rows}
    assert "ecmwf" in models and "gfs" in models, models
    assert all(45.0 <= r["max_temp_f"] <= 80.0 for r in rows)
    assert all(r["metric"] == "low" for r in rows)

@pytest.mark.live
def test_live_openmeteo_lead1_history():
    rows = forecasts.fetch_openmeteo_lead1(load_config(), date(2024, 7, 1),
                                           date(2024, 7, 31))
    assert len(rows) >= 25

@pytest.mark.live
def test_live_nws_gridpoint(tmp_path):
    conn = hdb.connect(tmp_path / "t.db")
    rows = forecasts.fetch_nws(load_config(), conn)
    assert any(60 <= r["max_temp_f"] <= 120 for r in rows)

@pytest.mark.live
def test_live_mos():
    rows = forecasts.fetch_mos(load_config())
    assert rows, "MOS endpoint/params need adapting - print raw payload and fix parse_mos"
    assert any(50 <= r["max_temp_f"] <= 120 for r in rows)

@pytest.mark.live
def test_live_mos_archive_small_window():
    # Bulk MOS archive service (request/mos.py?format=csv) - see
    # forecasts.fetch_mos_archive docstring/comment. Keep this window a
    # single week to stay a fast, polite smoke test rather than a real
    # backfill (that's Task 8's job, done separately via a one-off script).
    rows = forecasts.fetch_mos_archive(load_config(), date(2024, 7, 1), date(2024, 7, 7))
    assert rows, "MOS archive endpoint/params need adapting"
    assert all(r["model"] == "mos_mex" for r in rows)
    assert any(40 <= r["max_temp_f"] <= 120 for r in rows)
    # run_date < target_date for at least some rows (multi-day-ahead leads)
    assert any(r["run_date"] < r["target_date"] for r in rows)


def test_parse_mos_csv_keeps_overnight_min_for_the_low():
    """CSV path mirror of the JSON one — both share `_mos_row`, and the archive
    (CSV) path is the one that backfilled 5 years of LOW minima."""
    rows = forecasts.parse_mos_csv(MOS_CSV, "KLAX", "t0", "mos_mex",
                                   tz="America/Los_Angeles", extremum="min",
                                   std_utc_offset=-8)
    # row2: ftime 2024-07-02 12:00Z -> 05:00 local, min 60, run_date 06-30
    # row4: n_x == "M" -> still dropped, missing is missing for either extremum
    assert len(rows) == 1
    assert rows[0]["target_date"] == "2024-07-02"
    assert rows[0]["max_temp_f"] == 60.0


# Real NBS snippet (IEM request/mos.py?station=KLAX&model=NBS, fetched live
# 2026-07-26). NBS carries 3-hourly `tmp` and has NO `n_x` column at all --
# verified live: 759 rows over an 8-day window, `n_x` empty on every one.
NBS_CSV = """runtime,ftime,model,tmp,dpt,wdr,wsp,sky,station,txn,tsd
2026-07-24 00:00:00,2026-07-24 06:00:00,NBS,71,65,200,6,13,KLAX,,1
2026-07-24 00:00:00,2026-07-24 12:00:00,NBS,70,65,70,3,33,KLAX,70.0,2
2026-07-24 00:00:00,2026-07-24 18:00:00,NBS,80,66,210,7,30,KLAX,,2
2026-07-24 00:00:00,2026-07-24 21:00:00,NBS,79,65,250,9,38,KLAX,,1
2026-07-24 00:00:00,2026-07-25 06:00:00,NBS,68,64,190,5,20,KLAX,,1
2026-07-24 00:00:00,2026-07-25 18:00:00,NBS,77,64,220,8,25,KLAX,,2
2026-07-24 00:00:00,2026-07-25 21:00:00,NBS,M,64,220,8,25,KLAX,,2
"""


def test_parse_nbs_aggregates_3hourly_tmp_into_a_daily_extremum():
    """NBS has no daily max/min column, so the extremum is derived by
    aggregating 3-hourly `tmp` over the CLIMATE day (08:00Z->08:00Z)."""
    hi = forecasts.parse_nbs_csv(NBS_CSV, "KLAX", "t0", "nbm_nbs",
                                 tz="America/Los_Angeles", extremum="max",
                                 std_utc_offset=-8)
    lo = forecasts.parse_nbs_csv(NBS_CSV, "KLAX", "t0", "nbm_nbs",
                                 tz="America/Los_Angeles", extremum="min",
                                 std_utc_offset=-8)
    # Climate day 2026-07-24 spans 07-24 08:00Z -> 07-25 08:00Z, so it collects
    # the 12/18/21Z rows of the 24th AND the 06Z row of the 25th (68) -- that
    # 06:00Z row is BEFORE 08:00Z so it belongs to the previous climate day.
    hi_by = {r["target_date"]: r["max_temp_f"] for r in hi}
    lo_by = {r["target_date"]: r["max_temp_f"] for r in lo}
    assert hi_by["2026-07-24"] == 80.0        # max(70, 80, 79)
    assert lo_by["2026-07-24"] == 68.0        # min(70, 80, 79, 68@07-25 06Z)
    # The 07-24 06:00Z row (71) precedes the climate day start -> previous day.
    assert hi_by["2026-07-23"] == 71.0
    assert all(r["model"] == "nbm_nbs" for r in hi + lo)
    assert all(r["run_date"] == "2026-07-23" for r in hi + lo)  # 00Z = 5pm local


def test_parse_nbs_skips_missing_tmp():
    """`M` is missing, not a temperature. The 07-25 21Z row must not poison the
    07-25 aggregate."""
    hi = forecasts.parse_nbs_csv(NBS_CSV, "KLAX", "t0", "nbm_nbs",
                                 tz="America/Los_Angeles", extremum="max",
                                 std_utc_offset=-8)
    by = {r["target_date"]: r["max_temp_f"] for r in hi}
    assert by["2026-07-25"] == 77.0       # the M row is dropped, not read as 0


NBS_AUX_CSV = """runtime,ftime,model,tmp,dpt,wdr,wsp,sky,cig,lcb,sol,tsd,station
2026-07-24 00:00:00,2026-07-24 18:00:00,NBS,80,66,210,7,30,-88,-88,76,2,KLAX
2026-07-24 00:00:00,2026-07-24 21:00:00,NBS,79,65,90,9,10,25,20,94,1,KLAX
2026-07-24 00:00:00,2026-07-24 06:00:00,NBS,71,65,200,6,13,8,5,0,1,KLAX
"""


def test_nbs_aux_unlimited_ceiling_sentinel_is_not_treated_as_a_low_deck():
    """-88 is NBM's 'unlimited ceiling' (NO cloud deck). Taken literally,
    min(cig) over a window containing one clear period returns -88, so the
    CLEAREST sky scores as the LOWEST marine-layer deck — the signal exactly
    inverted, on precisely the hot burn-off days the feature exists to find."""
    rows = forecasts.parse_nbs_aux_csv(NBS_AUX_CSV, "KLAX", "t0",
                                       tz="America/Los_Angeles", extremum="max",
                                       std_utc_offset=-8)
    assert len(rows) == 1
    r = rows[0]
    # Window rows are the 18Z (11:00 local, cig=-88 -> clear) and 21Z (14:00
    # local, cig=25). The real deck at 25 must win over the clear sentinel.
    assert r["nbs_cig_min"] == 25.0
    assert r["nbs_lcb_min"] == 20.0
    assert r["nbs_cig_min"] > 0     # never negative


def test_nbs_aux_window_excludes_hours_outside_the_forming_period():
    """The 06Z row is 23:00 local the previous day — outside the max-forming
    window (09-17 local). Cloud cover at 11pm says nothing about a 2pm max."""
    rows = forecasts.parse_nbs_aux_csv(NBS_AUX_CSV, "KLAX", "t0",
                                       tz="America/Los_Angeles", extremum="max",
                                       std_utc_offset=-8)
    r = rows[0]
    assert r["nbs_n"] == 2                       # only the two in-window rows
    assert r["nbs_sol_mean"] == (76.0 + 94.0) / 2
    assert r["nbs_sky_mean"] == (30.0 + 10.0) / 2


def test_nbs_aux_offshore_fraction():
    """wdr 210 is onshore (sea breeze sector); 90 is offshore/downslope. Half the
    window offshore -> 0.5. This is the §5b Santa-Ana regime signal, derived
    without a second station's pressure feed."""
    rows = forecasts.parse_nbs_aux_csv(NBS_AUX_CSV, "KLAX", "t0",
                                       tz="America/Los_Angeles", extremum="max",
                                       std_utc_offset=-8)
    assert rows[0]["nbs_offshore"] == 0.5


def test_nbs_aux_low_uses_the_overnight_window():
    """The LOW's trough forms 00-08 local, so it must pick up the 06Z row
    (23:00 local previous day is NOT in 00-08 of that climate day) and drop the
    afternoon ones."""
    rows = forecasts.parse_nbs_aux_csv(NBS_AUX_CSV, "KLAX", "t0",
                                       tz="America/Los_Angeles", extremum="min",
                                       std_utc_offset=-8)
    # No 00-08-local rows exist for the 07-24 climate day in this fixture, and
    # the afternoon rows must NOT leak in.
    for r in rows:
        assert r["nbs_sol_mean"] != (76.0 + 94.0) / 2
