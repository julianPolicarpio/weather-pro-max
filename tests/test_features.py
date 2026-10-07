import pandas as pd
from datetime import datetime, timezone
from highcast import db as hdb, features
from highcast.config import load_config
from highcast.metric import get_metric

CFG = load_config()
ICAO = CFG.station.icao

def _seed(conn):
    hdb.upsert(conn, "cli_reports", [
        {"station": ICAO, "date": "2025-07-15", "high_f": 82, "low_f": 63, "raw": "{}", "fetched_at": "x"},
        {"station": ICAO, "date": "2025-07-16", "high_f": 84, "low_f": 64, "raw": "{}", "fetched_at": "x"},
    ])
    hdb.upsert(conn, "forecasts", [
        # lead-1 row: usable
        {"station": ICAO, "target_date": "2025-07-16", "model": "ecmwf",
         "run_date": "2025-07-15", "max_temp_f": 83.0, "fetched_at": "x"},
        # same-day row fetched AFTER the 09:00 local cutoff (16:00Z = 9am PDT; use 17:00Z): NOT usable
        {"station": ICAO, "target_date": "2025-07-16", "model": "gfs",
         "run_date": "2025-07-16", "max_temp_f": 99.0,
         "fetched_at": "2025-07-16T17:30:00+00:00"},
    ])
    hdb.upsert(conn, "obs", [
        # 08:00 PDT = 15:00Z on 2025-07-16, inside climate day, before cutoff
        {"station": ICAO, "valid_utc": "2025-07-16T15:00:00+00:00", "source": "metar",
         "tmpf": 68.0, "tmpf_precise": 68.2, "dwpf": 60.1, "sknt": 6.0, "drct": 250.0,
         "skyc": "OVC", "six_max_c": None, "raw": "", "fetched_at": "x"},
    ])

def test_dayahead_features_respect_cutoff(tmp_path):
    conn = hdb.connect(tmp_path / "t.db")
    _seed(conn)
    df = features.build_dayahead(conn, CFG, ["2025-07-16"])
    assert len(df) == 1
    row = df.iloc[0]
    assert row["fc_ecmwf"] == 83.0
    assert pd.isna(row["fc_gfs"])            # fetched after cutoff -> excluded
    assert row["y"] == 84
    assert row["yesterday_high"] == 82
    assert abs(row["morn_dwpf"] - 60.1) < 0.01
    assert row["morn_sky"] == 4.0            # OVC ordinal
    assert row["heat_alert"] == 0.0

def test_alert_flag_set_when_overlapping(tmp_path):
    conn = hdb.connect(tmp_path / "t.db")
    _seed(conn)
    hdb.upsert(conn, "alerts", [
        {"station": ICAO, "event": "Heat Advisory",
         "onset_utc": "2025-07-16T10:00:00+00:00",
         "expires_utc": "2025-07-17T04:00:00+00:00", "fetched_at": "x"},
    ])
    df = features.build_dayahead(conn, CFG, ["2025-07-16"])
    assert df.iloc[0]["heat_alert"] == 1.0

def test_alert_after_cutoff_not_leaked(tmp_path):
    # Alert issued at 20:00Z on 2025-07-16 — AFTER the 09:00 PDT (16:00Z)
    # prediction cutoff, though within the climate day. Must NOT be visible.
    conn = hdb.connect(tmp_path / "t.db")
    _seed(conn)
    hdb.upsert(conn, "alerts", [
        {"station": ICAO, "event": "Heat Advisory",
         "onset_utc": "2025-07-16T20:00:00+00:00",
         "expires_utc": "2025-07-17T04:00:00+00:00", "fetched_at": "x"},
    ])
    df = features.build_dayahead(conn, CFG, ["2025-07-16"])
    assert df.iloc[0]["heat_alert"] == 0.0


# --- LOW metric -----------------------------------------------------------
# For target D=2025-07-16 the low cutoff is 20:00 local on D-1 (2025-07-15).
# In PDT (UTC-7) that is 2025-07-16T03:00:00+00:00, and the cutoff's local
# date is 2025-07-15 — so a forecast RUN on 2025-07-15 is bounded by
# fetched_at <= cutoff, while a run on the target day is excluded entirely.

def _seed_low(conn):
    hdb.upsert(conn, "forecasts", [
        # prior-day (run_date = cutoff date 07-15) low forecast fetched BEFORE
        # the 03:00Z cutoff -> usable.
        {"station": ICAO, "target_date": "2025-07-16", "model": "ecmwf",
         "run_date": "2025-07-15", "max_temp_f": 60.0,
         "fetched_at": "2025-07-16T02:00:00+00:00", "metric": "low"},
        # prior-day low forecast fetched AFTER the low cutoff -> must be excluded.
        {"station": ICAO, "target_date": "2025-07-16", "model": "gfs",
         "run_date": "2025-07-15", "max_temp_f": 55.0,
         "fetched_at": "2025-07-16T05:00:00+00:00", "metric": "low"},
    ])
    hdb.upsert(conn, "obs", [
        # 19:00 PDT on 2025-07-15 = 2025-07-16T02:00Z: the evening-before METAR,
        # inside the low's 24h lookback ending at the 03:00Z cutoff.
        {"station": ICAO, "valid_utc": "2025-07-16T02:00:00+00:00", "source": "metar",
         "tmpf": 59.0, "tmpf_precise": 59.2, "dwpf": 55.5, "sknt": 5.0, "drct": 250.0,
         "skyc": "SCT", "six_max_c": None, "raw": "", "fetched_at": "x"},
    ])

def test_low_features_respect_cutoff(tmp_path):
    conn = hdb.connect(tmp_path / "t.db")
    _seed(conn)          # cli_reports (low_f 63/64) + a metric='high' forecast set
    _seed_low(conn)
    metric = get_metric(CFG, "low")
    df = features.build_features(conn, CFG, metric, ["2025-07-16"])
    assert len(df) == 1
    # No feature renamed / dropped: same columns as the high frame, plus
    # `fc_anchor` — a level anchor for the residual target, deliberately NOT a
    # model input (see HybridDayAheadModel / ACCURACY_IMPROVEMENTS.md §7).
    assert list(df.columns) == ["target_date", "y", *features.FEATURES, "fc_anchor"]
    assert "fc_anchor" not in features.FEATURES
    row = df.iloc[0]
    assert row["y"] == 64                      # low_f(D)
    assert row["yesterday_high"] == 63         # prior low_f(D-1), not prior high
    assert row["fc_ecmwf"] == 60.0             # fetched before cutoff -> kept
    assert pd.isna(row["fc_gfs"])              # fetched after cutoff -> excluded
    assert pd.isna(row["fc_mos_mex"])          # not a low model -> NaN, not dropped
    assert row["fc_mean"] == 60.0              # mean over only the kept forecast
    assert abs(row["morn_dwpf"] - 55.5) < 0.01  # evening-before METAR snapshot
    assert abs(row["morn_tmpf"] - 59.2) < 0.01  # tmpf_precise fallback
    assert row["morn_sky"] == 2.0              # SCT ordinal

def test_low_forecast_after_cutoff_excluded(tmp_path):
    # Isolate the leakage guard: the ONLY seeded low forecast is a prior-day
    # run fetched after the low cutoff. Under a target-date-bounded query it
    # would leak in; it must be excluded and its column left NaN.
    conn = hdb.connect(tmp_path / "t.db")
    _seed(conn)
    hdb.upsert(conn, "forecasts", [
        {"station": ICAO, "target_date": "2025-07-16", "model": "ecmwf",
         "run_date": "2025-07-15", "max_temp_f": 55.0,
         "fetched_at": "2025-07-16T05:00:00+00:00", "metric": "low"},
    ])
    df = features.build_features(conn, CFG, get_metric(CFG, "low"), ["2025-07-16"])
    assert pd.isna(df.iloc[0]["fc_ecmwf"])
