import argparse
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import yaml

from highcast import backtest as bt
from highcast import db as hdb
from highcast.cli import cmd_verify_hour
from highcast.config import load_config
from highcast.metric import get_metric

CFG = load_config()
ICAO = CFG.station.icao
TZ = ZoneInfo(CFG.station.tz)
UTC = ZoneInfo("UTC")

# January dates: PST == std_utc_offset (-8) year-round, so climate-day /
# asof-hour arithmetic can't trip on a DST mismatch.
START = date(2024, 1, 1)


def _dates(n: int) -> list[str]:
    return [(START + timedelta(days=i)).isoformat() for i in range(n)]


def _seed(conn, n: int, hour: int, obs_days: int | None = None, seed: int = 3):
    """Seed `n` consecutive days of cli_reports + a single 'ecmwf' forecast
    (so BaselineBlend can fit) for every day, and a METAR observation at
    `hour` station-local for only the first `obs_days` days (default: all).
    Fewer obs days than cli_reports days is how a real gap between the
    intraday and day-ahead checkpoint frames would arise."""
    if obs_days is None:
        obs_days = n
    dates = _dates(n)
    rng = np.random.default_rng(seed)
    y = np.round(75 + 8 * np.sin(np.arange(n) * 2 * np.pi / 365)
                + rng.normal(0, 1.5, n))

    hdb.upsert(conn, "cli_reports", [
        {"station": ICAO, "date": d, "high_f": float(v), "low_f": float(v - 15),
         "raw": "{}", "fetched_at": "x"}
        for d, v in zip(dates, y)
    ])
    hdb.upsert(conn, "forecasts", [
        # run_date = d-1 unconditionally satisfies the leakage guard
        # (run_date < cutoff_date), regardless of fetched_at.
        {"station": ICAO, "target_date": d, "model": "ecmwf",
         "run_date": (date.fromisoformat(d) - timedelta(days=1)).isoformat(),
         "max_temp_f": float(v), "fetched_at": "2000-01-01T00:00:00+00:00"}
        for d, v in zip(dates, y)
    ])
    obs_rows = []
    for d, v in list(zip(dates, y))[:obs_days]:
        day = date.fromisoformat(d)
        asof = datetime(day.year, day.month, day.day, hour, tzinfo=TZ)
        valid = (asof - timedelta(minutes=1)).astimezone(UTC).isoformat()
        obs_rows.append({
            "station": ICAO, "valid_utc": valid, "source": "metar",
            "tmpf": float(v - 2.0), "tmpf_precise": float(v - 2.0),
            "dwpf": float(v - 15.0), "sknt": 5.0, "drct": 250.0,
            "skyc": "CLR", "six_max_c": None, "raw": "", "fetched_at": "x",
        })
    hdb.upsert(conn, "obs", obs_rows)
    return dates


# --- 1. build_hour_frame: single-hour slice --------------------------------

def test_build_hour_frame_returns_only_requested_hour(tmp_path):
    conn = hdb.connect(tmp_path / "t.db")
    dates = _seed(conn, 40, hour=6)
    spec = get_metric(CFG, "high")
    frame = bt.build_hour_frame(conn, CFG, spec, dates, hour=6)
    assert len(frame) > 0
    assert (frame["hour"] == 6.0).all()


# --- 2. compare_at_hour: matched days are mandatory ------------------------

def test_compare_at_hour_restricts_to_matched_days(tmp_path):
    conn = hdb.connect(tmp_path / "t.db")
    n = 60
    dates = _seed(conn, n, hour=6, obs_days=45)   # intraday only sees days 0..44
    spec = get_metric(CFG, "high")
    arms = [
        {"label": "dayahead", "kind": "dayahead"},
        {"label": "intraday", "kind": "intraday", "residual": False},
    ]
    result = bt.compare_at_hour(conn, CFG, spec, dates, hour=6, arms=arms,
                               min_train=30, refit_every=100, n_boot=50, seed=0)

    # dayahead walks forward over all 60 days (predicts rows 30..59 -> 30 recs);
    # intraday only has 45 checkpoint rows (predicts rows 30..44 -> 15 recs).
    # Intersection is exactly the intraday side: 15 days.
    assert result["arms"]["intraday"]["dropped"] == 0
    assert result["arms"]["dayahead"]["dropped"] == 15
    n_matched = result["n_matched"]
    assert n_matched == 15
    for label in ("dayahead", "intraday"):
        assert result["arms"][label]["n"] == n_matched


# --- 3. paired bootstrap is deterministic given a seed ----------------------

def test_bootstrap_is_deterministic_given_seed(tmp_path):
    conn = hdb.connect(tmp_path / "t.db")
    n = 60
    dates = _seed(conn, n, hour=6, obs_days=45)
    spec = get_metric(CFG, "high")
    arms = [
        {"label": "dayahead", "kind": "dayahead"},
        {"label": "intraday", "kind": "intraday", "residual": False},
    ]
    r1 = bt.compare_at_hour(conn, CFG, spec, dates, hour=6, arms=arms,
                            min_train=30, refit_every=100, n_boot=50, seed=7)
    r2 = bt.compare_at_hour(conn, CFG, spec, dates, hour=6, arms=arms,
                            min_train=30, refit_every=100, n_boot=50, seed=7)
    assert r1["vs_baseline"]["intraday"]["dmae_ci"] == r2["vs_baseline"]["intraday"]["dmae_ci"]
    assert r1["vs_baseline"]["intraday"]["dcrps_ci"] == r2["vs_baseline"]["intraday"]["dcrps_ci"]


# --- 4. vs_baseline sign convention: arm[0] vs itself is exactly zero -------

def test_vs_baseline_arm_zero_is_exactly_zero(tmp_path):
    conn = hdb.connect(tmp_path / "t.db")
    n = 45
    dates = _seed(conn, n, hour=6, obs_days=n)
    spec = get_metric(CFG, "high")
    arms = [
        {"label": "dayahead", "kind": "dayahead"},
        {"label": "intraday", "kind": "intraday", "residual": False},
    ]
    result = bt.compare_at_hour(conn, CFG, spec, dates, hour=6, arms=arms,
                               min_train=30, refit_every=100, n_boot=50, seed=0)
    baseline = result["vs_baseline"]["dayahead"]
    assert baseline["dmae"] == 0.0
    assert baseline["dmae_ci"] == [0.0, 0.0]
    assert baseline["dcrps"] == 0.0
    assert baseline["dcrps_ci"] == [0.0, 0.0]


# --- 5. verify-hour CLI runs end-to-end -------------------------------------

def _write_config(tmp_path, db_path):
    raw = yaml.safe_load(open("config/klax.yaml").read())
    raw["db_path"] = str(db_path)
    cfg_path = tmp_path / "klax-test.yaml"
    cfg_path.write_text(yaml.safe_dump(raw))
    return str(cfg_path)


def test_verify_hour_cli_runs_end_to_end(tmp_path):
    db_path = tmp_path / "t.db"
    conn = hdb.connect(db_path)
    _seed(conn, 60, hour=6, obs_days=45)
    conn.close()
    cfg_path = _write_config(tmp_path, db_path)

    cmd_verify_hour(argparse.Namespace(
        config=cfg_path, metric="high", hour=6, arms="intraday,dayahead",
        refit_every=100, min_train=30, json=False))
