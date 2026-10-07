"""Intraday guidance capture — forward accumulation for a hypothesis that cannot
be tested on history.

WHY IT EXISTS. At the HIGH's h11 commit the guidance in use is 11.1 hours old:
measured for target 2026-07-30, every NWP channel including HRRR (a 3km model
built for same-day forecasting that refreshes HOURLY) was last fetched at 23:55
the previous night. `forecasts` does contain same-day runs, but every one arrived
at 06:55Z the NEXT morning -- after its target day ended -- so using them to ask
"would fresher guidance have helped" reads the answer sheet.

The load-bearing tests here are the two that protect the frozen markets
(`test_capture_never_writes_to_forecasts`) and the one that protects the future
experiment's honesty (`test_each_capture_is_a_separate_row_keyed_by_fetch_time`).
"""
import datetime as dt

import yaml

from highcast import backfill
from highcast import db as hdb
from highcast.config import load_config


def _cfg(tmp_path, src="config/klax.yaml"):
    raw = yaml.safe_load(open(src).read())
    raw["db_path"] = str(tmp_path / "t.db")
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(raw))
    return load_config(str(p))


def _row(station, target, model, fetched, temp=80.0, metric="high"):
    return {"station": station, "target_date": target, "model": model,
            "metric": metric, "fetched_at": fetched, "run_date": target,
            "max_temp_f": temp}


def test_each_capture_is_a_separate_row_keyed_by_fetch_time(tmp_path):
    """The whole point: the table must record WHAT WAS KNOWABLE AT EACH HOUR, not
    one row per model per day. If a later capture overwrote an earlier one the
    freshness question would be unanswerable."""
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    hdb.upsert(conn, "forecast_intraday", [
        _row("KLAX", "2026-07-31", "hrrr", "2026-07-31T16:00:00+00:00", 78.0),
        _row("KLAX", "2026-07-31", "hrrr", "2026-07-31T18:00:00+00:00", 81.0),
    ])
    rows = conn.execute(
        "SELECT fetched_at, max_temp_f FROM forecast_intraday ORDER BY fetched_at"
    ).fetchall()
    assert len(rows) == 2, "a later capture overwrote an earlier one"
    assert [r["max_temp_f"] for r in rows] == [78.0, 81.0]


def test_capture_never_writes_to_forecasts(tmp_path):
    """FREEZE CONTRACT. The LOW's feature cutoff is 20:00 local the PREVIOUS day,
    so a midday row landing in `forecasts` would fall before the next target's LOW
    cutoff and silently move `base_p50` for the frozen markets."""
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    before = conn.execute("SELECT count(*) FROM forecasts").fetchone()[0]
    hdb.upsert(conn, "forecast_intraday",
               [_row("KLAX", "2026-07-31", "hrrr", "2026-07-31T18:00:00+00:00")])
    after = conn.execute("SELECT count(*) FROM forecasts").fetchone()[0]
    assert before == after == 0


def test_nothing_reads_the_capture_table_yet():
    """It is data collection only. If a feature or model starts reading it, that
    is a model change and must go through a pre-registered gate first."""
    import pathlib
    src = pathlib.Path("src/highcast")
    readers = []
    for f in src.rglob("*.py"):
        if f.name in ("db.py", "backfill.py", "forecasts.py"):
            continue  # schema, the writer, and the fetcher itself
        # ⚠️ `staleness.py` is exempt, and the exemption is NARROW ON PURPOSE.
        # It is a LIVENESS check: it reads MAX(fetched_at) to answer "did this
        # feed stop delivering", and never touches a value column. That is not a
        # model read and does not need a pre-registered gate. The assertion
        # below enforces the narrowness so the exemption cannot be widened into
        # a feature by accident.
        if f.name == "staleness.py":
            continue
        if "forecast_intraday" in f.read_text():
            readers.append(str(f))
    assert not readers, f"forecast_intraday is being read by {readers}"

    # The staleness exemption above is only safe while it stays a liveness
    # check. If it ever selects a value column from these tables, that IS a
    # model change and this contract must fail again.
    stale_src = (src / "staleness.py").read_text()
    for table in ("forecast_hourly", "forecast_intraday"):
        for line in stale_src.splitlines():
            if table in line and "Feed(" not in line and not line.strip().startswith("#"):
                raise AssertionError(
                    f"staleness.py references {table} outside a Feed() "
                    f"declaration: {line.strip()}")
    assert "temp_f" not in stale_src and "max_temp_f" not in stale_src, \
        "staleness.py must never read a temperature column"



# ── the hourly rate limiter ─────────────────────────────────────────────────

def test_capture_is_due_when_the_table_is_empty(tmp_path):
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    assert backfill._intraday_capture_due(conn, cfg) is True


def test_capture_is_not_due_immediately_after_one(tmp_path):
    """The wake cycle runs every 300s; Open-Meteo refreshes hourly at best."""
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    hdb.upsert(conn, "forecast_intraday",
               [_row(cfg.station.icao, "2026-07-31", "hrrr", now)])
    assert backfill._intraday_capture_due(conn, cfg) is False


def test_capture_is_due_again_after_the_gap(tmp_path):
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    old = (dt.datetime.now(dt.timezone.utc)
           - dt.timedelta(minutes=backfill.INTRADAY_CAPTURE_MIN_GAP_MIN + 5))
    hdb.upsert(conn, "forecast_intraday",
               [_row(cfg.station.icao, "2026-07-31", "hrrr", old.isoformat())])
    assert backfill._intraday_capture_due(conn, cfg) is True


def test_the_limiter_is_per_station(tmp_path):
    """KMIA must not be starved because KLAX just captured."""
    klax = _cfg(tmp_path)
    conn = hdb.connect(klax.db_path)
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    hdb.upsert(conn, "forecast_intraday", [_row("KLAX", "2026-07-31", "hrrr", now)])
    kmia_dir = tmp_path / "m"
    kmia_dir.mkdir()
    kmia = _cfg(kmia_dir, "config/kmia.yaml")
    # point KMIA at the SAME db so the query is genuinely cross-station
    object.__setattr__(kmia, "db_path", klax.db_path)
    assert backfill._intraday_capture_due(conn, klax) is False
    assert backfill._intraday_capture_due(conn, kmia) is True


def test_limiter_fails_open_on_a_bad_timestamp(tmp_path):
    """An unparseable stamp must not silently stop the capture forever. One extra
    HTTP call costs nothing; a permanent hole cannot be backfilled."""
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    hdb.upsert(conn, "forecast_intraday",
               [_row(cfg.station.icao, "2026-07-31", "hrrr", "not-a-timestamp")])
    assert backfill._intraday_capture_due(conn, cfg) is True
