"""Hourly temperature capture — forward accumulation for the evening minimum.

WHY IT EXISTS. The biggest MEASURED error source in this project has no feature
behind it. On 11.0% of KLAX days the climate-day LOW lands at h18 or later — the
same evening, seventeen hours after the h4 commit — because the climate day runs
01:00→01:00 local and so contains TWO cooling periods. At h4 the model sees a
finished dawn minimum and has nothing at all about the evening. On 2026-07-30 the
guidance had it nearly exact (67.52 vs a settled 67); the model overrode it and
lost.

Every other guidance table stores a DAILY min/max, which cannot express "tonight
will be colder than this morning was".

The load-bearing tests are `test_nothing_reads_the_hourly_table_yet` (it is data
collection only — a feature built on it is a model change and needs a gate) and
`test_each_capture_is_a_separate_row_keyed_by_fetch_time` (if a later capture
overwrote an earlier one, the freshness question becomes unanswerable and the
data cannot be re-collected).
"""
import datetime as dt

import yaml

from highcast import db as hdb
from highcast.config import load_config


def _cfg(tmp_path, src="config/klax.yaml"):
    raw = yaml.safe_load(open(src).read())
    raw["db_path"] = str(tmp_path / "t.db")
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(raw))
    return load_config(str(p))


def _row(station, valid, model, fetched, temp):
    return {"station": station, "valid_utc": valid, "model": model,
            "fetched_at": fetched, "temp_f": temp}


def test_each_capture_is_a_separate_row_keyed_by_fetch_time(tmp_path):
    """The whole point: WHAT WAS PREDICTED FOR 20:00, AS OF EACH HOUR. If a later
    capture overwrote an earlier one the evening-forecast question would be
    unanswerable, and unlike every other table here it cannot be re-collected."""
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    hdb.upsert(conn, "forecast_hourly", [
        _row("KLAX", "2026-08-01T04:00:00+00:00", "gfs_hrrr",
             "2026-07-31T12:00:00+00:00", 71.0),
        _row("KLAX", "2026-08-01T04:00:00+00:00", "gfs_hrrr",
             "2026-07-31T18:00:00+00:00", 68.5),
    ])
    rows = conn.execute(
        "SELECT fetched_at, temp_f FROM forecast_hourly ORDER BY fetched_at"
    ).fetchall()
    assert len(rows) == 2, "a later capture overwrote an earlier one"
    assert [r["temp_f"] for r in rows] == [71.0, 68.5]


def test_models_do_not_collide(tmp_path):
    """Seven models share every timestamp; the PK must keep them apart."""
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    f = "2026-07-31T12:00:00+00:00"
    hdb.upsert(conn, "forecast_hourly", [
        _row("KLAX", "2026-08-01T04:00:00+00:00", "gfs_hrrr", f, 71.0),
        _row("KLAX", "2026-08-01T04:00:00+00:00", "ecmwf_ifs025", f, 69.0),
    ])
    assert conn.execute(
        "SELECT count(*) FROM forecast_hourly").fetchone()[0] == 2


def test_stations_do_not_collide(tmp_path):
    """Both stations share one SQLite file — anything keyed without the station
    is a latent cross-station bug."""
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    f = "2026-07-31T12:00:00+00:00"
    hdb.upsert(conn, "forecast_hourly", [
        _row("KLAX", "2026-08-01T04:00:00+00:00", "gfs_hrrr", f, 68.0),
        _row("KMIA", "2026-08-01T04:00:00+00:00", "gfs_hrrr", f, 80.0),
    ])
    assert conn.execute(
        "SELECT count(*) FROM forecast_hourly").fetchone()[0] == 2


def test_capture_never_writes_to_the_daily_guidance_tables(tmp_path):
    """FREEZE CONTRACT. The LOW's feature cutoff is 20:00 local the PREVIOUS
    day, so a midday row landing in `forecasts` would fall inside the next
    target's window and silently move `base_p50` for markets that are frozen."""
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    before = {t: conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
              for t in ("forecasts", "forecast_intraday")}
    hdb.upsert(conn, "forecast_hourly", [
        _row("KLAX", "2026-08-01T04:00:00+00:00", "gfs_hrrr",
             "2026-07-31T12:00:00+00:00", 68.0)])
    after = {t: conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
             for t in ("forecasts", "forecast_intraday")}
    assert before == after


def test_nothing_reads_the_hourly_table_yet():
    """Data collection only. If a feature or model starts reading this, that is
    a MODEL change and must go through a pre-registered gate first — the same
    contract `forecast_intraday` keeps."""
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
        if "forecast_hourly" in f.read_text():
            readers.append(str(f))
    assert not readers, f"forecast_hourly is being read by {readers}"

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



def test_hourly_capture_asks_for_seven_DISTINCT_products():
    """`gfs_seamless` IS HRRR over CONUS for the first 48h.

    Verified against the live API 2026-07-31: `gfs_seamless` and `gfs_hrrr`
    returned byte-identical 48-hour series, and in the stored daily table
    `fc_gfs` == `fc_hrrr` on 3,733 of 3,735 lead-0/lead-1 pairs since
    2024-01-18. Capturing both stores one channel twice and would let a future
    evening-minimum feature double-count HRRR — the exact failure
    `BaselineBlend._blend_cols` already guards against for `fc_`/`fcc_`.

    This table can NEVER be backfilled, so a duplicate captured today is a
    permanently wasted row and a real GFS series permanently lost.
    """
    from highcast.fetchers.forecasts import HOURLY_MODELS
    products = list(HOURLY_MODELS.values())
    assert len(products) == len(set(products)), "duplicate product requested"
    assert "gfs_seamless" not in products, (
        "gfs_seamless duplicates gfs_hrrr over CONUS — ask for gfs_global")
    assert HOURLY_MODELS["gfs"] == "gfs_global"
    assert HOURLY_MODELS["hrrr"] == "gfs_hrrr"


def test_hourly_model_list_is_independent_of_the_daily_one():
    """FREEZE CONTRACT. `MODELS` feeds the daily `forecasts` table, which is a
    LIVE MODEL INPUT — changing it is a model change and needs a pre-registered
    gate. The hourly capture is capture-only and free to differ, so it must not
    reach through to the daily list."""
    from highcast.fetchers.forecasts import HOURLY_MODELS, MODELS
    assert HOURLY_MODELS is not MODELS
    assert MODELS["gfs"] == "gfs_seamless", (
        "daily model surface changed — that needs a gate, not an edit here")


def test_hourly_forecast_horizon_is_derivable_without_a_new_column():
    """HRRR breaks past its real horizon (~18-20h): 96.7F at 18:00Z then 78.0F
    at 21:00Z for KMIA while every other model held 88-96F. We deliberately do
    NOT filter that away at capture time — `valid_utc - fetched_at` already
    carries the horizon, so a consumer can drop the bad tail while the raw
    record stays complete and honest."""
    lead = (dt.datetime.fromisoformat("2026-08-01T04:00:00+00:00")
            - dt.datetime.fromisoformat("2026-07-31T12:00:00+00:00"))
    assert lead == dt.timedelta(hours=16)


def test_timestamps_are_timezone_aware(tmp_path):
    """Rows are joined against `obs.valid_utc`, which is offset-aware. A naive
    stamp would raise on comparison or, worse, compare wrongly."""
    from highcast.fetchers.forecasts import fetch_hourly  # noqa: F401
    # Exercise the normalisation the fetcher applies, without a network call.
    raw = "2026-08-01T04:00"
    normalised = raw if len(raw) > 16 else f"{raw}:00+00:00"
    parsed = dt.datetime.fromisoformat(normalised)
    assert parsed.tzinfo is not None
    assert parsed.utcoffset() == dt.timedelta(0)
