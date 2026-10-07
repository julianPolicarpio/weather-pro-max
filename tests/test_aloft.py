from highcast import db as hdb
from highcast.fetchers import forecasts
from highcast.fetchers.forecasts import parse_openmeteo_aloft


def test_aloft_table_and_store(tmp_path):
    conn = hdb.connect(tmp_path / "t.db")
    cols = {r[1] for r in conn.execute("PRAGMA table_info(forecast_aloft)")}
    assert {"station", "target_date", "run_date", "metric", "t850", "t2m",
            "inversion", "fetched_at"} <= cols
    n = forecasts.store_aloft(conn, [{
        "station": "KLAX", "target_date": "2026-07-24", "run_date": "2026-07-24",
        "metric": "high", "t850": 72.0, "t2m": 66.0, "inversion": 6.0,
        "fetched_at": "2026-07-24T16:00:00+00:00"}])
    assert n == 1
    assert conn.execute("SELECT inversion FROM forecast_aloft").fetchone()[0] == 6.0


def _payload():
    # hours across 2 UTC days; only the 17:00 hour (9am PST) carries values,
    # so the parse must pick exactly that hour.
    times, t850, t2 = [], [], []
    for d in ("2026-07-24", "2026-07-25"):
        for h in range(24):
            times.append(f"{d}T{h:02d}:00")
            if h == 17:
                t850.append(72.0 if d == "2026-07-24" else 60.0)
                t2.append(66.0 if d == "2026-07-24" else 64.0)
            else:
                t850.append(None); t2.append(None)
    return {"hourly": {"time": times, "temperature_850hPa": t850,
                       "temperature_2m": t2}}


def test_parse_aloft_picks_9am_lst_and_computes_inversion():
    rows = parse_openmeteo_aloft(_payload(), "KLAX", "2026-07-24",
                                 "2026-07-24T16:00:00+00:00", -8, "high")
    by = {r["target_date"]: r for r in rows}
    assert by["2026-07-24"]["t850"] == 72.0 and by["2026-07-24"]["t2m"] == 66.0
    assert by["2026-07-24"]["inversion"] == 6.0     # 72-66: warm cap => holds
    assert by["2026-07-25"]["inversion"] == -4.0    # 60-64: unstable => clears
