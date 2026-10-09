from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

from citycast.sources.mos import chunks, parse_mos_runs, recent_chunks
from citycast.stations import Station

ST = Station("KNYC", "NYC", "NY_ASOS", "OKX", 40.78, -73.97, 40.0, "America/New_York", -5,
             "KXHIGHNY", "KXLOWTNYC")
F = "2026-10-07T00:00:00+00:00"

MEX = """runtime,ftime,model,n_x,tmp
2024-01-04 12:00:00,2024-01-05 00:00:00,MEX,45,40
2024-01-04 12:00:00,2024-01-05 12:00:00,MEX,30,31
2024-01-04 12:00:00,2024-01-06 00:00:00,MEX,48,44
2024-01-04 12:00:00,2024-01-06 12:00:00,MEX,,33
"""

NBS = """runtime,ftime,model,tmp,dpt
2024-01-05 01:00:00,2024-01-05 06:00:00,NBS,30,20
2024-01-05 01:00:00,2024-01-05 09:00:00,NBS,28,20
2024-01-05 01:00:00,2024-01-05 18:00:00,NBS,41,22
2024-01-05 01:00:00,2024-01-06 03:00:00,NBS,35,22
2024-01-05 01:00:00,2024-01-06 06:00:00,NBS,M,22
"""


def test_mex_uses_n_x_afternoon_for_high_morning_for_low_by_climate_day():
    rows = {(r["target_date"], r["metric"]): r for r in parse_mos_runs(MEX, ST, "MEX", F)}
    # 00Z Jan 5 = 19:00 local Jan 4 (afternoon) -> high of climate day Jan 4
    assert rows[("2024-01-04", "high")]["value_f"] == 45
    # 12Z Jan 5 = 07:00 local (morning) -> low of Jan 5; 00Z Jan 6 -> high of Jan 5
    assert rows[("2024-01-05", "low")]["value_f"] == 30
    assert rows[("2024-01-05", "high")]["value_f"] == 48
    assert ("2024-01-06", "low") not in rows                      # blank n_x skipped
    r = rows[("2024-01-04", "high")]
    assert r["runtime_utc"] == "2024-01-04T12:00:00+00:00" and r["n_points"] == 1
    assert r["model"] == "MEX" and r["station"] == "KNYC"


def test_nbs_aggregates_tmp_within_the_climate_day():
    rows = {(r["target_date"], r["metric"]): r for r in parse_mos_runs(NBS, ST, "NBS", F)}
    # climate day Jan 5 = 05:00Z Jan 5 .. 05:00Z Jan 6 -> 06Z, 09Z, 18Z, 03Z rows
    assert rows[("2024-01-05", "high")]["value_f"] == 41
    assert rows[("2024-01-05", "low")]["value_f"] == 28
    assert rows[("2024-01-05", "high")]["n_points"] == 4


def test_chunks_half_years_per_model_and_last_is_current():
    asked = []

    class FakePaced:
        def get_text(self, url, params):
            asked.append(params)
            return MEX
    cs = chunks(FakePaced(), {"KNYC": ST}, date(2025, 1, 1), date(2025, 9, 1), lambda: F,
                models=("MEX",))
    assert [c.key for c in cs] == ["KNYC:MEX:2025-01-01", "KNYC:MEX:2025-07-01"]
    assert [c.current for c in cs] == [False, True]
    cs[0].run()
    assert asked == [{"station": "KNYC", "model": "MEX", "sts": "2025-01-01T00:00Z",
                      "ets": "2025-07-01T00:00Z", "format": "csv"}]


def test_complete_before_decides_current_not_today():
    args = (object(), {"KNYC": ST}, date(2026, 1, 1), date(2026, 7, 1), lambda: F)
    kw = {"models": ("MEX",)}
    cur = {c.key: c.current for c in chunks(*args, complete_before=date(2026, 6, 29), **kw)}
    assert cur["KNYC:MEX:2026-01-01"] is True
    cur = {c.key: c.current for c in chunks(*args, complete_before=date(2026, 7, 2), **kw)}
    assert cur["KNYC:MEX:2026-01-01"] is False


def test_recent_chunks_fetch_the_last_days_of_runs_and_are_never_logged():
    asked = []

    class FakePaced:
        def get_text(self, url, params):
            asked.append(params)
            return MEX
    now = datetime(2026, 10, 9, 22, 37, tzinfo=ZoneInfo("Asia/Manila"))          # 14:37Z
    cs = recent_chunks(FakePaced(), {"KNYC": ST}, now, lambda: F, models=("MEX", "NBS"))
    assert [c.key for c in cs] == ["KNYC:MEX:recent", "KNYC:NBS:recent"]
    assert all(c.current for c in cs) and {c.source for c in cs} == {"mos"}
    table, rows = cs[0].run()[0]
    assert table == "mos_runs" and rows
    assert asked == [{"station": "KNYC", "model": "MEX", "sts": "2026-10-06T14:37Z",
                      "ets": "2026-10-10T14:37Z", "format": "csv"}]
    assert len(recent_chunks(FakePaced(), {"KNYC": ST}, now, lambda: F)) == 4          # every model by default
    recent_chunks(FakePaced(), {"KNYC": ST}, now.astimezone(timezone.utc), lambda: F, days=5,
                  models=("GFS",))[0].run()
    assert asked[-1]["sts"] == "2026-10-04T14:37Z"
