from datetime import date

from citycast.sources.obs import chunks, obs_rows
from citycast.stations import Station

ST = Station("KNYC", "NYC", "NY_ASOS", "OKX", 40.78, -73.97, 40.0, "America/New_York", -5,
             "KXHIGHNY", "KXLOWTNYC")
F = "2026-10-07T00:00:00+00:00"
TEXT = """station,valid,metar
NYC,2024-01-05 17:51,KNYC 051751Z 27008KT 10SM CLR 06/M06 A3012 RMK AO2 SLP199 T00561061 10067 20044
NYC,2024-01-05 18:51,KNYC 051851Z 28009KT 10SM CLR 05/M06 A3013 RMK AO2 SLP203 T00501061
"""


def test_obs_rows_parse_metar_including_six_hour_groups():
    rows = obs_rows(TEXT, ST, F)
    assert len(rows) == 2 and rows[0]["station"] == "KNYC" and rows[0]["source"] == "metar"
    assert rows[0]["valid_utc"] == "2024-01-05T17:51:00+00:00"
    assert rows[0]["six_max_c"] == 6.7 and rows[0]["six_min_c"] == 4.4


def test_chunks_are_months_with_current_month_current():
    asked = []

    class FakePaced:
        def get_text(self, url, params):
            asked.append(params)
            return TEXT
    cs = chunks(FakePaced(), {"KNYC": ST}, date(2026, 8, 1), date(2026, 10, 7), lambda: F)
    assert [c.key for c in cs] == ["KNYC:2026-08", "KNYC:2026-09", "KNYC:2026-10"]
    assert [c.current for c in cs] == [False, False, True]
    table, rows = cs[0].run()[0]
    assert table == "obs" and len(rows) == 2
    assert asked[0]["station"] == "NYC" and asked[0]["month1"] == 8 and asked[0]["month2"] == 9


def test_complete_before_decides_current_not_today():
    class P:
        pass
    args = (P(), {"KNYC": ST}, date(2026, 9, 1), date(2026, 10, 1), lambda: F)
    cur = {c.key: c.current for c in chunks(*args, complete_before=date(2026, 9, 29))}
    assert cur["KNYC:2026-09"] is True
    cur = {c.key: c.current for c in chunks(*args, complete_before=date(2026, 10, 2))}
    assert cur["KNYC:2026-09"] is False
