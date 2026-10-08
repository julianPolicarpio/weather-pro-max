import json
from datetime import date

from citycast.sources.cli_reports import chunks, cli_rows
from citycast.stations import Station

ST = Station("KNYC", "NYC", "NY_ASOS", "OKX", 40.78, -73.97, 40.0, "America/New_York", -5,
             "KXHIGHNY", "KXLOWTNYC")
PAYLOAD = {"results": [
    # issued the next morning: final
    {"valid": "2024-01-05", "high": 40, "low": 30, "product": "202401060714-KOKX-CDUS41-CLINYC"},
    # issued during the day itself: preliminary
    {"valid": "2024-01-06", "high": 38, "low": 29, "product": "202401062130-KOKX-CDUS41-CLINYC"},
    {"valid": "2024-01-07", "high": "M", "low": 20, "product": "202401080700-KOKX-CDUS41-CLINYC"},
]}


def test_cli_rows_marks_final_and_issue_time():
    rows = {r["date"]: r for r in cli_rows(PAYLOAD, ST, "2026-10-07T00:00:00+00:00")}
    assert set(rows) == {"2024-01-05", "2024-01-06"}       # missing high dropped
    assert rows["2024-01-05"]["final"] == 1 and rows["2024-01-06"]["final"] == 0
    assert rows["2024-01-05"]["issued_utc"] == "2024-01-06T07:14:00+00:00"
    assert rows["2024-01-05"]["high_f"] == 40 and rows["2024-01-05"]["low_f"] == 30
    assert json.loads(rows["2024-01-05"]["raw"])["valid"] == "2024-01-05"


def test_chunks_per_station_year_and_current_year_is_current():
    seen = []

    class FakePaced:
        def get_json(self, url, params):
            seen.append(params)
            return PAYLOAD
    cs = chunks(FakePaced(), {"KNYC": ST}, date(2024, 1, 1), date(2026, 10, 7),
                lambda: "2026-10-07T00:00:00+00:00")
    assert [c.key for c in cs] == ["KNYC:2024", "KNYC:2025", "KNYC:2026"]
    assert [c.current for c in cs] == [False, False, True]
    table, rows = cs[0].run()[0]
    assert table == "cli_daily" and len(rows) == 2
    assert seen == [{"station": "KNYC", "year": 2024}]


def test_cli_year_current_until_february_of_next_year():
    args = (object(), {"KNYC": ST}, date(2025, 1, 1), date(2026, 1, 1), lambda: "x")
    cur = {c.key: c.current for c in chunks(*args, complete_before=date(2025, 12, 30))}
    assert cur["KNYC:2025"] is True
    cur = {c.key: c.current for c in chunks(*args, complete_before=date(2026, 1, 31))}
    assert cur["KNYC:2025"] is True
    cur = {c.key: c.current for c in chunks(*args, complete_before=date(2026, 2, 1))}
    assert cur["KNYC:2025"] is False
