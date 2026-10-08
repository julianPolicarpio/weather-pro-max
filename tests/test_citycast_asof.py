from contextlib import closing
from datetime import date, datetime, timezone

from citycast import db
from citycast.asof import asof_points, lead_hours, same_day
from citycast.stations import Station

NYC = Station("KNYC", "NYC", "NY_ASOS", "OKX", 40.78, -73.97, 40.0, "America/New_York", -5,
              "KXHIGHNY", "KXLOWTNYC")
LAX = Station("KLAX", "LAX", "CA_ASOS", "LOX", 33.94, -118.39, 32.0, "America/Los_Angeles", -8,
              "KXHIGHLAX", "KXLOWTLAX")


def U(s):
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def test_grid_winter_new_york():
    pts = asof_points(NYC, date(2024, 1, 5))
    assert len(pts) == 14
    assert pts[0] == U("2024-01-04T11:00:00")        # 06:00 EST on D-1
    assert pts[-1] == U("2024-01-06T02:00:00")       # 21:00 EST on D
    assert same_day(NYC, date(2024, 1, 5), U("2024-01-05T05:00:00"))   # 00:00 EST = day start
    assert lead_hours(NYC, date(2024, 1, 5), U("2024-01-05T05:00:00")) == 24.0


def test_summer_dst_midnight_is_still_pre_day():
    d = date(2024, 7, 5)
    pts = asof_points(LAX, d)
    assert len(pts) == 14
    midnight_pdt = U("2024-07-05T07:00:00")           # 00:00 PDT; climate day starts 08:00Z
    assert midnight_pdt in pts and not same_day(LAX, d, midnight_pdt)


def test_new_tables_exist(tmp_path):
    with closing(db.connect(tmp_path / "h.db")) as c:
        names = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"wf_predictions", "model_choice", "live_predictions"} <= names
