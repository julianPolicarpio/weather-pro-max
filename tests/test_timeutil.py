from datetime import date, datetime, timezone
from highcast.timeutil import climate_day_bounds, round_half_up, c_to_f

def test_climate_day_is_local_standard_time_even_in_summer():
    # 2026-07-17 at KLAX (std offset -8): climate day is 08:00 UTC to 08:00 UTC next day
    start, end = climate_day_bounds(date(2026, 7, 17), -8)
    assert start == datetime(2026, 7, 17, 8, 0, tzinfo=timezone.utc)
    assert end == datetime(2026, 7, 18, 8, 0, tzinfo=timezone.utc)

def test_round_half_up_is_not_bankers():
    assert round_half_up(0.5) == 1      # Python round() would give 0
    assert round_half_up(1.5) == 2
    assert round_half_up(-0.5) == 0
    assert round_half_up(71.4) == 71

def test_c_to_f():
    assert c_to_f(22.0) == 71.6
