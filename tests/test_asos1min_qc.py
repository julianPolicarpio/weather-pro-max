"""QC for the 1-minute ASOS archive.

Measured 2026-08-12 over 1,540 KLAX and 2,015 KMIA days: the raw 1-minute
extreme matches the settled CLI value only 29.5-31.7% of the time on HIGHs and
runs ~+1F hot. Despike + 5-minute mean takes that to 94.6-98.5%. These tests pin
the behaviour each half of that pipeline is responsible for, so a later
"simplification" cannot quietly delete one.
"""
import numpy as np
import pytest

from highcast import asos1min


def _minutes(start: str, temps, step: int = 1):
    """(valid_utc list, tmpf list) at `step`-minute spacing from `start`."""
    from datetime import datetime, timedelta, timezone
    t0 = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    return ([(t0 + timedelta(minutes=i * step)).isoformat() for i in range(len(temps))],
            list(temps))


# ── the minute grid ─────────────────────────────────────────────────────────

def test_gaps_become_nan_not_interpolated():
    """A two-hour hole must stay a hole. Averaging across it would invent a
    reading — coverage is only ~1,250 of 1,440 minutes on a covered day."""
    v, t = _minutes("2026-01-01T00:00", [70.0, 71.0])
    v2, t2 = _minutes("2026-01-01T02:00", [80.0])
    grid, vals = asos1min._to_minute_grid(v + v2, t + t2)
    assert grid.size == 121                     # 00:00 .. 02:00 inclusive
    assert np.isnan(vals[2:120]).all()
    assert vals[0] == 70.0 and vals[-1] == 80.0


def test_out_of_order_input_is_sorted():
    v, t = _minutes("2026-01-01T00:00", [70.0, 71.0, 72.0])
    grid, vals = asos1min._to_minute_grid(list(reversed(v)), list(reversed(t)))
    assert list(vals) == [70.0, 71.0, 72.0]


def test_empty_input_does_not_explode():
    grid, vals = asos1min.qc_series([], [])
    assert grid.size == 0 and vals.size == 0


# ── despike: owns the TAIL, not the exact-match rate ────────────────────────

def test_despike_removes_the_46f_sensor_fault():
    """The real failure: KLAX days came back +46F and +56F above settlement."""
    temps = [70.0] * 20
    temps[10] = 116.0
    v, t = _minutes("2026-01-01T00:00", temps)
    _, vals = asos1min.qc_series(v, t, do_despike=True, do_mean=False)
    assert np.isnan(vals[10])
    assert np.nanmax(vals) == 70.0


def test_despike_leaves_a_real_ramp_alone():
    """A genuine 0.5F/min warming trend must survive — the tolerance exists to
    kill sensor faults, not to shave legitimate tails."""
    temps = [70.0 + 0.5 * i for i in range(40)]
    v, t = _minutes("2026-01-01T00:00", temps)
    _, vals = asos1min.qc_series(v, t, do_despike=True, do_mean=False)
    assert not np.isnan(vals).any()


def test_order_is_despike_then_mean():
    """⚠️ THE ORDERING BUG THIS GUARDS, and it only bites for MODERATE spikes.

    A +50F fault is caught either way — even smeared across five windows it
    still sits far enough from the local median to be removed. A +12F fault is
    the dangerous size: averaged first it becomes +2.4F, which is inside
    DESPIKE_TOL_F, so it survives and contaminates the daily max. Despiked
    first it is removed outright.

    This test used a +50F spike originally and PASSED for the wrong reason
    (both orders removed it), which would have let the ordering regress."""
    temps = [70.0] * 30
    temps[15] = 82.0
    v, t = _minutes("2026-01-01T00:00", temps)
    _, right = asos1min.qc_series(v, t, do_despike=True, do_mean=True)
    assert np.nanmax(right) == pytest.approx(70.0)
    # the wrong order, simulated: average first, then try to despike
    _, meaned = asos1min.qc_series(v, t, do_despike=False, do_mean=True)
    wrong = asos1min.despike(meaned)
    assert np.nanmax(wrong) > 70.0, "mean-first must leave contamination behind"


# ── the 5-minute mean: owns the +1F bias ────────────────────────────────────

def test_five_minute_mean_cuts_the_one_minute_overshoot():
    """The mechanism: the official extreme comes from 5-MINUTE observations, so
    the max of 1,440 one-minute samples is systematically higher."""
    temps = [70.0, 70.0, 71.0, 70.0, 70.0] * 6
    v, t = _minutes("2026-01-01T00:00", temps)
    _, raw = asos1min.qc_series(v, t, do_despike=False, do_mean=False)
    _, smooth = asos1min.qc_series(v, t, do_despike=False, do_mean=True)
    assert np.nanmax(raw) == 71.0
    assert np.nanmax(smooth) < 71.0


def test_mean_window_needs_min_samples():
    """Three isolated readings surrounded by holes must not produce a 5-minute
    mean — that would be an average of one sample wearing a window's name."""
    v, t = _minutes("2026-01-01T00:00", [70.0])
    v2, t2 = _minutes("2026-01-01T01:00", [90.0])
    _, vals = asos1min.qc_series(v + v2, t + t2, do_despike=False, do_mean=True)
    assert np.isnan(vals).all()


# ── daily extremes respect the CLIMATE day, not the UTC day ─────────────────

def test_daily_extreme_uses_the_climate_day_window():
    """KLAX's climate day is 08:00Z-08:00Z. A spike at 07:00Z belongs to the
    PREVIOUS day and must not raise today's max."""
    from datetime import date

    from highcast.config import load_config
    cfg = load_config("config/klax.yaml")
    assert cfg.station.std_utc_offset == -8   # PST; bounds() negates it

    temps = [70.0] * 1440
    v, t = _minutes("2026-06-15T08:00", temps)      # exactly one climate day
    early_v, early_t = _minutes("2026-06-15T07:00", [95.0] * 30)
    grid, vals = asos1min.qc_series(early_v + v, early_t + t)
    out = asos1min.daily_extremes(cfg, grid, vals, [date(2026, 6, 15)], "max",
                                  min_minutes=100)
    assert out["2026-06-15"] == pytest.approx(70.0)


def test_thin_days_are_dropped_not_scored():
    """A day with a fragment of coverage has probably missed the peak; scoring
    it would understate the method rather than report honestly."""
    from datetime import date

    from highcast.config import load_config
    cfg = load_config("config/klax.yaml")
    v, t = _minutes("2026-06-15T08:00", [70.0] * 50)
    grid, vals = asos1min.qc_series(v, t)
    out = asos1min.daily_extremes(cfg, grid, vals, [date(2026, 6, 15)], "max",
                                  min_minutes=1000)
    assert out == {}


# ── the table this must not silently leave ──────────────────────────────────

def test_obs_1min_is_not_obs(tmp_path):
    """⚠️ `latentmax.visible_running_extremum` reads `asos1` rows from `obs`
    WITHOUT filtering by source, and its output feeds the shipped `run_*`
    features. The archive lags 24-36h so it can only cover TRAINING rows —
    putting it in `obs` would train the model on a precision that does not exist
    at prediction time. See the comment on the table in `db.py`."""
    from highcast import db
    conn = db.connect(tmp_path / "t.db")
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(obs_1min)")}
    assert cols == {"station", "valid_utc", "tmpf", "fetched_at"}
    assert "source" not in cols, "obs_1min must not pretend to be an `obs` row"
