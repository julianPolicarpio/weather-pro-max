from highcast.config import load_config
from highcast.metric import get_metric

CFG = load_config()


def test_high_metric_defaults_match_current_behavior():
    m = get_metric(CFG, "high")
    assert m.key == "high"
    assert m.target_col == "high_f"
    assert m.openmeteo_var == "temperature_2m_max"
    assert m.nws_field == "maxTemperature"
    assert m.extremum == "max"
    assert m.tail_dir == "floor"
    assert m.cutoff_local_hour == 9 and m.cutoff_day_offset == 0
    assert m.kalshi_series == CFG.station.kalshi_series
    assert "mos_mex" in m.fc_models


def test_low_metric():
    m = get_metric(CFG, "low")
    assert m.target_col == "low_f"
    assert m.openmeteo_var == "temperature_2m_min"
    assert m.nws_field == "minTemperature"
    assert m.extremum == "min"
    assert m.tail_dir == "cap"           # <=T bet, not >=T
    assert m.cutoff_day_offset == -1     # evening before, mirrored timing
    assert m.kalshi_series == "KXLOWTLAX"
    # MOS IS available for the low (2026-07-26). The old assertion here encoded
    # the belief that "IEM MOS has no min-temp field for the low metric" — wrong:
    # the MOS N/X column carries the overnight min on its morning-valid rows, and
    # `_mos_row` was discarding them. Backfilled 16,255 LOW rows to 2021 on the
    # strength of it; MOS is the low's only station-calibrated guidance.
    assert m.mos_field == "n_x"
    assert "mos_mex" in m.fc_models


def test_unknown_metric_raises():
    import pytest
    with pytest.raises(ValueError):
        get_metric(CFG, "average")
