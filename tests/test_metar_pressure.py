"""SLP and 3-hour pressure tendency parsing.

These columns are DISPLAY/analysis only while the model is frozen. The test that
matters most is `test_slp_is_not_off_by_a_factor_of_ten` -- a throwaway query
during this work rendered SLP097 as 100.97 hPa, and a silent factor of ten in a
pressure feature is exactly the class of defect that cost a bucket on 2026-07-28.
"""
import pytest

from highcast.fetchers.metar import PRES_TEND_SHAPE, parse_metar

# Real reports pulled from the archive on 2026-07-30.
KLAX_FULL = ("METAR KLAX 292353Z 24012KT 10SM FEW010 24/18 A2982 RMK AO2 "
             "SLP097 T02440183 10267 20239 58013 $")
KLAX_NO_TEND = ("METAR KLAX 300153Z 25013KT 10SM SCT008 22/19 A2983 RMK AO2 "
                "SLP098 T02220189 $")
KMIA_FULL = ("METAR KMIA 292353Z 26008KT 10SM FEW035 SCT300 32/23 A2994 RMK AO2 "
             "SLP140 T03170233 10344 20311 55000 $")


def _p(raw):
    return parse_metar(raw, "2026-07-29T23:53:00+00:00", "KLAX", "")


def test_slp_is_not_off_by_a_factor_of_ten():
    """SLP097 is 1009.7 hPa. Surface pressure is never ~100."""
    assert _p(KLAX_FULL)["slp_hpa"] == pytest.approx(1009.7)
    assert _p(KMIA_FULL)["slp_hpa"] == pytest.approx(1014.0)
    assert 950.0 < _p(KLAX_FULL)["slp_hpa"] < 1050.0


def test_slp_uses_the_900s_branch_above_500():
    """SLP502 -> 950.2, not 1050.2. The >=500 rule is the whole decode."""
    raw = "METAR KXXX 010053Z 00000KT RMK AO2 SLP502 T01000050"
    assert _p(raw)["slp_hpa"] == pytest.approx(950.2)
    raw999 = "METAR KXXX 010053Z 00000KT RMK AO2 SLP999 T01000050"
    assert _p(raw999)["slp_hpa"] == pytest.approx(999.9)
    raw000 = "METAR KXXX 010053Z 00000KT RMK AO2 SLP000 T01000050"
    assert _p(raw000)["slp_hpa"] == pytest.approx(1000.0)


def test_slpno_yields_none_rather_than_a_bogus_value():
    raw = "METAR KXXX 010053Z 00000KT RMK AO2 SLPNO T01000050"
    assert _p(raw)["slp_hpa"] is None


def test_tendency_sign_comes_from_the_code_not_the_digits():
    """58013 -> code 8 (net LOWER), magnitude 1.3 -> -1.3 hPa.

    Reading ppp alone would report +1.3 and invert the advection signal.
    """
    got = _p(KLAX_FULL)
    assert got["pres_tend_code"] == 8
    assert got["pres_tend_hpa"] == pytest.approx(-1.3)


@pytest.mark.parametrize("code,expect_sign", [
    (0, +1), (1, +1), (2, +1), (3, +1), (4, 0), (5, -1), (6, -1), (7, -1), (8, -1),
])
def test_every_wmo_code_maps_to_the_right_direction(code, expect_sign):
    raw = f"METAR KXXX 010053Z 00000KT RMK AO2 SLP100 5{code}025 T01000050"
    got = _p(raw)
    assert got["pres_tend_code"] == code
    if expect_sign == 0:
        assert got["pres_tend_hpa"] == 0.0
    else:
        assert got["pres_tend_hpa"] == pytest.approx(expect_sign * 2.5)


def test_steady_code_forces_zero_even_with_a_nonzero_magnitude():
    """Code 4 means steady; a nonzero ppp beside it is self-contradictory.

    Note the group is `5appp` -- the leading 5 is the group INDICATOR, so a
    steady tendency is `54321` (5, a=4, ppp=321). `44321` is the 24-hour max/min
    group and must NOT be read as a tendency; asserted below.
    """
    raw = "METAR KXXX 010053Z 00000KT RMK AO2 SLP100 54321 T01000050"
    got = _p(raw)
    assert got["pres_tend_code"] == 4
    assert got["pres_tend_hpa"] == 0.0


def test_the_24_hour_group_is_not_read_as_a_tendency():
    """`4sTTTsTTT` shares no prefix with `5appp`, and a bare 4-group must be
    ignored rather than decoded as a pressure change."""
    raw = "METAR KXXX 010053Z 00000KT RMK AO2 SLP100 44321 T01000050"
    assert _p(raw)["pres_tend_hpa"] is None
    assert _p(raw)["pres_tend_code"] is None


def test_kmia_55000_is_a_real_zero_magnitude_fall_shape():
    """KMIA filed 55000: code 5 (falling then rising, net lower), magnitude 0."""
    got = parse_metar(KMIA_FULL, "2026-07-29T23:53:00+00:00", "KMIA", "")
    assert got["pres_tend_code"] == 5
    assert got["pres_tend_hpa"] == pytest.approx(0.0)


def test_absent_tendency_group_is_none_not_zero():
    """Only synoptic reports carry 5appp (~28% of rows). Absent must be NULL --
    coercing to 0.0 would fabricate 'steady' on three of every four rows."""
    got = _p(KLAX_NO_TEND)
    assert got["pres_tend_hpa"] is None
    assert got["pres_tend_code"] is None
    assert got["slp_hpa"] == pytest.approx(1009.8)


def test_altimeter_setting_is_not_mistaken_for_pressure():
    """A2982 is inches of mercury in the BODY, not the remarks. Matching outside
    RMK would let it or an RVR group collide with these patterns."""
    raw = "METAR KXXX 010053Z 00000KT 10SM FEW010 24/18 A2982"
    got = _p(raw)
    assert got["slp_hpa"] is None
    assert got["pres_tend_hpa"] is None


def test_peak_wind_group_cannot_be_read_as_a_tendency():
    """Mirror of the _SIX_MAX/`PK WND` regression from 2022. A direction cannot
    start with 5, but the (?!/) guard is asserted so it cannot be removed."""
    raw = ("METAR KXXX 010053Z 00000KT RMK AO2 PK WND 25036/2153 SLP100 "
           "T01000050")
    assert _p(raw)["pres_tend_hpa"] is None


def test_six_hour_groups_still_parse_unchanged():
    """Regression guard: the new patterns must not disturb 1sTTT / 2sTTT, which
    are the numbers that actually settle the bets."""
    got = _p(KLAX_FULL)
    assert got["six_max_c"] == pytest.approx(26.7)
    assert got["six_min_c"] == pytest.approx(23.9)
    assert got["tmpf_precise"] == pytest.approx(24.4 * 9 / 5 + 32)


def test_shape_table_covers_every_code():
    assert set(PRES_TEND_SHAPE) == set(range(9))
    assert "lower" in PRES_TEND_SHAPE[8]
    assert PRES_TEND_SHAPE[4] == "steady"
