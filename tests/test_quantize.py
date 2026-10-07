from highcast.quantize import candidates_from_reported_c, candidates_from_displayed_f, interval

def test_whole_celsius_maps_back_to_one_or_two_fahrenheit():
    # 71F -> 21.67C -> 22 ; 72F -> 22.22C -> 22 ; 73F -> 22.78C -> 23
    assert candidates_from_reported_c(22) == [71, 72]
    assert candidates_from_reported_c(23) == [73, 74]
    # every whole C from -30..45 must invert to 1 or 2 candidates
    for c in range(-30, 46):
        assert 1 <= len(candidates_from_reported_c(c)) <= 2

def test_displayed_f_roundtrips_through_c():
    # displayed F = round_half_up(C * 9/5 + 32); 22C displays as 72F
    assert candidates_from_displayed_f(72) == [71, 72]

def test_some_displayed_f_values_are_unreachable():
    # 25C displays 77F, 26C displays 79F - the feed can never show 78F.
    # Empty result is correct and documents the display-table gaps.
    assert candidates_from_displayed_f(78) == []

def test_interval():
    assert interval([71, 72]) == (71, 72)
