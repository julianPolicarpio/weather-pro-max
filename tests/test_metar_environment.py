"""Environment fields parsed from the raw METAR: visibility, cloud layers,
present weather.

WHY THEY EXIST. The KMIA HIGH is convection-limited and the nowcast has NO
precipitation input; `cur_sky` is the only storm-adjacent feature and it records
the HIGHEST layer, so a low deck is discarded on 82% of observations. Every field
here was already in text we store.

TWO BUGS THIS FILE PINS, both found by validating against real reports:

  1. `1/16SM` (dense fog) parsed as **16.0 SM** (nearly unlimited). The fraction
     branch only allowed single-digit denominators, so 1/16 fell through to the
     whole-number branch. 116 real KLAX fog reports were stored as the clearest
     days in the archive. An INVERTED variable is far worse than a missing one.
  2. `BKN032CB` was skipped entirely -- `\\b` does not match between `2` and `C`.
     Every convective (CB/TCU) layer was dropped, i.e. precisely the storm clouds
     these fields exist to capture. A real report parsed as 3 layers with a
     6000 ft ceiling instead of 4 layers at 3200 ft.

THE STANDING RULE: `skyc` must never change. It feeds `cur_sky`, a live model
feature, and the model is frozen.
"""
import pytest

from highcast.fetchers.metar import parse_metar

WHEN = "2026-07-15T12:00:00+00:00"


def p(raw):
    return parse_metar(raw, WHEN, "KMIA", "x")


# ── visibility ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expect,label", [
    ("KLAX 1Z 09003KT 1/16SM R25L/1800V3000FT FG VV001 11/10 A3005", 0.0625,
     "dense fog — the inverted-bug case"),
    ("KLAX 1Z 09003KT 1/8SM FG 11/10 A3005", 0.125, "eighth"),
    ("KLAX 1Z 09003KT 1/4SM FG 11/10 A3005", 0.25, "quarter"),
    ("KLAX 1Z 09003KT M1/4SM FG 11/10 A3005", 0.25, "less-than quarter"),
    ("KLAX 1Z 09003KT 1/2SM BR 11/10 A3005", 0.5, "half"),
    ("KLAX 1Z 09003KT 3/4SM BR 11/10 A3005", 0.75, "three quarters"),
    ("KLAX 1Z 09003KT 1 1/2SM BR 11/10 A3005", 1.5, "whole plus fraction"),
    ("KLAX 1Z 09003KT 10SM CLR 11/10 A3005", 10.0, "clear"),
])
def test_visibility_formats(raw, expect, label):
    assert p(raw)["vis_sm"] == pytest.approx(expect), label


def test_fog_is_never_read_as_clear():
    """The regression that matters. A fraction must never be mistaken for a
    whole number -- that turns the worst visibility into the best."""
    fog = p("KLAX 1Z 09003KT 1/16SM FG VV001 11/10 A3005")["vis_sm"]
    clear = p("KLAX 1Z 09003KT 10SM CLR 11/10 A3005")["vis_sm"]
    assert fog < 0.1
    assert fog < clear


def test_visibility_absent_is_none():
    assert p("KLAX 1Z 09003KT CLR 11/10 A3005")["vis_sm"] is None


# ── cloud layers ────────────────────────────────────────────────────────────

def test_convective_layers_are_not_dropped():
    """`BKN032CB` must count. The CB suffix marks cumulonimbus -- a
    thunderstorm -- and skipping it discards the most storm-specific datum in
    the report while also overstating the ceiling."""
    r = p("SPECI KMIA 1Z 24008G17KT 7SM TSRA BKN032CB BKN060 BKN110 OVC260 "
          "28/23 A2999")
    assert r["n_sky_layers"] == 4, "the CB-tagged layer was dropped"
    assert r["ceiling_ft"] == 3200, "ceiling taken from the wrong layer"
    assert r["sky_lowest_ft"] == 3200
    assert r["sky_cb"] == 1


def test_tcu_also_counts_as_convective():
    r = p("KMIA 1Z 24008KT 10SM SCT025TCU SCT090 30/23 A2999")
    assert r["sky_cb"] == 1
    assert r["sky_lowest_ft"] == 2500


def test_ceiling_is_the_lowest_broken_or_overcast():
    """SCT and FEW are not a ceiling: they do not cap insolation the same way."""
    r = p("KMIA 1Z 24008KT 10SM SCT032 SCT060 BKN110 OVC260 30/23 A2999")
    assert r["sky_lowest_ft"] == 3200      # lowest layer of any coverage
    assert r["ceiling_ft"] == 11000        # lowest BKN/OVC


def test_no_ceiling_when_only_scattered():
    r = p("KMIA 1Z 24008KT 10SM SCT038 SCT090 33/23 A2998")
    assert r["ceiling_ft"] is None
    assert r["sky_lowest_ft"] == 3800
    assert r["n_sky_layers"] == 2


def test_clear_sky_has_no_layers():
    r = p("KLAX 1Z 09003KT 10SM CLR 20/10 A3005")
    assert r["n_sky_layers"] is None
    assert r["ceiling_ft"] is None
    assert r["sky_cb"] == 0


# ── present weather ─────────────────────────────────────────────────────────

def test_thunderstorm_is_detected():
    r = p("SPECI KMIA 1Z 24008KT 7SM TSRA BKN032CB 28/23 A2999")
    assert r["wx_thunder"] == 1
    assert "TSRA" in r["wx_present"]


def test_rain_without_thunder():
    r = p("KMIA 1Z 24008KT 5SM -RA BKN020 26/24 A2999")
    assert r["wx_thunder"] == 0
    assert "RA" in r["wx_present"]


def test_no_weather_is_none_not_empty_string():
    r = p("KLAX 1Z 09003KT 10SM CLR 20/10 A3005")
    assert r["wx_present"] is None
    assert r["wx_thunder"] == 0


def test_remarks_cannot_fabricate_weather():
    """Only the BODY is scanned. A lookalike token in RMK must not register as
    observed weather."""
    r = p("KLAX 1Z 09003KT 10SM CLR 20/10 A3005 RMK AO2 TSB05E22 SLP097")
    assert r["wx_thunder"] == 0, "a remark was read as present weather"


# ── the freeze contract ─────────────────────────────────────────────────────

@pytest.mark.parametrize("raw", [
    "KMIA 1Z 24008KT 10SM SCT032 SCT060 SCT090 BKN260 31/24 A2998",
    "SPECI KMIA 1Z 24008G17KT 7SM TSRA BKN032CB BKN060 OVC260 28/23 A2999",
    "KLAX 1Z 09003KT 1/16SM FG VV001 11/10 A3005",
    "KLAX 1Z 09003KT 10SM CLR 20/10 A3005",
])
def test_skyc_is_still_the_last_token(raw):
    """`skyc` feeds `cur_sky`, a LIVE model feature. Adding layer parsing must
    not alter it by one character, or the frozen model's input changes."""
    import re
    tokens = re.findall(r"\b(FEW|SCT|BKN|OVC|CLR|SKC)\d*\b", raw)
    assert p(raw)["skyc"] == (tokens[-1] if tokens else None)
