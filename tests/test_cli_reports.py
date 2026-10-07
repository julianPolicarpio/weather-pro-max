from highcast.fetchers import cli_reports

SAMPLE = {
    "results": [
        {"station": "KLAX", "valid": "2025-07-16", "high": 84, "low": 64,
         "precip": 0.0},
        {"station": "KLAX", "valid": "2025-07-17", "high": None, "low": 63},
    ]
}

def test_parse_rows_skips_missing_high_and_maps_fields():
    rows = cli_reports.parse_rows(SAMPLE, "KLAX", "2026-07-17T00:00:00+00:00")
    assert len(rows) == 1
    r = rows[0]
    assert r["station"] == "KLAX"
    assert r["date"] == "2025-07-16"
    assert r["high_f"] == 84
    assert r["low_f"] == 64
    assert r["fetched_at"] == "2026-07-17T00:00:00+00:00"


# ── one row per date: the LATEST product wins, not the last one in the payload ──

def test_parse_rows_keeps_the_latest_product_when_two_exist_for_one_date():
    """Today whichever item lands last in the `results` list wins, which is
    arbitrary. Fix: keep the one issued LATER, regardless of payload order."""
    payload = {"results": [
        {"station": "KMIA", "valid": "2026-08-01", "high": 94, "low": 82,
         "product": "202608012033-KMFL-CDUS42-CLIMIA"},
        {"station": "KMIA", "valid": "2026-08-01", "high": 94, "low": 74,
         "product": "202608020600-KMFL-CDUS42-CLIMIA"},
    ]}
    rows = cli_reports.parse_rows(payload, "KMIA", "t")
    assert len(rows) == 1
    assert rows[0]["low_f"] == 74            # the LATER product wins


def test_parse_rows_keeps_the_latest_even_when_it_comes_first_in_the_payload():
    """Same pair, reversed order -- the result must not depend on arrival order."""
    payload = {"results": [
        {"station": "KMIA", "valid": "2026-08-01", "high": 94, "low": 74,
         "product": "202608020600-KMFL-CDUS42-CLIMIA"},
        {"station": "KMIA", "valid": "2026-08-01", "high": 94, "low": 82,
         "product": "202608012033-KMFL-CDUS42-CLIMIA"},
    ]}
    rows = cli_reports.parse_rows(payload, "KMIA", "t")
    assert len(rows) == 1
    assert rows[0]["low_f"] == 74


def test_parse_rows_keeps_a_known_issuance_time_over_an_unknown_one():
    """A row with no parseable `product` must never displace one whose
    issuance time IS known -- an unknown freshness is not evidence of a
    later report."""
    payload = {"results": [
        {"station": "KMIA", "valid": "2026-08-01", "high": 94, "low": 82,
         "product": "202608012033-KMFL-CDUS42-CLIMIA"},
        {"station": "KMIA", "valid": "2026-08-01", "high": 94, "low": 99},
    ]}
    rows = cli_reports.parse_rows(payload, "KMIA", "t")
    assert len(rows) == 1
    assert rows[0]["low_f"] == 82


import pytest
from highcast.config import load_config

@pytest.mark.live
def test_live_fetch_2025_has_a_full_year():
    cfg = load_config()
    rows = cli_reports.fetch_year(cfg, 2025)
    assert len(rows) >= 300, f"only {len(rows)} rows - check endpoint/params"
    highs = [r["high_f"] for r in rows]
    assert all(40 <= h <= 120 for h in highs)
    print("latest:", rows[-1]["date"], rows[-1]["high_f"])
