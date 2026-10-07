from highcast.fetchers.metar import parse_metar
from highcast.timeutil import norm_valid as _norm_valid

RAW = ("KLAX 171753Z 25008KT 10SM FEW015 24/17 A2992 "
       "RMK AO2 SLP132 T02390172 10244 20161 58008")

def test_parse_full_metar():
    row = parse_metar(RAW, "2026-07-17T17:53:00+00:00", "KLAX", "2026-07-17T18:00:00+00:00")
    assert row["source"] == "metar"
    assert row["tmpf"] == 75.2                    # 24 C coarse group
    assert abs(row["tmpf_precise"] - 75.02) < 0.01  # T02390172 -> 23.9 C
    assert abs(row["dwpf"] - 63.0) < 0.1          # 17.2 C from T-group
    assert row["sknt"] == 8.0 and row["drct"] == 250.0
    assert row["skyc"] == "FEW"
    assert row["six_max_c"] == 24.4               # 10244 -> +24.4 C

def test_parse_negative_t_group_and_no_remarks():
    raw = "KLAX 170353Z 00000KT 10SM CLR M01/M03 A3010 RMK AO2 T10061028"
    row = parse_metar(raw, "2026-01-17T03:53:00+00:00", "KLAX", "x")
    assert abs(row["tmpf_precise"] - 30.92) < 0.01  # -0.6 C
    assert row["six_max_c"] is None
    assert row["skyc"] == "CLR"


def test_parse_vrb_wind():
    raw = "KLAX 171753Z VRB03KT 10SM FEW015 24/17 A2992 RMK AO2 T02390172"
    row = parse_metar(raw, "2026-07-17T17:53:00+00:00", "KLAX", "x")
    assert row["drct"] is None
    assert row["sknt"] == 3.0


def test_parse_no_rmk_no_crash():
    raw = "KLAX 171753Z 25008KT 10SM FEW015 24/17 A2992"
    row = parse_metar(raw, "2026-07-17T17:53:00+00:00", "KLAX", "x")
    assert row["six_max_c"] is None
    assert row["tmpf"] == 75.2


def test_parse_rejects_implausible_six_max_vs_report_temp():
    # Real corrupted IEM archive remark: 10439 claims a 6-hr max of +43.9C
    # /111F while the report's own temperature (17/11, T01720111) is only
    # 17.2C/63F. A 6-hour max can't run ~48F above the current reading at a
    # surface station, so parse_metar must null it out rather than let it
    # poison downstream running-max bounds.
    raw = ("KLAX 032353Z 25008KT 10SM FEW015 17/11 A2992 "
           "RMK AO2 SLP132 T01720111 10439 20161 58008")
    row = parse_metar(raw, "2026-01-03T23:53:00+00:00", "KLAX", "x")
    assert row["six_max_c"] is None
    # A normal six-max (existing 10244 fixture, temp 24C, well within 30F)
    # must still parse through unaffected by the new guard.
    normal = parse_metar(RAW, "2026-07-17T17:53:00+00:00", "KLAX", "2026-07-17T18:00:00+00:00")
    assert normal["six_max_c"] == 24.4


def test_parse_six_min_group():
    # Genuine KLAX report: RMK carries both 6-hourly groups —
    # 10228 (6-hr max +22.8C) and 20211 (6-hr min +21.1C).
    raw = ("METAR KLAX 251153Z 12005KT 10SM FEW250 21/19 A2986 "
           "RMK AO2 SLP109 T02110189 10228 20211 58010")
    row = parse_metar(raw, "2026-07-25T11:53:00+00:00", "KLAX", "x")
    assert row["six_min_c"] == 21.1
    assert row["six_max_c"] == 22.8


def test_peak_wind_group_is_not_mistaken_for_a_six_hourly_group():
    """Real KLAX report, 2022-09-09T23:53Z. `PK WND 11032/2256` matches the
    naive 6-hr-max pattern as sign=1 (negative), digits=032 -> -3.2C, which is
    what shipped: six_max_c came out -3.2 on a 91F observation. The genuine
    groups on this same report are 10389 / 20322. Found by an invariant sweep
    (six_min_c > six_max_c) over 8,102 backfilled reports -- it was the only
    violation, and the mirrored _SIX_MIN would not have been harmless."""
    raw = ("KLAX 092353Z 12016KT 10SM -RA FEW090 SCT110 OVC130 33/17 A2953 "
           "RMK AO2 PK WND 11032/2256 RAB14 SLP998 VIRGA NW-N P0000 60000 "
           "T03280172 10389 20322 53003 $")
    row = parse_metar(raw, "2022-09-09T23:53:00+00:00", "KLAX", "x")
    assert row["six_max_c"] == 38.9
    assert row["six_min_c"] == 32.2
    assert row["six_min_c"] <= row["six_max_c"]


def test_parse_no_six_min_group_is_none():
    raw = "KLAX 170353Z 00000KT 10SM CLR M01/M03 A3010 RMK AO2 T10061028"
    row = parse_metar(raw, "2026-01-17T03:53:00+00:00", "KLAX", "x")
    assert row["six_min_c"] is None


def test_parse_rejects_implausible_six_min_vs_report_temp():
    # Mirror of the 6-hr-max guard: a 6-hr MIN cannot run more than 30F BELOW
    # the report's own temperature at a surface station. 21439 claims -43.9C
    # (-47F) while the report reads 17.2C/63F -> ~110F below. Null it out
    # rather than let it poison downstream running-min bounds.
    raw = ("KLAX 032353Z 25008KT 10SM FEW015 17/11 A2992 "
           "RMK AO2 SLP132 T01720111 10244 21439 58008")
    row = parse_metar(raw, "2026-01-03T23:53:00+00:00", "KLAX", "x")
    assert row["six_min_c"] is None
    # a plausible min on the same report still parses through
    ok = ("KLAX 032353Z 25008KT 10SM FEW015 17/11 A2992 "
          "RMK AO2 SLP132 T01720111 10244 20161 58008")
    assert parse_metar(ok, "2026-01-03T23:53:00+00:00", "KLAX", "x")["six_min_c"] == 16.1


def test_norm_valid_epoch_seconds():
    # 2026-07-17T00:53:00Z
    assert _norm_valid(1784249580) == "2026-07-17T00:53:00+00:00"


def test_norm_valid_iem_space_separated():
    assert _norm_valid("2025-07-01 00:53") == "2025-07-01T00:53:00+00:00"


def test_norm_valid_iso_z_with_fractional_seconds():
    assert _norm_valid("2026-07-17T00:53:00.000Z") == "2026-07-17T00:53:00+00:00"


def test_norm_valid_agrees_across_forms():
    epoch = _norm_valid(1784249580)
    iem = _norm_valid("2025-07-01 00:53")
    avwx = _norm_valid("2026-07-17T00:53:00.000Z")
    assert epoch == avwx == "2026-07-17T00:53:00+00:00"
    assert iem == "2025-07-01T00:53:00+00:00"


def test_store_roundtrip_preserves_tmpf_precise(tmp_path):
    from highcast import db
    from highcast.fetchers.metar import store

    row = parse_metar(RAW, "2026-07-17T17:53:00+00:00", "KLAX", "2026-07-17T18:00:00+00:00")
    conn = db.connect(tmp_path / "t.db")
    store(conn, [row])
    fetched = conn.execute(
        "SELECT tmpf_precise FROM obs WHERE station = ? AND valid_utc = ? AND source = ?",
        ("KLAX", "2026-07-17T17:53:00+00:00", "metar"),
    ).fetchone()
    assert fetched is not None
    assert abs(fetched["tmpf_precise"] - row["tmpf_precise"]) < 1e-9


import pytest
from datetime import date
from highcast.config import load_config
from highcast.fetchers import metar

@pytest.mark.live
def test_live_recent_metars():
    cfg = load_config()
    rows = metar.fetch_recent(cfg, hours=12)
    assert len(rows) >= 8
    assert any(r["tmpf_precise"] is not None for r in rows)

@pytest.mark.live
def test_live_archive_one_week():
    cfg = load_config()
    rows = metar.fetch_archive(cfg, date(2025, 7, 1), date(2025, 7, 7))
    assert len(rows) >= 150   # ~hourly + specials for a week
    assert any(r["six_max_c"] is not None for r in rows)
