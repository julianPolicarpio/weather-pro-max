"""The Station Table. Display only -- it must read and never write.

The load-bearing tests are `test_stall_searches_the_whole_climate_day` (my first
throwaway version of this query used a fixed lookback and under-reported the stall
by nearly five hours) and `test_onshore_projection_is_klax_only` (ONSHORE_DEG is
LAX's sea-breeze bearing; applying it to Miami is the station-constant
transplanting this project has had to unwind twice).
"""
import datetime as dt

import pytest
import yaml

from highcast import board
from highcast import db as hdb
from highcast.config import load_config

CFG = load_config()
ICAO = CFG.station.icao
# KLAX climate day is 08:00Z -> 08:00Z (std_utc_offset -8), DST-invariant.
DAY = dt.date(2026, 7, 15)
START = dt.datetime(2026, 7, 15, 8, 0, tzinfo=dt.timezone.utc)


def _cfg(tmp_path, src="config/klax.yaml"):
    raw = yaml.safe_load(open(src).read())
    raw["db_path"] = str(tmp_path / "t.db")
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(raw))
    return load_config(str(p))


def _obs(cfg, offsets_and_temps, **extra):
    """Seed hourly METARs at :53 past each offset hour from the climate-day start."""
    conn = hdb.connect(cfg.db_path)
    rows = []
    for hours, temp in offsets_and_temps:
        v = START + dt.timedelta(hours=hours, minutes=53)
        rows.append({"station": cfg.station.icao, "valid_utc": v.isoformat(),
                     "source": "metar", "tmpf": temp, "tmpf_precise": temp,
                     "dwpf": temp - 10, "sknt": 10.0, "drct": 250.0,
                     "skyc": "FEW", "six_max_c": None, "six_min_c": None,
                     "raw": f"METAR {cfg.station.icao} X {temp:.0f}/00 RMK AO2",
                     "fetched_at": "x", **extra})
    hdb.upsert(conn, "obs", rows)
    return conn


def test_stall_searches_the_whole_climate_day(tmp_path):
    """The minimum is set in hour 1 and never revisited; at hour 20 the stall is
    ~19h. A lookback window shorter than the day would clip it -- which is exactly
    what my first ad-hoc version did, reporting 13.1h where the truth was 17.9h.
    """
    cfg = _cfg(tmp_path)
    conn = _obs(cfg, [(1, 60.0)] + [(h, 75.0) for h in range(2, 21)])
    now = START + dt.timedelta(hours=20, minutes=55)
    got = board.row(conn, cfg, "low", now=now)
    assert got["stall_hours"] == pytest.approx(19.03, abs=0.1)
    assert got["stall_hours"] > board.STALL_NOTE_HOURS


def test_stall_resets_when_the_extremum_actually_moves(tmp_path):
    cfg = _cfg(tmp_path)
    conn = _obs(cfg, [(1, 60.0), (2, 75.0), (18, 59.0), (19, 75.0)])
    now = START + dt.timedelta(hours=19, minutes=55)
    got = board.row(conn, cfg, "low", now=now)
    # the min moved at hour 18, so the stall is ~2h, not ~19h
    assert got["stall_hours"] == pytest.approx(1.03, abs=0.15)


def test_onshore_projection_is_klax_only(tmp_path):
    """ONSHORE_DEG = 250 is LAX's sea-breeze bearing. KMIA must get None."""
    klax = _cfg(tmp_path, "config/klax.yaml")
    conn = _obs(klax, [(1, 70.0)])
    got = board.row(conn, klax, "low", now=START + dt.timedelta(hours=2))
    assert got["onshore_component"] is not None
    assert got["onshore_deg"] == 250.0

    kmia_dir = tmp_path / "m"
    kmia_dir.mkdir()
    kmia = _cfg(kmia_dir, "config/kmia.yaml")
    conn2 = _obs(kmia, [(1, 80.0)])
    got2 = board.row(conn2, kmia, "low",
                     now=START + dt.timedelta(hours=2))
    assert got2["onshore_component"] is None
    assert got2["onshore_deg"] is None


def test_wind_degrees_are_always_reported_even_without_a_projection(tmp_path):
    """The design rule: show the INPUT beside the verdict, never instead of it.
    240deg against a 250deg bearing is a borderline a human catches."""
    kmia_dir = tmp_path / "m"
    kmia_dir.mkdir()
    kmia = _cfg(kmia_dir, "config/kmia.yaml")
    conn = _obs(kmia, [(1, 80.0)])
    got = board.row(conn, kmia, "low", now=START + dt.timedelta(hours=2))
    assert got["drct"] == 250.0
    text = board.render([got])
    assert "250deg" in text


def test_tendency_falls_back_to_the_most_recent_group_with_an_age(tmp_path):
    """5appp files only at 3-hourly synoptic times, so the LATEST report usually
    lacks it. Printing 'unavailable' two thirds of the time would waste the
    column for a quantity that changes slowly."""
    cfg = _cfg(tmp_path)
    conn = _obs(cfg, [(1, 70.0)])
    # a tendency 3 hours before "now", then a newer report without one
    hdb.upsert(conn, "obs", [{
        "station": cfg.station.icao,
        "valid_utc": (START + dt.timedelta(hours=2, minutes=53)).isoformat(),
        "source": "metar", "tmpf": 71.0, "tmpf_precise": 71.0, "dwpf": 61.0,
        "sknt": 8.0, "drct": 250.0, "skyc": "FEW", "six_max_c": None,
        "six_min_c": None, "slp_hpa": 1010.0, "pres_tend_hpa": -1.3,
        "pres_tend_code": 8, "raw": "METAR X RMK AO2 SLP100 58013",
        "fetched_at": "x"}])
    hdb.upsert(conn, "obs", [{
        "station": cfg.station.icao,
        "valid_utc": (START + dt.timedelta(hours=5, minutes=53)).isoformat(),
        "source": "metar", "tmpf": 73.0, "tmpf_precise": 73.0, "dwpf": 62.0,
        "sknt": 9.0, "drct": 250.0, "skyc": "FEW", "six_max_c": None,
        "six_min_c": None, "slp_hpa": 1011.0, "pres_tend_hpa": None,
        "pres_tend_code": None, "raw": "METAR X RMK AO2 SLP110",
        "fetched_at": "x"}])
    now = START + dt.timedelta(hours=5, minutes=55)
    got = board.row(conn, cfg, "high", now=now)
    assert got["pres_tend_hpa"] == pytest.approx(-1.3)
    assert got["pres_tend_age_h"] == pytest.approx(3.03, abs=0.1)
    assert "3h old" in board.render([got])


def test_settles_to_is_the_conservative_end_of_the_interval(tmp_path):
    """For a LOW the settled value can be no HIGHER than the interval's upper
    bound; for a HIGH no LOWER than its lower bound. Rounding the wrong end would
    flatter the bound in the direction that loses money."""
    cfg = _cfg(tmp_path)
    conn = _obs(cfg, [(1, 60.4), (5, 79.6)])
    now = START + dt.timedelta(hours=6)
    lo = board.row(conn, cfg, "low", now=now)
    hi = board.row(conn, cfg, "high", now=now)
    assert lo["settles_to"] == round(lo["running_hi"])
    assert hi["settles_to"] == round(hi["running_lo"])


def test_render_only_shows_raw_when_asked(tmp_path):
    cfg = _cfg(tmp_path)
    conn = _obs(cfg, [(1, 70.0)])
    got = board.row(conn, cfg, "low", now=START + dt.timedelta(hours=2))
    assert "raw:" not in board.render([got])
    assert "raw:" in board.render([got], show_raw=True)


def test_board_writes_nothing(tmp_path):
    """Display only. If this ever changes, the freeze has been broken."""
    cfg = _cfg(tmp_path)
    conn = _obs(cfg, [(1, 70.0), (2, 75.0)])
    before = {t: conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
              for t in ("obs", "predictions", "app_bets", "signals")}
    board.row(conn, cfg, "low", now=START + dt.timedelta(hours=3))
    board.row(conn, cfg, "high", now=START + dt.timedelta(hours=3))
    after = {t: conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
             for t in ("obs", "predictions", "app_bets", "signals")}
    assert before == after


def test_no_observations_renders_without_crashing(tmp_path):
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    got = board.row(conn, cfg, "low", now=START + dt.timedelta(hours=3))
    assert got["tmpf"] is None
    assert "no observations" in board.render([got])


def test_commit_hour_stats_come_from_gating_not_the_current_hour(tmp_path):
    """The 2026-07-28 bug class: quoting the CURRENT hour's accuracy makes a
    transcription hour look like a forecasting one. The board must report the
    COMMIT hour's figures regardless of when it is run."""
    cfg = _cfg(tmp_path)
    conn = _obs(cfg, [(1, 70.0)])
    from highcast.gating import HOUR_ACCURACY, MIN_DECISION_HOUR
    ch = MIN_DECISION_HOUR[ICAO]["low"]
    for h in (2, 12, 22):
        got = board.row(conn, cfg, "low",
                        now=START + dt.timedelta(hours=h, minutes=55))
        assert got["commit_hour"] == ch
        assert got["commit_hit"] == HOUR_ACCURACY[ICAO]["low"][ch][1]


# ── reliability table ───────────────────────────────────────────────────────

def test_reliability_ranks_most_accurate_first():
    rows = board.reliability_table()
    hits = [r["hit"] for r in rows if r["hit"] is not None]
    assert hits == sorted(hits, reverse=True)
    # Hit is measured at each market's CURRENT commit hour, so moving one
    # reorders this table. KMIA HIGH moved 14 -> 13 on 2026-08-05 (on price,
    # not accuracy — see gating.MIN_DECISION_HOUR), taking it from 79.7% to
    # 67.6% and from first place to third:
    #
    #     KLAX high h13  69.5%   <- most accurate
    #     KLAX low  h6   68.2%
    #     KMIA high h13  67.6%
    #     KMIA low  h7   66.3%   <- least
    #
    # THIS TABLE RANKS ACCURACY ONLY, and that is the right job for it — it is
    # a display of how reliable each market is, not a recommendation of which
    # to trade. `paper_account.select_daily_market` is the one that must not
    # act on this ordering alone, and it filters on CONFIDENCE_FLOOR
    # eligibility for exactly that reason.
    assert (rows[0]["station"], rows[0]["metric"]) == ("KLAX", "high")
    assert (rows[-1]["station"], rows[-1]["metric"]) == ("KMIA", "low")


def test_reliability_covers_every_tradeable_market():
    rows = board.reliability_table()
    assert {(r["station"], r["metric"]) for r in rows} == {
        ("KLAX", "low"), ("KLAX", "high"), ("KMIA", "low"), ("KMIA", "high")}


def test_reliability_pairs_accuracy_with_still_ahead():
    """Hit without still-future is the trap that produced the 98.8% hours: a
    market looks brilliant purely because its answer already happened."""
    for r in board.reliability_table():
        assert r["hit"] is not None
        assert r["still_future"] is not None, "accuracy shown without context"


def test_reliability_reports_calibration_so_the_number_can_be_shaded():
    """2026-08-01: the hour move shrank KLAX HIGH's overconfidence from +6.71%
    (old h11) to +1.45% (new h13) -- the >=5-point bar this test used to assert
    no longer holds, because the hour move fixed most of what the bar was
    checking for. KLAX HIGH is still the more overconfident of the pair (KLAX
    LOW is now very slightly UNDER-confident), so the ordering survives; assert
    the new small magnitude instead of the stale threshold."""
    rows = {(r["station"], r["metric"]): r for r in board.reliability_table()}
    # KLAX HIGH is still the more overconfident of the two.
    assert rows[("KLAX", "high")]["calib"] > rows[("KLAX", "low")]["calib"]
    assert rows[("KLAX", "high")]["calib"] == pytest.approx(0.0145, abs=0.0005)
    # ...and it is no longer the >=5pt-overconfident market the old hour measured.
    assert rows[("KLAX", "high")]["calib"] < 0.05


def test_reliability_highlights_the_caller():
    rows = board.reliability_table(highlight=("KMIA", "high"))
    mine = [r for r in rows if r["is_current"]]
    assert len(mine) == 1
    assert (mine[0]["station"], mine[0]["metric"]) == ("KMIA", "high")


def test_render_reliability_shows_both_numbers():
    text = board.render_reliability(board.reliability_table())
    assert "bucket right" in text and "still ahead" in text
    # KLAX low hit at its new commit hour h6 (knee moved 4 -> 6 on 2026-07-31,
    # see gating.MIN_DECISION_HOUR; value from gating.HOUR_ACCURACY["KLAX"]["low"][6]).
    assert "68.2%" in text          # KLAX low hit
    # 2026-08-01: HOUR_CALIB was RE-MEASURED at the new commit hours. The
    # "KNOWN GAP" this test used to document -- HOUR_CALIB having no entry for
    # KLAX high at h13, so render_reliability printed "--" -- has been filled
    # in by that re-measurement (gating.HOUR_CALIB["KLAX"]["high"][13]). KLAX
    # high's overconfidence collapsed from +6.71% at the old h11 to +1.45% at
    # h13, because the later, more accurate hour is also more honest.
    assert "+1.5%" in text         # KLAX high overconfidence, now much smaller


# ── environment candidate columns (2026-07-31) ──────────────────────────────

def test_env_columns_never_enter_a_shipped_feature_list():
    """The KLAX HIGH unfreeze added candidate columns to the FRAME. They must not
    reach any model's feature list: the 2026-07-31 experiment measured them as
    indistinguishable from noise (deck alone -1.3% dHIT vs RAND -1.4%), so they
    are NOT shipped. If this fails, an unmeasured feature has been added."""
    from highcast.features import FEATURES, model_features
    from highcast.models.intraday import INTRADAY_FEATURES
    env = ("ceiling_ft", "sky_lowest_ft", "vis_sm", "pres_tend", "slp")
    for lst, name in ((INTRADAY_FEATURES, "INTRADAY_FEATURES"),
                      (FEATURES, "FEATURES"),
                      (model_features("high"), "model_features(high)"),
                      (model_features("low"), "model_features(low)")):
        leaked = [c for c in lst if any(e in c for e in env)]
        assert not leaked, f"{name} leaked {leaked}"


def test_shipped_feature_counts_are_pinned():
    """Guards the whole freeze in one assertion: the counts documented in
    HANDOFF. A change here means some model's inputs moved."""
    from highcast.features import FEATURES, model_features
    from highcast.models.intraday import INTRADAY_FEATURES
    assert len(INTRADAY_FEATURES) == 14
    assert len(FEATURES) == 42
    assert len(model_features("high")) == 20
    assert len(model_features("low")) == 38


# ── climatology bands: placing a reading against this station's own history ──

def _climo_obs(cfg, years=4, hour=12, span_days=45):
    """Seed a realistic pool: `span_days` either side of 15 July, at three
    consecutive UTC hours, across several years.

    Both conditions have to be satisfiable at once for a band to exist -- a flat
    200 days walking back from one anchor spans four months, of which only the
    ~61 inside the +/-30-day seasonal window survive, which is under
    CLIMO_MIN_SAMPLES. Real history has many YEARS of each calendar week, which
    is what makes the pool deep, and the fixture has to reproduce that shape.
    """
    conn = hdb.connect(cfg.db_path)
    rows = []
    for y in range(2022, 2022 + years):
        anchor = dt.datetime(y, 7, 15, hour, 53, tzinfo=dt.timezone.utc)
        for d in range(-span_days, span_days + 1):
            for dh in (-1, 0, 1):
                v = anchor + dt.timedelta(days=d, hours=dh)
                rows.append({
                    "station": cfg.station.icao, "valid_utc": v.isoformat(),
                    "source": "metar", "tmpf": 70.0, "tmpf_precise": 70.0,
                    "dwpf": 50.0 + (d % 20), "sknt": 8.0, "drct": 250.0,
                    "skyc": "FEW", "slp_hpa": 1013.0, "raw": "x",
                    "fetched_at": "x"})
    hdb.upsert(conn, "obs", rows)
    return conn


def test_bands_are_conditioned_on_the_hour(tmp_path):
    """Pooling every hour of the day would make a 4am reading look typical
    against 2pm values. Observations outside the +/-1h window must not count."""
    cfg = _cfg(tmp_path)
    conn = _climo_obs(cfg, hour=12)
    now = dt.datetime(2026, 7, 15, 12, 0, tzinfo=dt.timezone.utc)
    assert board.climatology_bands(conn, cfg, now)["dwpf"]["n"] > 0
    # 12 hours away: outside the window, so nothing comparable is left.
    far = dt.datetime(2026, 7, 15, 0, 0, tzinfo=dt.timezone.utc)
    assert "dwpf" not in board.climatology_bands(conn, cfg, far)


def test_bands_are_conditioned_on_the_season(tmp_path):
    """A July reading judged against January values is not judged at all."""
    cfg = _cfg(tmp_path)
    conn = _climo_obs(cfg, hour=12)
    jan = dt.datetime(2026, 1, 15, 12, 0, tzinfo=dt.timezone.utc)
    assert "dwpf" not in board.climatology_bands(conn, cfg, jan)


def test_a_thin_pool_yields_no_band_rather_than_a_shaky_one(tmp_path):
    cfg = _cfg(tmp_path)
    conn = _climo_obs(cfg, years=1, span_days=3, hour=12)
    now = dt.datetime(2026, 7, 15, 12, 0, tzinfo=dt.timezone.utc)
    assert board.climatology_bands(conn, cfg, now) == {}


def test_february_29_does_not_crash_the_band(tmp_path):
    """Day-of-year is taken on a leap reference year; a non-leap one raised."""
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    hdb.upsert(conn, "obs", [{
        "station": cfg.station.icao, "valid_utc": "2024-02-29T12:53:00+00:00",
        "source": "metar", "tmpf": 60.0, "tmpf_precise": 60.0, "dwpf": 50.0,
        "sknt": 5.0, "drct": 250.0, "skyc": "FEW", "raw": "x", "fetched_at": "x"}])
    now = dt.datetime(2024, 2, 29, 12, 0, tzinfo=dt.timezone.utc)
    assert board.climatology_bands(conn, cfg, now) == {}   # thin, but no raise


def test_band_position_places_the_reading_and_names_it():
    band = {"p10": 0.0, "p25": 2.5, "p50": 5.0, "p75": 7.5, "p90": 10.0, "n": 500}
    assert board.band_position(band, 5.0)["verdict"] == "typical"
    assert board.band_position(band, 5.0)["offset"] == pytest.approx(0.5)
    assert board.band_position(band, 1.0)["verdict"] == "uncommon"
    assert board.band_position(band, 12.0)["verdict"] == "unusual"


def test_a_reading_off_the_end_is_clamped_and_says_so():
    """The tick is pinned to the track end, so it must be distinguishable from
    a reading that genuinely sits at the edge -- otherwise a record-breaking
    value draws identically to an ordinary p90 one."""
    band = {"p10": 0.0, "p25": 2.5, "p50": 5.0, "p75": 7.5, "p90": 10.0, "n": 500}
    edge = board.band_position(band, 10.0)
    off = board.band_position(band, 40.0)
    assert edge["clamped"] is False
    assert off["clamped"] is True
    assert off["offset"] == 1.0


def test_band_position_survives_a_degenerate_band():
    """Every comparable hour identical -> no inside or outside, and no
    ZeroDivisionError on the request path."""
    flat = {"p10": 5.0, "p25": 5.0, "p50": 5.0, "p75": 5.0, "p90": 5.0, "n": 500}
    pos = board.band_position(flat, 5.0)
    assert pos["offset"] == 0.5 and pos["verdict"] == "typical"


def test_band_position_is_none_without_either_side():
    assert board.band_position(None, 5.0) is None
    assert board.band_position({"p10": 0, "p25": 1, "p50": 2, "p75": 3,
                                "p90": 4, "n": 9}, None) is None


def test_evidence_drops_missing_readings_rather_than_showing_blanks(tmp_path):
    """A row reading '--' invites the eye to treat absence as a reading, and
    pressure is absent from most reports by design."""
    row = {"depression": 5.0, "dwpf": None, "sknt": 8.0, "slp_hpa": None}
    band = {"p10": 0.0, "p25": 2.5, "p50": 5.0, "p75": 7.5, "p90": 10.0, "n": 500}
    ev = board.evidence(row, {"depression": band, "sknt": band})
    assert [e["key"] for e in ev] == ["depression", "sknt"]


def test_evidence_without_a_pool_still_lists_the_reading(tmp_path):
    """No history is a reason to omit the STRIP, not the number."""
    ev = board.evidence({"depression": 5.0}, {})
    assert len(ev) == 1 and ev[0]["band"] is None
    assert ev[0]["text"] == "5.0°F"


def test_climatology_bands_write_nothing(tmp_path):
    cfg = _cfg(tmp_path)
    conn = _climo_obs(cfg, hour=12)
    before = conn.execute("SELECT count(*) FROM obs").fetchone()[0]
    board.climatology_bands(conn, cfg,
                            dt.datetime(2026, 7, 15, 12, tzinfo=dt.timezone.utc))
    assert conn.execute("SELECT count(*) FROM obs").fetchone()[0] == before


# ── the gate verdict (`gating.decision_gates`) ──────────────────────────────

def test_the_board_shows_the_gate_refusals(tmp_path, monkeypatch):
    """The desk must see WHY it may not act, not merely that it may not."""
    from highcast import gating

    cfg = _cfg(tmp_path)
    conn = _obs(cfg, [(1, 70.0)])
    monkeypatch.setattr(
        gating, "decision_gates",
        lambda *a, **k: gating.GateVerdict(
            ok=False, refusals=("guidance_lead", "confidence_floor")))
    got = board.row(conn, cfg, "low", now=START + dt.timedelta(hours=2))
    text = board.render([got])
    assert "GATES: REFUSE" in text
    assert "guidance_lead" in text
    assert "confidence_floor" in text


def test_a_clear_gate_verdict_says_so(tmp_path, monkeypatch):
    """The flip side: when nothing refuses, the board says clear rather than
    staying silent -- silence would be indistinguishable from 'not measured'."""
    from highcast import gating

    cfg = _cfg(tmp_path)
    conn = _obs(cfg, [(1, 70.0)])
    monkeypatch.setattr(
        gating, "decision_gates",
        lambda *a, **k: gating.GateVerdict(ok=True, refusals=()))
    got = board.row(conn, cfg, "low", now=START + dt.timedelta(hours=2))
    text = board.render([got])
    assert "GATES: clear" in text
    assert "GATES: REFUSE" not in text
