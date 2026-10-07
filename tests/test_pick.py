"""Pick of the Day.

The load-bearing tests, in order of what would hurt most if they broke:
  * `test_never_writes_to_app_bets` -- the freeze contract. This is advice about
    Julian's money; `app_bets` is the immutable accuracy ledger and must not move.
  * `test_churn_window_ends_at_the_commit_hour` -- counting hours after the commit
    would use information that did not exist, which the freeze doc classes as a
    MODEL bug. Asserted to FAIL without the guard, so it is not tautological.
  * `test_parity_follows_the_live_ladder` -- an odd-floor bucket must be scored on
    the odd ladder. This project's own analysis got this wrong on 2026-07-30.
  * `test_unbuyable_outranks_everything` -- a pick you cannot transact is not a
    pick, however good the arithmetic.
"""
import datetime as dt
import json

import pytest
import yaml

from highcast import db as hdb
from highcast import pick
from highcast.config import load_config
from highcast.metric import get_metric
from highcast.timeutil import climate_day_bounds

TARGET = "2026-07-15"


def _final_product(cfg, d):
    """A CLI product id issued safely after `d`'s climate day closes -- what
    `pick.settle` now requires before it will grade a bet against it."""
    end = climate_day_bounds(dt.date.fromisoformat(d), cfg.station.std_utc_offset)[1]
    issued = end + dt.timedelta(hours=1)
    return issued.strftime("%Y%m%d%H%M") + "-TEST-CDUS00-CLITEST"


def _early_product(cfg, d):
    """A CLI product id issued one hour BEFORE `d`'s climate day closes --
    still preliminary."""
    end = climate_day_bounds(dt.date.fromisoformat(d), cfg.station.std_utc_offset)[1]
    issued = end - dt.timedelta(hours=1)
    return issued.strftime("%Y%m%d%H%M") + "-TEST-CDUS00-CLITEST"


def _cfg(tmp_path, src="config/klax.yaml", **staking):
    raw = yaml.safe_load(open(src).read())
    raw["db_path"] = str(tmp_path / "t.db")
    raw.setdefault("staking", {}).update(staking)
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(raw))
    return load_config(str(p))


def _out(bucket_lo, p_model, price, point=None, target=TARGET):
    """A minimal `signals.live_decision` result."""
    return {"target": target,
            "pred": {"point_f": point if point is not None else bucket_lo + 0.5},
            "market": [{"label": f"{bucket_lo}-{bucket_lo+1}", "lo": bucket_lo,
                        "hi": bucket_lo + 1, "p_model": p_model,
                        "market_price": price, "yes_ask": price}]}


def _at_commit(cfg, metric, plus=1):
    """A UTC instant at/after this market's commit hour, in station-local terms.

    Fixtures used to hardcode 12:00 UTC, which is 05:00 local at KLAX -- past the
    LOW's h4 commit but hours before the HIGH's h11. The commit-hour guard made
    that silently produce NO PICK (before commit hour) for every `high` test.
    """
    from zoneinfo import ZoneInfo
    from highcast.gating import MIN_DECISION_HOUR
    h = MIN_DECISION_HOUR.get(cfg.station.icao, {}).get(metric, 11) + plus
    return dt.datetime(2026, 7, 15, min(h, 23), 0,
                       tzinfo=ZoneInfo(cfg.station.tz)).astimezone(dt.timezone.utc)


def _score(tmp_path, bucket_lo, p_model, price, metric="low", **staking):
    cfg = _cfg(tmp_path, **staking)
    conn = hdb.connect(cfg.db_path)
    spec = get_metric(cfg, metric)
    now = _at_commit(cfg, metric)
    return cfg, conn, pick.score(conn, cfg, spec, now,
                                 out=_out(bucket_lo, p_model, price))


# ── the freeze contract ─────────────────────────────────────────────────────

def test_never_writes_to_app_bets(tmp_path):
    cfg = _cfg(tmp_path, bankroll=100.0, max_stake_frac=0.5)
    conn = hdb.connect(cfg.db_path)
    spec = get_metric(cfg, "low")
    now = _at_commit(cfg, "low")
    before = conn.execute("SELECT count(*) FROM app_bets").fetchone()[0]
    pick.record(conn, cfg, spec, now, "verdict", out=_out(68, 0.78, 0.62))
    after = conn.execute("SELECT count(*) FROM app_bets").fetchone()[0]
    assert before == after == 0
    # ...and it DID write its own row
    assert conn.execute("SELECT count(*) FROM picks").fetchone()[0] == 1


def test_declines_are_recorded_not_dropped(tmp_path):
    """A bar that only logs what it published can never be shown too strict."""
    cfg = _cfg(tmp_path, bankroll=100.0, max_stake_frac=0.5)
    conn = hdb.connect(cfg.db_path)
    spec = get_metric(cfg, "low")
    now = _at_commit(cfg, "low")
    pick.record(conn, cfg, spec, now, "verdict", out=_out(68, 0.60, 0.58))
    row = conn.execute("SELECT verdict, reason FROM picks").fetchone()
    assert row["verdict"].startswith("NO PICK")
    assert row["reason"]


def test_first_write_wins(tmp_path):
    cfg = _cfg(tmp_path, bankroll=100.0, max_stake_frac=0.5)
    conn = hdb.connect(cfg.db_path)
    spec = get_metric(cfg, "low")
    n0 = _at_commit(cfg, "low")
    pick.record(conn, cfg, spec, n0, "verdict", out=_out(68, 0.78, 0.62))
    pick.record(conn, cfg, spec, n0 + dt.timedelta(hours=1), "verdict",
                out=_out(72, 0.80, 0.64))
    rows = conn.execute("SELECT bucket FROM picks").fetchall()
    assert len(rows) == 1 and rows[0]["bucket"] == "68-69"


# ── verdict ordering ────────────────────────────────────────────────────────

def test_unbuyable_outranks_everything(tmp_path):
    """Great EV, unaffordable -> UNBUYABLE, and it names the fix."""
    _, _, s = _score(tmp_path, 68, 0.82, 0.66, bankroll=8.0, max_stake_frac=0.05)
    assert s["verdict"] == pick.UNBUYABLE
    assert s["ev"] > pick.MARGIN          # would otherwise have been a PICK
    assert "max_stake_frac" in s["reason"]


def test_price_ceiling_outranks_ev(tmp_path):
    """Even a positive EV above the ceiling is refused: expensive certainty."""
    _, _, s = _score(tmp_path, 68, 0.90, 0.80, bankroll=100.0, max_stake_frac=0.5)
    assert s["verdict"] == pick.NO_PICK_PRICE
    assert s["ev"] > pick.MARGIN


def test_margin_is_a_strict_bar(tmp_path):
    """EV exactly at MARGIN does not qualify -- the bar is `> MARGIN`.

    The fee must be read back from the scored row, not assumed: it is amortised
    over however many contracts the bankroll affords, so hardcoding the
    1-contract value overshoots. That amortisation is deliberate (Kalshi rounds
    up per ORDER), and an earlier draft of this test got it wrong.
    """
    cfg = _cfg(tmp_path, bankroll=100.0, max_stake_frac=0.5)
    conn = hdb.connect(cfg.db_path)
    spec = get_metric(cfg, "low")
    now = _at_commit(cfg, "low")
    price = 0.50
    # Engineering an exact-MARGIN case is a fixed point (p_model -> Kelly stake ->
    # contract count -> amortised fee -> EV), so walk p_model up in fine steps and
    # assert the flip happens exactly where EV crosses MARGIN, never below it.
    prev = None
    flipped = False
    for i in range(400):
        p_model = 0.50 + i * 0.001
        s = pick.score(conn, cfg, spec, now, out=_out(68, p_model, price))
        if s["verdict"] == pick.PICK:
            assert s["ev"] > pick.MARGIN, "published a pick at or below the bar"
            if prev is not None:
                assert prev["ev"] <= pick.MARGIN
                assert prev["verdict"] == pick.NO_PICK_EV
                flipped = True
            break
        if s["verdict"] == pick.NO_PICK_EV:
            assert s["ev"] <= pick.MARGIN, "declined a pick above the bar"
            prev = s
    assert flipped, "never crossed the bar — widen the sweep"


def test_a_clean_pick_publishes(tmp_path):
    _, _, s = _score(tmp_path, 68, 0.78, 0.62, bankroll=100.0, max_stake_frac=0.5)
    assert s["verdict"] == pick.PICK
    assert s["contracts"] >= 1
    assert s["ev"] > pick.MARGIN


def test_no_price_is_its_own_verdict(tmp_path):
    """37% of LOW days historically had no usable price -- distinguish that from
    a considered decline."""
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    spec = get_metric(cfg, "low")
    now = _at_commit(cfg, "low")
    s = pick.score(conn, cfg, spec, now,
                   out={"target": TARGET, "pred": {}, "market": []})
    assert s["verdict"] == pick.NO_PICK_NO_PRICE


def test_unscored_when_the_station_has_no_measured_calibration(tmp_path):
    """No cross-station fallback: an unmeasured market must degrade visibly."""
    cfg = _cfg(tmp_path)
    object.__setattr__(cfg.station, "icao", "KDEN")
    conn = hdb.connect(cfg.db_path)
    spec = get_metric(cfg, "low")
    now = _at_commit(cfg, "low")
    s = pick.score(conn, cfg, spec, now, out=_out(68, 0.78, 0.62))
    assert s["verdict"] == pick.UNSCORED
    assert s["p_hat"] is None


# ── the calibration shading ─────────────────────────────────────────────────

def test_p_hat_is_shaded_by_the_measured_offset(tmp_path):
    from highcast.gating import hour_calib
    cal = hour_calib("KLAX", "low", 6)
    _, _, s = _score(tmp_path, 68, 0.78, 0.62, bankroll=100.0, max_stake_frac=0.5)
    assert s["p_hat"] == pytest.approx(0.78 - cal)


def test_no_band_penalty_is_applied(tmp_path):
    """The >=80F band adjustment was refuted (+8.6% [-0.2,+17.9] vs +8.9%
    overall). A high reading must be shaded by the OVERALL offset only -- if this
    fails, the fudge has crept back."""
    from highcast.gating import hour_calib
    cfg = _cfg(tmp_path, bankroll=100.0, max_stake_frac=0.5)
    conn = hdb.connect(cfg.db_path)
    spec = get_metric(cfg, "high")
    now = _at_commit(cfg, "high")
    s = pick.score(conn, cfg, spec, now, out=_out(82, 0.80, 0.72, point=82.5))
    assert s["p_hat"] == pytest.approx(0.80 - hour_calib("KLAX", "high", 13))


def test_the_hard_band_appears_as_a_note_instead(tmp_path):
    """What survives of the >=80 finding: a display flag, not arithmetic."""
    assert pick.hard_band_note("KLAX", "high", 82.5) is not None
    assert "42%" in pick.hard_band_note("KLAX", "high", 82.5)
    assert pick.hard_band_note("KLAX", "high", 74.0) is None
    assert pick.hard_band_note("KMIA", "high", 95.0) is None   # KLAX-only finding


# ── stability, provisional and self-measuring ───────────────────────────────

def _seed_hours(conn, cfg, metric, hour_to_lo, tz_hour_base=0):
    """One intraday prediction per station-local hour, each favouring `lo`."""
    from zoneinfo import ZoneInfo
    tz = ZoneInfo(cfg.station.tz)
    rows = []
    for h, lo in hour_to_lo.items():
        issued = dt.datetime(2026, 7, 15, h, 5, tzinfo=tz).astimezone(dt.timezone.utc)
        pmf = {str(lo): 0.6, str(lo + 1): 0.3, str(lo + 2): 0.1}
        rows.append({"station": cfg.station.icao, "target_date": TARGET,
                     "issued_at_utc": issued.isoformat(), "kind": "intraday",
                     "metric": metric, "point_f": float(lo),
                     "quantiles": json.dumps({"0.5": float(lo)}),
                     "buckets": json.dumps(pmf)})
    hdb.upsert(conn, "predictions", rows)


def test_churn_window_ends_at_the_commit_hour(tmp_path):
    """LEAKAGE GUARD. The desk wobbles 68 -> 70 -> 72 but only the first two are
    at or before the commit hour (KLAX low h4). Churn must be 2, not 3."""
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    _seed_hours(conn, cfg, "low", {2: 68, 4: 70, 6: 72})
    got = pick.churn_to_commit(conn, cfg, TARGET, "low", 4, parity=0)
    assert got == 2, "hours after the commit leaked into churn"
    # and without the guard it WOULD be 3 -- proving the assertion bites
    assert pick.churn_to_commit(conn, cfg, TARGET, "low", 23, parity=0) == 3


def test_parity_follows_the_live_ladder(tmp_path):
    """69 and 70 are the SAME bucket on the odd ladder (69-70) and DIFFERENT
    buckets on the even one (68-69, 70-71). Churn must reflect the real ladder."""
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    _seed_hours(conn, cfg, "low", {2: 69, 3: 70})
    assert pick.churn_to_commit(conn, cfg, TARGET, "low", 4, parity=1) == 1
    assert pick.churn_to_commit(conn, cfg, TARGET, "low", 4, parity=0) == 2


def test_both_scores_are_stored_so_stability_measures_itself(tmp_path):
    cfg = _cfg(tmp_path, bankroll=100.0, max_stake_frac=0.5)
    conn = hdb.connect(cfg.db_path)
    spec = get_metric(cfg, "low")
    _seed_hours(conn, cfg, "low", {2: 68, 3: 70, 4: 68})
    now = _at_commit(cfg, "low")
    pick.record(conn, cfg, spec, now, "verdict", out=_out(68, 0.78, 0.62))
    r = conn.execute("SELECT * FROM picks").fetchone()
    assert r["score_without"] is not None
    assert r["score_with_stability"] is not None
    assert r["churn"] == 2
    # the stability term must not silently equal the plain score, or logging both
    # proves nothing
    assert r["score_with_stability"] != r["score_without"]


# ── grading and the kill criterion ──────────────────────────────────────────

def test_settle_grades_wins_and_losses(tmp_path):
    cfg = _cfg(tmp_path, bankroll=100.0, max_stake_frac=0.5)
    conn = hdb.connect(cfg.db_path)
    spec = get_metric(cfg, "low")
    now = _at_commit(cfg, "low")
    pick.record(conn, cfg, spec, now, "verdict", out=_out(68, 0.78, 0.62))
    hdb.upsert(conn, "cli_reports", [{
        "station": cfg.station.icao, "date": TARGET, "high_f": 80, "low_f": 69,
        "raw": json.dumps({"product": _final_product(cfg, TARGET)}),
        "fetched_at": "x"}])
    assert pick.settle(conn, cfg, now) == 1
    r = conn.execute("SELECT actual, won FROM picks").fetchone()
    assert r["actual"] == 69 and r["won"] == 1     # 69 is inside 68-69


def test_settle_records_a_loss_as_a_loss(tmp_path):
    cfg = _cfg(tmp_path, bankroll=100.0, max_stake_frac=0.5)
    conn = hdb.connect(cfg.db_path)
    spec = get_metric(cfg, "low")
    now = _at_commit(cfg, "low")
    pick.record(conn, cfg, spec, now, "verdict", out=_out(68, 0.78, 0.62))
    hdb.upsert(conn, "cli_reports", [{
        "station": cfg.station.icao, "date": TARGET, "high_f": 80, "low_f": 72,
        "raw": json.dumps({"product": _final_product(cfg, TARGET)}),
        "fetched_at": "x"}])
    pick.settle(conn, cfg, now)
    assert conn.execute("SELECT won FROM picks").fetchone()["won"] == 0


def test_settle_refuses_a_preliminary_cli_product(tmp_path):
    """A CLI product issued before the climate day closes must not settle a
    pick -- see scripts/migrate_prelim_settlement.py for the KMIA 2026-08-01
    incident this guards against. The pick stays UNGRADED and is retried on
    the next settle() call once a final product lands; never dropped."""
    cfg = _cfg(tmp_path, bankroll=100.0, max_stake_frac=0.5)
    conn = hdb.connect(cfg.db_path)
    spec = get_metric(cfg, "low")
    now = _at_commit(cfg, "low")
    pick.record(conn, cfg, spec, now, "verdict", out=_out(68, 0.78, 0.62))
    hdb.upsert(conn, "cli_reports", [{
        "station": cfg.station.icao, "date": TARGET, "high_f": 80, "low_f": 69,
        "raw": json.dumps({"product": _early_product(cfg, TARGET)}),
        "fetched_at": "x"}])
    assert pick.settle(conn, cfg, now) == 0
    r = conn.execute("SELECT actual, won, settled_utc FROM picks").fetchone()
    assert r["actual"] is None and r["won"] is None and r["settled_utc"] is None

    # A later wake with a FINAL product settles it normally -- retried, not lost.
    hdb.upsert(conn, "cli_reports", [{
        "station": cfg.station.icao, "date": TARGET, "high_f": 80, "low_f": 69,
        "raw": json.dumps({"product": _final_product(cfg, TARGET)}),
        "fetched_at": "y"}])
    assert pick.settle(conn, cfg, now) == 1
    r = conn.execute("SELECT actual, won FROM picks").fetchone()
    assert r["actual"] == 69 and r["won"] == 1


def test_graded_record_reports_realised_ev_not_win_rate(tmp_path):
    """A bar can post a fine win rate and still lose money -- the whole critique
    of nowcast.trade's 85%. The record must lead with realised EV."""
    cfg = _cfg(tmp_path, bankroll=100.0, max_stake_frac=0.5)
    conn = hdb.connect(cfg.db_path)
    spec = get_metric(cfg, "low")
    now = _at_commit(cfg, "low")
    pick.record(conn, cfg, spec, now, "verdict", out=_out(68, 0.78, 0.62))
    hdb.upsert(conn, "cli_reports", [{
        "station": cfg.station.icao, "date": TARGET, "high_f": 80, "low_f": 69,
        "raw": json.dumps({"product": _final_product(cfg, TARGET)}),
        "fetched_at": "x"}])
    pick.settle(conn, cfg, now)
    rec = pick.graded_record(conn, cfg)
    assert rec["wins"] == 1 and rec["losses"] == 0
    # Read the fee back rather than assuming a 1-contract $0.02: it is amortised
    # over the order, which is the point of `fees.fee_per_contract`.
    row = conn.execute("SELECT fee, price FROM picks").fetchone()
    # graded_record rounds to 4dp for display, so compare at that resolution.
    assert rec["realised_ev"] == pytest.approx(
        1.0 - row["price"] - row["fee"], abs=1e-4)
    assert rec["kill_triggered"] is None          # not enough picks to judge
    assert rec["kill_after"] == pick.KILL_AFTER_PICKS


def test_kill_criterion_is_preregistered_and_not_yet_triggerable(tmp_path):
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    rec = pick.graded_record(conn, cfg)
    assert rec["graded"] == 0
    assert rec["kill_triggered"] is None
    assert "priors, not measurements" in rec["note"]


def test_thresholds_are_documented_as_priors():
    """If either becomes a measured value this assertion should be updated
    deliberately, with the measurement cited."""
    assert pick.PRICE_CEILING == 0.75
    assert pick.MARGIN == 0.05
    assert "PRIORS" in pick.__doc__ or "priors" in pick.__doc__


# ── single-pick selection (pair deferred to post-test-week, 2026-07-30) ─────
#
# Julian reverted to ONE pick a day and then deferred the whole feature until
# after the test week. The module stays dormant -- nothing in cli/notify/web
# imports it -- but it is kept internally coherent so the suite stays green.

def _publish(conn, cfg, metric, bucket_lo, p_model, price, hour=None):
    """`hour` is now an OFFSET past the commit hour, not a UTC clock hour."""
    spec = get_metric(cfg, metric)
    now = _at_commit(cfg, metric, plus=1 if hour is None else 1)
    return pick.record(conn, cfg, spec, now, "verdict",
                       out=_out(bucket_lo, p_model, price))


def test_one_pick_per_day():
    assert pick.MAX_PICKS_PER_DAY == 1


def test_a_later_stronger_candidate_wins_the_slot(tmp_path):
    """The defect one slot exposes: KMIA LOW commits at 17:00 Manila, nine hours
    before the last candidate. Arrival order must NOT decide the pick."""
    cfg = _cfg(tmp_path, bankroll=100.0, max_stake_frac=0.5)
    conn = hdb.connect(cfg.db_path)
    _publish(conn, cfg, "low", 68, 0.74, 0.62)    # early, modest EV
    # Priced 65c, NOT 58c. At 58c this candidate's calibrated edge was 24.6
    # points — within 0.4 of `MAX_PLAUSIBLE_EDGE`, so a 0.65-point improvement
    # to KLAX HIGH's calibration on 2026-07-31 tipped it over and the guard
    # vetoed it, making this test fail for a reason it was never about. The
    # point here is that ARRIVAL ORDER must not decide the pick; keep the
    # fixture well clear of the edge cap so it tests only that.
    _publish(conn, cfg, "high", 80, 0.90, 0.65)   # later, stronger
    chosen = pick.day_pick(conn, TARGET)
    assert chosen["metric"] == "high"
    assert [s["metric"] for s in chosen["superseded"]] == ["low"]


def test_nothing_is_rewritten_when_a_pick_is_superseded(tmp_path):
    """Selection is a READ. The superseded row keeps its own EV on record, so
    upsert_ignore's first-write-wins anti-cheat contract still holds."""
    cfg = _cfg(tmp_path, bankroll=100.0, max_stake_frac=0.5)
    conn = hdb.connect(cfg.db_path)
    early = _publish(conn, cfg, "low", 68, 0.74, 0.62)
    _publish(conn, cfg, "high", 80, 0.90, 0.65)   # off the edge cap, see above
    row = conn.execute(
        "SELECT verdict, ev FROM picks WHERE metric='low'").fetchone()
    assert row["verdict"] == pick.PICK
    assert row["ev"] == pytest.approx(early["ev"])


def test_day_pick_is_none_when_nothing_qualifies(tmp_path):
    cfg = _cfg(tmp_path, bankroll=100.0, max_stake_frac=0.5)
    conn = hdb.connect(cfg.db_path)
    _publish(conn, cfg, "low", 68, 0.60, 0.58)   # below the bar
    assert pick.day_pick(conn, TARGET) is None


def test_near_ties_break_on_measured_reliability(tmp_path):
    """Where EV cannot separate two candidates, prefer the market historically
    more often right.

    2026-08-01: at the OLD hours KLAX low (62.4% at h4) was more reliable than
    KLAX high (52.4% at h11), so the tiebreak favoured LOW. At the NEW commit
    hours the ranking FLIPPED: KLAX high now measures 69.5% at h13, versus
    KLAX low's 68.2% at h6 (gating.HOUR_ACCURACY) -- HIGH is now (marginally)
    the more reliable of the pair. Re-engineered the fixture so LOW leads on
    raw EV by less than the tiebreak window, and confirm HIGH's better measured
    track record still pulls the win away from it -- it is the MECHANISM this
    test protects, not which market's name comes out on top.
    """
    from highcast.gating import hour_calib
    cfg = _cfg(tmp_path, bankroll=100.0, max_stake_frac=0.5)
    conn = hdb.connect(cfg.db_path)
    # The two metrics carry DIFFERENT calibration offsets (-0.0097 low vs
    # +0.0145 high), so equal p_model values land several EV points apart.
    # Offset the HIGH's p_model by its extra shading, then nudge it BEHIND by
    # less than the tiebreak window, so LOW leads on raw EV and reliability
    # must pull HIGH back in front.
    # Prices must MATCH too -- a mismatch alone puts the EVs points apart
    # regardless of calibration, which is what broke an earlier draft.
    PRICE = 0.62
    p_low = 0.78
    p_high = (p_low - hour_calib("KLAX", "low", 6)
              + hour_calib("KLAX", "high", 13) - 0.005)
    _publish(conn, cfg, "low", 68, p_low, PRICE)
    _publish(conn, cfg, "high", 80, p_high, PRICE)
    a = conn.execute("SELECT ev FROM picks WHERE metric='low'").fetchone()["ev"]
    b = conn.execute("SELECT ev FROM picks WHERE metric='high'").fetchone()["ev"]
    assert abs(a - b) <= pick.EV_TIEBREAK_WINDOW, "fixture is not a near-tie"
    assert a > b, "fixture must have LOW leading on raw EV for this to test anything"
    chosen = pick.day_pick(conn, TARGET)
    assert chosen["metric"] == "high", "tiebreak ignored measured reliability"


def test_graded_record_counts_supersessions(tmp_path):
    cfg = _cfg(tmp_path, bankroll=100.0, max_stake_frac=0.5)
    conn = hdb.connect(cfg.db_path)
    _publish(conn, cfg, "low", 68, 0.74, 0.62)
    _publish(conn, cfg, "high", 80, 0.90, 0.65)   # off the edge cap, see above
    rec = pick.graded_record(conn, cfg)
    assert rec["picks_per_day"] == 1
    assert rec["days_a_later_candidate_won_the_slot"] == 1


# ── guards found by running the module live, 2026-07-30 ─────────────────────

class _Vetoed:
    veto = True
    reason = "no-morning-obs regime"


def test_a_vetoed_read_is_never_priced(tmp_path):
    """A veto means the PROBABILITY is untrustworthy, which is categorically
    different from 'no edge'. appbet may ignore place=False; nothing may ignore
    this. Found live: pick.py was ignoring it."""
    cfg = _cfg(tmp_path, bankroll=100.0, max_stake_frac=0.5)
    conn = hdb.connect(cfg.db_path)
    spec = get_metric(cfg, "low")
    now = _at_commit(cfg, "low")
    out = {**_out(68, 0.78, 0.62), "decision": _Vetoed()}
    s = pick.score(conn, cfg, spec, now, out=out)
    assert s["verdict"] == pick.NO_PICK_VETOED
    assert s["ev"] is None, "a vetoed read must not even be scored"


def test_nothing_publishes_before_the_commit_hour(tmp_path):
    """KLAX LOW commits at h6 (moved from h4 on 2026-07-31, see
    gating.MIN_DECISION_HOUR). At h1 the measured hit rate is 38.4% (n=194) --
    the worst hour on the curve. Found live: it returned PICK at h1.

    This is the ONE test in the suite whose whole subject IS the commit hour,
    so -- unlike the fixtures elsewhere in this file -- the hour is not
    incidental here and must track the NEW knee, not just move past it."""
    cfg = _cfg(tmp_path, bankroll=100.0, max_stake_frac=0.5)
    conn = hdb.connect(cfg.db_path)
    spec = get_metric(cfg, "low")
    from zoneinfo import ZoneInfo
    tz = ZoneInfo(cfg.station.tz)
    early = dt.datetime(2026, 7, 15, 1, 30, tzinfo=tz).astimezone(dt.timezone.utc)
    s = pick.score(conn, cfg, spec, early, out=_out(68, 0.78, 0.62))
    assert s["verdict"] == pick.NO_PICK_TOO_EARLY
    # ...and at/after the (new) commit hour the same read is scorable again
    late = dt.datetime(2026, 7, 15, 6, 0, tzinfo=tz).astimezone(dt.timezone.utc)
    assert pick.score(conn, cfg, spec, late,
                      out=_out(68, 0.78, 0.62))["verdict"] == pick.PICK


def test_an_implausible_edge_is_refused_not_published(tmp_path):
    """The live failure: bucket '<=60' at 52% against a 1c market, a +51 point
    edge on a station whose lows run 69F. Four times on real money an edge this
    size has meant a broken model.

    `now` moved from h5 -> h6: h5 was past KLAX LOW's OLD h4 knee but is BEFORE
    the NEW h6 knee, so this guard's own subject (the edge check) never used to
    be about the commit hour, and the fixture just needs to sit past the new
    knee to keep testing that."""
    cfg = _cfg(tmp_path, bankroll=100.0, max_stake_frac=0.5)
    conn = hdb.connect(cfg.db_path)
    spec = get_metric(cfg, "low")
    from zoneinfo import ZoneInfo
    now = dt.datetime(2026, 7, 15, 6, 0,
                      tzinfo=ZoneInfo(cfg.station.tz)).astimezone(dt.timezone.utc)
    s = pick.score(conn, cfg, spec, now, out=_out(60, 0.52, 0.01))
    assert s["verdict"] == pick.NO_PICK_ABSURD_EDGE
    assert "broken model" in s["reason"]


def test_a_normal_edge_still_publishes(tmp_path):
    """The guard must not swallow ordinary opportunities -- only absurd ones.
    `now` moved from h5 -> h6 for the same reason as the implausible-edge test
    above: h5 is now before KLAX LOW's new h6 knee."""
    cfg = _cfg(tmp_path, bankroll=100.0, max_stake_frac=0.5)
    conn = hdb.connect(cfg.db_path)
    spec = get_metric(cfg, "low")
    from zoneinfo import ZoneInfo
    now = dt.datetime(2026, 7, 15, 6, 0,
                      tzinfo=ZoneInfo(cfg.station.tz)).astimezone(dt.timezone.utc)
    s = pick.score(conn, cfg, spec, now, out=_out(68, 0.78, 0.62))
    assert (s["p_hat"] - s["price"]) < pick.MAX_PLAUSIBLE_EDGE
    assert s["verdict"] == pick.PICK


def test_churn_uses_chronological_cutoff_not_local_hour(tmp_path):
    """AUDIT 2026-07-30: the old guard compared local_hour > commit_hour, which
    leaked any prediction issued on a LATER date at a small local hour (the hour
    wraps midnight; the proxy can't see the date). Measured 1 such row per KLAX
    market. This pins the chronological fix."""
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    from zoneinfo import ZoneInfo
    tz = ZoneInfo(cfg.station.tz)
    # KLAX low commits h4. Seed: two pre-commit buckets on the target date, then a
    # LATER-DATE row at local hour 2 (< commit hour 4) favouring a third bucket.
    rows = []
    for h, lo, day in [(2, 68, "2026-07-15"), (3, 70, "2026-07-15"),
                       (2, 74, "2026-07-16")]:  # <- next day, small hour: the trap
        issued = dt.datetime(int(day[:4]), 7, int(day[8:]), h, 5,
                             tzinfo=tz).astimezone(dt.timezone.utc)
        rows.append({"station": cfg.station.icao, "target_date": "2026-07-15",
                     "issued_at_utc": issued.isoformat(), "kind": "intraday",
                     "metric": "low", "point_f": float(lo),
                     "quantiles": json.dumps({"0.5": float(lo)}),
                     "buckets": json.dumps({str(lo): 0.6, str(lo + 1): 0.4})})
    hdb.upsert(conn, "predictions", rows)
    # The next-day row must be EXCLUDED: churn is 2 (68-69, 70-71), not 3.
    got = pick.churn_to_commit(conn, cfg, "2026-07-15", "low", 4, parity=0)
    assert got == 2, "a later-date prediction leaked into pre-commit churn"
