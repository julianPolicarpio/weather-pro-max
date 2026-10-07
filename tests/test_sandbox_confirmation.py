"""Fresh evidence windows, report segmentation and read-only surfaces."""
from contextlib import closing
import json
import logging
import sqlite3

import pytest
from sandbox import scoreboard as sb
from sandbox.db import connect
from sandbox.cli import main
from sandbox.cycle import ensure_accounts
from test_sandbox_scoreboard import _settled_events, T


def test_confirmation_passes_two_disjoint_windows_and_stays_unlocked():
    with closing(connect(':memory:')) as c:
        _settled_events(c, 100, .9, True)
        assert sb.confirmation_status(c, 'mos')['state'] == 'LOCKED'
        sb.advance_confirmation(c, 'mos', T)
        s = sb.confirmation_status(c, 'mos')
        assert s['state'] == 'PROVISIONAL' and s['marker_id'] == 100
        assert s['fresh_settled_bets'] == 0
        _settled_events(c, 99, .9, True)
        sb.advance_confirmation(c, 'mos', T)
        assert sb.confirmation_status(c, 'mos')['state'] == 'PROVISIONAL'
        _settled_events(c, 1, .9, True)
        sb.advance_confirmation(c, 'mos', T)
        assert sb.confirmation_status(c, 'mos')['state'] == 'UNLOCKED'
        before = c.total_changes
        sb.advance_confirmation(c, 'mos', T)
        assert c.total_changes == before


def test_failed_window_restarts_after_all_existing_orders():
    with closing(connect(':memory:')) as c:
        _settled_events(c, 100, .9, True)
        sb.advance_confirmation(c, 'mos', T)
        _settled_events(c, 100, .9, False)
        sb.advance_confirmation(c, 'mos', T)
        s = sb.confirmation_status(c, 'mos')
        assert s['state'] == 'LOCKED' and s['attempt_start_id'] == 200
        assert s['resolved_at'] == T
        assert sb.report(c, 'mos', after_order_id=200).settled_bets == 0
        sb.advance_confirmation(c, 'mos', T)
        assert sb.confirmation_status(c, 'mos') == s


def test_confirmation_waits_for_event_floor():
    with closing(connect(':memory:')) as c:
        _settled_events(c, 100, .9, True)
        sb.advance_confirmation(c, 'mos', T)
        _settled_events(c, 20, .9, True, snapshots=5)
        sb.advance_confirmation(c, 'mos', T)
        s = sb.confirmation_status(c, 'mos')
        assert s['state'] == 'PROVISIONAL'
        assert (s['fresh_settled_bets'], s['fresh_n_events']) == (100, 20)


def test_segments_boundary_unknown_station_and_read_only():
    with closing(connect(':memory:')) as c:
        _settled_events(c, 3, .9, True)
        c.execute("UPDATE orders SET decided_at='2026-10-04T04:59:59+00:00' WHERE id=1")
        c.execute("UPDATE orders SET decided_at='2026-10-04T05:00:00+00:00' WHERE id=2")
        c.execute("UPDATE markets SET station=NULL WHERE event_ticker='EV2'")
        c.commit()
        c.execute('PRAGMA query_only=ON')
        before = c.total_changes
        pre = sb.report(c, 'mos', segment='pre_day')
        same = sb.report(c, 'mos', segment='same_day')
        total = sb.report(c, 'mos')
        sb.confirmation_status(c, 'mos')
        assert (pre.settled_bets, same.settled_bets) == (1, 2)
        assert pre.profit + same.profit == pytest.approx(total.profit)
        assert c.total_changes == before
        with pytest.raises(ValueError):
            sb.report(c, 'mos', segment='typo')


def test_cli_verdict_is_persisted_and_database_unchanged(tmp_path, capsys):
    p = tmp_path / 'sandbox.db'
    with closing(connect(p)) as c:
        ensure_accounts(c)
        _settled_events(c, 100, .9, True)
    before = p.read_bytes()
    assert main(['--db', str(p), 'report']) == 0
    out = capsys.readouterr().out
    assert 'UNLOCKED' not in out and 'mos       LOCKED' in out
    assert 'pre_day' in out and 'same_day' in out
    assert p.read_bytes() == before


def test_readonly_connection_does_not_migrate_or_create(tmp_path):
    from sandbox.db import connect_readonly
    p = tmp_path / 'old.db'
    with closing(sqlite3.connect(p)) as c:
        c.execute('CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)')
    before = p.read_bytes()
    with closing(connect_readonly(p)) as c:
        assert c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()[0][0] == 'meta'
        with pytest.raises(sqlite3.OperationalError):
            c.execute("INSERT INTO meta VALUES ('x','y')")
    assert p.read_bytes() == before
    missing = tmp_path / 'missing.db'
    with pytest.raises(sqlite3.OperationalError):
        connect_readonly(missing)
    assert not missing.exists()


def _payouts(c, upto):
    return [r[0] for r in c.execute("SELECT payout FROM orders WHERE id <= ? ORDER BY id", (upto,))]


def test_provisional_waits_for_the_scored_floors_then_unlocks():
    """I1: two filled buckets per (snapshot, event) give 100 settled bets but only 50
    scored samples. The old trigger graded that window at once and failed it on the floor."""
    with closing(connect(':memory:')) as c:
        _settled_events(c, 100, .9, True)
        sb.advance_confirmation(c, 'mos', T)
        assert sb.confirmation_status(c, 'mos')['state'] == 'PROVISIONAL'
        _settled_events(c, 50, .9, True, buckets=2, b_price=.10)
        assert sb.advance_confirmation(c, 'mos', T) is None
        s = sb.confirmation_status(c, 'mos')
        assert s['state'] == 'PROVISIONAL' and s['resolved_at'] is None
        assert (s['fresh_settled_bets'], s['fresh_n_events']) == (100, 50)
        assert (s['fresh_brier_n'], s['fresh_brier_events']) == (50, 50)
        _settled_events(c, 50, .9, True, buckets=2, b_price=.10)
        assert sb.advance_confirmation(c, 'mos', T) == ('PROVISIONAL', 'UNLOCKED')
        s = sb.confirmation_status(c, 'mos')
        assert s['state'] == 'UNLOCKED' and s['resolved_at'] == T
        assert (s['fresh_brier_n'], s['fresh_brier_events']) == (0, 0)   # zero outside PROVISIONAL


def test_losing_fresh_window_is_evaluated_once_the_scored_floors_are_met():
    with closing(connect(':memory:')) as c:
        _settled_events(c, 100, .9, True)
        sb.advance_confirmation(c, 'mos', T)
        _settled_events(c, 50, .9, False, buckets=2)       # A -0.52, B +0.48 at 0.50 each
        assert sb.advance_confirmation(c, 'mos', T) is None   # 100 settled, 50 scored: wait
        s = sb.confirmation_status(c, 'mos')
        assert s['state'] == 'PROVISIONAL' and s['resolved_at'] is None
        _settled_events(c, 50, .9, False, buckets=2)
        assert sb.advance_confirmation(c, 'mos', T) == ('PROVISIONAL', 'LOCKED')
        s = sb.confirmation_status(c, 'mos')
        assert s['state'] == 'LOCKED' and s['resolved_at'] == T
        assert s['attempt_start_id'] == 300 and s['failed_attempts'] == 1


def test_a_window_without_candidates_keeps_waiting_instead_of_failing():
    with closing(connect(':memory:')) as c:
        _settled_events(c, 100, .9, True)
        sb.advance_confirmation(c, 'mos', T)
        _settled_events(c, 100, .9, True)
        c.execute('DELETE FROM candidates WHERE snapshot_id > 100')
        c.commit()
        assert sb.advance_confirmation(c, 'mos', T) is None
        s = sb.confirmation_status(c, 'mos')
        assert s['state'] == 'PROVISIONAL' and s['resolved_at'] is None
        assert (s['fresh_settled_bets'], s['fresh_brier_n']) == (100, 0)


def test_fresh_losers_lock_although_the_whole_history_would_pass():
    """M1: 400 stage-1 winners then 100 fresh losers. A verdict that read the whole history
    would still unlock, so only a window-isolated verdict can fail this."""
    with closing(connect(':memory:')) as c:
        _settled_events(c, 400, .9, True)
        assert sb.advance_confirmation(c, 'mos', T) == ('LOCKED', 'PROVISIONAL')
        _settled_events(c, 100, .9, False)
        assert sb.report(c, 'mos').unlocked          # the history alone would pass
        assert sb.advance_confirmation(c, 'mos', T) == ('PROVISIONAL', 'LOCKED')
        assert sb.confirmation_status(c, 'mos')['state'] == 'LOCKED'
        assert _payouts(c, 400) == [1.0] * 400       # the later helper call rewrote nothing


def test_new_attempt_after_a_failure_passes_stage_one_on_new_bets_only():
    with closing(connect(':memory:')) as c:
        _settled_events(c, 100, .9, True)
        sb.advance_confirmation(c, 'mos', T)
        _settled_events(c, 100, .9, False)
        sb.advance_confirmation(c, 'mos', T)
        assert sb.confirmation_status(c, 'mos')['attempt_start_id'] == 200
        _settled_events(c, 100, .9, True)
        assert not sb.report(c, 'mos').unlocked      # old losers would drag a whole-history grade down
        assert sb.advance_confirmation(c, 'mos', T) == ('LOCKED', 'PROVISIONAL')
        s = sb.confirmation_status(c, 'mos')
        assert s['state'] == 'PROVISIONAL' and s['marker_id'] == 300
        assert _payouts(c, 200) == [1.0] * 100 + [0.0] * 100


# ---- M3: a window counts only city-days that are new to it -------------------

def test_new_events_only_drops_events_already_bet_at_or_before_the_marker():
    with closing(connect(':memory:')) as c:
        _settled_events(c, 3, .9, True)                      # EV0..EV2, order ids 1..3
        _settled_events(c, 3, .9, True, start=1)             # EV1, EV2 again, EV3 new: ids 4..6
        fresh = sb.report(c, 'mos', after_order_id=3, new_events_only=True)
        assert (fresh.settled_bets, fresh.n_events, fresh.brier_n, fresh.brier_events) == (1, 1, 1, 1)
        assert fresh.profit == pytest.approx(.48)
        plain = sb.report(c, 'mos', after_order_id=3)
        assert (plain.settled_bets, plain.n_events) == (3, 3)
        assert sb.report(c, 'mos').settled_bets == 6          # lifetime scorecard unaffected
        assert sb.report(c, 'mos', segment='same_day').settled_bets == 6
        with pytest.raises(ValueError):
            sb.report(c, 'mos', new_events_only=True)


def test_provisional_window_ignores_new_bets_on_events_the_first_window_used():
    with closing(connect(':memory:')) as c:
        _settled_events(c, 100, .9, True)
        sb.advance_confirmation(c, 'mos', T)
        _settled_events(c, 100, .9, True, start=0)           # same 100 city-days again
        assert sb.advance_confirmation(c, 'mos', T) is None
        s = sb.confirmation_status(c, 'mos')
        assert s['state'] == 'PROVISIONAL'
        assert (s['fresh_settled_bets'], s['fresh_n_events'], s['fresh_brier_n']) == (0, 0, 0)
        _settled_events(c, 100, .9, True)                    # 100 genuinely new city-days
        assert sb.advance_confirmation(c, 'mos', T) == ('PROVISIONAL', 'UNLOCKED')


def test_a_new_attempt_after_a_failure_ignores_events_of_the_failed_attempt():
    with closing(connect(':memory:')) as c:
        _settled_events(c, 100, .9, True)
        sb.advance_confirmation(c, 'mos', T)
        _settled_events(c, 100, .9, False)                   # EV100..EV199
        sb.advance_confirmation(c, 'mos', T)
        assert sb.confirmation_status(c, 'mos')['state'] == 'LOCKED'
        _settled_events(c, 100, .9, False, start=100)        # the failed attempt's events, again
        assert sb.advance_confirmation(c, 'mos', T) is None
        _settled_events(c, 100, .9, True)
        assert sb.advance_confirmation(c, 'mos', T) == ('LOCKED', 'PROVISIONAL')


# ---- M4: attempt history is visible ------------------------------------------

def _failed_attempt(c):
    """Stage 1 passes on 100 winners, then 100 fresh losers fail the confirmation window."""
    _settled_events(c, 100, .9, True)
    sb.advance_confirmation(c, 'mos', T)
    _settled_events(c, 100, .9, False)
    return sb.advance_confirmation(c, 'mos', T)


def test_failed_attempts_counter_survives_later_transitions():
    with closing(connect(':memory:')) as c:
        assert sb.confirmation_status(c, 'mos')['failed_attempts'] == 0
        assert _failed_attempt(c) == ('PROVISIONAL', 'LOCKED')
        assert sb.confirmation_status(c, 'mos')['failed_attempts'] == 1
        _settled_events(c, 100, .9, True)
        sb.advance_confirmation(c, 'mos', T)                 # PROVISIONAL again
        s = sb.confirmation_status(c, 'mos')
        assert s['state'] == 'PROVISIONAL' and s['failed_attempts'] == 1
        _settled_events(c, 100, .9, False)
        assert sb.advance_confirmation(c, 'mos', T) == ('PROVISIONAL', 'LOCKED')
        s = sb.confirmation_status(c, 'mos')
        assert s['failed_attempts'] == 2
        assert 'confirmation failed 2026-10-04 18:00 Manila (2 failed attempts)' in sb.verdict_text(s)
        _settled_events(c, 100, .9, True)
        sb.advance_confirmation(c, 'mos', T)
        _settled_events(c, 100, .9, True)
        assert sb.advance_confirmation(c, 'mos', T) == ('PROVISIONAL', 'UNLOCKED')
        s = sb.confirmation_status(c, 'mos')
        assert s['failed_attempts'] == 2
        assert sb.verdict_text(s) == 'UNLOCKED (after 2 failed attempts)'


def test_a_stored_confirmation_without_the_counter_reads_as_zero():
    from sandbox.db import set_meta
    with closing(connect(':memory:')) as c:
        set_meta(c, 'confirm:mos', json.dumps({
            'state': 'UNLOCKED', 'attempt_start_id': 0, 'marker_id': 5,
            'provisional_at': T, 'resolved_at': T}))
        s = sb.confirmation_status(c, 'mos')
        assert s['failed_attempts'] == 0 and sb.verdict_text(s) == 'UNLOCKED'


def test_transitions_are_returned_and_logged(caplog):
    caplog.set_level(logging.INFO, logger='sandbox.scoreboard')
    with closing(connect(':memory:')) as c:
        _settled_events(c, 100, .9, True)
        assert sb.advance_confirmation(c, 'mos', T) == ('LOCKED', 'PROVISIONAL')
        assert sb.advance_confirmation(c, 'mos', T) is None          # nothing new: no transition
        _settled_events(c, 100, .9, False)
        assert sb.advance_confirmation(c, 'mos', T) == ('PROVISIONAL', 'LOCKED')
    lines = [r.getMessage() for r in caplog.records if r.name == 'sandbox.scoreboard']
    assert len(lines) == 2
    assert 'mos' in lines[0] and 'LOCKED' in lines[0] and 'PROVISIONAL' in lines[0]
    assert 'PROVISIONAL' in lines[1] and 'LOCKED' in lines[1]
    assert '100' in lines[1]                                         # the fresh counts


def test_verdict_text_wording():
    base = {'state': 'LOCKED', 'resolved_at': None, 'failed_attempts': 0}
    assert sb.verdict_text(base) == 'LOCKED'
    assert sb.verdict_text({**base, 'state': 'UNLOCKED'}) == 'UNLOCKED'
    assert sb.verdict_text({**base, 'resolved_at': '2026-10-04T10:00:00+00:00',
                            'failed_attempts': 3}) == (
        'LOCKED — confirmation failed 2026-10-04 18:00 Manila (3 failed attempts)')
    assert sb.verdict_text({**base, 'state': 'PROVISIONAL', 'fresh_settled_bets': 7,
                            'fresh_n_events': 5, 'fresh_brier_n': 4,
                            'fresh_brier_events': 3}) == (
        'PROVISIONAL (fresh 7/100 settled, 5/30 events, 4/100 scored, 3/30 scored events)')


def test_cli_provisional_counts_and_failure_timestamp(tmp_path, capsys):
    p = tmp_path / 'sandbox.db'
    with closing(connect(p)) as c:
        ensure_accounts(c)
        _settled_events(c, 100, .9, True)
        sb.advance_confirmation(c, 'mos', T)
    main(['--db', str(p), 'report'])
    assert ('PROVISIONAL (fresh 0/100 settled, 0/30 events, 0/100 scored, 0/30 scored events)'
            in capsys.readouterr().out)
    with closing(connect(p)) as c:
        _settled_events(c, 100, .9, False)
        sb.advance_confirmation(c, 'mos', T)
    before = p.read_bytes()
    main(['--db', str(p), 'report'])
    assert ('LOCKED — confirmation failed 2026-10-04 18:00 Manila (1 failed attempt)'
            in capsys.readouterr().out)
    assert p.read_bytes() == before


def test_verdict_counts_failed_attempts_in_every_state():
    base = {'state': 'PROVISIONAL', 'resolved_at': None, 'fresh_settled_bets': 7,
            'fresh_n_events': 5, 'fresh_brier_n': 4, 'fresh_brier_events': 3}
    assert sb.verdict_text({**base, 'failed_attempts': 2}) == (
        'PROVISIONAL (fresh 7/100 settled, 5/30 events, 4/100 scored, 3/30 scored events, '
        'after 2 failed attempts)')
    assert sb.verdict_text({**base, 'failed_attempts': 1}).endswith(', after 1 failed attempt)')
    assert sb.verdict_text({'state': 'UNLOCKED', 'failed_attempts': 1}) == (
        'UNLOCKED (after 1 failed attempt)')
    assert sb.verdict_text({'state': 'LOCKED', 'resolved_at': '2026-10-04T10:00:00+00:00',
                            'failed_attempts': 1}).endswith('(1 failed attempt)')


def test_require_initialised_lets_real_database_errors_through():
    from sandbox.db import NotInitialised, require_initialised

    class Locked:
        def execute(self, *a):
            raise sqlite3.OperationalError('database is locked')

    with pytest.raises(sqlite3.OperationalError, match='locked'):
        require_initialised(Locked())
    with closing(sqlite3.connect(':memory:')) as blank, pytest.raises(NotInitialised):
        require_initialised(blank)                          # no accounts table: not initialised
