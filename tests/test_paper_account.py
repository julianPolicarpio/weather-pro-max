"""THE ACCEPTANCE TEST for the paper demo account.

`paper_account.unit_replay` must reproduce the archive's hand-computed
record EXACTLY: 14W-6L over 20 graded bets; rule-said-PLACE 3/8 won, net
+$0.015/contract; rule-said-SKIP 11/12 won, net +$1.990/contract. Those
numbers were computed directly from the real `app_bets` table on
2026-08-02 (see the task brief) — `ARCHIVE_ROWS` below is a byte-for-byte
copy of that table's 22 rows, not a synthetic fixture, so this test is the
actual reconciliation gate, not an approximation of it.

Every other test in this file exercises the rest of the module (opening
balance seeding, the accumulation plan's stage ladder and above-cap
override, and the commit-hour variant comparison) against small, purpose-
built fixtures.
"""
import dataclasses
import sqlite3

import pytest

from highcast import db, fees, paper_account
from highcast.config import load_config

KLAX = load_config("config/klax.yaml")
KMIA = load_config("config/kmia.yaml")


# ── the real archive, byte-for-byte (22 rows) ────────────────────────────────
# columns: station, metric, target_date, placed_utc, price, p_model, bucket,
#          reason, actual, won, settled_utc
ARCHIVE_ROWS = [
    ("KLAX", "low", "2026-07-27", "2026-07-27T11:04:28.807263+00:00", 0.66, 0.8781, "69–70", None, 69, 1, "2026-07-28T07:00:57+00:00"),
    ("KLAX", "high", "2026-07-27", "2026-07-27T18:04:31.706305+00:00", 0.74, 0.4692, "79–80", "no edge: 79–80 model 47% vs market 74% = -27% < 5% required — do not place", 79, 1, "2026-07-28T07:00:57+00:00"),
    ("KLAX", "low", "2026-07-28", "2026-07-28T11:00:52.561047+00:00", 0.80, 0.7376, "68–69", "no edge: 68–69 model 74% vs market 80% = -6% < 5% required — do not place", 69, 1, "2026-07-29T07:01:08+00:00"),
    ("KLAX", "high", "2026-07-28", "2026-07-28T18:05:08.276456+00:00", 0.69, 0.6136, "78–79", "no edge: 78–79 model 61% vs market 69% = -8% < 5% required — do not place", 79, 1, "2026-07-29T07:01:08+00:00"),
    ("KMIA", "low", "2026-07-29", "2026-07-29T09:02:16.714805+00:00", 0.62, 0.4435, "79–80", "no edge: 79–80 model 44% vs market 62% = -18% < 5% required — do not place", 79, 1, "2026-07-30T04:02:43+00:00"),
    ("KLAX", "low", "2026-07-29", "2026-07-29T11:03:15.459859+00:00", 0.58, 0.8120, "67–68", "PLACE 67–68: 81% vs market 58% (edge +23%) -> USD 0.40 = 5.0% of bankroll (hard-capped), ~0 contracts", 68, 1, "2026-07-30T08:48:11+00:00"),
    ("KMIA", "high", "2026-07-29", "2026-07-29T15:02:15.792928+00:00", 0.56, 0.6350, "94–95", "PLACE 94–95: 64% vs market 56% (edge +7%) -> USD 0.34 = 4.3% of bankroll, ~0 contracts", 94, 1, "2026-07-30T04:02:43+00:00"),
    ("KLAX", "high", "2026-07-29", "2026-07-29T18:01:42.219977+00:00", 0.40, 0.5951, "81–82", "PLACE 81–82: 60% vs market 40% (edge +20%) -> USD 0.40 = 5.0% of bankroll (hard-capped), ~1 contracts", 80, 0, "2026-07-30T08:48:11+00:00"),
    ("KMIA", "low", "2026-07-30", "2026-07-30T09:02:15.885317+00:00", 0.51, 0.5257, "80–81", "no edge: 80–81 model 53% vs market 51% = +2% < 5% required — do not place", 80, 1, "2026-07-31T04:50:21+00:00"),
    ("KLAX", "low", "2026-07-30", "2026-07-30T11:02:29.633077+00:00", 0.34, 0.8078, "≥69", "PLACE ≥69: 81% vs market 34% (edge +47%) -> USD 0.40 = 5.0% of bankroll (hard-capped), ~1 contracts", 67, 0, "2026-07-31T09:52:04+00:00"),
    ("KMIA", "high", "2026-07-30", "2026-07-30T15:02:19.590878+00:00", 0.47, 0.7221, "94–95", "PLACE 94–95: 72% vs market 47% (edge +25%) -> USD 0.40 = 5.0% of bankroll (hard-capped), ~0 contracts", 96, 0, "2026-07-31T04:50:21+00:00"),
    ("KLAX", "high", "2026-07-30", "2026-07-30T18:02:41.495372+00:00", 0.38, 0.8093, "76–77", "PLACE 76–77: 81% vs market 38% (edge +43%) -> USD 0.40 = 5.0% of bankroll (hard-capped), ~1 contracts", 78, 0, "2026-07-31T09:52:04+00:00"),
    ("KMIA", "low", "2026-07-31", "2026-07-31T09:48:53.206894+00:00", 0.11, 0.5560, "≥81", "PLACE ≥81: 56%->55% calibrated vs market 11% (edge +44%) -> USD 0.40 = 5.0% of bankroll (hard-capped), 3 contracts", 81, 1, "2026-08-01T02:52:23+00:00"),
    ("KLAX", "low", "2026-07-31", "2026-07-31T11:00:26.872806+00:00", 0.71, 0.4963, "67–68", "no edge: 67–68 model 50%->46% calibrated vs market 71% = -25% < 3% required — do not place", 67, 1, "2026-08-01T07:04:52+00:00"),
    ("KMIA", "high", "2026-07-31", "2026-07-31T15:02:17.623331+00:00", 0.55, 0.5000, "95–96", "no edge: 95–96 model 50%->48% calibrated vs market 55% = -7% < 3% required — do not place", 96, 1, "2026-08-01T02:52:23+00:00"),
    ("KLAX", "high", "2026-07-31", "2026-07-31T20:00:59.017253+00:00", 0.975, 0.7743, "80–81", "no edge: 80–81 model 77%->73% calibrated vs market 98% = -25% < 3% required — do not place", 81, 1, "2026-08-01T07:04:52+00:00"),
    ("KMIA", "low", "2026-08-01", "2026-08-01T11:02:14.436952+00:00", 0.64, 0.4242, "≥81", "no edge: ≥81 model 42%->45% calibrated vs market 64% = -19% < 3% required — do not place", 74, 0, "2026-08-02T04:02:24+00:00"),
    ("KLAX", "low", "2026-08-01", "2026-08-01T13:02:16.641414+00:00", 0.005, 0.7261, "68–69", "PLACE 68–69 — edge +73%, but bankroll is unset (config staking.bankroll=0): set it to get an amount", 67, 0, "2026-08-02T07:00:25+00:00"),
    ("KMIA", "high", "2026-08-01", "2026-08-01T18:02:13.497494+00:00", 0.97, 0.9850, "94–95", "no edge: 94–95 model 98%->100% calibrated vs market 97% = +3% < 3% required — do not place", 94, 1, "2026-08-02T04:02:24+00:00"),
    ("KLAX", "high", "2026-08-01", "2026-08-01T20:00:42.239970+00:00", 0.935, 0.8702, "80–81", "no edge: 80–81 model 87%->76% calibrated vs market 94% = -17% < 3% required — do not place", 80, 1, "2026-08-02T07:00:25+00:00"),
    ("KMIA", "low", "2026-08-02", "2026-08-02T11:02:33.407531+00:00", 0.695, 0.4954, "78–79", "no edge: 78–79 model 50%->52% calibrated vs market 70% = -17% < 3% required — do not place", None, None, None),
    ("KLAX", "low", "2026-08-02", "2026-08-02T13:06:24.976270+00:00", 0.755, 0.6756, "69–70", "no edge: 69–70 model 68%->69% calibrated vs market 76% = -7% < 3% required — do not place", None, None, None),
]


def _load_archive(conn):
    for (station, metric, target, placed_utc, price, p_model, bucket, reason,
        actual, won, settled_utc) in ARCHIVE_ROWS:
        conn.execute(
            """INSERT INTO app_bets
               (station, metric, target_date, placed_utc, placed_manila,
                placed_local_hour, bucket, p_model, price, stake,
                payout_if_win, actual, won, settled_utc, reason)
               VALUES (?,?,?,?,'00:00',0,?,?,?,1.0,?,?,?,?,?)""",
            (station, metric, target, placed_utc, bucket, p_model, price,
             (1.0 / price if price else None), actual, won, settled_utc,
             reason))
    conn.commit()


@pytest.fixture
def archive_conn(tmp_path):
    conn = db.connect(tmp_path / "archive.db")
    _load_archive(conn)
    return conn


# ── the reconciliation gate ───────────────────────────────────────────────────

def test_unit_replay_reproduces_the_archive_exactly(archive_conn):
    out = paper_account.unit_replay(archive_conn, opening_balance=5.50)
    assert out["n_graded"] == 20
    assert out["wins"] == 14
    assert out["losses"] == 6
    rec = out["reconciliation"]
    assert rec["place"]["n"] == 8
    assert rec["place"]["wins"] == 3
    assert rec["place"]["net_per_contract"] == pytest.approx(0.015, abs=1e-6)
    assert rec["skip"]["n"] == 12
    assert rec["skip"]["wins"] == 11
    assert rec["skip"]["net_per_contract"] == pytest.approx(1.990, abs=1e-6)


def test_unit_replay_leaves_ungraded_bets_as_open_positions(archive_conn):
    out = paper_account.unit_replay(archive_conn, opening_balance=5.50)
    assert len(out["open_positions"]) == 2
    keys = {(p["station"], p["metric"], p["target_date"])
           for p in out["open_positions"]}
    assert ("KMIA", "low", "2026-08-02") in keys
    assert ("KLAX", "low", "2026-08-02") in keys


def test_unit_replay_never_mutates_app_bets(archive_conn):
    before = [dict(r) for r in archive_conn.execute(
        "SELECT * FROM app_bets ORDER BY placed_utc")]
    paper_account.unit_replay(archive_conn, opening_balance=5.50)
    after = [dict(r) for r in archive_conn.execute(
        "SELECT * FROM app_bets ORDER BY placed_utc")]
    assert before == after
    assert len(after) == 22


def test_refresh_persists_ledger_without_touching_app_bets(archive_conn, tmp_path):
    cfgs = {KLAX.station.icao: dataclasses.replace(KLAX, db_path=":memory:"),
           KMIA.station.icao: dataclasses.replace(KMIA, db_path=":memory:")}
    before = archive_conn.execute("SELECT COUNT(*) c FROM app_bets").fetchone()["c"]
    out = paper_account.refresh(archive_conn, cfgs, env={}, fallback_usd=5.50,
                             credentials_path=tmp_path / "no-such-creds")
    after = archive_conn.execute("SELECT COUNT(*) c FROM app_bets").fetchone()["c"]
    assert before == after == 22
    rows = archive_conn.execute(
        "SELECT COUNT(*) c FROM paper_account_txns WHERE mode='unit_replay'"
    ).fetchone()["c"]
    # 22 OPEN rows + 20 SETTLE rows (2 bets stay open, no settle row)
    assert rows == 22 + 20
    assert out["seed"]["source"] == "fallback"


def test_refresh_persist_false_writes_nothing(archive_conn, tmp_path):
    """The dashboard READS this account; it must never be a writer.

    `persist=False` exists for exactly one caller (the web view, which is hit
    by every page load and by a poll timer). It must return the same numbers
    as a persisting run while leaving `paper_account_txns` and `meta`
    untouched — otherwise a browser refresh races the CLI's own rebuild for
    the same rows.
    """
    cfgs = {KLAX.station.icao: dataclasses.replace(KLAX, db_path=":memory:"),
           KMIA.station.icao: dataclasses.replace(KMIA, db_path=":memory:")}
    creds = tmp_path / "no-such-creds"
    out = paper_account.refresh(archive_conn, cfgs, env={}, fallback_usd=5.50,
                                credentials_path=creds, persist=False)
    assert archive_conn.execute(
        "SELECT COUNT(*) c FROM paper_account_txns").fetchone()["c"] == 0
    assert db.get_meta(archive_conn,
                       "paper_account_opening_balance_usd") is None
    # ...and the payload is identical to what the persisting call returns.
    persisted = paper_account.refresh(archive_conn, cfgs, env={},
                                      fallback_usd=5.50, credentials_path=creds)
    assert out["unit_replay"]["closing_balance"] == \
        persisted["unit_replay"]["closing_balance"]
    assert out["daily_plan"].get("closing_balance") == \
        persisted["daily_plan"].get("closing_balance")


def test_refresh_is_idempotent(archive_conn, tmp_path):
    cfgs = {KLAX.station.icao: dataclasses.replace(KLAX, db_path=":memory:"),
           KMIA.station.icao: dataclasses.replace(KMIA, db_path=":memory:")}
    creds_path = tmp_path / "no-such-creds"
    paper_account.refresh(archive_conn, cfgs, env={}, fallback_usd=5.50,
                           credentials_path=creds_path)
    n1 = archive_conn.execute(
        "SELECT COUNT(*) c FROM paper_account_txns").fetchone()["c"]
    paper_account.refresh(archive_conn, cfgs, env={}, fallback_usd=5.50,
                           credentials_path=creds_path)
    n2 = archive_conn.execute(
        "SELECT COUNT(*) c FROM paper_account_txns").fetchone()["c"]
    assert n1 == n2


# ── opening balance seeding ───────────────────────────────────────────────────

def test_seed_falls_back_without_credentials(tmp_path):
    missing = tmp_path / "no-such-file"
    out = paper_account.seed_opening_balance(
        {}, credentials_path=missing, fallback_usd=3.25)
    assert out["source"] == "fallback"
    assert out["balance_usd"] == 3.25
    assert "fallback" in out["note"]


def test_seed_reads_a_real_balance_via_injected_transport(tmp_path):
    """Mirrors tests/test_kalshi_reads_need_no_arming.py's fake transport --
    never touches the network, but exercises the real signing path with a
    genuine ephemeral RSA key."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()).decode()
    # A PEM is multi-line, so it cannot be inlined into the KEY=VALUE
    # credentials file directly -- point at it via KALSHI_PRIVATE_KEY_FILE,
    # the same pattern `kalshi_client.load_credentials` itself documents.
    keyfile = tmp_path / "key.pem"
    keyfile.write_text(pem)
    keyfile.chmod(0o600)
    creds = tmp_path / "credentials"
    creds.write_text(f"KALSHI_API_KEY_ID=test-key\nKALSHI_PRIVATE_KEY_FILE={keyfile}\n")
    creds.chmod(0o600)

    class _Resp:
        status_code = 200
        def json(self):
            return {"balance": 550, "portfolio_value": 0}

    class _Transport:
        def get(self, url, headers=None, **kw):
            return _Resp()

    out = paper_account.seed_opening_balance(
        {}, credentials_path=creds, transport=_Transport())
    assert out["source"] == "kalshi_balance"
    assert out["balance_usd"] == 5.50


def test_seed_falls_back_when_the_read_fails(tmp_path):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()).decode()
    keyfile = tmp_path / "key.pem"
    keyfile.write_text(pem)
    keyfile.chmod(0o600)
    creds = tmp_path / "credentials"
    creds.write_text(f"KALSHI_API_KEY_ID=test-key\nKALSHI_PRIVATE_KEY_FILE={keyfile}\n")
    creds.chmod(0o600)

    class _Resp:
        status_code = 500
        def json(self):
            return {}

    class _Transport:
        def get(self, url, headers=None, **kw):
            return _Resp()

    out = paper_account.seed_opening_balance(
        {}, credentials_path=creds, fallback_usd=1.0, transport=_Transport())
    assert out["source"] == "fallback"
    assert out["balance_usd"] == 1.0


def test_seed_never_touches_arming():
    """Reads are credentials-only; this function must never construct an
    armed client, mirroring kalshi_client's own read/trade split."""
    import inspect
    src = inspect.getsource(paper_account.seed_opening_balance)
    assert "armed=False" in src
    assert "armed=True" not in src


# ── stage ladder ──────────────────────────────────────────────────────────────

def test_stage_ladder_boundaries_fall_out_of_the_cap():
    ladder = paper_account.stage_ladder(max_stake_frac=0.05, price_ceiling=0.84)
    # 40c / 0.05 = 8, 60c / 0.05 = 12, 0.84 / 0.05 = 16.80, 1.00 / 0.05 = 20
    highs = [s["hi"] for s in ladder]
    assert highs[0] == pytest.approx(8.0)
    assert highs[1] == pytest.approx(12.0)
    assert highs[2] == pytest.approx(16.80)
    assert highs[3] == pytest.approx(20.0)
    assert highs[4] is None


def test_stage_for_balance_places_the_real_opening_balance_in_stage_one():
    ladder = paper_account.stage_ladder(0.05, 0.84)
    stage = paper_account.stage_for_balance(5.50, ladder)
    assert stage["stage"] == 1
    assert stage["hi"] == pytest.approx(8.0)


def test_stage_for_balance_at_twenty_is_the_final_stage():
    ladder = paper_account.stage_ladder(0.05, 0.84)
    stage = paper_account.stage_for_balance(25.0, ladder)
    assert stage["stage"] == 5
    assert stage["hi"] is None


# ── daily plan: selection, override, gates ────────────────────────────────────

def test_select_daily_market_picks_the_highest_hit_rate():
    m = paper_account.select_daily_market()
    assert m is not None
    # KMIA HIGH remains the daily market after the commit hour moved 14 -> 13
    # on 2026-08-05. On RAW hit rate the move demotes it to third (67.6% at
    # h13 vs KLAX HIGH's 69.5%), but KLAX HIGH is not an eligible candidate --
    # see the eligibility test below.
    assert (m["station"], m["metric"]) == ("KMIA", "high")
    assert m["hour"] == 13


def test_only_markets_with_a_measured_confidence_floor_are_eligible():
    """THE REGRESSION THIS EXISTS FOR. Ranking purely on hit rate made the
    commit-hour move silently hand the daily slot to KLAX HIGH, which
    `gating.CONFIDENCE_FLOOR` deliberately EXCLUDES on two independent
    grounds: no stated-probability threshold reaches a 0.80 lower bound, and
    at h13 its favoured bucket trades 0.75-0.96 against a model that can
    justify ~69c.

    A market the bot may never fire on is not a market the daily plan should
    be simulating. The floor's own comment already states the policy — "a
    market with no such threshold is ABSENT here, and an absent market never
    fires" — this makes the selector obey it too."""
    from highcast import gating
    m = paper_account.select_daily_market()
    key = (m["station"], m["metric"], m["hour"])
    assert key in gating.CONFIDENCE_FLOOR, key
    # ...and KLAX HIGH really is the higher-hit-rate candidate being skipped,
    # so this test fails loudly if the eligibility filter is ever removed.
    klax_hit = gating.HOUR_ACCURACY["KLAX"]["high"][
        gating.MIN_DECISION_HOUR["KLAX"]["high"]][1]
    assert klax_hit > m["hit"]
    assert ("KLAX", "high", 13) not in gating.CONFIDENCE_FLOOR


def test_daily_plan_above_cap_override_buys_exactly_one_contract(archive_conn):
    cfgs = {KMIA.station.icao: KMIA, KLAX.station.icao: KLAX}
    out = paper_account.daily_plan_sim(archive_conn, cfgs, opening_balance=5.50)
    assert out["market"]["station"] == "KMIA"
    assert out["market"]["metric"] == "high"
    overrides = [d for d in out["detail"] if d.get("override")]
    assert len(overrides) >= 1
    for d in overrides:
        assert d["contracts"] == 1
        assert "ABOVE" in d["reason"]
        assert "cap" in d["reason"]
        # never silently rounds past what was actually accepted
        fee = fees.taker_fee(1, d["price"])
        assert d["price"] + fee <= 5.50 + 1e-9


def test_daily_plan_refuses_above_the_price_ceiling(archive_conn):
    cfgs = {KMIA.station.icao: KMIA, KLAX.station.icao: KLAX}
    out = paper_account.daily_plan_sim(archive_conn, cfgs, opening_balance=5.50)
    ceiling_refusals = [d for d in out["detail"]
                        if not d["trade"] and "ceiling" in d["reason"]]
    # 2026-08-01 KMIA high priced 0.97, above the 0.84 ceiling
    assert any(d["target_date"] == "2026-08-01" for d in ceiling_refusals)


def test_daily_plan_never_rounds_up_when_totally_unaffordable():
    """A price under the ceiling but above the CURRENT balance must be a
    plain NO_TRADE, never an override -- the override only ever accepts what
    is actually affordable in absolute dollars."""
    row = {"price": 0.80, "p_model": 0.95, "reason": None}  # under the 0.84 ceiling
    ladder = paper_account.stage_ladder(0.05, 0.84)
    d = paper_account._daily_plan_decision(row, KMIA, balance=0.50,
                                          price_ceiling=0.84, ladder=ladder)
    assert d["trade"] is False
    assert "insufficient bankroll" in d["reason"]


def test_daily_plan_stake_compounds_with_the_running_balance():
    """Sizing must use the LIVE balance, not the fixed opening figure --
    a bigger running balance should be able to afford more contracts at the
    same price."""
    row = {"price": 0.50, "p_model": 0.90, "reason": None}
    ladder = paper_account.stage_ladder(0.05, 0.84)
    small = paper_account._daily_plan_decision(row, KMIA, balance=5.50,
                                               price_ceiling=0.84, ladder=ladder)
    big = paper_account._daily_plan_decision(row, KMIA, balance=100.0,
                                             price_ceiling=0.84, ladder=ladder)
    assert small["contracts"] == 1 and small.get("override")
    assert big["contracts"] > small["contracts"]
    assert not big.get("override", False)


def test_daily_plan_never_touches_app_bets(archive_conn):
    before = [dict(r) for r in archive_conn.execute(
        "SELECT * FROM app_bets ORDER BY placed_utc")]
    cfgs = {KMIA.station.icao: KMIA, KLAX.station.icao: KLAX}
    paper_account.daily_plan_sim(archive_conn, cfgs, opening_balance=5.50)
    after = [dict(r) for r in archive_conn.execute(
        "SELECT * FROM app_bets ORDER BY placed_utc")]
    assert before == after


# ── commit-hour variants ──────────────────────────────────────────────────────

def _insert_signal(conn, station, metric, target_date, created_utc, price,
                   p_model, bucket="94–95"):
    conn.execute(
        """INSERT INTO signals (station, target_date, metric, kind,
             created_utc, committed, within_window, place, bucket, p_model,
             price, market_json)
           VALUES (?,?,?,?,?,0,1,0,?,?,?,'[]')""",
        (station, target_date, metric, "intraday", created_utc, bucket,
         p_model, price))
    conn.commit()


def _insert_final_cli(conn, cfg, target_date, high=None, low=None):
    import json
    from datetime import timedelta
    from highcast.timeutil import climate_day_bounds
    end = climate_day_bounds(
        __import__("datetime").date.fromisoformat(target_date),
        cfg.station.std_utc_offset)[1]
    issued = end + timedelta(hours=1)
    product = issued.strftime("%Y%m%d%H%M") + "-TEST-CDUS00-CLITEST"
    conn.execute(
        """INSERT INTO cli_reports (station, date, high_f, low_f, raw, fetched_at)
           VALUES (?,?,?,?,?,?)""",
        (cfg.station.icao, target_date, high, low,
         json.dumps({"product": product}), "t"))
    conn.commit()


def test_commit_hour_variants_compares_default_and_extra_hours(tmp_path):
    conn = db.connect(tmp_path / "v.db")
    kmia = dataclasses.replace(KMIA, db_path=tmp_path / "v.db")
    # two settled days, each with a signal at h13 and h14, KMIA HIGH
    _insert_signal(conn, "KMIA", "high", "2026-07-29",
                   "2026-07-29T17:00:00+00:00", 0.52, 0.70)   # h13 local (UTC-4)
    _insert_signal(conn, "KMIA", "high", "2026-07-29",
                   "2026-07-29T18:00:00+00:00", 0.75, 0.80)   # h14 local
    _insert_signal(conn, "KMIA", "high", "2026-07-30",
                   "2026-07-30T17:00:00+00:00", 0.50, 0.65)
    _insert_signal(conn, "KMIA", "high", "2026-07-30",
                   "2026-07-30T18:00:00+00:00", 0.70, 0.75)
    _insert_final_cli(conn, kmia, "2026-07-29", high=94)
    _insert_final_cli(conn, kmia, "2026-07-30", high=94)

    market = {"station": "KMIA", "metric": "high", "hour": 14,
             "mae": 0.5, "hit": 0.8}
    out = paper_account.commit_hour_variants(
        conn, {"KMIA": kmia}, market, extra_hours=(13,))
    hours = {v["hour"]: v for v in out["variants"]}
    assert set(hours) == {13, 14}
    assert hours[14]["is_default"] is True
    assert hours[13]["is_default"] is False
    assert hours[13]["n_graded"] == 2
    assert hours[14]["n_graded"] == 2
    # both graded bets are on bucket 94-95, actual 94 -> both win at both hours
    assert hours[13]["wins"] == 2
    assert hours[14]["wins"] == 2
    # h13's mean entry price is cheaper -> better mean net/bet
    assert hours[13]["mean_entry_price"] < hours[14]["mean_entry_price"]
    assert hours[13]["mean_net_per_bet"] > hours[14]["mean_net_per_bet"]


def test_commit_hour_variants_never_touches_signals_or_app_bets(tmp_path):
    conn = db.connect(tmp_path / "v2.db")
    kmia = dataclasses.replace(KMIA, db_path=tmp_path / "v2.db")
    _insert_signal(conn, "KMIA", "high", "2026-07-29",
                   "2026-07-29T18:00:00+00:00", 0.75, 0.80)
    _insert_final_cli(conn, kmia, "2026-07-29", high=94)
    before_sig = [dict(r) for r in conn.execute("SELECT * FROM signals")]
    before_bets = [dict(r) for r in conn.execute("SELECT * FROM app_bets")]
    market = {"station": "KMIA", "metric": "high", "hour": 14,
             "mae": 0.5, "hit": 0.8}
    paper_account.commit_hour_variants(conn, {"KMIA": kmia}, market)
    after_sig = [dict(r) for r in conn.execute("SELECT * FROM signals")]
    after_bets = [dict(r) for r in conn.execute("SELECT * FROM app_bets")]
    assert before_sig == after_sig
    assert before_bets == after_bets


def test_commit_hour_variants_with_no_market_returns_empty():
    out = paper_account.commit_hour_variants(None, {}, None)
    assert out["variants"] == []
