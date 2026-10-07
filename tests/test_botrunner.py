"""Wiring the order bot into the wake cycle.

`botrunner` is the ONLY place the three pieces meet: the market snapshot, the
gate verdict, and the transport. It holds no trading capability of its own —
`kalshi_client` has that — and no decision logic — `orderbot` has that.

THE TWO THINGS THAT MATTER HERE:

1. IDEMPOTENCY. The wake cycle fires every 5 minutes, so the commit hour brings
   ~12 invocations. Exactly one decision may be recorded and at most one order
   sent. This is enforced by the PRIMARY KEY on `bot_orders` plus INSERT OR
   IGNORE, not by a flag someone can forget to check.

2. REFUSALS ARE RECORDED TOO. Two weeks of "why not" is the evidence the arming
   decision rests on. A runner that only logged orders would produce an empty
   table and no way to tell "never qualified" from "never ran".
"""
import datetime as dt

import pytest
import yaml

from highcast import botrunner
from highcast import db as hdb
from highcast.config import load_config


def _cfg(tmp_path, src="config/kmia.yaml"):
    raw = yaml.safe_load(open(src).read())
    raw["db_path"] = str(tmp_path / "t.db")
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(raw))
    return load_config(str(p))


def _at(hour, day="2026-08-02", tz="America/New_York"):
    from zoneinfo import ZoneInfo
    d = dt.date.fromisoformat(day)
    return dt.datetime(d.year, d.month, d.day, hour, 5,
                       tzinfo=ZoneInfo(tz)).astimezone(dt.timezone.utc)


def _market(ask=0.75, p_model=0.90, ticker="KXHIGHMIA-26AUG02-B94.5"):
    """A ladder row shaped like the real one — INCLUDING the ticker.

    It was omitted at first, which is how a fixture teaches you the wrong
    lesson: the code then constructed a ticker instead, and no test could
    notice because no fixture ever carried the real one.
    """
    return [{"label": "94–95", "lo": 94, "hi": 95, "p_model": p_model,
             "ticker": ticker,
             "market_price": ask, "yes_bid": ask - 0.01, "yes_ask": ask}]


def test_it_records_a_decision_even_when_it_refuses(tmp_path):
    """The whole point of the dry-run fortnight. A runner that only logged
    orders would leave an empty table, and 'never qualified' would be
    indistinguishable from 'never ran'."""
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    # gates_ok/calibrated are stubbed so the CEILING is what is being tested.
    # Without them the real gates refuse first on an empty test DB, and the test
    # passes for the wrong reason — which is how it was first written.
    botrunner.run_once(conn, cfg, now=_at(14), ranges=_market(ask=0.95),
                       env={}, gates_ok=True, calibrated=0.90)
    rows = conn.execute("SELECT * FROM bot_orders").fetchall()
    assert len(rows) == 1
    assert rows[0]["place"] == 0
    assert "ceiling" in rows[0]["reason"].lower()


def test_only_the_first_decision_of_the_hour_is_kept(tmp_path):
    """~12 wake cycles land inside the commit hour. The PRIMARY KEY makes this
    structural rather than a flag someone forgets to check."""
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    for _ in range(5):
        botrunner.run_once(conn, cfg, now=_at(14), ranges=_market(), env={})
    rows = conn.execute("SELECT * FROM bot_orders").fetchall()
    assert len(rows) == 1


def test_it_does_nothing_outside_the_commit_hour(tmp_path):
    """h14 is where every number behind this bot was measured."""
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    botrunner.run_once(conn, cfg, now=_at(11), ranges=_market(), env={})
    assert conn.execute("SELECT count(*) FROM bot_orders").fetchone()[0] == 0


def test_dry_run_records_the_order_it_would_have_placed(tmp_path):
    """The dry-run row must carry the ACTIONABLE detail — bucket, limit, size —
    or the fortnight produces no usable evidence about entry prices."""
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    botrunner.run_once(conn, cfg, now=_at(14), ranges=_market(ask=0.75),
                       env={}, gates_ok=True, calibrated=0.90)
    r = conn.execute("SELECT * FROM bot_orders").fetchone()
    assert r["place"] == 1
    assert r["dry_run"] == 1
    assert r["placed"] == 0
    assert r["bucket"] == "94–95"
    assert r["limit_price"] == 0.75
    assert r["contracts"] == 1
    assert r["ask"] == 0.75


def test_a_crash_anywhere_does_not_break_the_wake_cycle(tmp_path):
    """This runs inside a cycle that also fetches observations and issues
    predictions. A trading bug must never cost the weather pipeline."""
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)

    class Exploding(list):
        """Non-empty so it is truthy — `ranges or []` short-circuits on an EMPTY
        list, so the first version of this test never actually crashed and
        passed vacuously."""
        def __init__(self):
            super().__init__([{"p_model": 0.9}])

        def __iter__(self):
            raise RuntimeError("boom")

    out = botrunner.run_once(conn, cfg, now=_at(14), ranges=Exploding(), env={})
    assert out["place"] is False
    assert "error" in out["reason"].lower()


def test_it_never_sends_while_unarmed_even_with_credentials(tmp_path):
    """Credentials being present is not arming. During the fortnight the key IS
    installed, so this is the live safety property, not a hypothetical."""
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)

    class Boom:
        def post(self, *a, **k):
            raise AssertionError("sent an order while unarmed")

    out = botrunner.run_once(
        conn, cfg, now=_at(14), ranges=_market(), gates_ok=True,
        calibrated=0.90,
        env={"KALSHI_API_KEY_ID": "k", "KALSHI_PRIVATE_KEY": "pem"},
        transport=Boom())
    assert out["dry_run"] is True
    assert out["placed"] is False


# ── THE TICKER ──────────────────────────────────────────────────────────────
#
# `_ticker` used to CONSTRUCT one: f"{series}-{target}-{bucket}", which gives
# "KXHIGHMIA-2026-08-02-94–95". That contains an EN DASH (from the bucket label)
# and Kalshi's date suffix is "26AUG02", not "2026-08-02". Every order would
# have been rejected — at the commit hour, on the one day of the fortnight the
# bot finally wanted to trade, and logged as an exchange error rather than as
# our own bad string.

def test_the_order_uses_the_ladder_s_real_ticker_not_a_constructed_one(tmp_path):
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    sent = {}

    class Spy:
        def post(self, url, json=None, headers=None):
            sent.update(json or {})

            class R:
                status_code = 201

                def json(self):
                    return {"order": {"order_id": "o1"}}
            return R()

    m = _market(ask=0.75)
    # `_armed_env` supplies the arming WINDOW as well as the literal. Without it
    # the unattended bot stays dry and this test would pass vacuously — it would
    # assert nothing about the ticker, because nothing would be sent.
    botrunner.run_once(conn, cfg, now=_at(14), ranges=m, gates_ok=True,
                       calibrated=0.90, transport=Spy(), env=_armed_env())
    assert sent.get("ticker") == "KXHIGHMIA-26AUG02-B94.5"
    assert "–" not in sent.get("ticker", ""), "an en dash reached the exchange"


def test_a_ladder_without_a_ticker_refuses_rather_than_guessing(tmp_path):
    """No ticker means no order that could be accepted. A refusal we can name
    beats a rejection we would have to diagnose at 02:00 Manila."""
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)

    class Boom:
        def post(self, *a, **k):
            raise AssertionError("sent an order with no real ticker")

    from highcast import kalshi_client as kc
    botrunner.run_once(conn, cfg, now=_at(14),
                       ranges=_market(ask=0.75, ticker=None),
                       gates_ok=True, calibrated=0.90, transport=Boom(),
                       env={kc.ARM_ENV: kc.ARM_VALUE,
                            kc.KEY_ID_ENV: "kid", kc.PRIVATE_KEY_ENV: _pem()})
    r = conn.execute("SELECT * FROM bot_orders").fetchone()
    assert r["place"] == 0
    assert "ticker" in r["reason"]


def _pem():
    import pytest
    pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    k = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return k.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()).decode()


# ── THE BOT WILL NOT TRADE PAST ITS WINDOW ──────────────────────────────────

def _armed_env(until="2026-08-02"):
    from highcast import kalshi_client as kc
    e = {kc.ARM_ENV: kc.ARM_VALUE, kc.KEY_ID_ENV: "kid",
         kc.PRIVATE_KEY_ENV: _pem()}
    if until is not None:
        e[kc.ARM_UNTIL_ENV] = until
    return e


def test_the_literal_alone_does_not_arm_the_unattended_bot(tmp_path):
    """THE property that makes 'arm it for one day' real. The literal has to sit
    in the launchd plist for the bot to fire at 02:00 while Julian sleeps, and a
    plist entry survives until someone removes it. Without a window, that is an
    open-ended live bot — so no window means DRY RUN."""
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)

    class Boom:
        def post(self, *a, **k):
            raise AssertionError("traded with no arming window set")

    out = botrunner.run_once(conn, cfg, now=_at(14), ranges=_market(ask=0.75),
                             gates_ok=True, calibrated=0.90, transport=Boom(),
                             env=_armed_env(until=None))
    assert out["dry_run"] is True
    assert out["placed"] is False


def test_the_bot_disarms_itself_after_the_window(tmp_path):
    """Aug 2 with a window ending Aug 1: the plist can be left in place and the
    bot goes quiet on its own. Forgetting to clean up fails SAFE."""
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)

    class Boom:
        def post(self, *a, **k):
            raise AssertionError("traded after the arming window expired")

    out = botrunner.run_once(conn, cfg, now=_at(14, day="2026-08-02"),
                             ranges=_market(ask=0.75), gates_ok=True,
                             calibrated=0.90, transport=Boom(),
                             env=_armed_env(until="2026-08-01"))
    assert out["dry_run"] is True


def test_the_bot_trades_inside_its_window(tmp_path):
    """The positive control. Without this the two tests above would pass against
    a bot that never trades at all."""
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    sent = {}

    class Spy:
        def post(self, url, json=None, headers=None):
            sent.update(json or {})

            class R:
                status_code = 201

                def json(self):
                    return {"order": {"order_id": "o1"}}
            return R()

    out = botrunner.run_once(conn, cfg, now=_at(14, day="2026-08-02"),
                             ranges=_market(ask=0.75), gates_ok=True,
                             calibrated=0.90, transport=Spy(),
                             env=_armed_env(until="2026-08-02"))
    assert out["dry_run"] is False and out["placed"] is True
    assert sent["ticker"] == "KXHIGHMIA-26AUG02-B94.5"


def test_a_refusal_still_records_the_bucket_it_was_looking_at(tmp_path):
    """THE EVIDENCE GAP. `orderbot.decide` returns on the gate check BEFORE it
    selects a bucket, so `intent.bucket` is None on every refusal — and refusals
    are most of the fortnight. Without this, two weeks of 'confidence_floor'
    rows would say the gate fired but not what it fired ON, which is exactly the
    question the arming review has to answer.

    `place` distinguishes ordered from considered, so the column can safely mean
    'the day's pick'.
    """
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    botrunner.run_once(conn, cfg, now=_at(14), ranges=_market(ask=0.75),
                       env={}, gates_ok=False, calibrated=0.51)
    r = conn.execute("SELECT * FROM bot_orders").fetchone()
    assert r["place"] == 0
    assert r["bucket"] == "94–95", "the refusal lost the bucket it was scoring"
    assert r["calibrated"] == 0.51
    assert r["ask"] == 0.75


# ── THE EDGE IS LOGGED ON EVERY DECISION ────────────────────────────────────
#
# `gating.MAX_EDGE` is provisional because too little data exists to tune it.
# `bot_orders.edge` is what lets that measurement happen later -- but only if
# it is written on every row, not only the ones the gate trips.

def test_the_edge_is_recorded_on_a_placed_order(tmp_path):
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    botrunner.run_once(conn, cfg, now=_at(14), ranges=_market(ask=0.75),
                       env={}, gates_ok=True, calibrated=0.90)
    r = conn.execute("SELECT * FROM bot_orders").fetchone()
    assert r["place"] == 1
    assert r["edge"] == pytest.approx(0.15)   # 0.90 calibrated - 0.75 ask


def test_the_edge_is_recorded_on_a_max_edge_refusal(tmp_path):
    """The gate this whole feature exists to add: a calibrated number wildly
    above a cheap ask must refuse AND still log the edge it refused on."""
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    botrunner.run_once(conn, cfg, now=_at(14), ranges=_market(ask=0.05),
                       env={}, gates_ok=True, calibrated=0.99)
    r = conn.execute("SELECT * FROM bot_orders").fetchone()
    assert r["place"] == 0
    assert "edge_implausible" in r["reason"]
    assert r["bucket"] == "94–95"
    assert r["edge"] == pytest.approx(0.94)


def test_the_edge_is_recorded_even_when_the_gate_verdict_refuses_first(tmp_path):
    """`orderbot.decide` never reaches its own market-selection step when
    `gates_ok=False`, so `intent.edge` alone would be None here. `botrunner`
    must compute it independently from the calibrated value and the ladder's
    ask it already holds -- same principle as the bucket-recording fix above,
    applied to the new column."""
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    botrunner.run_once(conn, cfg, now=_at(14), ranges=_market(ask=0.75),
                       env={}, gates_ok=False, calibrated=0.90)
    r = conn.execute("SELECT * FROM bot_orders").fetchone()
    assert r["place"] == 0
    assert r["edge"] == pytest.approx(0.15)
