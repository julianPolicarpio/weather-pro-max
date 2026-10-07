"""Reads need credentials; TRADES need arming. Both halves pinned.

WHY THIS FILE EXISTS. `positions()` used to return empty unless `self.armed` --
the LIVE-ORDER arming flag (the literal `ARM_LIVE_ORDERS_I_MEAN_IT` plus an
unexpired date). That gate exists because the bot fires at 02:00 Manila while
Julian sleeps, so spending money must be hard to switch on.

Gating a READ behind it inverted the property it protects: to look at your own
balance you had to grant permission to place live orders, which trains the habit
of setting the trading literal for harmless work. Kalshi's own API issues keys
scoped `read::portfolio_balance`, narrower than any trading right, precisely so
reads need no trade permission.

So the split is: reads -> credentials only. place_order -> still armed.

THE DANGER THIS FILE GUARDS AGAINST is the refactor going one step too far and
ungating `place_order` along with the reads. That would be catastrophic and
silent -- an unarmed bot that suddenly transacts. Relaxing the read gate and
ungating a trade look nearly identical in a diff, so this asserts them
separately rather than trusting review.
"""
import pytest

from highcast.kalshi_client import BALANCE_PATH, POSITIONS_PATH, KalshiClient


def _real_pem() -> str:
    """A GENUINE ephemeral RSA key, generated once for this module.

    A placeholder string does not work and fails misleadingly: `sign()` loads the
    PEM for real, a bogus one raises ValueError, and `_send`'s broad
    `except Exception` converts that into `{"reason": "GET failed: ValueError"}`
    -- which looks exactly like an auth/permission failure. The first version of
    this file used a fake PEM and every test failed identically, pointing at the
    arming gate rather than at the key.

    Using a real key also means these tests exercise the actual RSA-PSS signing
    path instead of stubbing past it. The key never leaves this process.
    """
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    k = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return k.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()).decode()


KEY, PEM = "key-abc", _real_pem()


class _Transport:
    """Records requests and returns a canned body. Never touches the network.

    Mirrors the httpx-style surface `_send` actually uses -- `.get`/`.post`/
    `.delete`, not `.request`. Getting this wrong made every test in this file
    fail identically on the first run, which is a useful reminder that a fake
    which does not match the real interface tests nothing.
    """

    def __init__(self, body=None):
        self.calls = []
        self._body = body or {}

    def get(self, url, headers=None, **kw):
        self.calls.append(("GET", url, None))
        return _Resp(self._body)

    def post(self, url, json=None, headers=None, **kw):
        self.calls.append(("POST", url, json))
        return _Resp(self._body)

    def delete(self, url, headers=None, **kw):
        self.calls.append(("DELETE", url, None))
        return _Resp(self._body)


class _Resp:
    status_code = 200

    def __init__(self, body):
        self._body = body

    def json(self):
        return self._body

    @property
    def text(self):
        return str(self._body)


def _client(armed: bool, body=None, creds=True):
    return KalshiClient(KEY if creds else None, PEM if creds else None,
                        armed=armed, transport=_Transport(body))


# ── the read half ───────────────────────────────────────────────────────────

def test_balance_is_readable_while_unarmed():
    """The whole point. An UNARMED client must still see its own money."""
    c = _client(armed=False, body={"balance": 1234, "portfolio_value": 5678})
    out = c.balance()
    assert out.get("reason") != "no credentials"
    assert out["balance_usd"] == 12.34


def test_positions_are_readable_while_unarmed():
    c = _client(armed=False, body={"positions": [{"ticker": "X"}]})
    assert c.positions().get("reason") is None


def test_reads_still_require_credentials():
    """No key = no read. Failing open here would send unsigned requests."""
    c = _client(armed=True, creds=False)
    assert c.balance()["balance_usd"] is None
    assert c.balance()["reason"] == "no credentials"
    assert c.positions()["positions"] == []


# ── the units half ──────────────────────────────────────────────────────────

def test_balance_converts_cents_to_dollars_exactly_once():
    """Kalshi returns CENTS. Everything downstream -- bankroll, stakes, caps --
    is DOLLARS, so a missed conversion sizes every position 100x too large.
    This is the single most expensive possible bug in this file."""
    c = _client(armed=False, body={"balance": 5600, "portfolio_value": 100})
    out = c.balance()
    assert out["balance_usd"] == 56.00
    assert out["portfolio_value_usd"] == 1.00
    # the raw cents field must survive untouched for auditing
    assert out["balance"] == 5600


def test_balance_handles_a_missing_amount():
    c = _client(armed=False, body={})
    out = c.balance()
    assert out["balance_usd"] is None
    assert out["portfolio_value_usd"] is None


def test_balance_hits_the_documented_path():
    c = _client(armed=False, body={"balance": 1})
    c.balance()
    method, url, _ = c._transport.calls[-1]
    assert method == "GET"
    assert url.endswith(BALANCE_PATH)
    assert BALANCE_PATH == "/trade-api/v2/portfolio/balance"
    assert POSITIONS_PATH == "/trade-api/v2/portfolio/positions"


# ── the trade half: THIS MUST NOT RELAX ─────────────────────────────────────

def test_place_order_still_refuses_while_unarmed():
    """The load-bearing assertion in this file.

    If relaxing the read gate ever ungates ordering, this fails. An unarmed
    client must not transact under any circumstance -- it must not even build a
    transport.
    """
    c = _client(armed=False, body={"order": {"order_id": "should-never-happen"}})
    res = c.place_order(ticker="KXHIGHMIA-T", side="yes", count=1,
                        limit_price=0.01)
    assert not res.get("placed", False), (
        "UNARMED CLIENT PLACED AN ORDER — the read/trade split has ungated "
        "trading. Do not ship.")
    assert not c._transport.calls, (
        "unarmed client sent an HTTP request while placing an order")


def test_reading_does_not_flip_the_armed_flag():
    c = _client(armed=False, body={"balance": 1})
    c.balance()
    c.positions()
    assert c.armed is False


@pytest.mark.parametrize("armed", [True, False])
def test_arming_does_not_change_read_behaviour(armed):
    """Reads must behave identically armed or not -- otherwise the arming flag
    is still secretly load-bearing for reads."""
    c = _client(armed=armed, body={"balance": 4200})
    assert c.balance()["balance_usd"] == 42.00
