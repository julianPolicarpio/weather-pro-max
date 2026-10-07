"""The only code in this project that can spend money. Guard it accordingly.

Design: `docs/superpowers/specs/2026-08-01-kmia-high-order-bot-design.md` §5-6.

THE TWO LOAD-BEARING TESTS are `test_dry_run_never_posts` and
`test_arming_requires_the_exact_literal`. Everything else here is hygiene; those
two are what stand between a config typo and real orders.

⚠️ A GUARANTEE THAT EXPIRED, LEFT HERE ON PURPOSE.
This file used to say the dry-run was PHYSICALLY incapable of ordering, because
`cryptography` was not installed and nothing supplied an HTTP transport. Both
stopped being true on 2026-08-01: the signing library went in so `creds-check`
could verify a real key, and `default_transport` went in because otherwise
ARMING WOULD HAVE BEEN A NO-OP at the end of the two-week dry-run — the bot
would have reported ARMED and sent nothing.

So the guarantee is now `is_armed` plus the caps in `orderbot`, and the two
load-bearing tests below carry the whole weight. Which is why the last section
of this file exists: those two prove an order is NOT sent, and nothing proved
that one CAN be. A signature Kalshi rejects, or a renamed field, would have
looked exactly like a gate refusal in the log, and would have been found on the
first day the bot actually wanted to trade.
"""
import base64

import pytest

from highcast import kalshi_client as kc


# ── ARMING ──────────────────────────────────────────────────────────────────

def test_dry_run_is_the_default_with_no_environment():
    assert kc.is_armed({}) is False


@pytest.mark.parametrize("value", [
    "", "0", "false", "no", "true", "TRUE", "1", "yes", "y",
    "ARMED", "armed", " ARM_LIVE_ORDERS ", "arm_live_orders",
])
def test_arming_requires_the_exact_literal(value):
    """A truthy-looking value must NOT arm it. `if os.environ.get(...)` would
    arm on "0", "false" and "no" — every one of which reads as OFF to a human.
    Only one exact string arms it, and it is not a word anyone types by accident.
    """
    assert kc.is_armed({kc.ARM_ENV: value}) is False


def test_the_one_literal_that_arms_it():
    assert kc.is_armed({kc.ARM_ENV: kc.ARM_VALUE}) is True


def test_dry_run_never_posts():
    """THE test. In dry-run the client must not attempt a network call at all —
    not a POST that is later discarded, not a request built and dropped."""
    calls = []

    class Boom:
        def post(self, *a, **k):
            calls.append(a)
            raise AssertionError("dry-run attempted a POST")

    c = kc.KalshiClient(key_id="k", private_key_pem=None, armed=False,
                        transport=Boom())
    res = c.place_order(ticker="KXHIGHMIA-26AUG01-B94", side="yes",
                        count=1, limit_price=0.75)
    assert res["dry_run"] is True
    assert res["placed"] is False
    assert calls == []


def test_an_armed_client_without_a_key_refuses_rather_than_posting():
    """Arming is necessary but not sufficient. A missing key must fail closed,
    never fall through to an unsigned request."""
    class Boom:
        def post(self, *a, **k):
            raise AssertionError("posted without a signature")

    c = kc.KalshiClient(key_id=None, private_key_pem=None, armed=True,
                        transport=Boom())
    res = c.place_order(ticker="X", side="yes", count=1, limit_price=0.5)
    assert res["placed"] is False
    assert "credential" in res["reason"].lower()


# ── SIGNING ─────────────────────────────────────────────────────────────────

def test_the_signed_message_is_timestamp_method_path():
    """Kalshi signs `timestamp + METHOD + path`, with the path EXCLUDING the
    query string. Including the query would produce a signature the server
    cannot reproduce, and the failure would look like a permissions error."""
    msg = kc.signing_message(1735689600000, "POST",
                             "/trade-api/v2/portfolio/orders?foo=1")
    assert msg == "1735689600000POST/trade-api/v2/portfolio/orders"


def test_signature_verifies_against_the_public_key():
    """Round-trip with a throwaway key: sign, then verify with the public half.
    Pins the algorithm (RSA-PSS, SHA-256, digest-length salt) so a silent change
    is caught here rather than as an opaque 401 at 02:00 Manila."""
    crypto = pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding, rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()).decode()

    msg = kc.signing_message(1735689600000, "GET", "/trade-api/v2/portfolio")
    sig = kc.sign(pem, msg)

    key.public_key().verify(
        base64.b64decode(sig), msg.encode(),
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=hashes.SHA256.digest_size),
        hashes.SHA256())


def test_signing_without_a_key_raises_rather_than_returning_empty():
    """An empty signature would be sent and rejected; a raise stops the request
    before it leaves the machine."""
    with pytest.raises(Exception):
        kc.sign(None, "anything")


# ── CREDENTIAL HYGIENE ──────────────────────────────────────────────────────

def test_the_private_key_never_appears_in_repr_or_errors():
    """A key echoed into a log or a traceback is a leaked key. Errors from this
    module get read in terminals and pasted into chats."""
    secret = "-----BEGIN PRIVATE KEY-----\nSUPERSECRETMATERIAL\n-----END PRIVATE KEY-----"
    c = kc.KalshiClient(key_id="kid123", private_key_pem=secret, armed=False)
    assert "SUPERSECRETMATERIAL" not in repr(c)
    assert "SUPERSECRETMATERIAL" not in str(c)
    res = c.place_order(ticker="X", side="yes", count=1, limit_price=0.5)
    assert "SUPERSECRETMATERIAL" not in str(res)


def test_credentials_are_read_from_the_environment_not_a_config_file():
    """Config files get committed. This project already ships a Discord webhook
    in config/klax.yaml — a private key must never follow it."""
    env = {kc.KEY_ID_ENV: "abc", kc.PRIVATE_KEY_ENV: "pem-material"}
    kid, pem = kc.credentials_from_env(env)
    assert (kid, pem) == ("abc", "pem-material")
    assert kc.credentials_from_env({}) == (None, None)


# ── WHERE THE CREDENTIALS ACTUALLY LIVE ─────────────────────────────────────
#
# The bot runs under launchd, which does NOT inherit a login shell's
# environment. Exports in ~/.zshrc work when Julian runs it by hand and silently
# do nothing at 02:00 — the worst possible failure, because it looks configured.
# And the launchd plists live IN THE REPO, so a key cannot go there either.
# Hence a credentials file outside the repo.

def test_a_credentials_file_outside_the_repo_is_loaded(tmp_path):
    f = tmp_path / "credentials"
    f.write_text("KALSHI_API_KEY_ID=kid-123\nKALSHI_PRIVATE_KEY=inline-pem\n")
    f.chmod(0o600)
    kid, pem = kc.load_credentials({}, path=f)
    assert (kid, pem) == ("kid-123", "inline-pem")


def test_a_real_environment_wins_over_the_file(tmp_path):
    """So an interactive run or a test can override without editing the file."""
    f = tmp_path / "credentials"
    f.write_text("KALSHI_API_KEY_ID=from-file\nKALSHI_PRIVATE_KEY=from-file\n")
    f.chmod(0o600)
    kid, pem = kc.load_credentials(
        {kc.KEY_ID_ENV: "from-env", kc.PRIVATE_KEY_ENV: "from-env"}, path=f)
    assert (kid, pem) == ("from-env", "from-env")


def test_the_key_may_be_a_PATH_to_a_pem_rather_than_inline(tmp_path):
    """Pasting a multi-line PEM into a KEY=VALUE file is error-prone, so the
    file may instead point at the .pem Kalshi downloaded."""
    pem = tmp_path / "k.pem"
    pem.write_text("-----BEGIN PRIVATE KEY-----\nMATERIAL\n-----END PRIVATE KEY-----")
    pem.chmod(0o600)
    f = tmp_path / "credentials"
    f.write_text(f"KALSHI_API_KEY_ID=kid\nKALSHI_PRIVATE_KEY_FILE={pem}\n")
    f.chmod(0o600)
    kid, got = kc.load_credentials({}, path=f)
    assert kid == "kid"
    assert "MATERIAL" in got


def test_a_world_readable_credentials_file_is_REFUSED(tmp_path):
    """A private key readable by every process on the machine is not a secret.
    Refusing is better than warning — a warning in a launchd log is never read."""
    f = tmp_path / "credentials"
    f.write_text("KALSHI_API_KEY_ID=kid\nKALSHI_PRIVATE_KEY=pem\n")
    f.chmod(0o644)
    kid, pem = kc.load_credentials({}, path=f)
    assert (kid, pem) == (None, None)


def test_a_missing_credentials_file_is_not_an_error(tmp_path):
    """Absent credentials is the NORMAL state during the dry-run fortnight. It
    must return empty, not raise, or the wake cycle dies every 5 minutes."""
    assert kc.load_credentials({}, path=tmp_path / "nope") == (None, None)


def test_comments_and_blank_lines_are_tolerated(tmp_path):
    f = tmp_path / "credentials"
    f.write_text("# my kalshi creds\n\nKALSHI_API_KEY_ID=kid\n"
                 "  KALSHI_PRIVATE_KEY = pem-with-spaces  \n")
    f.chmod(0o600)
    assert kc.load_credentials({}, path=f) == ("kid", "pem-with-spaces")


def test_the_default_location_is_outside_the_repo():
    """If it defaulted inside the project it would eventually be committed —
    the same way config/klax.yaml already carries a live Discord webhook."""
    import pathlib
    d = str(kc.DEFAULT_CREDENTIALS_PATH)
    assert str(pathlib.Path.cwd()) not in d
    assert d.startswith(str(pathlib.Path.home()))


# ── DEMO (PAPER) ENVIRONMENT ────────────────────────────────────────────────
#
# Kalshi runs a full paper-trading sandbox with fake funds. It is where order
# placement gets tested: same auth, same endpoints, same fills, no money.
# Credentials are NOT shared with production, so a demo key cannot reach the
# live book even by mistake.

def test_the_demo_base_is_a_distinct_host_from_production():
    assert kc.DEMO_BASE != kc.BASE
    assert "demo" in kc.DEMO_BASE


def test_a_demo_client_is_not_gated_by_the_production_arming_literal():
    """Demo money is fake, so requiring the live-trading literal here would make
    testing harder for no safety gain — and would train the habit of setting
    the arming variable, which is the exact habit that must stay rare."""
    c = kc.KalshiClient(key_id="k", private_key_pem="pem", armed=True,
                        base=kc.DEMO_BASE)
    assert c.is_demo is True


def test_a_production_client_is_not_marked_demo():
    c = kc.KalshiClient(key_id="k", private_key_pem="pem", armed=True)
    assert c.is_demo is False


def test_the_order_result_says_which_environment_it_used():
    """A fill log that does not name its environment is how a demo test gets
    mistaken for a live one, and vice versa."""
    c = kc.KalshiClient(key_id="k", private_key_pem=None, armed=False,
                        base=kc.DEMO_BASE)
    res = c.place_order(ticker="X", side="yes", count=1, limit_price=0.5)
    assert res["env"] == "demo"
    assert kc.KalshiClient(key_id="k", private_key_pem=None, armed=False
                           ).place_order(ticker="X", side="yes", count=1,
                                         limit_price=0.5)["env"] == "production"


# ── THE PATH ACTUALLY WORKING ───────────────────────────────────────────────
#
# Everything above is about NOT sending an order. These are about the opposite
# question, which nothing else in this suite asks: when it finally does send
# one, does the request get built, does the id come back, can it be cancelled?


class FakeResp:
    def __init__(self, status=201, body=None):
        self.status_code = status
        self._body = body or {}

    def json(self):
        return self._body


class Recorder:
    """Records calls instead of making them. Not a network stub for its own
    sake — the assertions below are about WHAT gets sent."""

    def __init__(self, responses=None):
        self.calls = []
        self._responses = responses or {}

    def _r(self, verb, url):
        self.calls.append((verb, url))
        return self._responses.get(verb, FakeResp())

    def post(self, url, json=None, headers=None):
        self.calls.append(("POST", url, json))
        return self._responses.get("POST", FakeResp(
            201, {"order": {"order_id": "ord-abc", "status": "resting"}}))

    def get(self, url, headers=None):
        return self._r("GET", url)

    def delete(self, url, headers=None):
        return self._r("DELETE", url)


def _armed_client(transport, base=None):
    crypto_pem = _throwaway_pem()
    kw = {"base": base} if base else {}
    return kc.KalshiClient(key_id="kid", private_key_pem=crypto_pem,
                           armed=True, transport=transport, **kw)


def _throwaway_pem():
    pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa as _rsa
    k = _rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return k.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()).decode()


def _install_demo_creds(tmp_path, monkeypatch):
    """Point a demo credentials file at a throwaway key.

    Via KALSHI_PRIVATE_KEY_FILE, not an inline value: a PEM is multi-line and
    the credentials file is KEY=VALUE, so inlining it truncates the key at the
    first newline and every signature fails. Which is the whole reason the
    _FILE form exists — and these tests hit the same wall real users would.
    """
    pem = tmp_path / "demo.pem"
    pem.write_text(_throwaway_pem())
    pem.chmod(0o600)
    f = tmp_path / "credentials-demo"
    f.write_text(f"KALSHI_API_KEY_ID=demo-kid\nKALSHI_PRIVATE_KEY_FILE={pem}\n")
    f.chmod(0o600)
    monkeypatch.setattr(kc, "DEMO_CREDENTIALS_PATH", f)
    return f


def test_an_armed_order_returns_the_order_id():
    """The id is the ONLY handle on a resting order. Without it there is no way
    to cancel what was just placed — a smoke test becomes an abandoned order."""
    t = Recorder()
    res = _armed_client(t).place_order(ticker="KXHIGHMIA-X", side="yes",
                                       count=1, limit_price=0.01)
    assert res["placed"] is True
    assert res["order_id"] == "ord-abc"
    verb, url, body = t.calls[0]
    assert verb == "POST"
    assert body["yes_price"] == 1 and body["count"] == 1
    assert body["type"] == "limit" and body["action"] == "buy"


def test_arming_alone_is_enough_to_send__no_transport_needed():
    """THE REGRESSION THIS EXISTS FOR. `place_order` used to answer 'armed but
    no transport configured' when nobody injected one — and production injects
    nothing. So the two-week dry-run would have ended with the arming literal
    set and the bot sending precisely nothing, while the log said ARMED. This
    asserts a real client gets built on demand."""
    c = kc.KalshiClient(key_id="kid", private_key_pem=_throwaway_pem(),
                        armed=True)
    assert c._transport is None
    client = c._client()
    assert client is not None
    assert hasattr(client, "post") and hasattr(client, "delete")


def test_an_unarmed_client_never_builds_a_transport():
    """The counterpart: lazily defaulting the transport must not weaken dry-run.
    An unarmed client must not so much as construct an HTTP client."""
    c = kc.KalshiClient(key_id="kid", private_key_pem="pem", armed=False)
    c.place_order(ticker="X", side="yes", count=1, limit_price=0.5)
    c.get_order("ord-1")
    c.cancel_order("ord-1")
    c.list_markets("KXHIGHMIA")
    c.positions()
    assert c._transport is None


def test_cancel_uses_DELETE_on_the_order_id():
    t = Recorder({"DELETE": FakeResp(200, {})})
    res = _armed_client(t).cancel_order("ord-abc")
    assert res["ok"] is True
    assert t.calls[0][0] == "DELETE"
    assert t.calls[0][1].endswith("/ord-abc")


def test_a_query_string_is_excluded_from_the_signature_but_kept_in_the_url():
    """`list_markets` is the first caller with a query string. Kalshi signs the
    path WITHOUT it; the request must still carry it, or the server filters
    nothing and returns every market on the exchange."""
    t = Recorder({"GET": FakeResp(200, {"markets": [{"ticker": "T-1"}]})})
    res = _armed_client(t).list_markets("KXHIGHMIA")
    url = t.calls[0][1]
    assert "series_ticker=KXHIGHMIA" in url and "status=open" in url
    assert res["body"]["markets"][0]["ticker"] == "T-1"
    assert kc.signing_message(1, "GET", "/p?a=1") == "1GET/p"


def test_a_transport_exception_never_leaks_the_message():
    """An exception message can carry a URL — and a URL can carry a signature.
    Only the exception TYPE is reported."""
    class Leaky:
        def post(self, *a, **k):
            raise RuntimeError("failed for https://x/?sig=SECRETSIGNATURE")

    res = _armed_client(Leaky()).place_order(ticker="X", side="yes", count=1,
                                             limit_price=0.01)
    assert res["placed"] is False
    assert "SECRETSIGNATURE" not in str(res)
    assert "RuntimeError" in res["reason"]


# ── THE SMOKE TEST ──────────────────────────────────────────────────────────

def test_smoke_test_places_reads_and_cancels(tmp_path, monkeypatch):
    _install_demo_creds(tmp_path, monkeypatch)
    t = Recorder({"GET": FakeResp(200, {"markets": [{"ticker": "T-1"}],
                                        "order": {"status": "resting"}}),
                  "DELETE": FakeResp(200, {})})
    lines, ok = kc.smoke_test({}, demo=True, transport=t)
    out = "\n".join(lines)
    assert ok, out
    assert "ORDER PATH WORKS END TO END" in out
    assert [c[0] for c in t.calls] == ["GET", "POST", "GET", "DELETE"]


def test_smoke_test_cancels_even_when_the_read_back_fails(tmp_path, monkeypatch):
    """An abandoned resting order is the only genuinely bad outcome this
    procedure can produce, so cleanup must not be conditional on the step
    before it succeeding."""
    _install_demo_creds(tmp_path, monkeypatch)

    calls = []

    class ReadFails(Recorder):
        def get(self, url, headers=None):
            calls.append(("GET", url))
            if "markets" in url:
                return FakeResp(200, {"markets": [{"ticker": "T-1"}]})
            return FakeResp(500, {})

        def delete(self, url, headers=None):
            calls.append(("DELETE", url))
            return FakeResp(200, {})

    lines, ok = kc.smoke_test({}, demo=True, transport=ReadFails())
    assert ok is False
    assert any(c[0] == "DELETE" for c in calls), "the cancel was skipped"


def test_a_production_smoke_test_refuses_unless_armed():
    """One way to enable real orders, not two. The manual test rehearses the
    exact ritual the bot will need."""
    lines, ok = kc.smoke_test({}, demo=False)
    assert ok is False
    assert "not armed" in "\n".join(lines).lower()


def test_the_test_price_is_hard_capped(tmp_path, monkeypatch):
    """A 'test' order that can fill at a real price is trading with extra
    steps. 1 contract at <=10c cannot fill on anything worth buying."""
    _install_demo_creds(tmp_path, monkeypatch)
    t = Recorder({"GET": FakeResp(200, {"markets": [{"ticker": "T-1"}],
                                        "order": {"status": "resting"}}),
                  "DELETE": FakeResp(200, {})})
    kc.smoke_test({}, demo=True, price_cents=99, transport=t)
    post = [c for c in t.calls if c[0] == "POST"][0]
    assert post[2]["yes_price"] == kc.MAX_TEST_PRICE_CENTS


def test_the_demo_smoke_test_ignores_exported_production_credentials(tmp_path,
                                                                     monkeypatch):
    """If KALSHI_API_KEY_ID is exported for production, an env-wins load would
    send the PRODUCTION key id to the demo host. Nothing unsafe happens, but
    the failure looks like a broken demo signup rather than a mixed-up key."""
    f = _install_demo_creds(tmp_path, monkeypatch)
    kid, _pem = kc.load_credentials({kc.KEY_ID_ENV: "PRODUCTION-KID",
                                     kc.PRIVATE_KEY_ENV: "prod-pem"},
                                    path=f, use_env=False)
    assert kid == "demo-kid"


def test_missing_demo_credentials_explains_that_demo_is_a_separate_account(tmp_path,
                                                                          monkeypatch):
    monkeypatch.setattr(kc, "DEMO_CREDENTIALS_PATH", tmp_path / "nope")
    lines, ok = kc.smoke_test({}, demo=True)
    assert ok is False
    assert "separate account" in "\n".join(lines).lower()


# ── AT-MARKET: A REAL POSITION, NOT A TEST ──────────────────────────────────
#
# `allow_fill=True` is meant to fill. It proves the one thing the 1c round trip
# cannot — that a fill happens — and the cost of that proof is a real position.

def test_at_market_sizes_to_the_budget_and_never_lowers_the_price(tmp_path,
                                                                  monkeypatch):
    """Size gives way to the budget; price never does. A cheaper price is a
    DIFFERENT order that may not fill, which is the thing being tested."""
    _install_demo_creds(tmp_path, monkeypatch)
    t = Recorder({"GET": FakeResp(200, {"order": {"status": "executed"}})})
    kc.smoke_test({}, demo=True, ticker="T-1", price_cents=56, count=99,
                  allow_fill=True, transport=t)
    body = [c for c in t.calls if c[0] == "POST"][0][2]
    assert body["yes_price"] == 56, "the price was lowered to fit the budget"
    assert body["count"] == 1, "1 x 56c is all that fits in $1"


def test_the_dollar_cap_cannot_be_raised_by_the_caller(tmp_path, monkeypatch):
    """`--dollars 50` must not buy $50 of anything. The ceiling lives next to
    the code that spends, not in the CLI where a flag can move it."""
    _install_demo_creds(tmp_path, monkeypatch)
    t = Recorder({"GET": FakeResp(200, {"order": {"status": "executed"}})})
    kc.smoke_test({}, demo=True, ticker="T-1", price_cents=10, count=500,
                  allow_fill=True, max_dollars=50.0, transport=t)
    body = [c for c in t.calls if c[0] == "POST"][0][2]
    assert body["count"] == 10, "10 x 10c = $1.00, the hard cap"


def test_at_market_does_not_cancel_by_default(tmp_path, monkeypatch):
    """A filled order cannot be cancelled; attempting it would log a failure
    that actually means 'it worked'."""
    _install_demo_creds(tmp_path, monkeypatch)
    t = Recorder({"GET": FakeResp(200, {"order": {"status": "executed"}})})
    lines, ok = kc.smoke_test({}, demo=True, ticker="T-1", price_cents=56,
                              count=1, allow_fill=True, transport=t)
    assert "DELETE" not in [c[0] for c in t.calls]
    assert ok is True
    assert "real position" in "\n".join(lines).lower()


def test_at_market_can_still_be_told_to_cancel(tmp_path, monkeypatch):
    _install_demo_creds(tmp_path, monkeypatch)
    t = Recorder({"GET": FakeResp(200, {"order": {"status": "resting"}}),
                  "DELETE": FakeResp(200, {})})
    kc.smoke_test({}, demo=True, ticker="T-1", price_cents=56, count=1,
                  allow_fill=True, cancel=True, transport=t)
    assert "DELETE" in [c[0] for c in t.calls]


def test_a_price_above_the_cap_refuses_rather_than_buying_zero(tmp_path,
                                                               monkeypatch):
    """At $1 nothing priced over $1 is affordable. Sending count=0 would be an
    exchange error; saying so is a refusal."""
    _install_demo_creds(tmp_path, monkeypatch)
    t = Recorder()
    lines, ok = kc.smoke_test({}, demo=True, ticker="T-1", price_cents=99,
                              count=1, allow_fill=True, max_dollars=0.50,
                              transport=t)
    assert ok is False
    assert t.calls == []
    assert "cap" in "\n".join(lines).lower()


def test_the_round_trip_mode_is_unaffected_by_the_new_flags(tmp_path,
                                                            monkeypatch):
    """The default must stay the free, unfillable, always-cancelled probe."""
    _install_demo_creds(tmp_path, monkeypatch)
    t = Recorder({"GET": FakeResp(200, {"markets": [{"ticker": "T-1"}],
                                        "order": {"status": "resting"}}),
                  "DELETE": FakeResp(200, {})})
    lines, ok = kc.smoke_test({}, demo=True, count=99, price_cents=99,
                              transport=t)
    body = [c for c in t.calls if c[0] == "POST"][0][2]
    assert body["count"] == 1 and body["yes_price"] == kc.MAX_TEST_PRICE_CENTS
    assert "DELETE" in [c[0] for c in t.calls]
    assert ok is True


# ── SELF-EXPIRING ARMING ────────────────────────────────────────────────────
#
# The bot exists so Julian can sleep through the 02:00 Manila commit, so the
# arming literal must live in the launchd plist (a shell export dies with the
# shell). But a literal in a plist is armed EVERY day until removed. The window
# makes forgetting fail safe instead of leaving a live bot running unattended.

@pytest.mark.parametrize("env", [
    {},                                          # nothing set
    {kc.ARM_UNTIL_ENV: ""},                      # blank
    {kc.ARM_UNTIL_ENV: "   "},                   # whitespace
    {kc.ARM_UNTIL_ENV: "tomorrow"},              # not a date
    {kc.ARM_UNTIL_ENV: "2026-13-45"},            # impossible date
    {kc.ARM_UNTIL_ENV: "08/01/2026"},            # wrong format
])
def test_a_missing_or_malformed_window_is_NOT_armed(env):
    """Fails closed on everything. An operator who sets the literal and forgets
    the window gets a dry run, not an open-ended live bot."""
    assert kc.arming_window_ok(env, "2026-08-01") is False


def test_the_window_admits_its_last_day_and_expires_after_it():
    env = {kc.ARM_UNTIL_ENV: "2026-08-01"}
    assert kc.arming_window_ok(env, "2026-07-31") is True   # before
    assert kc.arming_window_ok(env, "2026-08-01") is True   # the day itself
    assert kc.arming_window_ok(env, "2026-08-02") is False  # the day after


def test_the_window_is_independent_of_the_literal():
    """Two separate conditions, deliberately. `is_armed` keeps its exact meaning
    so the interactive `order-test` -- run by hand, watched, one-shot -- is
    unaffected; only the UNATTENDED bot needs the end date."""
    assert kc.is_armed({kc.ARM_ENV: kc.ARM_VALUE}) is True
    assert kc.arming_window_ok({kc.ARM_ENV: kc.ARM_VALUE}, "2026-08-01") is False
