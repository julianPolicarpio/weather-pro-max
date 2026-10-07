import pytest

from highcast.fetchers import kalshi
from highcast.web import app as app_module


@pytest.fixture(autouse=True)
def _no_real_discord(request, monkeypatch):
    """⚠️ THE TEST SUITE COULD SEND JULIAN A REAL DISCORD MESSAGE. Caught
    2026-08-12, after it actually did.

    `tests/test_auto.py` calls `auto_cycle(conn, CFG, ...)` with the LIVE config,
    and `config/klax.yaml` carries a real `discord_webhook_url`. The Kalshi stub
    below covers the market boundary but never covered the webhook, so
    `notify.send` was free to POST from a unit test.

    It went unnoticed until `notify_missed_commit` shipped (§2.12): that alert
    fires when a commit hour has passed with no bet recorded, and a `tmp_path`
    database has no bets EVER, so every commit looks missed. It only fires while
    the window is still open, which makes it wall-clock dependent — the suite
    passed all morning and then posted an alert in the afternoon.

    A unit test must never produce an outward-facing side effect.

    ⚠️ THE GUARD IS ON `httpx.post`, NOT ON `notify.send`. Blocking `send` was the
    first attempt and it was too blunt: `test_no_webhook_configured_is_a_silent_no_op`
    and `test_send_never_raises_on_a_network_error` legitimately CALL `send` to
    exercise its own logic, and neither reaches the network. Guarding the actual
    HTTP boundary lets those pass while still making a real POST impossible.

    A test that patches `httpx.post` itself still wins — monkeypatch applies the
    fixture first and the test's own patch second. `@pytest.mark.live` is exempt,
    matching the Kalshi convention.
    """
    if not request.node.get_closest_marker("live"):
        monkeypatch.setattr("httpx.post", lambda *a, **k: pytest.fail(
            "httpx.post() called in a non-live test — a unit test must not send "
            "anything outward (this is the Discord webhook path). Stub it in the "
            "test, or mark it @pytest.mark.live."))


@pytest.fixture(autouse=True)
def _kalshi_test_isolation(request, monkeypatch):
    """Keep the Kalshi live-market integration out of the default test run.

    Two concerns:
    1. The in-process cache in highcast.web.app is module-level and keyed by
       calendar date, which is identical across every test in a single pytest
       process -- well within the 120s TTL. Without clearing it, whichever
       test runs first "wins" the cache for that date and later tests that
       stub the fetch differently would silently get a stale result. Clear
       it before every test (including @pytest.mark.live ones, where it's a
       no-op).
    2. Every non-live test defaults kalshi.get_json (the network boundary,
       not fetch_ranges itself) to an empty payload, so /api/today never
       makes a real HTTP call by default -- but tests exercising
       fetch_ranges's own logic (highcast/tests/test_kalshi_fetcher.py) can
       still monkeypatch get_json themselves to see their own fixture
       payload, and app-level tests can still monkeypatch fetch_ranges
       directly to inject ready-made ranges. @pytest.mark.live tests are
       exempted so they can exercise the real network path end to end.
    """
    app_module._KALSHI_CACHE.clear()
    if not request.node.get_closest_marker("live"):
        monkeypatch.setattr(kalshi, "get_json", lambda *a, **k: {"markets": []})
