"""The ONLY module in this project that can spend money.

Design: `docs/superpowers/specs/2026-08-01-kmia-high-order-bot-design.md` §5-6.

SEPARATION OF CONCERNS. This file does transport and authentication and knows
nothing about weather; `orderbot.py` decides WHAT to order and knows nothing
about HTTP. Neither is testable if they are one file: the money rails need to be
exercised without a network, and the signing needs to be exercised without a
model.

DRY-RUN IS THE DEFAULT.

⚠️ THE ORIGINAL "IT IS PHYSICALLY IMPOSSIBLE" CLAIM NO LONGER HOLDS, AND SAYING
SO MATTERS MORE THAN THE CLAIM DID. It rested on two accidents: `cryptography`
was not installed, and nothing supplied an HTTP transport. Both changed on
2026-08-01 -- the signing library went in so `creds-check` could verify a real
key, and `default_transport` went in because "armed but no transport configured"
would have made ARMING ITSELF A NO-OP after the two-week dry-run. A safety
property that quietly decays into a false comment is worse than one that was
never claimed, so it is written down here instead.

WHAT ACTUALLY STOPS AN ORDER NOW, in order of how much weight each carries:

  1. `is_armed` -- one exact literal, tested against every truthy-looking
     near-miss. This is the real gate.
  2. `orderbot` caps -- KMIA HIGH only, one order per day, <= $1, price
     ceiling 0.80, and a confidence floor that refuses most days outright.
  3. The commit-hour check -- the runner is inert 23 hours a day.

ARMING TAKES ONE EXACT LITERAL. Not "true", not "1", not "yes" -- those are what
a hurried person types, and `if os.environ.get(...)` would arm on the strings
"0", "false" and "no", every one of which reads as OFF to a human.

CREDENTIALS COME FROM THE ENVIRONMENT AND ARE NEVER ECHOED. Not in `repr`, not in
an exception, not in a returned dict. This project already ships a live Discord
webhook inside `config/klax.yaml`; a private key must never follow it there.

WHAT THIS MODULE DELIBERATELY DOES NOT DO: manage exits. Julian's 99c sell limit
stays his. An exit leg is a separate decision needing its own gate.
"""
from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Any

BASE = "https://api.elections.kalshi.com"
# Kalshi's paper-trading sandbox: same API, same auth, fake funds. This is where
# order placement gets TESTED. Credentials are NOT shared with production, so a
# demo key physically cannot reach the live book.
DEMO_BASE = "https://external-api.demo.kalshi.co"
ORDERS_PATH = "/trade-api/v2/portfolio/orders"
POSITIONS_PATH = "/trade-api/v2/portfolio/positions"
# Verified against docs.kalshi.com/api-reference/portfolio/get-balance,
# 2026-08-02. Returns CENTS -- see `KalshiClient.balance`.
BALANCE_PATH = "/trade-api/v2/portfolio/balance"
MARKETS_PATH = "/trade-api/v2/markets"

# Read from the environment; never from a config file, never a literal here.
KEY_ID_ENV = "KALSHI_API_KEY_ID"
PRIVATE_KEY_ENV = "KALSHI_PRIVATE_KEY"

# The single literal that arms live trading. Deliberately not a word anyone
# types by accident, and deliberately not truthy-shaped.
ARM_ENV = "HIGHCAST_TRADING"
ARM_VALUE = "ARM_LIVE_ORDERS_I_MEAN_IT"


def is_armed(env: dict[str, str]) -> bool:
    """Live trading only when the env var holds EXACTLY `ARM_VALUE`.

    Exact equality, no strip, no case-fold: "arm_live_orders_i_mean_it" and
    " ARM_LIVE_ORDERS_I_MEAN_IT " both stay dry. Somebody who half-remembers the
    value does not get to place orders.
    """
    return env.get(ARM_ENV) == ARM_VALUE


# The LAST target date the unattended bot may trade, as an ISO date.
#
# WHY THIS EXISTS. The bot's whole purpose is to trade at 02:00 Manila so Julian
# can sleep, which means the arming literal has to live in the launchd plist --
# a shell `export` dies with the shell, and launchd inherits nothing from it. But
# a literal sitting in a plist is armed EVERY day until somebody remembers to
# take it out, which is the exact opposite of "arm it for one day".
#
# So the unattended path requires BOTH the literal and an explicit end date, and
# a MISSING end date means NOT ARMED. Forgetting to clean up disarms the bot
# instead of leaving it live: the failure mode of forgetting now points the safe
# way. `is_armed` alone is unchanged, so the interactive `order-test` -- run by
# hand, watched, one-shot -- still needs only the literal.
ARM_UNTIL_ENV = "HIGHCAST_TRADING_UNTIL"


def arming_window_ok(env: dict[str, str], target_date: str) -> bool:
    """May the UNATTENDED bot trade `target_date`? Fails closed on everything.

    Absent, malformed, or past — all False. An operator who sets the literal and
    forgets the window gets a dry run, not an open-ended live bot.
    """
    raw = (env.get(ARM_UNTIL_ENV) or "").strip()
    if not raw:
        return False
    try:
        import datetime as _dt
        return _dt.date.fromisoformat(target_date) <= _dt.date.fromisoformat(raw)
    except (TypeError, ValueError):
        return False


def credentials_from_env(env: dict[str, str]) -> tuple[str | None, str | None]:
    """(key_id, private_key_pem) or (None, None). Callers must fail closed."""
    return env.get(KEY_ID_ENV) or None, env.get(PRIVATE_KEY_ENV) or None


# WHERE THE CREDENTIALS LIVE, and why not the obvious places.
#
# The bot runs under launchd, which does NOT inherit a login shell's
# environment. `export` in ~/.zshrc works when Julian runs the command by hand
# and silently does nothing at 02:00 -- the worst failure mode available,
# because it looks configured. And the launchd plists are checked into
# `launchd/*.plist`, so a key cannot go there either.
#
# So: a KEY=VALUE file OUTSIDE the repo, mode 0600. `config/klax.yaml` already
# ships a live Discord webhook; this must not follow it.
DEFAULT_CREDENTIALS_PATH = Path.home() / ".highcast" / "credentials"

# Demo is a SEPARATE ACCOUNT on a separate host with its own key, so it gets its
# own file. One file holding both would make "which key am I about to send?" a
# question answered by a flag, and the answer would be wrong exactly once.
DEMO_CREDENTIALS_PATH = Path.home() / ".highcast" / "credentials-demo"

# Alternative to pasting a multi-line PEM into a KEY=VALUE file.
PRIVATE_KEY_FILE_ENV = "KALSHI_PRIVATE_KEY_FILE"


def _parse_env_file(text: str) -> dict[str, str]:
    """Minimal KEY=VALUE, tolerating blank lines, # comments and stray spaces.

    Deliberately not a full dotenv parser: fewer rules means fewer ways for a
    hand-edited secrets file to be misread.
    """
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def load_credentials(env: dict[str, str],
                     path: "Path | str | None" = None,
                     use_env: bool = True) -> tuple[str | None, str | None]:
    """(key_id, private_key_pem), from the real environment or the creds file.

    A real environment variable WINS, so an interactive run or a test can
    override without editing the file.

    `use_env=False` reads ONLY the file. That exists for the demo account: if
    KALSHI_API_KEY_ID is exported in the shell for production, an env-wins load
    would quietly send the PRODUCTION key id to the demo host. Nothing unsafe
    happens (it just fails auth), but the failure looks like a broken demo
    signup rather than a mixed-up key, and that costs an evening.

    REFUSES a file that is group- or world-readable. A private key every process
    on the machine can read is not a secret, and refusing beats warning: a
    warning in a launchd log is never read.

    Never raises. Absent credentials is the NORMAL state during the dry-run
    fortnight, and this runs inside a wake cycle that fires every five minutes.
    """
    if not use_env:
        env = {}
    kid = env.get(KEY_ID_ENV) or None
    pem = env.get(PRIVATE_KEY_ENV) or None
    if kid and pem:
        return kid, pem

    p = Path(path) if path is not None else DEFAULT_CREDENTIALS_PATH
    data: dict[str, str] = {}
    try:
        if p.exists():
            mode = p.stat().st_mode
            if mode & 0o077:
                # group/other can read: treat as absent rather than trust it
                return None, None
            data = _parse_env_file(p.read_text())
    except OSError:
        return None, None

    kid = kid or data.get(KEY_ID_ENV) or None
    pem = pem or data.get(PRIVATE_KEY_ENV) or None
    if not pem:
        keyfile = env.get(PRIVATE_KEY_FILE_ENV) or data.get(PRIVATE_KEY_FILE_ENV)
        if keyfile:
            try:
                kp = Path(keyfile).expanduser()
                if kp.exists() and not (kp.stat().st_mode & 0o077):
                    pem = kp.read_text()
            except OSError:
                pem = None
    return kid, pem


def signing_message(timestamp_ms: int, method: str, path: str) -> str:
    """Kalshi signs `timestamp + METHOD + path`, path WITHOUT the query string.

    Including the query would yield a signature the server cannot reproduce, and
    the failure surfaces as an opaque auth error rather than anything pointing at
    the query string.
    """
    return f"{timestamp_ms}{method.upper()}{path.split('?', 1)[0]}"


def sign(private_key_pem: str | None, message: str) -> str:
    """base64 RSA-PSS/SHA-256 signature, digest-length salt.

    Raises when there is no key: an empty signature would be transmitted and
    rejected, whereas a raise stops the request before it leaves the machine.

    `cryptography` is imported HERE, not at module scope, so the rest of the
    application -- and the whole dry-run path -- runs without it installed.
    """
    if not private_key_pem:
        raise ValueError("no private key: refusing to build an unsigned request")
    try:
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import padding
    except ImportError as e:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "signing needs the optional 'trade' extra: "
            "pip install -e '.[trade]'") from e
    key = serialization.load_pem_private_key(
        private_key_pem.encode(), password=None)
    sig = key.sign(
        message.encode(),
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=hashes.SHA256.digest_size),
        hashes.SHA256())
    return base64.b64encode(sig).decode()


def default_transport():
    """A real HTTP client, built only when a request is about to be made.

    WHY THIS EXISTS. `place_order` used to answer "armed but no transport
    configured" whenever nobody injected one -- and production injects nothing.
    So the two-week dry-run would have ended with Julian setting the arming
    literal and the bot continuing to send precisely nothing, while the log said
    it was armed. That is this project's least favourite failure: configured-
    looking and inert. The same shape as the launchd env-var trap the credential
    loader was written to avoid.

    `httpx` is already a hard dependency (`highcast.http`), so this adds nothing
    to install. It is imported lazily anyway: no import happens on the dry-run
    path, which keeps the money rails out of every other module's import graph.
    """
    import httpx
    return httpx.Client(timeout=20.0)


class KalshiClient:
    """Signed access to Kalshi's trading API. Dry-run unless explicitly armed."""

    def __init__(self, key_id: str | None, private_key_pem: str | None,
                 armed: bool = False, transport: Any = None,
                 base: str = BASE) -> None:
        self._key_id = key_id
        self._pem = private_key_pem          # never rendered; see __repr__
        self.armed = bool(armed)
        self._transport = transport
        self._base = base
        # Demo is fake money, so it is not gated behind the production arming
        # literal -- requiring it there would add no safety and would train the
        # habit of setting that variable, which must stay rare.
        self.is_demo = base == DEMO_BASE

    def __repr__(self) -> str:
        """Never renders the key. Errors from this module get read in terminals
        and pasted into chats."""
        have = "yes" if self._pem else "no"
        return (f"KalshiClient(key_id={self._key_id!r}, private_key=<{have}>, "
                f"armed={self.armed})")

    __str__ = __repr__

    def _client(self):
        """The injected transport, or a real one. NEVER reached while unarmed:
        every caller checks `self.armed` first, so an unarmed client builds no
        HTTP client at all."""
        if self._transport is None:
            self._transport = default_transport()
        return self._transport

    def _send(self, method: str, path: str, *, json_body=None,
              require_arm: bool = True) -> dict:
        """Signed request → {ok, status, body, reason}. Never raises, never
        returns credential material.

        Used by everything EXCEPT the dry-run branch of `place_order`, which
        returns before any of this is touched.

        `require_arm` DEFAULTS TO TRUE so anything added later is gated unless
        its author opts out deliberately. Only the two portfolio READS pass
        False -- see the read/trade note above `_can_read`. `place_order` uses
        the default and stays armed-only; `tests/test_kalshi_reads_need_no_arming
        .py::test_place_order_still_refuses_while_unarmed` fails loudly if that
        ever stops being true.
        """
        if require_arm and not self.armed:
            return {"ok": False, "status": 0, "body": {},
                    "reason": f"not armed (set {ARM_ENV} to arm)"}
        if not self._key_id or not self._pem:
            return {"ok": False, "status": 0, "body": {},
                    "reason": "credentials missing — refusing to send unsigned"}
        try:
            headers = self._headers(method, path)
            c = self._client()
            url = self._base + path
            if method == "POST":
                r = c.post(url, json=json_body, headers=headers)
            elif method == "DELETE":
                r = c.delete(url, headers=headers)
            else:
                r = c.get(url, headers=headers)
            status = getattr(r, "status_code", 0)
            try:
                body = r.json()
            except Exception:  # noqa: BLE001 - a non-JSON error page is normal
                body = {}
            return {"ok": status in (200, 201), "status": status, "body": body,
                    "reason": f"HTTP {status}"}
        except Exception as e:  # noqa: BLE001 - unattended callers must not die
            # Type only: an exception message can carry a URL with a signature.
            return {"ok": False, "status": 0, "body": {},
                    "reason": f"{method} failed: {type(e).__name__}"}

    def _headers(self, method: str, path: str) -> dict[str, str]:
        ts = int(time.time() * 1000)
        return {
            "KALSHI-ACCESS-KEY": self._key_id or "",
            "KALSHI-ACCESS-SIGNATURE": sign(self._pem,
                                            signing_message(ts, method, path)),
            "KALSHI-ACCESS-TIMESTAMP": str(ts),
            "Content-Type": "application/json",
        }

    def place_order(self, *, ticker: str, side: str, count: int,
                    limit_price: float) -> dict:
        """Place a limit buy, or describe what would have been placed.

        Returns a dict rather than raising, so an unattended caller logs a
        refusal instead of dying mid-wake-cycle. The return NEVER contains
        credential material.
        """
        intent = {
            "ticker": ticker, "side": side, "count": int(count),
            "limit_price_cents": int(round(limit_price * 100)),
            "dry_run": not self.armed, "placed": False, "reason": "",
            # Named explicitly: a fill log that does not say which environment it
            # used is how a demo test gets mistaken for a live one.
            "env": "demo" if self.is_demo else "production",
        }
        # DRY-RUN RETURNS BEFORE ANYTHING IS BUILT OR SENT. No request is
        # constructed, no signature attempted, no transport touched.
        if not self.armed:
            intent["reason"] = ("dry-run: not armed "
                                f"(set {ARM_ENV} to arm)")
            return intent
        if not self._key_id or not self._pem:
            intent["reason"] = ("armed but credentials are missing — refusing "
                                "to send an unsigned order")
            return intent
        body = {"ticker": ticker, "action": "buy", "side": side,
                "count": int(count), "type": "limit",
                "yes_price": int(round(limit_price * 100))}
        res = self._send("POST", ORDERS_PATH, json_body=body)
        intent["placed"] = res["ok"]
        intent["reason"] = res["reason"]
        # THE ORDER ID IS THE ONLY HANDLE ON A RESTING ORDER. Without it there
        # is no way to cancel what was just placed, which is the difference
        # between a smoke test and an abandoned live order.
        intent["order_id"] = ((res["body"] or {}).get("order") or {}).get("order_id")
        return intent

    def get_order(self, order_id: str) -> dict:
        """Read one order back. Proves the exchange actually accepted it —
        an HTTP 201 says the request was well-formed, not that an order rests."""
        return self._send("GET", f"{ORDERS_PATH}/{order_id}")

    def cancel_order(self, order_id: str) -> dict:
        """Cancel a resting order. The smoke test's cleanup step, and the reason
        a live placement test costs nothing: place, read, cancel, seconds apart."""
        return self._send("DELETE", f"{ORDERS_PATH}/{order_id}")

    def list_markets(self, series_ticker: str, *, status: str = "open") -> dict:
        """Open markets for a series. The smoke test needs a ticker that really
        exists in THIS environment — demo's markets are not production's, and a
        guessed ticker fails in a way that looks like an auth problem."""
        return self._send(
            "GET", f"{MARKETS_PATH}?series_ticker={series_ticker}&status={status}")

    # ── READS vs TRADES ──────────────────────────────────────────────────────
    # `self.armed` is the LIVE-ORDER arming flag: the literal plus an unexpired
    # date, deliberately hard to set because it lets the bot spend money while
    # Julian sleeps. It gates `place_order` and must keep gating it.
    #
    # It should NEVER have gated a READ. Reading a balance cannot lose money, and
    # requiring the arming literal to look at your own account inverts the safety
    # property: it trains you to set the trade-arming variable for a harmless
    # operation, which is exactly the habit the arming design exists to prevent.
    # Kalshi agrees -- the API issues keys scoped `read::portfolio_balance`,
    # narrower than any trading permission, precisely so reads need no trade
    # rights (docs.kalshi.com/api-reference/portfolio/get-balance).
    #
    # So reads require CREDENTIALS ONLY. `place_order` is untouched below and
    # still requires `self.armed`; `tests/test_kalshi_reads_need_no_arming.py`
    # pins both halves so this split can never drift into ungating a trade.
    def _can_read(self) -> bool:
        return bool(self._key_id and self._pem)

    def positions(self) -> dict:
        """Current portfolio positions. Read-only: credentials, not arming."""
        if not self._can_read():
            return {"positions": [], "reason": "no credentials"}
        res = self._send("GET", POSITIONS_PATH, require_arm=False)
        return res["body"] if res["ok"] else {"positions": [],
                                              "reason": res["reason"]}

    def balance(self) -> dict:
        """Account balance. Read-only: credentials, not arming.

        ⚠️ KALSHI RETURNS CENTS. `balance` and `portfolio_value` are integer
        CENTS; `balance_dollars` is a decimal string ("0.5600"). Verified against
        docs.kalshi.com/api-reference/portfolio/get-balance on 2026-08-02.
        Everything downstream of this -- `StakingConfig.bankroll`, every stake,
        every cap -- is in DOLLARS, so treating cents as dollars would size every
        position 100x too large. The conversion happens HERE, once, and
        `balance_usd` is the only field callers should use.
        """
        if not self._can_read():
            return {"balance_usd": None, "reason": "no credentials"}
        res = self._send("GET", BALANCE_PATH, require_arm=False)
        if not res["ok"]:
            return {"balance_usd": None, "reason": res["reason"]}
        body = dict(res["body"] or {})
        cents = body.get("balance")
        body["balance_usd"] = (None if cents is None else float(cents) / 100.0)
        pv = body.get("portfolio_value")
        body["portfolio_value_usd"] = (None if pv is None else float(pv) / 100.0)
        return body


MAX_TEST_PRICE_CENTS = 10

# The hard spend ceiling for anything this function can do, in dollars. Julian's
# stated cap. It is enforced HERE rather than in the CLI so a typo'd flag cannot
# raise it -- a number that guards money belongs next to the code that spends it.
MAX_TEST_DOLLARS = 1.00


def smoke_test(env: dict[str, str], *, demo: bool = True,
               series: str = "KXHIGHMIA", ticker: str | None = None,
               price_cents: int = 1, count: int = 1, allow_fill: bool = False,
               cancel: bool | None = None, max_dollars: float = MAX_TEST_DOLLARS,
               transport=None) -> tuple[list[str], bool]:
    """Prove the whole order path end to end: place → read back → cancel.

    THE POINT. Every safety property in this file is about NOT sending an order.
    None of them answer the opposite question -- when it finally does send one,
    does it work? A 401 from a malformed signature, a rejected ticker format, a
    field Kalshi renamed: all of those look identical to "the gates refused" in
    the log, and all of them would be discovered on the first day the bot
    actually wanted to trade. This finds them on a day of our choosing.

    WHY IT COSTS NOTHING EVEN AGAINST REAL MONEY. The order is 1 contract at 1c,
    which cannot fill on a bucket anyone is willing to sell, and it is cancelled
    seconds later. Worst case in the entire procedure is one cent. The price is
    hard-capped at MAX_TEST_PRICE_CENTS: a test order is not a position, and a
    "test" that can fill at a real price is just trading with extra steps.

    THE CANCEL RUNS EVEN IF THE READ-BACK FAILS. An abandoned resting order is
    the only genuinely bad outcome available here, so cleanup is unconditional.

    ── TWO MODES, AND THE DIFFERENCE IS NOT COSMETIC ────────────────────────

    `allow_fill=False` (default) — a 1c order that cannot fill, cancelled
    seconds later. Proves auth, signing, ticker format, order body, order id,
    read-back and cancel: every mechanical step. Costs nothing. Takes no
    position.

    `allow_fill=True` — a marketable order that is MEANT to fill, and is
    therefore a REAL POSITION, not a test. It proves the one thing the 1c
    version cannot: that a fill actually happens. It is not cancelled by
    default, because a filled order cannot be cancelled and pretending
    otherwise would log a spurious failure. Total spend is capped at
    `max_dollars` and `count` is reduced to fit -- the cap is enforced here, not
    by the caller, so a typo'd flag cannot raise it.
    """
    # The caller may only LOWER the ceiling, never raise it. Passing the flag
    # through unclamped would have made the "a typo'd flag cannot raise it"
    # claim above false, which is worse than not making the claim.
    max_dollars = min(float(max_dollars), MAX_TEST_DOLLARS)
    if allow_fill:
        price_cents = max(1, min(int(price_cents), 99))
        # Reduce SIZE to fit the budget; never the price. A cheaper price is a
        # different order that may not fill, which is the thing being tested.
        affordable = int((max_dollars * 100) // price_cents)
        count = max(0, min(int(count), affordable))
        cancel = False if cancel is None else cancel
    else:
        price_cents = max(1, min(int(price_cents), MAX_TEST_PRICE_CENTS))
        count = 1
        cancel = True if cancel is None else cancel

    base = DEMO_BASE if demo else BASE
    where = "DEMO (fake money)" if demo else "PRODUCTION (REAL MONEY)"
    lines = [f"KALSHI ORDER SMOKE TEST — {where}", f"  host   {base}"]
    if allow_fill:
        lines += [f"  mode   AT-MARKET — this is meant to FILL. "
                  f"{count} @ {price_cents}c = ${count * price_cents / 100:.2f}",
                  "         A filled order is a POSITION, not a test."]
    else:
        lines.append(f"  mode   round-trip — {count} @ {price_cents}c, cannot "
                     f"fill, cancelled immediately")
    lines.append("")
    ok = True

    if count < 1:
        return lines + [f"  [FAIL] {price_cents}c exceeds the "
                        f"${max_dollars:.2f} cap even for one contract"], False

    if demo:
        # Demo reads ONLY its own file: an exported production KALSHI_API_KEY_ID
        # would otherwise be sent to the demo host and fail as a mystery.
        kid, pem = load_credentials(env, path=DEMO_CREDENTIALS_PATH,
                                    use_env=False)
        armed = True  # fake money; the production literal is not the gate here
        if not (kid and pem):
            return lines + [
                f"  [FAIL] no demo credentials at {DEMO_CREDENTIALS_PATH}",
                "",
                "Demo is a SEPARATE ACCOUNT from your real one: sign up at",
                "demo.kalshi.co, create an API key there, then put it in that",
                "file (chmod 600) the same way as the production one.",
            ], False
    else:
        kid, pem = load_credentials(env)
        armed = is_armed(env)
        if not armed:
            return lines + [
                "  [FAIL] not armed — refusing to touch the real book",
                "",
                f"A real-money test needs {ARM_ENV} set to the exact arming",
                "literal, same as the bot. That is deliberate: there is ONE way",
                "to enable real orders, and this rehearses it.",
            ], False
        if not (kid and pem):
            return lines + ["  [FAIL] armed but no credentials loaded — run "
                            "`highcast creds-check`"], False

    c = KalshiClient(key_id=kid, private_key_pem=pem, armed=armed,
                     transport=transport, base=base)

    # 1. A ticker that exists IN THIS ENVIRONMENT. Demo's markets are not
    #    production's, and a guessed ticker fails looking like an auth error.
    if ticker is None:
        res = c.list_markets(series)
        markets = (res["body"] or {}).get("markets") or []
        if not res["ok"] or not markets:
            return lines + [f"  [FAIL] could not list {series} markets: "
                            f"{res['reason']}"], False
        ticker = markets[0]["ticker"]
        lines.append(f"  [ ok ] markets            {len(markets)} open in "
                     f"{series}; using {ticker}")
    else:
        lines.append(f"  [ -- ] markets            using supplied {ticker}")

    # 2. Place.
    intent = c.place_order(ticker=ticker, side="yes", count=count,
                           limit_price=price_cents / 100.0)
    oid = intent.get("order_id")
    if not intent.get("placed"):
        return lines + [f"  [FAIL] place              {intent['reason']}"], False
    lines.append(f"  [ ok ] place              {count} @ {price_cents}c  "
                 f"({intent['reason']}, id {oid or 'not returned'})")

    try:
        # 3. Read it back. HTTP 201 means the request parsed; only a read-back
        #    proves an order actually rests on the book.
        if oid:
            got = c.get_order(oid)
            state = ((got["body"] or {}).get("order") or {}).get("status", "?")
            lines.append(f"  [{' ok ' if got['ok'] else 'FAIL'}] read back          "
                         f"{got['reason']}, status={state}")
            ok = ok and got["ok"]
        else:
            lines.append("  [FAIL] read back          no order id returned — "
                         "cannot verify or cancel")
            ok = False
    finally:
        # 4. Cancel. Unconditional WITHIN the round-trip mode: an abandoned
        #    resting order is the only bad outcome that mode can produce, so
        #    cleanup does not depend on the step before it succeeding.
        #
        #    In at-market mode the order is MEANT to fill, and a filled order
        #    cannot be cancelled — attempting it would log a failure that means
        #    "it worked".
        if oid and cancel:
            can = c.cancel_order(oid)
            lines.append(f"  [{' ok ' if can['ok'] else 'FAIL'}] cancel             "
                         f"{can['reason']}")
            ok = ok and can["ok"]
            if not can["ok"]:
                lines.append("         ⚠️  CANCEL FAILED — an order may still be "
                             "resting. Check the Kalshi web UI now.")
        elif oid:
            lines.append(f"  [ -- ] cancel             SKIPPED — order {oid} is "
                         f"left in place on purpose")

    if allow_fill:
        lines += ["", "ORDER SENT AND ACCEPTED — it is now a real position or a",
                  "real resting order. The exit is yours: this bot does not sell.",
                  f"Manage it at kalshi.com. Order id: {oid}"]
        return lines, ok
    lines += ["", "ORDER PATH WORKS END TO END" if ok else
              "SMOKE TEST FAILED — see FAIL lines above"]
    return lines, ok


def setup_report(env: dict[str, str]) -> tuple[list[str], bool]:
    """(lines, ok) describing whether trading is correctly configured.

    Lives HERE, not in `cli.py`, for two reasons. The module that owns
    credentials should own the diagnostic; and the report necessarily names the
    signing algorithm and the endpoint path, which `tests/test_freeze_contracts`
    bans everywhere else -- prose about capability would otherwise be
    indistinguishable from capability.

    NEVER RETURNS KEY MATERIAL. A short fingerprint stands in for the key, so the
    output is safe to paste into a chat when something is wrong.
    """
    import hashlib
    import stat as statmod

    lines: list[str] = ["KALSHI TRADING SETUP", ""]
    ok = True

    p = DEFAULT_CREDENTIALS_PATH
    if not p.exists():
        lines.append(f"  [FAIL] credentials file   MISSING at {p}")
        ok = False
    else:
        mode = p.stat().st_mode
        loose = bool(mode & 0o077)
        lines.append(f"  [{'FAIL' if loose else ' ok '}] credentials file   {p} "
                     f"(mode {statmod.filemode(mode)})"
                     + ("  <- group/other can read; run: chmod 600" if loose else ""))
        ok = ok and not loose

    kid, pem = load_credentials(env)
    placeholder = (kid or "").startswith("PUT-YOUR")
    lines.append(f"  [{'FAIL' if (not kid or placeholder) else ' ok '}] key id             "
                 + ("MISSING" if not kid else
                    ("PLACEHOLDER — not filled in yet" if placeholder else kid)))
    ok = ok and bool(kid) and not placeholder

    if not pem:
        lines.append("  [FAIL] private key        NOT LOADED")
        ok = False
    else:
        fp = hashlib.sha256(pem.encode()).hexdigest()[:12]
        lines.append(f"  [ ok ] private key        loaded ({len(pem)} bytes, sha256:{fp})")

    try:
        sign(pem, signing_message(1, "GET", POSITIONS_PATH))
        lines.append("  [ ok ] signing            works")
    except Exception as e:  # noqa: BLE001
        lines.append(f"  [FAIL] signing            {type(e).__name__}: {e}")
        ok = False

    import datetime as _dt

    armed = is_armed(env)
    today = _dt.date.today().isoformat()
    window = (env.get(ARM_UNTIL_ENV) or "").strip()
    window_ok = arming_window_ok(env, today)
    bot_live = armed and window_ok

    lines.append(
        f"  [ {'!!' if armed else 'ok'} ] arming literal     "
        + ("SET" if armed else "not set (dry-run)"))
    if not window:
        lines.append(f"  [ ok ] arming window      not set — the unattended bot "
                     f"stays DRY ({ARM_UNTIL_ENV})")
    else:
        lines.append(
            f"  [ {'!!' if window_ok else 'ok'} ] arming window      "
            f"through {window} — "
            + ("ACTIVE today" if window_ok else f"EXPIRED (today is {today})"))
    lines += ["", "WHAT HAPPENS IF THE BOT RUNS RIGHT NOW:"]
    if bot_live:
        lines.append("   ⚠️  It can place a REAL order (KMIA HIGH only, <= $1/day, "
                     "one per day, price ceiling 0.80).")
        lines.append(f"   It disarms itself automatically after {window}.")
    else:
        lines.append("   It decides, logs the order it WOULD place, and sends nothing.")
        if armed and not window_ok:
            lines.append(f"   The literal is set but the window is not active, so the "
                         f"UNATTENDED bot is dry. Set {ARM_UNTIL_ENV} to arm it.")
    lines += ["", "SETUP IS GOOD" if ok else "SETUP INCOMPLETE — see FAIL lines above"]
    if ok and not armed:
        lines.append(f"Credentials are ready; it stays dry-run until you set {ARM_ENV}.")
    return lines, ok
