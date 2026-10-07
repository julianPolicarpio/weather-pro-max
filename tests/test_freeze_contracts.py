"""Trading capability is CONFINED to one module. These tests are the boundary.

Originally (2026-07-31) this asserted NO trading capability existed anywhere.
That was right while the accuracy work was landing. On 2026-08-01 the KMIA HIGH
order bot was built, so the contract CHANGED SHAPE rather than being dropped:

    trading code may live in EXACTLY ONE file, and nowhere else.

Narrowing beats deleting. The original risk — auth or order placement creeping
into the model, the dashboard, the fetchers — is undiminished, and a
single-file exemption still catches it everywhere it matters.

Note `bankroll` stays 0 and `app_bets` stays paper: the bot is DRY-RUN and its
live path is gated behind an exact-literal env var plus an optional dependency
that is not installed. See tests/test_kalshi_client.py.
"""
import pathlib
import re

import yaml

SRC = pathlib.Path("src/highcast")

# The single file permitted to contain trading capability.
TRADING_MODULE = "kalshi_client.py"


def banned_tokens():
    """Word-boundary patterns for real trading capability. Matched precisely so
    the guard cannot fire on ordinary prose — an earlier version banned the bare
    substring `rsa` and tripped on the word "universal"."""
    return [
        r"\bprivate_key\b", r"\brsa\b", r"\brsa_pss\w*",
        r"load_pem_private_key", r"cryptography\.hazmat",
        r"\bsign_request\b", r"/portfolio\b", r"/trade-api/v2/portfolio",
        # NOTE: `place_order` / `create_order` are deliberately NOT banned.
        # They are METHOD NAMES, and a call site is not a capability — banning
        # them would force every orchestration layer into the exemption, which
        # inverts the point of the guard. What is banned is the ability to
        # AUTHENTICATE and to reach a trading ENDPOINT; without those a call site
        # can do nothing. `botrunner.py` calls `place_order` and holds no
        # capability of its own.
    ]


def test_no_trading_authentication_anywhere_in_src():
    """The app is read-only on Kalshi: the public markets endpoint, no auth.
    Signing keys or a portfolio read would mean phase 2 landed early.

    ⚠️ TOKENS ARE MATCHED ON WORD BOUNDARIES, NOT AS BARE SUBSTRINGS.
    The first version of this test banned the substring `rsa`, which fired on the
    ENGLISH WORD "universal" in a gating.py comment. That was "fixed" by rewording
    the comment — a fix that would have broken again on "versatile",
    "conversation" or "traversal". A guard that cries wolf on prose gets
    reworded around until it is worthless, so the matching is precise instead:
    real RSA usage appears as `import rsa`, `rsa_pss`, or a cryptography import,
    never as a fragment inside an ordinary word.
    """
    banned = banned_tokens()
    hits = []
    for f in SRC.rglob("*.py"):
        # THE ONE EXEMPTION. Everything trading-related is confined here, and it
        # has its own dedicated guards in tests/test_kalshi_client.py (dry-run
        # posts nothing; arming needs an exact literal; keys never printed).
        if f.name == TRADING_MODULE:
            continue
        low = f.read_text().lower()
        for token in banned:
            if re.search(token, low):
                hits.append(f"{f}: {token}")
    assert not hits, (
        f"trading capability leaked outside {TRADING_MODULE}: {hits}. Keep auth "
        f"and order placement in that one file — the model, the fetchers and the "
        f"dashboard must stay incapable of spending money.")


def test_the_trading_module_is_the_only_exemption_and_it_exists():
    """A typo'd exemption would silently disable the guard for every file."""
    assert (SRC / TRADING_MODULE).exists(), (
        f"{TRADING_MODULE} is exempted but does not exist — the exemption is "
        f"now disabling nothing, or hiding a rename")


def test_the_order_bot_itself_carries_no_trading_capability():
    """`orderbot.py` decides WHAT to order; it must never learn to send one.
    Keeping the decision layer network-free is what makes the money rails
    testable without a network."""
    low = (SRC / "orderbot.py").read_text().lower()
    for token in banned_tokens():
        assert not re.search(token, low), (
            f"orderbot.py gained trading capability ({token}) — transport "
            f"belongs in {TRADING_MODULE}")


def test_bankroll_is_zero_in_both_configs():
    """Paper mode. At bankroll 0 the ledger records the model's genuine opinion
    with no sizing constraint, so app-vs-Julian measures the MODEL."""
    for p in ("config/klax.yaml", "config/kmia.yaml"):
        raw = yaml.safe_load(open(p).read())
        assert raw["staking"]["bankroll"] == 0.0, f"{p} left paper mode"


def test_nothing_reads_the_hourly_capture_yet():
    """`forecast_hourly` is capture-only. A feature built on it is a MODEL change
    and needs its own pre-registered gate."""
    readers = []
    for f in SRC.rglob("*.py"):
        if f.name in ("db.py", "backfill.py", "forecasts.py"):
            continue
        # ⚠️ `staleness.py` is exempt, and the exemption is NARROW ON PURPOSE.
        # It is a LIVENESS check: it reads MAX(fetched_at) to answer "did this
        # feed stop delivering", and never touches a value column. That is not a
        # model read and does not need a pre-registered gate. The assertion
        # below enforces the narrowness so the exemption cannot be widened into
        # a feature by accident.
        if f.name == "staleness.py":
            continue
        if "forecast_hourly" in f.read_text():
            readers.append(str(f))
    assert not readers, f"forecast_hourly is being read by {readers}"

    # The staleness exemption above is only safe while it stays a liveness
    # check. If it ever selects a value column from these tables, that IS a
    # model change and this contract must fail again.
    stale_src = (SRC / "staleness.py").read_text()
    for table in ("forecast_hourly", "forecast_intraday"):
        for line in stale_src.splitlines():
            if table in line and "Feed(" not in line and not line.strip().startswith("#"):
                raise AssertionError(
                    f"staleness.py references {table} outside a Feed() "
                    f"declaration: {line.strip()}")
    assert "temp_f" not in stale_src and "max_temp_f" not in stale_src, \
        "staleness.py must never read a temperature column"



# A CALL to an individual gate, not a mention of one. See the test below for
# why the distinction is load-bearing.
_GATE_CALL = re.compile(r"\bconfidence_floor_ok\s*\(")


def _reimplements_gate_logic(text: str) -> bool:
    return bool(_GATE_CALL.search(text)) and "decision_gates" not in text


def test_the_decision_gates_aggregator_is_the_only_composed_verdict():
    """No surface may re-implement the gate logic. If a file CALLS the
    individual gates directly instead of `decision_gates`, the surfaces will
    drift -- which is the whole reason the aggregator exists.

    ⚠️ MATCHED AS A CALL, NOT AS A SUBSTRING, and that is the second time this
    file has learned the lesson. The original version asked whether the string
    `confidence_floor_ok` appeared anywhere, so on 2026-08-01 it failed on
    `boundregime.py` for a DOCSTRING that named the function while explaining
    which convention it follows -- prose, referencing the rule, calling nothing.

    The tempting fix is to reword the docstring. That is exactly what was done
    the first time, when the banned substring `rsa` fired on the English word
    "universal", and the note above `test_no_trading_authentication_anywhere_in_src`
    records why it was wrong: a guard that cries wolf on prose gets reworded
    around until it is worthless. Cross-referencing a rule in a comment is good
    practice and must not carry a tax. So the matcher is precise instead, and
    the negative control below is what keeps it honest.
    """
    offenders = [str(f) for f in SRC.rglob("*.py")
                 if f.name != "gating.py" and _reimplements_gate_logic(f.read_text())]
    assert not offenders, f"gate logic re-implemented in {offenders}"


def test_the_gate_guard_catches_a_real_call_but_not_prose_about_one():
    """THE NEGATIVE CONTROL. A guard nobody can see fail is not a guard.

    Without this, tightening the matcher above could quietly weaken it to
    nothing -- the failure mode is silent, because a guard that matches no file
    passes exactly like a guard that matches the right files.
    """
    assert _reimplements_gate_logic("if confidence_floor_ok(station, m, h, p):")
    assert _reimplements_gate_logic("x = confidence_floor_ok (a, b)")
    # Prose: names the rule, calls nothing.
    assert not _reimplements_gate_logic("the rule `confidence_floor_ok` follows")
    assert not _reimplements_gate_logic("# see confidence_floor_ok for the pattern")
    # A real call is fine when the file also goes through the aggregator.
    assert not _reimplements_gate_logic(
        "v = decision_gates(...)\nif confidence_floor_ok(s, m, h, p): pass")
