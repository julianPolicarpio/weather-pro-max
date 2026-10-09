"""Sandbox bots. Each bot sees one Snapshot and returns intents + candidates.

Every bot bets 1 contract on the YES side. Stake sizing is out of scope (spec §9).
"""
import json
import logging
import math
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import NamedTuple
from zoneinfo import ZoneInfo

from highcast.buckets import pmf_from_buckets, pmf_mass
from highcast.fees import fee_per_contract
from highcast.gating import kind_at_hour
from highcast.fetchers.forecasts import IEM_MOS, parse_mos
from highcast.http import get_json
from highcast.orderbot import MIN_EDGE  # 0.03 — one constant shared with the live desk
from sandbox.markets import usable_ask, usable_bid
from sandbox.models import Candidate, Intent, Market, Snapshot
from sandbox.pricing import devig
from sandbox.stations import STATIONS, event_started

log = logging.getLogger(__name__)

SIGMA_F = 3.0   # MOS spread prior in °F — a guess, not a measurement (spec §6)

__all__ = ["MIN_EDGE", "SIGMA_F", "Decision", "evaluate_bucket", "normal_bucket_probs",
           "minutes_between", "ForecastBot", "FavoriteBot", "IemMos", "MosBot",
           "HIGHCAST_STATIONS", "HighcastBot", "CitycastBot"]


class Decision(NamedTuple):
    intents: list[Intent]
    candidates: list[Candidate]


def evaluate_bucket(p: float, ask: float | None) -> tuple[bool, float | None, float | None, str]:
    if ask is None:
        return False, None, None, "no_ask"
    fpc = fee_per_contract(1, ask)
    edge = round(p - ask - fpc, 6)
    if edge >= MIN_EDGE:
        return True, fpc, edge, "edge"
    return False, fpc, edge, "below_min_edge"


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def normal_bucket_probs(markets: list[Market], mu: float, sigma: float) -> dict[str, float]:
    """Normal(mu, sigma) over integer °F outcomes, summed into each bucket."""
    raw = {}
    for m in markets:
        upper = 1.0 if m.hi is None else _phi((m.hi + 0.5 - mu) / sigma)
        lower = 0.0 if m.lo is None else _phi((m.lo - 0.5 - mu) / sigma)
        raw[m.ticker] = max(upper - lower, 0.0)
    total = sum(raw.values())
    return {t: v / total for t, v in raw.items()} if total > 0 else {}


def minutes_between(start_iso: str, end_iso: str | None) -> float | None:
    if not end_iso:
        return None
    delta = datetime.fromisoformat(end_iso) - datetime.fromisoformat(start_iso)
    return delta.total_seconds() / 60.0


class ForecastBot:
    """Base for bots that produce a probability per bucket. Logs every bucket.

    `decide(snapshot, now_iso)`: `now_iso` is the decision time (defaults to the
    snapshot's `taken_at`). It stamps each candidate and, for `pre_day_only` bots,
    skips every event whose station-local climate day has begun at that time."""
    name = "forecast"
    pre_day_only = False

    def pmf_for_event(self, markets: list[Market], snapshot: Snapshot) -> dict[str, float] | None:
        """The pmf half of `forecast_for_event`. A subclass overrides this or
        `forecast_for_event`; each defaults to the other, so overriding neither recurses."""
        return self.forecast_for_event(markets, snapshot)[0]

    def forecast_for_event(self, markets: list[Market],
                           snapshot: Snapshot) -> tuple[dict[str, float] | None, dict | None]:
        """(pmf, the forecast behind it for the candidate log). Override to log the forecast."""
        return self.pmf_for_event(markets, snapshot), None

    def forecast_at_decision(self, markets, snapshot, now_iso):
        return self.forecast_for_event(markets, snapshot)

    def decide(self, snapshot: Snapshot, now_iso: str | None = None) -> Decision:
        now = now_iso or snapshot.taken_at
        intents: list[Intent] = []
        candidates: list[Candidate] = []
        for markets in snapshot.events().values():
            if self.pre_day_only and event_started(markets[0], now):
                continue
            pmf, forecast = self.forecast_at_decision(markets, snapshot, now)
            if not pmf:
                continue
            market_p = devig(markets)
            pmf_json = json.dumps(pmf, sort_keys=True)
            forecast_json = None if forecast is None else json.dumps(forecast, sort_keys=True)
            for m in markets:
                p = pmf.get(m.ticker)
                if p is None:
                    continue
                ask, bid = usable_ask(m), usable_bid(m)
                ok, fpc, edge, reason = evaluate_bucket(p, ask)
                candidates.append(Candidate(
                    bot=self.name, ticker=m.ticker, series=m.series, station=m.station,
                    metric=m.metric, event_date=m.event_date, snapshot_id=snapshot.id,
                    decided_at=now, p_bot=p, pmf_json=pmf_json,
                    p_market=market_p.get(m.ticker), yes_bid=bid, yes_ask=ask,
                    spread=None if ask is None or bid is None else round(ask - bid, 4),
                    ask_size=m.yes_ask_size,
                    minutes_to_close=minutes_between(now, m.close_time),
                    fee_per_contract=fpc, edge_after_fee=edge, placed=ok, reason=reason,
                    forecast_json=forecast_json))
                if ok:
                    intents.append(Intent(account=self.name, ticker=m.ticker, side="yes",
                                          contracts=1, price_cap=ask,
                                          snapshot_id=snapshot.id, p_bot=p,
                                          p_market=market_p.get(m.ticker), priority=edge))
        return Decision(intents, candidates)


class FavoriteBot:
    """Control: buy YES on each event's most likely bucket at the ask, before the
    station's climate day starts. If that bucket has no usable ask the event is
    skipped: buying a runner-up would not be "follow the crowd"."""
    name = "favorite"

    def decide(self, snapshot: Snapshot, now_iso: str | None = None) -> Decision:
        now = now_iso or snapshot.taken_at
        intents: list[Intent] = []
        for markets in snapshot.events().values():
            if event_started(markets[0], now):
                continue
            probs = devig(markets)
            if not probs:
                continue
            # max() keeps the first of equal values, so ties go to ladder order
            top = max((m for m in markets if m.ticker in probs), key=lambda m: probs[m.ticker])
            ask = usable_ask(top)
            if ask is None:
                continue
            p = probs[top.ticker]
            intents.append(Intent(account=self.name, ticker=top.ticker, side="yes",
                                  contracts=1, price_cap=ask, snapshot_id=snapshot.id,
                                  p_bot=p, p_market=p, priority=p))
        return Decision(intents, [])


class IemMos:
    """MOS MEX daily max/min per (station, metric, climate day), fetched once per
    station per instance. Build a fresh instance each cycle. After
    `MAX_CONSECUTIVE_FAILURES` failed fetches in a row the circuit opens: no more
    fetches this instance (an IEM outage must not cost 24 timeouts per cycle)."""

    MAX_CONSECUTIVE_FAILURES = 3

    def __init__(self, fetch=get_json, model: str = "MEX"):
        self.fetch = fetch
        self.model = model
        self._payloads: dict[str, dict] = {}
        self._failures = 0
        self._circuit_logged = False

    def _payload(self, icao: str) -> dict | None:
        if icao in self._payloads:
            return self._payloads[icao]
        if self._failures >= self.MAX_CONSECUTIVE_FAILURES:
            if not self._circuit_logged:
                log.warning("MOS circuit open after %d consecutive fetch failures; "
                            "no more MOS fetches this cycle", self._failures)
                self._circuit_logged = True
            return None
        try:
            self._payloads[icao] = self.fetch(IEM_MOS, {"station": icao, "model": self.model})
            self._failures = 0
        except Exception as e:  # noqa: BLE001 - no forecast this cycle, not a crash
            log.warning("MOS fetch failed for %s: %r", icao, e)
            self._payloads[icao] = {}
            self._failures += 1
        return self._payloads[icao]

    def value(self, icao: str, metric: str, event_date: str) -> float | None:
        if metric not in ("high", "low"):
            return None
        if icao not in STATIONS:
            return None
        payload = self._payload(icao)
        if payload is None:
            return None
        tz, offset = STATIONS[icao]
        try:
            rows = parse_mos(payload, icao, "", f"mos_{self.model.lower()}",
                             tz, "max" if metric == "high" else "min", offset)
        except Exception as e:  # noqa: BLE001 - malformed payload, not a crash
            log.warning("MOS parse failed for %s: %r", icao, e)
            return None
        matches = sorted((r for r in rows if r["target_date"] == event_date),
                         key=lambda r: r["run_date"])
        return float(matches[-1]["max_temp_f"]) if matches else None


class MosBot(ForecastBot):
    """Normal(MOS, SIGMA_F) over each event's buckets. Uncalibrated by design.

    Pre-day only: MOS knows nothing of today's observations, while the market does
    once the station's climate day has started."""
    name = "mos"
    pre_day_only = True

    def __init__(self, source, sigma: float = SIGMA_F):
        self.source = source
        self.sigma = sigma

    def forecast_for_event(self, markets, snapshot):
        first = markets[0]
        if first.station is None or first.metric is None:
            return None, None
        mu = self.source.value(first.station, first.metric, first.event_date)
        if mu is None:
            return None, None
        source = f"mos_{getattr(self.source, 'model', 'MEX').lower()}"
        return (normal_bucket_probs(markets, mu, self.sigma),
                {"source": source, "mu": mu, "sigma": self.sigma})


def _open_ro(path: Path) -> sqlite3.Connection:
    """Read-only handle (URI `mode=ro`) on a database another process writes.

    timeout=30 rides out a write lock rather than losing an hourly decision after
    sqlite's 5 s default."""
    return sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=30)


def _normalise_over_markets(pmf: dict[int, float], markets: list[Market]) -> dict[str, float] | None:
    """The pmf's mass in each market's bucket, rescaled to sum to 1 over the ladder.
    None when the ladder holds no mass."""
    raw = {m.ticker: pmf_mass(pmf, m.lo, m.hi) or 0.0 for m in markets}
    total = sum(raw.values())
    if total <= 0:
        return None
    return {t: v / total for t, v in raw.items()}


HIGHCAST_STATIONS = {"KLAX", "KMIA"}


class HighcastBot(ForecastBot):
    """The shipped highcast model, read from highcast.db READ-ONLY.

    Not pre-day gated: highcast's intraday kind sees the day's observations. Reads
    only the `dayahead` and `intraday` kinds (an allowlist, so any new kind such as
    `evening` fails closed) issued at or before the snapshot's `taken_at`."""
    name = "highcast"
    KINDS = ("dayahead", "intraday")

    def __init__(self, highcast_db: str):
        self.highcast_db = highcast_db

    def _latest_prediction(self, station: str, metric: str, event_date: str,
                           at_or_before: str, preferred: str) -> tuple[dict[int, float], str, str] | None:
        """Newest available preferred kind, otherwise the other allowed kind."""
        path = Path(self.highcast_db)
        if not path.exists():
            return None
        try:
            # The live desk writes this DB (journal_mode=delete); _open_ro's 30 s timeout
            # rides out its write lock.
            with closing(_open_ro(path)) as c:
                row = c.execute(
                    f"""SELECT buckets, kind, issued_at_utc FROM predictions
                       WHERE station=? AND metric=? AND target_date=? AND issued_at_utc<=?
                         AND buckets IS NOT NULL
                         AND kind IN ({", ".join("?" for _ in self.KINDS)})
                       ORDER BY (kind=?) DESC, issued_at_utc DESC LIMIT 1""",
                    (station, metric, event_date, at_or_before, *self.KINDS, preferred)).fetchone()
        except sqlite3.Error as e:
            log.warning("highcast read failed for %s %s %s: %r", station, metric, event_date, e)
            return None
        if not row:
            return None
        try:
            return pmf_from_buckets(json.loads(row[0])), row[1], row[2]
        except (ValueError, TypeError) as e:
            log.warning("highcast read failed for %s %s %s: %r", station, metric, event_date, e)
            return None

    def forecast_for_event(self, markets, snapshot):
        return self.forecast_at_decision(markets, snapshot, snapshot.taken_at)

    def forecast_at_decision(self, markets, snapshot, now_iso):
        first = markets[0]
        if first.station not in HIGHCAST_STATIONS or first.metric is None:
            return None, None
        local_now = datetime.fromisoformat(now_iso).astimezone(ZoneInfo(STATIONS[first.station][0]))
        preferred = kind_at_hour(first.station, first.metric,
                                 local_now.date().isoformat() == first.event_date, local_now.hour)
        found = self._latest_prediction(first.station, first.metric, first.event_date,
                                        snapshot.taken_at, preferred)
        if not found or not found[0]:
            return None, None
        pmf, kind, issued = found
        probs = _normalise_over_markets(pmf, markets)
        if probs is None:
            return None, None
        return probs, {"source": "highcast", "kind": kind, "issued_at_utc": issued}


class CitycastBot(ForecastBot):
    """citycast's 24-city model (spec 2026-10-07 §3.9). Reads data/history.db read-only and
    uses only station/metrics whose model passed the gate and whose prediction was made by
    the currently chosen variant; skips predictions older than 2 h."""
    name = "citycast"
    MAX_AGE_S = 7200

    def __init__(self, history_db: str):
        self.history_db = history_db

    def _latest(self, station, metric, event_date, at_or_before):
        path = Path(self.history_db)
        if not path.exists():
            return None
        try:
            with closing(_open_ro(path)) as c:
                return c.execute(
                    """SELECT lp.pmf_json, lp.variant, lp.issued_utc FROM live_predictions lp
                       JOIN model_choice mc ON mc.station=lp.station AND mc.metric=lp.metric
                                           AND mc.variant=lp.variant
                       WHERE mc.gate_pass=1 AND lp.station=? AND lp.metric=? AND lp.event_date=?
                         AND lp.issued_utc<=? ORDER BY lp.issued_utc DESC LIMIT 1""",
                    (station, metric, event_date, at_or_before)).fetchone()
        except sqlite3.Error as e:
            # history.db gains these tables only at the first `citycast` run: until then it is
            # expected, not worth a warning per event per hour.
            if not (isinstance(e, sqlite3.OperationalError) and "no such table" in str(e)):
                log.warning("citycast read failed for %s %s %s: %r", station, metric, event_date, e)
            return None

    def forecast_for_event(self, markets, snapshot):
        first = markets[0]
        if first.station is None or first.metric is None:
            return None, None
        row = self._latest(first.station, first.metric, first.event_date, snapshot.taken_at)
        if row is None:
            return None, None
        age = (datetime.fromisoformat(snapshot.taken_at) - datetime.fromisoformat(row[2])).total_seconds()
        if age > self.MAX_AGE_S:
            return None, None
        try:
            pmf = {int(k): float(v) for k, v in json.loads(row[0]).items()}
        except (ValueError, TypeError, AttributeError) as e:
            log.warning("citycast pmf unreadable for %s: %r", first.event_ticker, e)
            return None, None
        probs = _normalise_over_markets(pmf, markets)
        if probs is None:
            return None, None
        return probs, {"source": "citycast", "variant": row[1], "issued_utc": row[2]}
