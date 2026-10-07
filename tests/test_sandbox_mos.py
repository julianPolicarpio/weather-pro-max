import json
import logging

from sandbox_helpers import three_buckets

from sandbox.bots import IemMos, MosBot
from sandbox.models import Snapshot

T = "2026-10-04T04:00:00+00:00"            # before KNYC's 2026-10-04 climate day (05:00Z)

# KNYC, standard offset -5. 00Z on Oct 5 = 20:00 EDT Oct 4 (afternoon row -> MAX),
# 12Z on Oct 4 = 08:00 EDT (morning row -> MIN). Both fall on climate day Oct 4.
PAYLOAD = {"data": [
    {"runtime": "2026-10-04 12:00", "ftime": "2026-10-05 00:00", "n_x": 71},
    {"runtime": "2026-10-04 00:00", "ftime": "2026-10-04 12:00", "n_x": 58},
]}


def test_iem_mos_value_picks_the_extremum_for_the_climate_day():
    calls = []
    src = IemMos(fetch=lambda url, params: calls.append(params) or PAYLOAD)
    assert src.value("KNYC", "high", "2026-10-04") == 71.0
    assert src.value("KNYC", "low", "2026-10-04") == 58.0
    assert calls == [{"station": "KNYC", "model": "MEX"}]  # cached per station


def test_iem_mos_unknown_station_or_fetch_error_is_none():
    def boom(url, params):
        raise RuntimeError("down")
    assert IemMos(fetch=boom).value("KNYC", "high", "2026-10-04") is None
    assert IemMos(fetch=lambda u, p: PAYLOAD).value("KXXX", "high", "2026-10-04") is None


def test_fetch_failure_is_logged_and_cached(caplog):
    calls = []

    def boom(url, params):
        calls.append(params)
        raise RuntimeError("down")

    src = IemMos(fetch=boom)
    with caplog.at_level("WARNING", logger="sandbox.bots"):
        assert src.value("KNYC", "high", "2026-10-04") is None
        assert src.value("KNYC", "high", "2026-10-04") is None
    assert len(calls) == 1
    warnings = [r for r in caplog.records
                if r.name == "sandbox.bots" and r.levelname == "WARNING"]
    assert len(warnings) == 1 and "KNYC" in warnings[0].getMessage()


def test_unknown_metric_is_none():
    assert IemMos(fetch=lambda u, p: PAYLOAD).value("KNYC", "mean", "2026-10-04") is None


def test_non_dict_payload_is_none_not_a_crash():
    assert IemMos(fetch=lambda u, p: None).value("KNYC", "high", "2026-10-04") is None


class FakeSource:
    def __init__(self, v):
        self.v = v
        self.asked = []

    def value(self, icao, metric, event_date):
        self.asked.append((icao, metric, event_date))
        return self.v


def test_mos_bot_uses_station_metric_and_date_from_the_market():
    src = FakeSource(69.5)
    d = MosBot(src).decide(Snapshot(id=1, taken_at=T, markets=tuple(three_buckets())))
    assert src.asked == [("KNYC", "high", "2026-10-04")]
    assert len(d.candidates) == 3 and all(c.bot == "mos" for c in d.candidates)
    assert [i.ticker for i in d.intents] == ["KXHIGHNY-26OCT04-T69"]


def test_mos_bot_skips_events_without_a_forecast():
    d = MosBot(FakeSource(None)).decide(Snapshot(id=1, taken_at=T, markets=tuple(three_buckets())))
    assert d.candidates == []


# ---- F2: MOS has no observation input, so it bets only before the climate day ----

def test_mos_decides_until_the_climate_day_starts():
    snap = Snapshot(id=1, taken_at="2026-10-04T04:55:00+00:00", markets=tuple(three_buckets()))
    src = FakeSource(69.5)
    d = MosBot(src).decide(snap, "2026-10-04T04:59:59+00:00")
    assert len(d.candidates) == 3
    src = FakeSource(69.5)
    d = MosBot(src).decide(snap, "2026-10-04T05:00:00+00:00")
    assert d.candidates == [] and d.intents == []
    assert src.asked == []                              # no MOS fetch for a started event


def test_mos_skips_an_unknown_station():
    src = FakeSource(69.5)
    snap = Snapshot(id=1, taken_at=T, markets=tuple(three_buckets(station="KXXX")))
    assert MosBot(src).decide(snap).candidates == [] and src.asked == []
    assert MosBot.pre_day_only is True


# ---- F16: the forecast behind the pmf is logged with each candidate --------------

def test_mos_candidates_record_the_forecast():
    d = MosBot(FakeSource(69.5)).decide(Snapshot(id=1, taken_at=T, markets=tuple(three_buckets())))
    assert {c.forecast_json for c in d.candidates} == {
        json.dumps({"mu": 69.5, "sigma": 3.0, "source": "mos_mex"}, sort_keys=True)}


# ---- F14: an IEM outage stops fetching after 3 consecutive failures --------------

STATIONS_5 = ["KNYC", "KLAX", "KMIA", "KDEN", "KBOS"]


def test_mos_circuit_opens_after_three_consecutive_failures(caplog):
    calls = []

    def boom(url, params):
        calls.append(params["station"])
        raise RuntimeError("down")

    src = IemMos(fetch=boom)
    with caplog.at_level(logging.WARNING, logger="sandbox.bots"):
        for st in STATIONS_5:
            assert src.value(st, "high", "2026-10-04") is None
    assert calls == STATIONS_5[:3]
    opened = [r for r in caplog.records if "MOS circuit open" in r.getMessage()]
    assert len(opened) == 1


def test_a_success_resets_the_failure_count():
    calls = []

    def flaky(url, params):
        calls.append(params["station"])
        if params["station"] == "KMIA":
            return PAYLOAD
        raise RuntimeError("down")

    src = IemMos(fetch=flaky)
    for st in ["KNYC", "KLAX", "KMIA", "KDEN", "KBOS"]:
        src.value(st, "high", "2026-10-04")
    assert calls == ["KNYC", "KLAX", "KMIA", "KDEN", "KBOS"]   # never 3 failures in a row
