# tests/test_citycast_http.py
import pytest

from citycast.http import HttpError, Paced, RateLimited

IEM = "https://mesonet.agron.iastate.edu/x"
KAL = "https://api.elections.kalshi.com/trade-api/v2/x"


class Clock:
    def __init__(self):
        self.t = 0.0
        self.sleeps = []

    def clock(self):
        return self.t

    def sleep(self, s):
        self.sleeps.append(s)
        self.t += s


def _paced(responses, clk):
    calls = []

    def fetch(url, params):
        calls.append(url)
        return responses.pop(0)
    return Paced(fetch=fetch, sleep=clk.sleep, clock=clk.clock), calls


def test_paces_per_host():
    clk = Clock()
    p, _ = _paced([(200, "a"), (200, "b"), (200, "c")], clk)
    p.get_text(IEM)
    p.get_text(KAL)          # other host: no wait
    p.get_text(IEM)          # same host again: waits the rest of 1.0 s
    assert clk.sleeps == [1.0]


def test_backs_off_on_429_and_too_many_requests_then_succeeds():
    clk = Clock()
    p, calls = _paced([(429, ""), (200, "Too many requests from your IP"), (200, "ok")], clk)
    assert p.get_text(KAL) == "ok"
    assert len(calls) == 3
    assert [s for s in clk.sleeps if s >= 5] == [5, 15]


def test_raises_rate_limited_after_backoffs_exhausted():
    clk = Clock()
    p, _ = _paced([(429, "")] * 4, clk)
    with pytest.raises(RateLimited):
        p.get_text(KAL)


def test_http_error_carries_status():
    clk = Clock()
    p, _ = _paced([(404, "not found")], clk)
    with pytest.raises(HttpError) as e:
        p.get_text(KAL)
    assert e.value.status == 404


def test_get_json_parses():
    clk = Clock()
    p, _ = _paced([(200, '{"a": 1}')], clk)
    assert p.get_json(KAL) == {"a": 1}


def test_paced_is_thread_safe_and_spaces_fetch_starts():
    import threading
    import time as _t
    lock = threading.Lock()
    state = {"t": 0.0}
    starts = []

    def clock():
        with lock:
            return state["t"]

    def sleep(s):
        with lock:
            state["t"] += s
        _t.sleep(0.0005)

    def fetch(url, params):
        starts.append(clock())
        return 200, "ok"
    p = Paced(fetch=fetch, sleep=sleep, clock=clock)
    assert hasattr(p, "_lock")

    def work():
        for _ in range(10):
            p.get_text(KAL)
    ts = [threading.Thread(target=work) for _ in range(2)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    starts.sort()
    assert len(starts) == 20
    assert all(b - a >= 0.2 - 1e-9 for a, b in zip(starts, starts[1:]))
