from citycast.sources.kalshi_markets import (chunks, event_date, fetch_series_markets,
                                             interval, market_rows)

IDX = {"KXHIGHNY": ("KNYC", "high")}


def test_event_date_handles_modern_legacy_and_double_dash():
    assert event_date("KXHIGHNY-26OCT04") == "2026-10-04"
    assert event_date("HIGHNY-21AUG06") == "2021-08-06"
    assert event_date("HIGHAUS--23MAY11") == "2023-05-11"
    assert event_date("NODATE") is None


def test_interval_from_strike_fields_matches_live_parser():
    assert interval({"strike_type": "between", "floor_strike": 66, "cap_strike": 67}) == (66, 67)
    assert interval({"strike_type": "greater", "floor_strike": 72}) == (73, None)
    assert interval({"strike_type": "less", "cap_strike": 60}) == (None, 59)


def test_interval_from_legacy_rules_text():
    gt = {"rules_primary": "If the highest temperature ... is strictly greater than 86°F, then ..."}
    lt = {"rules_primary": "... is strictly less than 50°F, then the market resolves to Yes."}
    bt = {"rules_primary": "... is between 66-67°, then the market resolves to Yes."}
    assert interval(gt) == (87, None) and interval(lt) == (None, 49) and interval(bt) == (66, 67)
    assert interval({"rules_primary": "something else"}) is None


def test_market_rows_station_from_series_and_unusable_kept():
    ms = [{"ticker": "HIGHNY-21AUG06-T86", "event_ticker": "HIGHNY-21AUG06",
           "rules_primary": "strictly greater than 86°F", "close_time": "2021-08-07T03:59:00Z",
           "open_time": "2021-08-05T14:00:00Z", "result": "yes", "expiration_value": "88"},
          {"ticker": "HIGHNY-21AUG06-X", "event_ticker": "HIGHNY-21AUG06",
           "rules_primary": "unparseable", "close_time": "2021-08-07T03:59:00Z", "result": "no"},
          {"ticker": "KXHIGHNY-26OCT04-B70.5", "event_ticker": "KXHIGHNY-26OCT04",
           "market_type": "scalar", "close_time": "2026-10-05T04:59:00Z"}]
    rows = {r["ticker"]: r for r in market_rows(ms, "KXHIGHNY", IDX)}
    assert set(rows) == {"HIGHNY-21AUG06-T86", "HIGHNY-21AUG06-X"}   # non-binary dropped
    a = rows["HIGHNY-21AUG06-T86"]
    assert (a["station"], a["metric"], a["event_date"], a["lo"], a["hi"]) == (
        "KNYC", "high", "2021-08-06", 87, None)
    assert a["close_time"] == "2021-08-07T03:59:00+00:00" and a["expiration_value"] == "88"
    b = rows["HIGHNY-21AUG06-X"]
    assert b["lo"] is None and b["hi"] is None


def test_fetch_series_markets_pages_both_endpoints_live_wins():
    pages = {
        ("/historical/markets", None): {"markets": [{"ticker": "A", "v": "hist"}], "cursor": "c1"},
        ("/historical/markets", "c1"): {"markets": [{"ticker": "B", "v": "hist"}], "cursor": ""},
        ("/markets", None): {"markets": [{"ticker": "B", "v": "live"}], "cursor": ""},
    }
    calls = []

    class FakePaced:
        def get_json(self, url, params):
            path = url.split("/trade-api/v2")[1]
            calls.append((path, params))
            return pages[(path, params.get("cursor"))]
    got = {m["ticker"]: m["v"] for m in fetch_series_markets(FakePaced(), "KXHIGHNY")}
    assert got == {"A": "hist", "B": "live"}
    assert calls[-1][1]["status"] == "settled"


def test_chunks_one_per_series_always_current():
    cs = chunks(object(), {"KXHIGHNY": ("KNYC", "high"), "KXLOWTNYC": ("KNYC", "low")})
    assert sorted(c.key for c in cs) == ["KXHIGHNY", "KXLOWTNYC"] and all(c.current for c in cs)
