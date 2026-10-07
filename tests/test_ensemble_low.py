"""ENSEMBLE FOR THE LOW — and the latent bug that blocked it.

WHAT WAS WRONG. `parse_openmeteo_ensemble` takes `metric_key` and stamps it on
every row, but the aggregation was hardcoded `max(vals)`. Asking it for the LOW
would have produced rows LABELLED "low" carrying each member's daily MAXIMUM —
silently, with no error, into a table the desk reads. Nothing had ever asked
for the low, so the bug had never fired.

WHY IT MATTERS NOW. `forecast_ens` held 164 KLAX-high and 145 KMIA-high rows
and ZERO low rows for either station, so both LOW markets showed no ensemble on
the station table. The lows are where KMIA keeps getting hurt: on 2026-08-06 it
settled 75F against a 78-79 call, and its big misses are all one-directional
(33 walk-forward days >4F too WARM, zero >4F too cold).

⚠️ `ens_frac_ge_82` IS HIGH-ONLY AND STAYS THAT WAY. 82F is the KLAX marine-layer
seam's upper edge — "what fraction of members clear the seam". It is meaningless
for a low, and picking some new threshold for the lows would be inventing a
constant nobody measured. It comes back None there instead.
"""
import pytest

from highcast.fetchers import forecasts


def _payload(vals_by_member: dict[str, list[float]], times: list[str]) -> dict:
    return {"hourly": {"time": times, **vals_by_member}}


TIMES = [f"2026-08-06T{h:02d}:00" for h in range(24)] + \
        [f"2026-08-07T{h:02d}:00" for h in range(12)]


def _members(n: int, shape):
    # The real column shape is `temperature_2m_member00_ncep_gefs025` — the
    # model suffix is part of what `_ENS_MEMBER_RE` requires.
    return {f"temperature_2m_member{i:02d}_ncep_gefs025":
            [shape(i, h) for h in range(len(TIMES))] for i in range(n)}


def test_low_takes_the_daily_MINIMUM_not_the_maximum():
    """THE BUG. A rising-then-falling day whose min and max are far apart: if
    the aggregation is still `max`, p50 lands near the peak instead of the
    trough and the row is a lie wearing the right label."""
    # member i: dips to 60+i in the small hours, peaks at 90+i mid-afternoon
    shape = lambda i, h: (60 + i) if h % 24 in (5, 6, 7) else (90 + i)
    p = _payload(_members(30, shape), TIMES)
    lo = forecasts.parse_openmeteo_ensemble(p, "KMIA", "2026-08-06", "x", 5, "low")
    hi = forecasts.parse_openmeteo_ensemble(p, "KMIA", "2026-08-06", "x", 5, "high")
    assert lo and hi
    assert lo[0]["metric"] == "low" and hi[0]["metric"] == "high"
    assert lo[0]["ens_p50"] < 80, lo[0]["ens_p50"]
    assert hi[0]["ens_p50"] > 80, hi[0]["ens_p50"]


def test_the_high_is_completely_unchanged():
    """Every existing row was built by the high path; it must be byte-identical
    or 164 KLAX rows silently change meaning."""
    shape = lambda i, h: 70 + i + (h % 5)
    p = _payload(_members(40, shape), TIMES)
    default = forecasts.parse_openmeteo_ensemble(p, "KLAX", "2026-08-06", "x", 8)
    explicit = forecasts.parse_openmeteo_ensemble(p, "KLAX", "2026-08-06", "x", 8,
                                                  "high")
    assert default == explicit


def test_frac_ge_82_is_high_only():
    """82F is the KLAX marine-layer seam's upper edge. It has no meaning for a
    low, and inventing a low-side threshold would be a constant nobody
    measured."""
    shape = lambda i, h: 60 + i
    p = _payload(_members(40, shape), TIMES)
    lo = forecasts.parse_openmeteo_ensemble(p, "KLAX", "2026-08-06", "x", 8, "low")
    hi = forecasts.parse_openmeteo_ensemble(p, "KLAX", "2026-08-06", "x", 8, "high")
    assert lo[0]["ens_frac_ge_82"] is None
    assert hi[0]["ens_frac_ge_82"] is not None


def test_spread_and_bimodality_still_computed_for_the_low():
    """The two fields the station table actually shows must survive."""
    shape = lambda i, h: (55 + i * 2) if h % 24 in (5, 6) else (85 + i)
    p = _payload(_members(40, shape), TIMES)
    lo = forecasts.parse_openmeteo_ensemble(p, "KMIA", "2026-08-06", "x", 5, "low")
    assert lo[0]["ens_std"] is not None and lo[0]["ens_std"] > 0
    assert lo[0]["ens_bimodality"] is not None


def test_below_the_member_floor_everything_is_none(monkeypatch):
    shape = lambda i, h: 60 + i
    p = _payload(_members(2, shape), TIMES)
    lo = forecasts.parse_openmeteo_ensemble(p, "KMIA", "2026-08-06", "x", 5, "low")
    assert lo[0]["n_members"] == 2
    assert lo[0]["ens_std"] is None


def test_fetch_asks_for_both_metrics(monkeypatch):
    """One HTTP call, both metrics — the member temperatures are the same
    payload, so fetching twice would double the request for nothing."""
    from highcast.config import load_config
    cfg = load_config("config/kmia.yaml")
    shape = lambda i, h: (60 + i) if h % 24 in (5, 6) else (90 + i)
    calls = []

    def fake_get_json(url, params):
        calls.append(params)
        return _payload(_members(30, shape), TIMES)

    monkeypatch.setattr(forecasts, "get_json", fake_get_json)
    rows = forecasts.fetch_openmeteo_ensemble(cfg)
    assert len(calls) == 1, "must not fetch the same members twice"
    assert {r["metric"] for r in rows} == {"high", "low"}
