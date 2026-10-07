import math

from highcast.fetchers.forecasts import parse_openmeteo_ensemble

# KLAX std_utc_offset is -8 (PST). climate_day_bounds(date(2026,7,24), -8) =
# 2026-07-24T08:00Z .. 2026-07-25T08:00Z. Build 24 hourly UTC stamps inside it.
def _hours():
    return [f"2026-07-24T{h:02d}:00" for h in range(8, 24)] + \
           [f"2026-07-25T{h:02d}:00" for h in range(0, 8)]


def _payload(member_series: dict):
    return {"hourly": {"time": _hours(),
                       # bare control column present but must be IGNORED:
                       "temperature_2m_ncep_gefs025": [70.0] * 24,
                       **member_series}}


def test_bimodal_split_summary():
    # 20 members: 10 peak at ~78 (layer holds), 10 peak at ~84 (clears).
    series = {}
    for i in range(1, 11):
        series[f"temperature_2m_member{i:02d}_ncep_gefs025"] = [78.0] * 24
    for i in range(11, 21):
        series[f"temperature_2m_member{i:02d}_ncep_gefs025"] = [84.0] * 24
    rows = parse_openmeteo_ensemble(_payload(series), "KLAX",
                                    "2026-07-24", "2026-07-24T16:00:00+00:00", -8)
    row = next(r for r in rows if r["target_date"] == "2026-07-24")
    assert row["n_members"] == 20
    assert row["ens_p10"] == 78.0 and row["ens_p90"] == 84.0
    assert math.isclose(row["ens_frac_ge_82"], 0.5)
    # a 6°F empty gap between two clumps, std 3 => bimodality ~2, clearly high
    assert row["ens_bimodality"] > 1.0
    assert set(row) >= {"station", "run_date", "metric", "fetched_at", "ens_mean"}


def test_unimodal_low_bimodality():
    series = {f"temperature_2m_member{i:02d}_ncep_gefs025":
              [77.0 + (i - 1) * 0.1] * 24 for i in range(1, 21)}
    rows = parse_openmeteo_ensemble(_payload(series), "KLAX",
                                    "2026-07-24", "2026-07-24T16:00:00+00:00", -8)
    row = next(r for r in rows if r["target_date"] == "2026-07-24")
    assert row["ens_bimodality"] < 1.0          # tight cluster, no gap
    assert row["ens_frac_ge_82"] == 0.0


def test_below_floor_writes_nan():
    series = {f"temperature_2m_member{i:02d}_ncep_gefs025": [80.0] * 24
              for i in range(1, 6)}            # only 5 members < floor 10
    rows = parse_openmeteo_ensemble(_payload(series), "KLAX",
                                    "2026-07-24", "2026-07-24T16:00:00+00:00", -8)
    row = next(r for r in rows if r["target_date"] == "2026-07-24")
    assert row["n_members"] == 5
    assert row["ens_std"] is None and row["ens_frac_ge_82"] is None
