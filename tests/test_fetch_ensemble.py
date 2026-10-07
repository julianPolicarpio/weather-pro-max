from highcast.config import load_config
from highcast.fetchers import forecasts


def test_fetch_openmeteo_ensemble_parses_mocked_payload(monkeypatch):
    cfg = load_config()
    captured = {}

    def fake_get_json(url, params=None):
        captured["url"] = url
        captured["params"] = params
        hrs = [f"2026-07-24T{h:02d}:00" for h in range(8, 24)] + \
              [f"2026-07-25T{h:02d}:00" for h in range(0, 8)]
        series = {f"temperature_2m_member{i:02d}_ncep_gefs025": [79.0 + (i % 4)] * 24
                  for i in range(1, 21)}
        return {"hourly": {"time": hrs, **series}}

    monkeypatch.setattr(forecasts, "get_json", fake_get_json)
    rows = forecasts.fetch_openmeteo_ensemble(cfg, past_days=1, forecast_days=1)
    assert captured["url"] == forecasts.ENSEMBLE_URL
    assert captured["params"]["models"] == forecasts.ENSEMBLE_MODELS
    assert captured["params"]["past_days"] == 1
    assert captured["params"]["temperature_unit"] == "fahrenheit"
    assert captured["params"]["timezone"] == "GMT"
    assert any(r["target_date"] == "2026-07-24" and r["n_members"] == 20
               for r in rows)
