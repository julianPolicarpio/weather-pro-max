from highcast.fetchers import fivemin
from highcast.config import load_config, AppConfig

SAMPLE = {
    "STATION": [{
        "OBSERVATIONS": {
            "date_time": ["2026-07-17T20:00:00Z", "2026-07-17T20:05:00Z"],
            "air_temp_set_1": [22.0, 23.0],
        }
    }]
}

def test_parse_synoptic_payload():
    rows = fivemin.parse_rows(SAMPLE, "KLAX", "2026-07-17T20:06:00+00:00")
    assert len(rows) == 2
    assert rows[0]["source"] == "syn5"
    assert rows[0]["tmpf"] == 71.6           # 22 C -> displayed F equivalent
    assert rows[0]["valid_utc"].startswith("2026-07-17T20:00")

def test_blank_token_returns_empty():
    cfg = load_config()
    assert cfg.synoptic_token == ""
    assert fivemin.fetch_recent(cfg) == []
