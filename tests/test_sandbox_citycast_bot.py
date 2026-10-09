import json
from contextlib import closing
from dataclasses import replace
from pathlib import Path

from citycast import db as hdb
from sandbox.bots import CitycastBot
from sandbox_helpers import make_market

ISSUED = "2026-10-08T00:00:00+00:00"


def _history(tmp_path, gate=1, issued=ISSUED, variant="shared", choice="shared"):
    p = tmp_path / "history.db"
    with closing(hdb.connect(p)) as c, c:
        c.execute("INSERT INTO model_choice VALUES ('KNYC','high',?,1,1,-0.1,?,1,1,90,?,'x')",
                  (choice, gate, gate))
        c.execute("INSERT INTO live_predictions VALUES ('KNYC','high','2026-10-08',?,?,70,NULL,'{}',?,1.0)",
                  (issued, variant, json.dumps({"69": 0.2, "70": 0.5, "71": 0.3})))
    return str(p)


def _markets():
    ms = [make_market("KXHIGHNY-26OCT08-B69.5", lo=69, hi=70, station="KNYC", metric="high"),
          make_market("KXHIGHNY-26OCT08-T70", lo=71, hi=None, station="KNYC", metric="high")]
    return [replace(m, event_date="2026-10-08") for m in ms]


class Snap:
    taken_at = "2026-10-08T01:00:00+00:00"


def test_reads_latest_gated_prediction(tmp_path):
    probs, fj = CitycastBot(_history(tmp_path)).forecast_for_event(_markets(), Snap())
    assert abs(probs["KXHIGHNY-26OCT08-B69.5"] - 0.7) < 1e-9 and fj["source"] == "citycast"


def test_ungated_stale_or_missing_db_gives_nothing(tmp_path):
    assert CitycastBot(_history(tmp_path, gate=0)).forecast_for_event(_markets(), Snap()) == (None, None)
    old = "2026-10-07T20:00:00+00:00"                          # 5 h before the snapshot
    assert CitycastBot(_history(tmp_path / "o", issued=old)).forecast_for_event(_markets(), Snap()) == (None, None)
    assert CitycastBot(str(tmp_path / "none.db")).forecast_for_event(_markets(), Snap()) == (None, None)


def test_prediction_variant_must_match_current_choice(tmp_path):
    p = _history(tmp_path, variant="city", choice="shared")
    assert CitycastBot(p).forecast_for_event(_markets(), Snap()) == (None, None)


def test_never_writes_history_db(tmp_path):
    p = _history(tmp_path)
    before = Path(p).read_bytes()
    CitycastBot(p).forecast_for_event(_markets(), Snap())
    assert Path(p).read_bytes() == before
