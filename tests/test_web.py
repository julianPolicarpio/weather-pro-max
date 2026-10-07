import json
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from fastapi.testclient import TestClient
from highcast import db as hdb
from highcast.config import load_config
from highcast.fetchers import kalshi
from highcast.timeutil import climate_day_bounds, round_half_up
from highcast.web import app as app_module
from highcast.web.app import create_app

def _client(tmp_path, seed_tomorrow=False):
    cfg = load_config()
    object.__setattr__(cfg, "db_path", tmp_path / "t.db")
    conn = hdb.connect(cfg.db_path)
    today = datetime.now(ZoneInfo(cfg.station.tz)).date()
    hdb.upsert(conn, "predictions", [{
        "station": cfg.station.icao, "target_date": today.isoformat(),
        "issued_at_utc": "2026-07-17T02:00:00+00:00", "kind": "dayahead",
        "point_f": 84.0, "quantiles": json.dumps({"0.5": 84.0}),
        "buckets": json.dumps({"83": 0.2, "84": 0.5, "85": 0.3}),
    }])
    if seed_tomorrow:
        hdb.upsert(conn, "predictions", [{
            "station": cfg.station.icao,
            "target_date": (today + timedelta(days=1)).isoformat(),
            # The tomorrow card asks for the EVENING forecast by name: at 17:00
            # local nothing else exists for tomorrow. See db.KIND_EVENING.
            "issued_at_utc": "2026-07-18T01:00:00+00:00", "kind": hdb.KIND_EVENING,
            "point_f": 86.0, "quantiles": json.dumps({"0.5": 86.0}),
            "buckets": json.dumps({"85": 0.3, "86": 0.4, "87": 0.3}),
        }])
    return TestClient(create_app(cfg))

def test_today_returns_latest_prediction(tmp_path):
    r = _client(tmp_path).get("/api/today")
    assert r.status_code == 200
    body = r.json()
    assert body["dayahead"]["buckets"]["84"] == 0.5

def test_index_renders(tmp_path):
    r = _client(tmp_path).get("/")
    assert r.status_code == 200
    assert "highcast" in r.text.lower()

def test_index_has_panel_anchors(tmp_path):
    r = _client(tmp_path).get("/")
    assert r.status_code == 200
    assert 'id="timing-panel"' in r.text
    assert 'id="bucket-board"' in r.text
    assert 'id="dist-chart"' in r.text
    assert 'id="place-rec"' in r.text
    assert 'id="tomorrow-card"' in r.text
    assert 'id="model-pick"' in r.text
    assert "TODAY" in r.text
    assert "TOMORROW" in r.text

def test_clock_endpoint(tmp_path):
    r = _client(tmp_path).get("/api/clock")
    assert any(e["name"] == "Settlement" for e in r.json())

def test_timing_endpoint(tmp_path):
    r = _client(tmp_path).get("/api/timing")
    assert r.status_code == 200
    body = r.json()
    assert body["phase"] in {
        "overnight", "morning_obs", "late_morning", "peak_heating",
        "post_peak", "report_out",
    }
    assert "la" in body["next_transition"] and "manila" in body["next_transition"]

def test_today_has_freshness_fields_on_dayahead(tmp_path):
    body = _client(tmp_path).get("/api/today").json()
    da = body["dayahead"]
    assert isinstance(da["age_minutes"], int)
    assert isinstance(da["stale"], bool)
    assert da["age_minutes"] >= 0

def test_today_kalshi_buckets_sum_to_one_and_not_dead(tmp_path):
    body = _client(tmp_path).get("/api/today").json()
    buckets = body["kalshi_buckets"]
    assert buckets, "expected at least one kalshi bucket"
    for b in buckets:
        assert set(b.keys()) >= {"label", "lo", "hi", "p_dayahead", "p_intraday", "dead"}
    total = sum(b["p_dayahead"] for b in buckets if b["p_dayahead"] is not None)
    assert abs(total - 1.0) <= 0.01
    # no obs seeded -> no running max -> nothing can be "dead" yet
    assert all(b["dead"] is False for b in buckets)
    # intraday has no prediction in this fixture -> null, not zero
    assert all(b["p_intraday"] is None for b in buckets)

def test_today_tomorrow_null_when_no_row(tmp_path):
    body = _client(tmp_path).get("/api/today").json()
    tm = body["tomorrow"]
    cfg = load_config()
    exp = (datetime.now(ZoneInfo(cfg.station.tz)).date()
           + timedelta(days=1)).isoformat()
    assert tm["target_date"] == exp
    assert tm["prediction"] is None
    assert tm["kalshi_buckets"] == []

def test_today_tomorrow_populated(tmp_path):
    body = _client(tmp_path, seed_tomorrow=True).get("/api/today").json()
    tm = body["tomorrow"]
    pred = tm["prediction"]
    assert pred["point"] == 86.0
    assert pred["buckets"]["86"] == 0.4
    assert isinstance(pred["age_minutes"], int)
    assert isinstance(pred["stale"], bool)
    buckets = tm["kalshi_buckets"]
    assert buckets, "expected kalshi buckets for tomorrow"
    # nothing observed for a future date: intraday null, nothing dead
    assert all(b["p_intraday"] is None for b in buckets)
    assert all(b["dead"] is False for b in buckets)
    total = sum(b["p_dayahead"] for b in buckets if b["p_dayahead"] is not None)
    assert abs(total - 1.0) <= 0.01
    # today's response must be untouched by the tomorrow row
    assert body["dayahead"]["point"] == 84.0

# Real-market ranges shaped like fetch_ranges' output, spanning the seeded
# dayahead pmf ({"83": 0.2, "84": 0.5, "85": 0.3}) so mass-summing can be
# checked against a known total.
_FIXTURE_RANGES = [
    {"label": "≤83", "lo": None, "hi": 83, "yes_bid": 0.19, "yes_ask": 0.21, "last_price": 0.20},
    {"label": "84–85", "lo": 84, "hi": 85, "yes_bid": 0.78, "yes_ask": 0.80, "last_price": 0.79},
    {"label": "≥86", "lo": 86, "hi": None, "yes_bid": 0.0, "yes_ask": 0.01, "last_price": None},
]

def test_kalshi_market_appears_when_fetch_ranges_available(tmp_path, monkeypatch):
    monkeypatch.setattr(kalshi, "fetch_ranges", lambda cfg, target_date, metric=None: _FIXTURE_RANGES)
    body = _client(tmp_path).get("/api/today").json()
    km = body["kalshi_market"]
    assert km is not None
    assert len(km) == 3
    for item in km:
        assert set(item.keys()) >= {
            "label", "lo", "hi", "market_price", "p_model_dayahead", "p_model_intraday"}
    total = sum(i["p_model_dayahead"] for i in km if i["p_model_dayahead"] is not None)
    assert abs(total - 1.0) <= 0.01
    # last_price present -> used directly
    top = next(i for i in km if i["label"] == "84–85")
    assert top["market_price"] == 0.79
    # last_price missing -> falls back to bid/ask mid
    tail = next(i for i in km if i["label"] == "≥86")
    assert abs(tail["market_price"] - 0.005) < 1e-6
    # intraday has no prediction in this fixture -> null, not zero
    assert all(i["p_model_intraday"] is None for i in km)
    # synthetic fallback stays present alongside the live market
    assert body["kalshi_buckets"]

def test_kalshi_market_null_when_fetch_ranges_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(kalshi, "fetch_ranges", lambda cfg, target_date, metric=None: None)
    body = _client(tmp_path).get("/api/today").json()
    assert body["kalshi_market"] is None
    # synthetic fallback still present
    buckets = body["kalshi_buckets"]
    assert buckets
    total = sum(b["p_dayahead"] for b in buckets if b["p_dayahead"] is not None)
    assert abs(total - 1.0) <= 0.01

def test_kalshi_market_populated_for_tomorrow_too(tmp_path, monkeypatch):
    monkeypatch.setattr(kalshi, "fetch_ranges", lambda cfg, target_date, metric=None: _FIXTURE_RANGES)
    body = _client(tmp_path, seed_tomorrow=True).get("/api/today").json()
    tm_km = body["tomorrow"]["kalshi_market"]
    assert tm_km is not None
    assert len(tm_km) == 3
    # no obs constrain a future date: intraday stays null for every range
    assert all(i["p_model_intraday"] is None for i in tm_km)

# ---------------- public-feed display temperature (upgrade 4, item 1) -----

def _client_with_running_max(tmp_path, tmpf_precise):
    """Seed a single metar obs (tmpf_precise controls the running-max hi
    bound exactly: hi = tmpf_precise + METAR_TOL) inside today's climate day,
    and return a TestClient against that DB."""
    cfg = load_config()
    object.__setattr__(cfg, "db_path", tmp_path / "t.db")
    conn = hdb.connect(cfg.db_path)
    today = datetime.now(ZoneInfo(cfg.station.tz)).date()
    start, _end = climate_day_bounds(today, cfg.station.std_utc_offset)
    valid_utc = (start + timedelta(hours=2)).isoformat()
    hdb.upsert(conn, "obs", [{
        "station": cfg.station.icao, "valid_utc": valid_utc, "source": "metar",
        "tmpf": None, "tmpf_precise": tmpf_precise, "dwpf": None, "sknt": None,
        "drct": None, "skyc": None, "six_max_c": None, "raw": "",
        "fetched_at": "x",
    }])
    return TestClient(create_app(cfg))

def test_public_display_f_present_and_differs_on_double_rounding(tmp_path):
    # tmpf_precise=78.08 -> hi = 78.18: rounds to whole C (26) then back to F
    # (79), one higher than round_half_up(78.18) == 78 -- exactly the public
    # double-rounding artifact this feature surfaces.
    client = _client_with_running_max(tmp_path, 78.08)
    body = client.get("/api/today").json()
    rm = body["running_max"]
    assert rm["hi"] == 78.18
    assert rm["public_display_f"] == 79
    assert round_half_up(rm["hi"]) == 78

def test_public_display_f_agrees_when_no_double_rounding_effect(tmp_path):
    # tmpf_precise=76.9 -> hi = 77.0 exactly: whole-degree bound, no
    # C-rounding artifact, so public display agrees with the precise max.
    client = _client_with_running_max(tmp_path, 76.9)
    body = client.get("/api/today").json()
    rm = body["running_max"]
    assert rm["hi"] == 77.0
    assert rm["public_display_f"] == 77
    assert rm["public_display_f"] == round_half_up(rm["hi"])

def test_public_display_f_absent_when_no_running_max(tmp_path):
    body = _client(tmp_path).get("/api/today").json()
    assert body["running_max"] is None

def test_public_display_anchor_present_in_index_template(tmp_path):
    r = _client(tmp_path).get("/")
    assert r.status_code == 200
    assert 'id="public-display"' in r.text

# ---------------- model pick in PLACE recommendation (upgrade 4, item 2) --

def _freeze_now(monkeypatch, when):
    """Monkeypatch highcast.web.app's `datetime` symbol so datetime.now(tz)
    returns a fixed instant -- makes /api/timing's phase/target_date
    deterministic without a freezegun dependency."""
    class _Frozen(datetime):
        @classmethod
        def now(cls, tz=None):
            return when if tz is None else when.astimezone(tz)
    monkeypatch.setattr(app_module, "datetime", _Frozen)

def _seed_prediction(conn, cfg, target_date, kind, issued_at_utc, buckets):
    hdb.upsert(conn, "predictions", [{
        "station": cfg.station.icao, "target_date": target_date,
        "issued_at_utc": issued_at_utc, "kind": kind,
        "point_f": 74.0, "quantiles": json.dumps({"0.5": 74.0}),
        "buckets": json.dumps(buckets),
    }])

def test_model_pick_today_intraday_synthetic_buckets(tmp_path, monkeypatch):
    cfg = load_config()
    object.__setattr__(cfg, "db_path", tmp_path / "t.db")
    conn = hdb.connect(cfg.db_path)
    _seed_prediction(conn, cfg, "2026-07-18", "intraday",
                      "2026-07-18T08:00:00+00:00",
                      {"73": 0.1, "74": 0.6, "75": 0.3})
    monkeypatch.setattr(kalshi, "fetch_ranges", lambda cfg, target_date, metric=None: None)
    # 2026-07-18T20:30Z == 1:30pm PDT -> peak_heating, target_date=2026-07-18
    _freeze_now(monkeypatch, datetime(2026, 7, 18, 20, 30, 0, tzinfo=timezone.utc))
    body = TestClient(create_app(cfg)).get("/api/timing").json()
    assert body["recommendation"]["target_date"] == "2026-07-18"
    mp = body["recommendation"]["model_pick"]
    assert (mp["label"], mp["prob"], mp["source"]) == ("73–74", 0.7, "intraday")
    assert "actionable" in mp and "gate" in mp   # gating attached to the pick

def test_model_pick_today_falls_back_to_dayahead_when_no_intraday(tmp_path, monkeypatch):
    cfg = load_config()
    object.__setattr__(cfg, "db_path", tmp_path / "t.db")
    conn = hdb.connect(cfg.db_path)
    _seed_prediction(conn, cfg, "2026-07-18", "dayahead",
                      "2026-07-18T02:00:00+00:00",
                      {"73": 0.1, "74": 0.6, "75": 0.3})
    monkeypatch.setattr(kalshi, "fetch_ranges", lambda cfg, target_date, metric=None: None)
    _freeze_now(monkeypatch, datetime(2026, 7, 18, 20, 30, 0, tzinfo=timezone.utc))
    body = TestClient(create_app(cfg)).get("/api/timing").json()
    mp = body["recommendation"]["model_pick"]
    assert (mp["label"], mp["prob"], mp["source"]) == ("73–74", 0.7, "dayahead")
    assert "actionable" in mp and "gate" in mp

def test_model_pick_tomorrow_dayahead_live_market(tmp_path, monkeypatch):
    cfg = load_config()
    object.__setattr__(cfg, "db_path", tmp_path / "t.db")
    conn = hdb.connect(cfg.db_path)
    _seed_prediction(conn, cfg, "2026-07-19", "dayahead",
                      "2026-07-18T01:00:00+00:00",
                      {"83": 0.2, "84": 0.5, "85": 0.3})
    monkeypatch.setattr(kalshi, "fetch_ranges", lambda cfg, target_date, metric=None: _FIXTURE_RANGES)
    # 2026-07-19T00:00Z == 5pm PDT Jul 18 -> report_out, target_date=2026-07-19
    _freeze_now(monkeypatch, datetime(2026, 7, 19, 0, 0, 0, tzinfo=timezone.utc))
    body = TestClient(create_app(cfg)).get("/api/timing").json()
    assert body["recommendation"]["target_date"] == "2026-07-19"
    mp = body["recommendation"]["model_pick"]
    assert (mp["label"], mp["prob"], mp["source"]) == ("84–85", 0.8, "dayahead")
    assert "actionable" in mp and "gate" in mp

def test_model_pick_null_when_no_prediction(tmp_path, monkeypatch):
    cfg = load_config()
    object.__setattr__(cfg, "db_path", tmp_path / "t.db")
    hdb.connect(cfg.db_path)  # empty DB, no predictions seeded
    monkeypatch.setattr(kalshi, "fetch_ranges", lambda cfg, target_date, metric=None: None)
    _freeze_now(monkeypatch, datetime(2026, 7, 18, 20, 30, 0, tzinfo=timezone.utc))
    body = TestClient(create_app(cfg)).get("/api/timing").json()
    assert body["recommendation"]["model_pick"] is None


def test_is_stale_schedule_aware():
    from datetime import datetime, timezone
    from highcast.config import load_config
    from highcast.web.app import _is_stale

    cfg = load_config()
    utc = timezone.utc
    evening_issue = datetime(2026, 7, 19, 0, 31, tzinfo=utc)   # 17:31 PDT Jul 18

    # 04:00 PDT: overnight aging of the evening forecast is on schedule
    assert _is_stale("dayahead", evening_issue, datetime(2026, 7, 19, 11, 0, tzinfo=utc), cfg) is False
    # 10:00 PDT with the 9am refresh missed -> genuinely stale
    assert _is_stale("dayahead", evening_issue, datetime(2026, 7, 19, 17, 0, tzinfo=utc), cfg) is True
    # 10:00 PDT with a fresh 09:05 refresh -> fine
    morning_issue = datetime(2026, 7, 19, 16, 5, tzinfo=utc)
    assert _is_stale("dayahead", morning_issue, datetime(2026, 7, 19, 17, 0, tzinfo=utc), cfg) is False
    # intraday inside the climate day: 45 min old means the auto-cycle is failing
    now_in_day = datetime(2026, 7, 19, 20, 0, tzinfo=utc)      # 13:00 PDT
    assert _is_stale("intraday", datetime(2026, 7, 19, 19, 15, tzinfo=utc), now_in_day, cfg) is True
    assert _is_stale("intraday", datetime(2026, 7, 19, 19, 50, tzinfo=utc), now_in_day, cfg) is False
    # 00:30 PDT sits before the 01:00 climate-day start: nothing to regenerate
    boundary = datetime(2026, 7, 19, 7, 30, tzinfo=utc)
    assert _is_stale("intraday", datetime(2026, 7, 19, 1, 0, tzinfo=utc), boundary, cfg) is False


# ---------------- field notes (upgrade 5) ---------------------------------

def test_post_note_valid_returns_id(tmp_path):
    client = _client(tmp_path)
    today = datetime.now(ZoneInfo(load_config().station.tz)).date()
    r = client.post("/api/note", json={
        "target_date": today.isoformat(),
        "sky": "Marine layer never cleared — grey till 3pm"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert isinstance(body["id"], int)

def test_post_note_then_get_notes_has_context_and_phase(tmp_path):
    client = _client(tmp_path)
    today = datetime.now(ZoneInfo(load_config().station.tz)).date()
    client.post("/api/note", json={
        "target_date": today.isoformat(), "sky": "Burned off by 9am, full sun"})
    r = client.get("/api/notes")
    assert r.status_code == 200
    notes = r.json()
    assert len(notes) == 1
    note = notes[0]
    assert note["sky"] == "Burned off by 9am, full sun"
    assert note["target_date"] == today.isoformat()
    assert note["created_utc"]
    assert note["actual_high"] is None
    ctx = note["context"]
    assert set(ctx.keys()) == {"running_max", "dayahead", "intraday", "phase"}
    assert ctx["phase"] in {
        "overnight", "morning_obs", "late_morning", "peak_heating",
        "post_peak", "report_out"}

def test_post_note_context_captures_dayahead_pick(tmp_path):
    client = _client(tmp_path)
    today = datetime.now(ZoneInfo(load_config().station.tz)).date()
    client.post("/api/note", json={
        "target_date": today.isoformat(), "sky": "Hot and dry, Santa Ana"})
    note = client.get("/api/notes").json()[0]
    ctx = note["context"]
    # seeded dayahead buckets {"83": 0.2, "84": 0.5, "85": 0.3} -> odd-pair
    # bucket 83-84 carries 0.7, beating 85-86's 0.3
    assert ctx["dayahead"]["point"] == 84.0
    assert ctx["dayahead"]["pick_label"] == "83–84"
    assert abs(ctx["dayahead"]["pick_prob"] - 0.7) < 1e-6
    assert ctx["intraday"] is None
    assert ctx["running_max"] is None

def test_post_note_context_captures_running_max(tmp_path):
    cfg = load_config()
    object.__setattr__(cfg, "db_path", tmp_path / "t.db")
    conn = hdb.connect(cfg.db_path)
    today = datetime.now(ZoneInfo(cfg.station.tz)).date()
    start, _end = climate_day_bounds(today, cfg.station.std_utc_offset)
    valid_utc = (start + timedelta(hours=2)).isoformat()
    hdb.upsert(conn, "obs", [{
        "station": cfg.station.icao, "valid_utc": valid_utc, "source": "metar",
        "tmpf": None, "tmpf_precise": 76.9, "dwpf": None, "sknt": None,
        "drct": None, "skyc": None, "six_max_c": None, "raw": "",
        "fetched_at": "x",
    }])
    client = TestClient(create_app(cfg))
    client.post("/api/note", json={"target_date": today.isoformat(), "sky": "Sunny all day"})
    note = client.get("/api/notes").json()[0]
    assert note["context"]["running_max"] == {"lo": 76.8, "hi": 77.0}

def test_post_note_empty_sky_400(tmp_path):
    client = _client(tmp_path)
    r = client.post("/api/note", json={"target_date": "2026-07-18", "sky": "   "})
    assert r.status_code == 400

def test_post_note_oversized_sky_400(tmp_path):
    client = _client(tmp_path)
    r = client.post("/api/note", json={"target_date": "2026-07-18", "sky": "x" * 501})
    assert r.status_code == 400

def test_post_note_bad_date_400(tmp_path):
    client = _client(tmp_path)
    r = client.post("/api/note", json={"target_date": "not-a-date", "sky": "hot and dry"})
    assert r.status_code == 400

def test_get_notes_joins_actual_high_when_settled(tmp_path):
    cfg = load_config()
    object.__setattr__(cfg, "db_path", tmp_path / "t.db")
    conn = hdb.connect(cfg.db_path)
    client = TestClient(create_app(cfg))
    target = "2026-07-18"
    hdb.upsert(conn, "cli_reports", [{
        "station": cfg.station.icao, "date": target, "high_f": 81, "low_f": 60,
        "raw": "{}", "fetched_at": "2026-07-18T23:59:00+00:00"}])
    client.post("/api/note", json={"target_date": target, "sky": "Hot and dry"})
    note = client.get("/api/notes").json()[0]
    assert note["actual_high"] == 81

def test_get_notes_newest_first_and_limited_to_50(tmp_path):
    client = _client(tmp_path)
    for i in range(3):
        client.post("/api/note", json={"target_date": "2026-07-18", "sky": f"note {i}"})
    notes = client.get("/api/notes").json()
    assert [n["sky"] for n in notes] == ["note 2", "note 1", "note 0"]

# ---------------- track record / scorecard --------------------------------

def test_scorecard_endpoint_returns_records_and_summary(tmp_path):
    cfg = load_config()
    object.__setattr__(cfg, "db_path", tmp_path / "t.db")
    conn = hdb.connect(cfg.db_path)
    target = "2026-07-18"
    hdb.upsert(conn, "cli_reports", [{
        "station": cfg.station.icao, "date": target, "high_f": 84, "low_f": 60,
        "raw": "{}", "fetched_at": "2026-07-18T23:59:00+00:00"}])
    hdb.upsert(conn, "predictions", [{
        "station": cfg.station.icao, "target_date": target,
        "issued_at_utc": "2026-07-18T01:00:00+00:00", "kind": "dayahead",
        "point_f": 84.0, "quantiles": json.dumps({"0.5": 84.0}),
        "buckets": json.dumps({"83": 0.2, "84": 0.5, "85": 0.3}),
    }])
    client = TestClient(create_app(cfg))
    r = client.get("/api/scorecard")
    assert r.status_code == 200
    body = r.json()
    assert "records" in body and "summary" in body
    records = body["records"]
    assert len(records) == 1
    rec = records[0]
    assert rec["target_date"] == target
    assert rec["kind"] == "dayahead"
    assert rec["actual"] == 84
    assert rec["point"] == 84
    assert rec["bucket_hit"] is True
    assert "tail_label" in rec and rec["tail_label"].startswith("≥")
    assert "tail_hit" in rec and isinstance(rec["tail_hit"], bool)
    summary = body["summary"]
    assert set(summary.keys()) >= {"dayahead", "overall"}
    for key in ("dayahead", "overall"):
        assert set(summary[key].keys()) == {
            "n", "point_hit_rate", "within1_rate", "bucket_hit_rate",
            "tail_hit_rate", "mae"}
    assert summary["dayahead"]["n"] == 1
    assert summary["overall"]["n"] == 1
    assert summary["dayahead"]["tail_hit_rate"] == 1.0

def test_index_has_field_notes_anchor(tmp_path):
    r = _client(tmp_path).get("/")
    assert r.status_code == 200
    assert 'id="field-notes"' in r.text


def test_timing_and_bet_agree_on_the_pick(tmp_path, monkeypatch):
    """The timing card and the Decision Desk must read the SAME prediction.
    They used to diverge (timing showed the nowcast, desk showed day-ahead)."""
    import json as _j
    from highcast.fetchers import kalshi as _k
    from zoneinfo import ZoneInfo as _Z
    from datetime import datetime as _dt
    monkeypatch.setattr(_k, "fetch_ranges", lambda cfg, t, metric=None: None)
    cfg = load_config()
    object.__setattr__(cfg, "db_path", tmp_path / "t.db")
    conn = hdb.connect(cfg.db_path)
    today = _dt.now(_Z(cfg.station.tz)).date().isoformat()

    def _p(kind, issued, point):
        return {"station": cfg.station.icao, "target_date": today,
                "issued_at_utc": issued, "kind": kind, "point_f": point,
                "quantiles": _j.dumps({"0.5": point}),
                "buckets": _j.dumps({str(int(point)): 1.0}), "metric": "high"}
    hdb.upsert(conn, "predictions", [_p("dayahead", "2026-01-01T16:00:00+00:00", 79.0),
                                     _p("intraday", "2026-01-01T17:00:00+00:00", 85.0)])
    client = TestClient(create_app(cfg))
    timing = client.get("/api/timing?metric=high").json()
    bet = client.get("/api/bet?metric=high").json()
    pick = (timing.get("recommendation") or {}).get("model_pick")
    if pick and bet.get("kind"):
        assert pick["source"] == bet["kind"], (
            f"timing card reads {pick['source']} but the desk reads {bet['kind']}")


def test_no_route_opens_a_migrating_connection():
    """⚠️ THE §2.4 BUG, WHICH CAME BACK. `db.connect()` defaults to migrate=True
    and runs `executescript(SCHEMA)` + 8 migrations on EVERY call. The dashboard
    opens a connection per request against the same file the 5-minute wake cycle
    writes, and `data/serve.err` carries `sqlite3.OperationalError: database is
    locked` tracebacks raised INSIDE that schema call while merely reading.

    The original fix added `migrate=False` to `_conn()` — and MISSED two routes
    (`/api/timing` and `/api/bet`) that called `db.connect(c.db_path)` directly,
    both of which document themselves as read-only. They kept issuing DDL until
    2026-08-15.

    Every request-path connection now goes through `_reader()`. Exactly ONE bare
    `db.connect(` may remain: the deliberate migrating connection in
    `create_app`, which runs once at startup so the schema is present before any
    request is served.
    """
    import pathlib
    import re

    src = (pathlib.Path(__file__).resolve().parents[1]
           / "src" / "highcast" / "web" / "app.py").read_text()
    calls = [ln.strip() for ln in src.splitlines()
             if re.search(r"(?<!\w)db\.connect\(", ln)
             and not ln.strip().startswith(("#", "⚠️"))]
    bare = [c for c in calls if "migrate=False" not in c]
    assert len(bare) == 1, (
        f"expected exactly one migrating connect (the startup one), found "
        f"{len(bare)}: {bare}")
    assert bare[0].startswith("db.connect(cfg.db_path).close()"), (
        f"the one migrating connect must be the startup call, got: {bare[0]}")


def test_freshness_reports_ACTIVE_alerts_not_the_age_of_the_last_row(tmp_path):
    """⚠️ ALERTS IS EVENT-DRIVEN, SO ITS AGE IS NOT A HEALTH SIGNAL.

    KLAX went twelve days with no NWS advisory — entirely normal — and the
    dashboard chip read "ALERTS 292h 38m" in warning amber, implying a dead
    feed. `highcast.staleness` exempts the table outright for exactly this
    reason; the freshness row was the last place still treating silence as a
    fault.

    ⚠️ THIS TESTS THE API HALF ONLY. The chip's live-alert rendering is JS and
    is NOT covered here — see the note in the handoff. This at least guarantees
    the UI is handed the right fact.
    """
    import dataclasses
    from datetime import datetime, timedelta, timezone

    from highcast import db as hdb
    from highcast.config import load_config
    from highcast.web.app import _freshness

    cfg = load_config()
    icao = cfg.station.icao
    conn = hdb.connect(tmp_path / "t.db")
    now = datetime.now(timezone.utc)

    # 1. an OLD, expired alert must NOT count as active
    hdb.upsert(conn, "alerts", [{
        "station": icao, "event": "Heat Advisory",
        "onset_utc": (now - timedelta(days=12)).isoformat(),
        "expires_utc": (now - timedelta(days=12, hours=-8)).isoformat(),
        "fetched_at": (now - timedelta(days=12)).isoformat()}])
    f = _freshness(conn, icao, cfg)
    assert f["alerts_active"] == 0, "an expired advisory is not active"
    assert f["alerts_active_events"] == []
    assert f["alerts"] is not None, "the age is still reported, just not as health"

    # 2. a LIVE one must be named
    hdb.upsert(conn, "alerts", [{
        "station": icao, "event": "Excessive Heat Warning",
        "onset_utc": (now - timedelta(hours=1)).isoformat(),
        "expires_utc": (now + timedelta(hours=4)).isoformat(),
        "fetched_at": now.isoformat()}])
    f = _freshness(conn, icao, cfg)
    assert f["alerts_active"] == 1
    assert f["alerts_active_events"] == ["Excessive Heat Warning"]


def test_freshness_alerts_do_not_leak_across_stations(tmp_path):
    """Two stations share one database — KMIA's Heat Advisory must not show on
    the KLAX board. Same cross-station bug class as the `metric:date` cache."""
    from datetime import datetime, timedelta, timezone

    from highcast import db as hdb
    from highcast.config import load_config
    from highcast.web.app import _freshness

    cfg = load_config()
    conn = hdb.connect(tmp_path / "t.db")
    now = datetime.now(timezone.utc)
    hdb.upsert(conn, "alerts", [{
        "station": "KMIA", "event": "Heat Advisory",
        "onset_utc": (now - timedelta(hours=1)).isoformat(),
        "expires_utc": (now + timedelta(hours=4)).isoformat(),
        "fetched_at": now.isoformat()}])
    assert _freshness(conn, cfg.station.icao, cfg)["alerts_active"] == 0
    assert _freshness(conn, "KMIA", cfg)["alerts_active"] == 1


def test_api_today_is_station_aware(tmp_path):
    """⚠️ ADDED 2026-08-15. `/api/today` feeds the freshness row including the
    ALERTS chip, and it took NO `station` parameter — so it always answered for
    KLAX and a KMIA advisory could not be shown there at all. `/api/bet` and
    `/api/timing` already took one; this was the odd endpoint out.

    Asserting on `running_max` rather than the alert count, because two stations
    can legitimately BOTH have zero active alerts — that comparison would pass
    while the parameter was ignored. The running extremum cannot coincide.
    """
    from fastapi.testclient import TestClient

    from highcast.web.app import create_app
    from highcast.config import load_config

    cfg = load_config()
    app = create_app(cfg, config_path="config/klax.yaml")
    with TestClient(app) as client:
        a = client.get("/api/today?metric=high").json()
        b = client.get("/api/today?metric=high&station=KMIA").json()
    assert a["target_date"] and b["target_date"]
    # both payloads must carry the alert fields the chip reads
    for p in (a, b):
        assert "alerts_active" in p["freshness"]
        assert "alerts_active_events" in p["freshness"]
    # and the two stations must not be the same query
    assert a["running_max"] != b["running_max"] or a["kalshi_buckets"] != b["kalshi_buckets"], (
        "KLAX and KMIA returned identical payloads — the station parameter is "
        "being ignored")


def test_api_today_defaults_to_the_servers_own_station(tmp_path):
    """Every existing caller omits `station`. That path must stay byte-identical
    to before the parameter existed."""
    from fastapi.testclient import TestClient

    from highcast.web.app import create_app
    from highcast.config import load_config

    cfg = load_config()
    app = create_app(cfg, config_path="config/klax.yaml")
    with TestClient(app) as client:
        bare = client.get("/api/today?metric=high").json()
        named = client.get(f"/api/today?metric=high&station={cfg.station.icao}").json()
    assert bare["running_max"] == named["running_max"]
    assert bare["freshness"]["metar"] == named["freshness"]["metar"]
