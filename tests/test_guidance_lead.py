"""A precautionary freshness guard. ITS ORIGINAL JUSTIFICATION WAS WITHDRAWN.

Added 2026-07-31 believing the evening `dayahead_tomorrow` run saw two-day-old
guidance and extrapolated cold. On 2026-08-01 that was traced to a BAD QUERY,
not a bad feed: the reconstruction used `WHERE fetched_at <= issued_at`, but
`db.upsert` is INSERT OR REPLACE and the `forecasts` PK excludes `fetched_at`,
so every re-fetch rewrites it. Rows present at 17:04 looked absent because their
stamp had been overwritten to 23:55.

The cold bias itself is REAL (measured against settled CLI, independent of
`fetched_at`). Its cause is the all-NaN `morn_*` regime that `morning_obs_ready`
already detects -- split by that guard, KLAX HIGH day-ahead runs +0.31F (n=18,
MAE 1.35) with morning obs and -5.35F (n=13, MAE 5.35) without.

These tests therefore pin BEHAVIOUR ONLY -- that the function refuses lead-2 and
fails closed. They do not assert that the lead-2 regime occurs in production,
because that has never been demonstrated.
"""
import yaml

from highcast import db as hdb, gating
from highcast.config import load_config


def _cfg(tmp_path, src="config/klax.yaml"):
    raw = yaml.safe_load(open(src).read())
    raw["db_path"] = str(tmp_path / "t.db")
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(raw))
    return load_config(str(p))


def _fc(station, target, run_date, model="gfs", metric="high", val=80.0):
    return {"station": station, "target_date": target, "model": model,
            "run_date": run_date, "max_temp_f": val, "metric": metric,
            "fetched_at": f"{run_date}T12:00:00+00:00"}


def test_lead_1_guidance_is_ready(tmp_path):
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    hdb.upsert(conn, "forecasts", [_fc("KLAX", "2026-08-02", "2026-08-01")])
    assert gating.guidance_lead_ready(conn, cfg, "2026-08-02") is True


def test_lead_2_guidance_is_refused(tmp_path):
    """THE bug. Two-day-old guidance is a regime the model was not trained on."""
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    hdb.upsert(conn, "forecasts", [_fc("KLAX", "2026-08-02", "2026-07-31")])
    assert gating.guidance_lead_ready(conn, cfg, "2026-08-02") is False


def test_newest_run_wins_when_both_leads_exist(tmp_path):
    """A stale row alongside a fresh one must not veto — only the NEWEST matters."""
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    hdb.upsert(conn, "forecasts", [
        _fc("KLAX", "2026-08-02", "2026-07-31"),
        _fc("KLAX", "2026-08-02", "2026-08-01"),
    ])
    assert gating.guidance_lead_ready(conn, cfg, "2026-08-02") is True


def test_no_guidance_at_all_fails_closed(tmp_path):
    """Missing data and 'no' must be the same answer for an unattended caller."""
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    assert gating.guidance_lead_ready(conn, cfg, "2026-08-02") is False


def test_the_other_station_cannot_satisfy_this_one(tmp_path):
    """Both stations share one SQLite file — anything not keyed by station is a
    latent cross-station bug."""
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    hdb.upsert(conn, "forecasts", [_fc("KMIA", "2026-08-02", "2026-08-01")])
    assert gating.guidance_lead_ready(conn, cfg, "2026-08-02") is False
