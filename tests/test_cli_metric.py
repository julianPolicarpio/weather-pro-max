from datetime import date, timedelta

import yaml

from highcast import db as hdb
from highcast.cli import build_parser, cmd_predict
from highcast.config import load_config

CFG = load_config()
ICAO = CFG.station.icao


# --- parse-level: every metric-aware subcommand gains --metric --------------

def test_predict_metric_parses():
    assert build_parser().parse_args(["predict"]).metric == "high"
    assert build_parser().parse_args(["predict", "--metric", "low"]).metric == "low"


def test_backtest_metric_parses():
    assert build_parser().parse_args(["backtest"]).metric == "high"
    assert build_parser().parse_args(["backtest", "--metric", "low"]).metric == "low"


def test_log_metric_parses():
    assert build_parser().parse_args(["log"]).metric == "high"
    assert build_parser().parse_args(["log", "--metric", "low"]).metric == "low"


def test_verify_quantization_metric_parses():
    assert build_parser().parse_args(["verify-quantization"]).metric == "high"
    assert build_parser().parse_args(
        ["verify-quantization", "--metric", "low"]).metric == "low"


def test_metric_rejects_unknown_choice():
    import pytest
    with pytest.raises(SystemExit):
        build_parser().parse_args(["predict", "--metric", "dewpoint"])


# --- hermetic end-to-end: predict --metric low writes a metric='low' row ----

def _write_config(tmp_path, db_path):
    raw = yaml.safe_load(open("config/klax.yaml").read())
    raw["db_path"] = str(db_path)
    cfg_path = tmp_path / "klax-test.yaml"
    cfg_path.write_text(yaml.safe_dump(raw))
    return str(cfg_path)


def _seed_history(conn, n=70, start="2025-04-01"):
    d0 = date.fromisoformat(start)
    rows = []
    for i in range(n):
        d = d0 + timedelta(days=i)
        # seasonal-ish lows around 58F, highs around 78F
        low = 58 + (i % 7) - 3
        high = 78 + (i % 5) - 2
        rows.append({"station": ICAO, "date": d.isoformat(),
                     "high_f": high, "low_f": low, "raw": "{}", "fetched_at": "x"})
    hdb.upsert(conn, "cli_reports", rows)
    # target date is the day AFTER the seeded history
    return (d0 + timedelta(days=n)).isoformat()


def test_predict_metric_low_writes_low_row(tmp_path):
    db_path = tmp_path / "t.db"
    conn = hdb.connect(db_path)
    target = _seed_history(conn)
    conn.close()
    cfg_path = _write_config(tmp_path, db_path)

    import argparse
    cmd_predict(argparse.Namespace(
        config=cfg_path, date=target, live=False, metric="low"))

    conn = hdb.connect(db_path)
    rows = list(conn.execute(
        "SELECT metric, kind, point_f FROM predictions WHERE target_date=?",
        (target,)))
    assert len(rows) == 1
    assert rows[0]["metric"] == "low"
    assert rows[0]["kind"] == "dayahead"
    assert 40.0 <= rows[0]["point_f"] <= 80.0     # plausible low


def test_predict_metric_high_default_writes_high_row(tmp_path):
    # Guard: the default (no --metric) still stores a metric='high' row, and a
    # namespace with NO metric attribute (the auto_cycle/_ns path) does not crash.
    db_path = tmp_path / "t.db"
    conn = hdb.connect(db_path)
    target = _seed_history(conn)
    conn.close()
    cfg_path = _write_config(tmp_path, db_path)

    import argparse
    # deliberately omit `metric` to mimic auto_cycle's _ns(config=,date=,live=)
    cmd_predict(argparse.Namespace(config=cfg_path, date=target, live=False))

    conn = hdb.connect(db_path)
    rows = list(conn.execute(
        "SELECT metric FROM predictions WHERE target_date=?", (target,)))
    assert len(rows) == 1
    assert rows[0]["metric"] == "high"
