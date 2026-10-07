import argparse
import csv
import dataclasses
from datetime import datetime, timezone

from highcast import db, signals
from highcast.cli import EXPORT_HEADERS, cmd_export_bets
from highcast.config import StakingConfig, load_config
from highcast.metric import get_metric

CFG = load_config()
SPEC = get_metric(CFG, "high")
LOW_SPEC = get_metric(CFG, "low")
MARKET = [{"label": "82-83", "lo": 82, "hi": 83, "p_model": 0.40, "market_price": 0.30},
          {"label": "84-85", "lo": 84, "hi": 85, "p_model": 0.30, "market_price": 0.25}]
LOW_MARKET = [{"label": "68-69", "lo": 68, "hi": 69, "p_model": 0.6,
               "market_price": 0.74}]


def _cfg(tmp_path):
    return dataclasses.replace(CFG, db_path=tmp_path / "t.db",
                               staking=StakingConfig(bankroll=8.0))


def _ns(cfg_path, **kw):
    base = dict(config=cfg_path, metric="both", kind="manual", out=None)
    base.update(kw)
    return argparse.Namespace(**base)


def _write_cfg(tmp_path):
    """cmd_export_bets loads config from a path, so point it at a copy whose
    db_path is the temp DB."""
    import yaml
    src = yaml.safe_load(open("config/klax.yaml"))
    src["db_path"] = str(tmp_path / "t.db")
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(src))
    return str(p)


def _rows(capsys):
    out = capsys.readouterr().out
    return list(csv.reader(out.strip().splitlines()))


def test_headers_are_exactly_the_tracked_columns(tmp_path, capsys):
    cfg_path = _write_cfg(tmp_path)
    db.connect(tmp_path / "t.db")
    cmd_export_bets(_ns(cfg_path))
    rows = _rows(capsys)
    assert rows[0] == EXPORT_HEADERS
    # the first seven are Julian's own sheet, in his order -- do not reorder
    assert rows[0][:7] == [
        "prediction(highest temp/lowest temp)",
        "date(date placed)",
        "time(time placed)",
        "bucket(what bucket did i bet on)",
        "bet amount(how much did i bet)",
        "bet winning(the amount i will win if the bet is correct)",
        "did the prediction hit Y/N",
    ]


def test_unsettled_bet_leaves_hit_blank_not_N(tmp_path, capsys):
    """A bet that has not settled yet must not read as a loss."""
    cfg = _cfg(tmp_path)
    conn = db.connect(cfg.db_path)
    signals.record_manual(conn, cfg, SPEC, "2026-07-24", "82-83", 3, 0.30,
                          MARKET, datetime(2026, 7, 24, 13, tzinfo=timezone.utc))
    cmd_export_bets(_ns(_write_cfg(tmp_path)))
    rows = _rows(capsys)
    assert len(rows) == 2
    assert rows[1][6] == ""            # hit column blank, not "N"
    assert rows[1][8] == ""            # settled actual blank


def test_amounts_and_metric_label(tmp_path, capsys):
    cfg = _cfg(tmp_path)
    conn = db.connect(cfg.db_path)
    signals.record_manual(conn, cfg, LOW_SPEC, "2026-07-26", "68-69", 2, 0.74,
                          LOW_MARKET, datetime(2026, 7, 26, 13, tzinfo=timezone.utc))
    cmd_export_bets(_ns(_write_cfg(tmp_path)))
    r = _rows(capsys)[1]
    assert r[0] == "lowest temp"
    assert float(r[4]) == 1.48         # 2 x 0.74 cost
    assert float(r[5]) == 0.52         # 2 x (1 - 0.74) profit if correct
    assert r[7] == "2026-07-26"        # LA climate day kept separate


def test_placed_time_is_rendered_in_manila(tmp_path, capsys):
    """13:00Z == 21:00 Manila same day. The sheet records when Julian placed it
    from his chair, not UTC."""
    cfg = _cfg(tmp_path)
    conn = db.connect(cfg.db_path)
    signals.record_manual(conn, cfg, LOW_SPEC, "2026-07-26", "68-69", 1, 0.74,
                          LOW_MARKET, datetime(2026, 7, 26, 13, tzinfo=timezone.utc))
    cmd_export_bets(_ns(_write_cfg(tmp_path)))
    r = _rows(capsys)[1]
    assert r[1] == "2026-07-26"
    assert r[2] == "21:00"


def test_settled_bet_reports_Y_or_N(tmp_path, capsys):
    cfg = _cfg(tmp_path)
    conn = db.connect(cfg.db_path)
    signals.record_manual(conn, cfg, SPEC, "2026-07-24", "82-83", 3, 0.30,
                          MARKET, datetime(2026, 7, 24, 13, tzinfo=timezone.utc))
    conn.execute("INSERT OR REPLACE INTO cli_reports(station,date,high_f,low_f,raw,"
                 "fetched_at) VALUES(?,?,?,?,?,?)",
                 (cfg.station.icao, "2026-07-24", 83, 68, "{}", "x"))
    conn.commit()
    signals.settle_signals(conn, cfg, datetime(2026, 7, 25, 18, tzinfo=timezone.utc))
    cmd_export_bets(_ns(_write_cfg(tmp_path)))
    r = _rows(capsys)[1]
    assert r[6] == "Y"                 # 83 lands inside 82-83
    assert r[8] == "83"


def test_metric_filter(tmp_path, capsys):
    cfg = _cfg(tmp_path)
    conn = db.connect(cfg.db_path)
    signals.record_manual(conn, cfg, SPEC, "2026-07-24", "82-83", 1, 0.30,
                          MARKET, datetime(2026, 7, 24, 13, tzinfo=timezone.utc))
    signals.record_manual(conn, cfg, LOW_SPEC, "2026-07-26", "68-69", 1, 0.74,
                          LOW_MARKET, datetime(2026, 7, 26, 13, tzinfo=timezone.utc))
    cmd_export_bets(_ns(_write_cfg(tmp_path), metric="low"))
    rows = _rows(capsys)
    assert len(rows) == 2
    assert rows[1][0] == "lowest temp"
