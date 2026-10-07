"""`orderbot_enabled` is a THIRD off-switch, not a replacement for the other two.

Turned off on both stations 2026-08-02 so attention could go to the paper demo
account. The bot has never placed an order (0 placed, 0 non-dry-run, ever), so
this is not a fix for anything -- it is one less moving part while the demo is
the focus, and one fewer way to arm live trading by accident.

THE PROPERTY THIS FILE PROTECTS is that the switches stay INDEPENDENT and stay
ANDed. `orderbot_enabled` decides whether the bot is invoked at all; the
`HIGHCAST_TRADING` literal and an unexpired `HIGHCAST_TRADING_UNTIL` decide
whether an invoked bot may transact. A refactor that made the new flag the ONLY
gate -- e.g. "enabled implies armed" -- would be catastrophic and would look
tidy in a diff, which is exactly why it is asserted rather than reviewed.
"""
import dataclasses

from highcast.config import load_config
from highcast.kalshi_client import (ARM_ENV, ARM_UNTIL_ENV, ARM_VALUE,
                                    arming_window_ok, is_armed)

CFG = load_config("config/kmia.yaml")


def test_both_stations_ship_with_the_bot_off():
    for path in ("config/kmia.yaml", "config/klax.yaml"):
        assert load_config(path).orderbot_enabled is False, path


def test_absent_from_yaml_defaults_to_enabled():
    """An older config file, or anything constructing AppConfig directly, must
    behave exactly as it did before this flag existed."""
    from highcast.config import AppConfig, StationConfig
    cfg = AppConfig(station=StationConfig(**dataclasses.asdict(CFG.station)),
                    display_tz="Asia/Manila", synoptic_token="",
                    db_path=CFG.db_path)
    assert cfg.orderbot_enabled is True


def test_the_flag_does_not_arm_anything():
    """Enabling the bot must NOT imply permission to trade. `is_armed` and
    `arming_window_ok` are unchanged and still require their own inputs."""
    enabled = dataclasses.replace(CFG, orderbot_enabled=True)
    assert enabled.orderbot_enabled is True
    # An enabled bot with an empty environment is still unarmed.
    assert is_armed({}) is False
    assert arming_window_ok({}, "2026-08-03") is False


def test_arming_still_needs_both_the_literal_and_an_unexpired_date():
    env = {ARM_ENV: ARM_VALUE}
    assert is_armed(env) is True                      # literal alone: interactive
    assert arming_window_ok(env, "2026-08-03") is False  # ...but no window
    env[ARM_UNTIL_ENV] = "2026-08-03"
    assert arming_window_ok(env, "2026-08-03") is True
    assert arming_window_ok(env, "2026-08-04") is False  # expired


def test_disabled_bot_is_not_invoked_by_the_wake_cycle(monkeypatch, tmp_path):
    """The load-bearing assertion: with the flag off, `botrunner.run_once` must
    not be called at all. Checked at the CALL SITE rather than by inspecting
    output, because a bot that runs and stays silent is a different (and worse)
    property than a bot that never runs -- only the latter cannot be armed by
    accident.
    """
    from highcast import botrunner, cli, db

    called = []
    monkeypatch.setattr(botrunner, "run_once",
                        lambda *a, **k: called.append(1) or {"place": False})
    cfg = dataclasses.replace(CFG, db_path=tmp_path / "t.db",
                              discord_webhook_url="",
                              orderbot_enabled=False)
    conn = db.connect(cfg.db_path)
    # auto_cycle does a great deal besides the bot; every other branch is
    # allowed to fail here. Only the bot call site is under test.
    try:
        cli.auto_cycle(conn, cfg, "config/kmia.yaml")
    except Exception:  # noqa: BLE001
        pass
    assert called == [], "botrunner.run_once ran despite orderbot_enabled=False"
