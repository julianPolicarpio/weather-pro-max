import ast
import pathlib

from highcast.config import ModelConfig, load_config


def test_load_klax_config():
    cfg = load_config("config/klax.yaml")
    assert cfg.station.icao == "KLAX"
    assert cfg.station.iem_id == "LAX"
    assert cfg.station.std_utc_offset == -8
    assert cfg.station.lat == 33.9425
    assert cfg.display_tz == "Asia/Manila"
    assert str(cfg.db_path).endswith("highcast.db")
    # CAZ366 = current KLAX forecast zone (LOX renumbered zones in 2023);
    # CAZ041 = the same area's pre-2023 code. Both are needed so the VTEC
    # archive fetch covers years on either side of the renumbering.
    assert "CAZ366" in cfg.station.nws_zones
    assert "CAZ041" in cfg.station.nws_zones


# ── matched-hour bandwidth is PER-METRIC ────────────────────────────────────
# Measured 2026-08-02 (`scripts/klax_bandwidth_eval.py`): k=1 passed the
# better-or-neutral-on-all-three gate for the KLAX LOW and did NOT for the KLAX
# HIGH (dHIT -0.7 pts, leaning the wrong way). A single station-wide int cannot
# express that, so shipping the LOW's result would have silently dragged the
# HIGH along with it.


def test_klax_ships_k1_on_the_low_only():
    cfg = load_config("config/klax.yaml")
    assert cfg.model.intraday_bandwidth_for("low") == 1
    assert cfg.model.intraday_bandwidth_for("high") == 0


def test_kmia_station_wide_setting_still_applies_to_both():
    """KMIA sets the station-wide int and no overrides; it must not regress."""
    cfg = load_config("config/kmia.yaml")
    assert cfg.model.intraday_bandwidth_for("low") == 1
    assert cfg.model.intraday_bandwidth_for("high") == 1


def test_override_falls_back_to_the_station_default():
    m = ModelConfig(intraday_hour_bandwidth=2)
    assert m.intraday_bandwidth_for("low") == 2
    assert m.intraday_bandwidth_for("high") == 2
    # An override of 0 must WIN over a non-zero default -- `or 0` style fallback
    # logic would treat it as "unset" and silently apply the default instead.
    m = ModelConfig(intraday_hour_bandwidth=2, intraday_hour_bandwidth_high=0)
    assert m.intraday_bandwidth_for("high") == 0
    assert m.intraday_bandwidth_for("low") == 2


def test_no_shipped_code_reads_the_bandwidth_field_directly():
    """The accessor is the only sanctioned reader.

    A direct `cfg.model.intraday_hour_bandwidth` read applies one metric's
    measured answer to the other. This project has been bitten by exactly this
    shape three times -- `orderbot.CONFIDENCE_FLOOR` shadowing `gating`'s,
    `fc_anchor` re-entering the blend it was derived from, and a settlement
    predicate that guarded fetching but not grading. This test fails if a direct
    read reappears anywhere in `src/`, rather than trusting review to catch it.
    """
    root = pathlib.Path(__file__).resolve().parent.parent
    # `scripts/` IS IN SCOPE, and that is the point rather than thoroughness for
    # its own sake: `scripts/hour_curve.py` produces `gating.HOUR_ACCURACY`,
    # which sets `MIN_DECISION_HOUR` -- the live commit hours. A script reading
    # the station-wide field re-derives those hours from a model nobody runs,
    # which that file's own comment already warns about. 13 scripts had this
    # exact defect the day the field became per-metric.
    EXEMPT = {
        "config.py",              # defines the field
        # Sweeps k=0 vs k=1 as its whole experiment, so it must set the arms
        # itself rather than ask what is configured. The only legitimate
        # direct reader.
        "klax_bandwidth_eval.py",
    }
    offenders = []
    paths = [*(root / "src" / "highcast").rglob("*.py"),
             *(root / "scripts").glob("*.py")]
    for path in paths:
        if path.name in EXEMPT:
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if (isinstance(node, ast.Attribute)
                    and node.attr == "intraday_hour_bandwidth"):
                offenders.append(f"{path.name}:{node.lineno}")
            # getattr(model, "intraday_hour_bandwidth", ...) evades the above
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "getattr"
                    and len(node.args) >= 2
                    and isinstance(node.args[1], ast.Constant)
                    and node.args[1].value == "intraday_hour_bandwidth"):
                offenders.append(f"{path.name}:{node.lineno} (getattr)")
    assert not offenders, (
        "read intraday_hour_bandwidth directly instead of via "
        f"intraday_bandwidth_for(metric): {offenders}")
