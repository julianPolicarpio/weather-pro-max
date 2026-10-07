import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

MAX_BANKROLL = 1e9  # sanity cap; a bankroll above this is almost certainly a typo

# Matches the single `bankroll:` KEY line (indent + key + numeric value +
# optional trailing comment). The word "bankroll" also appears inside comments
# on other lines, but only the real key sits at line-start followed by ':'.
_BANKROLL_LINE = re.compile(
    r"^([ \t]*bankroll[ \t]*:[ \t]*)([0-9]*\.?[0-9]+)([ \t]*)(#.*)?$",
    re.MULTILINE,
)


@dataclass(frozen=True)
class StationConfig:
    key: str
    name: str
    icao: str
    iem_id: str
    iem_network: str
    wfo: str
    lat: float
    lon: float
    tz: str
    std_utc_offset: int
    nws_zones: list[str] = field(default_factory=list)
    kalshi_series: str = ""
    kalshi_low_series: str = ""


@dataclass(frozen=True)
class StakingConfig:
    """Decision-support staking policy. The policy sets the rule + amount from
    these knobs; the human places the trade. Deliberately conservative defaults
    (quarter-Kelly, hard 5% cap) — the point is to CURB emotional overbetting,
    not chase it. `bankroll` defaults to 0 so the tool never assumes the user's
    money: it must be set explicitly before any dollar amount is produced."""
    bankroll: float = 0.0          # user sets this; 0 => tool sizes to $0 and warns
    kelly_fraction: float = 0.25   # fraction of full-Kelly (0.25 = quarter-Kelly)
    max_stake_frac: float = 0.05   # hard cap: never risk >5% of bankroll per bucket
    min_edge: float = 0.05         # need >=5 percentage-points edge to place at all
    # CEILING on the edge, mirroring `gating.MAX_EDGE` which has guarded the
    # ORDER BOT since 2026-08-02. The desk had none, so it recommended bets the
    # bot in the same repo refuses -- one project, two answers.
    #
    # A very large edge is usually OUR error, not free money: corr(edge, won)
    # = -0.571 over 48 settled bets, and of the 18 PLACE bets the 6 with edge
    # > 0.40 went 1-5. 0.40 is REUSED from the bot's own derivation, NOT fitted
    # to those 18 -- the desk record makes 0.25 look better, which is exactly
    # the swept-threshold error QUEUE item 1 forbids.
    #
    # ⚠️ The P&L gain is NOT significant at n=18 (D net/bet +0.049,
    # CI [-0.095,+0.195]). What carries it is the correlation on the larger
    # sample. 0 disables the ceiling. See `tests/test_max_edge_desk.py`.
    max_edge: float = 0.40
    max_spread: float = 0.10       # skip if bid/ask spread > this (illiquid — slippage eats edge)
    min_confidence: float = 0.80   # model's top 2F bucket must be >= this to act
    # OFF BY DEFAULT, and it should stay off until there is price history to
    # justify it. When true, `staking.evaluate` refuses to PLACE below the
    # market's MEASURED `gating.CONFIDENCE_FLOOR`. Measured on the 14 settled
    # PLACE bets — the only price-aware evidence that exists — gating at 0.80
    # would have deleted the group that MADE money (hit 33%, ask 0.31,
    # +0.009/contract) and kept the group that lost it (hit 60%, ask 0.60,
    # -0.016/contract): the ask rises to meet the confidence and eats it. See
    # `tests/test_confidence_gate.py` for the full table and the countervailing
    # 574-day hit-rate evidence.
    confidence_gate: bool = False
    stability_readings: int = 3    # consecutive reads for the (informational) "stable" flag
    # Must sit AFTER the metric's gating.MIN_DECISION_HOUR — see config/klax.yaml.
    personal_cutoff: str = "09:00"      # HIGH: last actionable wall-time (display_tz) = LA 18:00
    low_personal_cutoff: str = "04:00"  # LOW: = LA 13:00, after the 18Z 6h-min group lands
    currency: str = "USD"


@dataclass(frozen=True)
class CalibrationConfig:
    """Regime-conditional seam calibration. `off` = global PIT only (default,
    live behavior unchanged). `width` widens the distribution at the marine-layer
    seam; `center_width` also shifts its center. Default stays `off` until the
    walk-forward backtest picks a winner."""
    mode: str = "off"          # off | width | center_width


@dataclass(frozen=True)
class ModelConfig:
    """Day-ahead model structure. `hybrid_anchor` blends a residual(MOS)-target
    model into the hot tail, where a gradient-boosted tree cannot extrapolate
    past its training targets. Backtested 2026-07-25 over 1,664 walk-forward
    days: overall dMAE -0.021 [95% CI -0.044, -0.0004], days below 80F
    unchanged (1.48), true hot days >=88F MAE 6.99 -> 4.68. Set false to run
    the plain direct-target model.

    `intraday_residual_low` makes the LOW nowcast regress the UNDERCUT (the
    remaining move below the observed running minimum) instead of the
    temperature. Measured 2026-07-26 at the LOW decision hour, N=1333 matched
    days: MAE 0.806 -> 0.765, CRPS 0.561 -> 0.526, dMAE -0.041
    [95% CI -0.064, -0.019]. The HIGH equivalent is OFF because it has not been
    measured — the HIGH nowcast is not what the desk reads in-window anyway."""
    hybrid_anchor: bool = True
    # Per-metric, because a metric-specific win must not be generalised on faith
    # (that mistake was made twice in this project — the desk-model preference
    # and the day-ahead hybrid both had to be walked back for the LOW).
    intraday_residual_low: bool = True
    intraday_residual_high: bool = False
    # TRAINING-HOUR BANDWIDTH for the nowcast. 0 = train on the prediction hour
    # alone (the long-standing default). k = train on {h-k .. h+k}, wrapping the
    # clock, so the hour still gates the regime but the fit is not starved.
    #
    # This is bandwidth selection: k=0 is zero bandwidth, k=24 is infinite. Only
    # the endpoints had ever been measured, and per-hour beat full pooling -- but
    # the optimum in local regression is essentially never at an endpoint.
    # Swept on KMIA 2026-07-29 (`scripts/bandwidth_eval.py`, holdout 2025-01-01,
    # bucket hit PARITY-AVERAGED), versus k=0:
    #
    #   HIGH  k=+/-1  MAE 1.070 -> 1.052   hit 55.1% -> 55.7%   churn 1.81 -> 1.73
    #         k=+/-2  MAE        1.052     hit        55.1%     churn        1.68
    #         k=all   MAE 1.179 (worse)    hit 61.2% (-2.6)     churn 2.03 (worse)
    #   LOW   k=+/-1  dMAE -0.014   dHIT +0.2 pts   churn +0.01 (flat)
    #         k=+/-3  dMAE -0.016   dHIT +0.3 pts   churn +0.01
    #         k=all   dMAE +0.028   dHIT -2.6 pts   churn +0.14 (worse)
    #
    # k=1 DOMINATES k=0 for the HIGH -- better on all three axes with no
    # trade-off, the only unambiguous win found. The LOW gains accuracy only.
    # Full pooling loses clearly in both: it has to represent "2am approaching a
    # minimum" and "6pm long past one" with shared tree structure, and that
    # smoothing destroys the dawn regime (-12.5 pts at the LOW's hour 1).
    #
    # DEFAULT STAYS 0 -- but the REASON changed on 2026-08-02. It was the
    # live-testing freeze ("changing KLAX's model mid-run would void the
    # scorecard"); that freeze is lifted on all four markets
    # (an internal design note).
    # What remains is that the sweep above is a KMIA measurement and k=1 has
    # never been measured on KLAX. Opt in per station in the config file, or
    # measure KLAX and move the default -- the objection is now evidence, not
    # permission.
    intraday_hour_bandwidth: int = 0
    # PER-METRIC OVERRIDES. None = fall back to `intraday_hour_bandwidth` above,
    # so KMIA's station-wide `intraday_hour_bandwidth: 1` is unchanged.
    #
    # WHY THIS HAD TO BECOME PER-METRIC (measured 2026-08-02,
    # `scripts/klax_bandwidth_eval.py`, nested walk-forward, 15 folds, n=1306/1307
    # days, positive control passed at 0.30F injection):
    #
    #   KLAX LOW   dMAE +0.000 [-0.009,+0.008]  dHIT +0.5 [-0.6,+1.6]
    #              churn 1.279 -> 1.248, -0.031 [-0.053,-0.010]   -> SHIP
    #   KLAX HIGH  dMAE -0.006 [-0.022,+0.008]  dHIT -0.7 [-1.9,+0.4]
    #              churn -0.012 [-0.037,+0.013]                   -> NEUTRAL
    #
    # The gate (better-or-neutral on MAE, HIT and churn) passed for the LOW and
    # not the HIGH. A single station-wide int cannot express that: shipping the
    # LOW's result would have dragged the HIGH along, and the HIGH's only
    # non-flat signal is a -0.7 pt bucket-hit point estimate -- not significant,
    # but leaning the wrong way, and there is no reason to buy it.
    #
    # ⚠️ THE ACCURACY CASE IS EMPTY, deliberately recorded here so nobody later
    # reads k=1 as an accuracy win. The eval reports its own resolving floor at
    # this n as 0.008F and the LOW's observed |dMAE| was 0.000F -- BELOW what the
    # comparison can distinguish. The measured gain is CHURN ONLY: 2.4% fewer
    # modal-bucket flips across h5/h6/h7, i.e. the preview at h5 agrees with the
    # commit at h6 more often. That is stability, not skill.
    intraday_hour_bandwidth_low: int | None = None
    intraday_hour_bandwidth_high: int | None = None
    # Refuse a commit that lands so late the extremum has largely already formed
    # (`appbet` RULE 3b). Correct behaviour, and OFF BY DEFAULT ANYWAY: switching
    # it on narrows the commit window (KLAX's LOW goes from hours 4-23 to 4-5),
    # so on a late-wake day the app produces NO bet where it used to produce a
    # late one. That is a change to WHEN A BET COMMITS, which the live-testing
    # freeze forbade outright; the freeze was lifted 2026-08-02 on all four
    # markets (see the full-unfreeze decision record), so this is now a live
    # choice rather than a blocked one. It stays OFF by default because the
    # trade-off is real and unmeasured -- fewer late bets is not obviously
    # better than some late bets -- not because permission is missing.
    late_commit_guard: bool = False

    def intraday_residual_for(self, metric: str) -> bool:
        """Whether `metric`'s nowcast regresses the undercut."""
        return bool(self.intraday_residual_low if metric == "low"
                    else self.intraday_residual_high)

    def intraday_bandwidth_for(self, metric: str) -> int:
        """`metric`'s matched-hour training bandwidth k, override or default.

        ASK THIS, NEVER READ `intraday_hour_bandwidth` DIRECTLY. The two metrics
        at one station measured differently (see the constant's comment), so a
        direct read silently applies one metric's answer to the other -- the same
        class of bug as `orderbot`'s duplicated CONFIDENCE_FLOOR and the
        parity-averaged calibration that inverted a conclusion. Pinned by
        `tests/test_config.py`.
        """
        override = (self.intraday_hour_bandwidth_low if metric == "low"
                    else self.intraday_hour_bandwidth_high)
        k = self.intraday_hour_bandwidth if override is None else override
        return int(k or 0)


@dataclass(frozen=True)
class AppConfig:
    station: StationConfig
    display_tz: str
    synoptic_token: str
    db_path: Path
    # Discord webhook for bet alerts. Empty = notifications off, same pattern as
    # `synoptic_token`: an unconfigured optional integration is a silent no-op,
    # never an error.
    discord_webhook_url: str = ""
    # DEAD MAN'S SWITCH. URL of a hosted check that alerts when pings STOP.
    #
    # The missed-commit Discord alert cannot cover a power cut: the outage that
    # stops the wake cycle stops the alert too. Nothing running on this machine
    # can report that this machine is off, so the alerting has to live
    # somewhere that survives the cut and the app only pings it. See
    staking: StakingConfig = field(default_factory=StakingConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    # Does the wake cycle run the order bot at all?
    #
    # A THIRD, OUTERMOST OFF-SWITCH, and deliberately not a replacement for the
    # other two. Arming still needs `HIGHCAST_TRADING` set to its exact literal
    # AND an unexpired `HIGHCAST_TRADING_UNTIL`; this decides whether
    # `botrunner.run_once` is invoked in the first place. The three are
    # independent: with this False the bot cannot be armed by accident, because
    # it is never asked.
    #
    # DEFAULT TRUE preserves the shipped behaviour for anything that constructs
    # an AppConfig directly. Both station yamls set it False as of 2026-08-02 to
    # focus on the paper demo account -- see the comment there.
    #
    # ⚠️ TURNING THIS OFF STOPS `bot_orders` ACCRUING. That table is one input to
    # the n≈60 edge re-test (QUEUE.md item 1). It is an acceptable loss because
    # `signals` (~250 rows/day, all four markets) and `app_bets` keep accruing
    # from the same wake cycle and carry strictly more information -- the demo
    # account replays THOSE, not `bot_orders`. If the re-test ever needs the
    # bot's own decision log specifically, turn this back on first.
    orderbot_enabled: bool = True


def load_config(path: str | Path = "config/klax.yaml") -> AppConfig:
    raw = yaml.safe_load(Path(path).read_text())
    return AppConfig(
        station=StationConfig(**raw["station"]),
        display_tz=raw["display_tz"],
        synoptic_token=raw.get("synoptic_token") or "",
        db_path=Path(raw["db_path"]),
        discord_webhook_url=raw.get("discord_webhook_url") or "",
        # CONFIG WINS when set; the env var is the convenience path so one
        # export covers BOTH stations without editing two yamls. A blank or
        # whitespace-only variable reads as OFF -- `export VAR=` is how people
        # try to switch something off, and it must not look configured.
        staking=StakingConfig(**(raw.get("staking") or {})),
        calibration=CalibrationConfig(**(raw.get("calibration") or {})),
        model=ModelConfig(**(raw.get("model") or {})),
        # Absent from the yaml = True, matching the dataclass default, so an
        # older config file behaves exactly as it did before this flag existed.
        orderbot_enabled=bool(raw.get("orderbot_enabled", True)),
    )


def set_bankroll(path: str | Path, value: float) -> float:
    """Persist a new staking.bankroll to the yaml at `path`, preserving every
    comment and other line (Option A: the yaml stays the single source of
    truth). Validates the value is finite, >= 0, and <= MAX_BANKROLL. Writes
    atomically (temp file + os.replace) and re-parses the result BEFORE swapping
    it in, so a crash or a bad edit never corrupts the live config. Returns the
    persisted bankroll as a float; raises ValueError on a bad value or if the
    file has no unambiguous `bankroll:` line."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"bankroll must be a number, got {value!r}")
    if not math.isfinite(v):
        raise ValueError("bankroll must be a finite number")
    if v < 0:
        raise ValueError("bankroll must be >= 0")
    if v > MAX_BANKROLL:
        raise ValueError(f"bankroll must be <= {MAX_BANKROLL:g}")

    path = Path(path)
    text = path.read_text()
    matches = _BANKROLL_LINE.findall(text)
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one 'bankroll:' line in {path}, found {len(matches)}")

    def _repl(m: "re.Match[str]") -> str:
        pre, _old, gap, comment = m.group(1), m.group(2), m.group(3), m.group(4)
        return f"{pre}{v!r}{gap}{comment or ''}"

    new_text = _BANKROLL_LINE.sub(_repl, text, count=1)

    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(new_text)
    try:
        parsed = load_config(tmp)
        if not math.isclose(parsed.staking.bankroll, v, rel_tol=0, abs_tol=1e-9):
            raise ValueError("post-write verification failed; config unchanged")
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, path)
    return v
