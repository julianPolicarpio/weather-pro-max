"""DOES THE QC ACTUALLY REPRODUCE THE SETTLED VALUE? Measured against CLI.

The raw 1-minute extreme matches the CLI high only ~30% of the time and runs
~+1F hot (QUEUE item 13). `highcast.asos1min` proposes two fixes -- despike, then
5-minute rolling mean. This scores them SEPARATELY and TOGETHER, because a
combination that works tells you nothing about which half did the work, and the
half that does nothing is code nobody should keep.

⚠️ THE BAR, stated before running: the 5-minute mean is a real hypothesis about
mechanism (the official extreme comes from 5-minute observations), so it should
move highs substantially. Despiking should barely touch the exact-match rate --
spikes are ~7.5% of days -- but should collapse the long tail (+36, +46, +56F).
If despiking moves exact-match a lot, I have mis-diagnosed something. If the
5-minute mean does NOT move highs, the mechanism story is wrong and the whole
approach should be dropped rather than tuned.

Same day filter as the raw measurement (>=1000 covered minutes) so the numbers
are comparable to the table in QUEUE item 13.

Usage: PYTHONPATH=src .venv/bin/python scripts/asos1min_qc.py
"""
import collections
from datetime import date

import numpy as np

from highcast import asos1min
from highcast.config import load_config
from highcast.db import connect
from highcast.timeutil import round_half_up

VARIANTS = (
    ("raw", False, False),
    ("despike only", True, False),
    ("5-min mean only", False, True),
    ("despike + 5-min mean", True, True),
)


def score(pred: dict[str, float], cli: dict[str, int]) -> tuple[int, dict, float, int]:
    gaps = collections.Counter()
    for d, v in pred.items():
        if d in cli:
            gaps[round_half_up(v) - cli[d]] += 1
    n = sum(gaps.values())
    if not n:
        return 0, gaps, float("nan"), 0
    bias = sum(g * c for g, c in gaps.items()) / n
    tail = sum(c for g, c in gaps.items() if abs(g) >= 3)
    return n, gaps, bias, tail


print(__doc__.split("Usage:")[0])
for cfgp in ("config/klax.yaml", "config/kmia.yaml"):
    cfg = load_config(cfgp)
    icao = cfg.station.icao
    conn = connect(cfg.db_path, migrate=False)
    valid, temps = asos1min.load_station(conn, icao)
    print(f"\n{'='*74}\n{icao}: {len(valid):,} raw 1-minute rows\n{'='*74}")

    cli_rows = list(conn.execute(
        """SELECT date, high_f, low_f FROM cli_reports WHERE station=?
           ORDER BY date""", (icao,)))
    days = [date.fromisoformat(r["date"]) for r in cli_rows]
    cli_hi = {r["date"]: r["high_f"] for r in cli_rows if r["high_f"] is not None}
    cli_lo = {r["date"]: r["low_f"] for r in cli_rows if r["low_f"] is not None}

    for metric, extremum, cli in (("HIGH", "max", cli_hi), ("LOW", "min", cli_lo)):
        print(f"\n  --- {icao} {metric} ---")
        print(f"    {'variant':<22}{'n':>6}{'exact':>8}{'within 1':>10}"
              f"{'mean bias':>11}{'|gap|>=3':>10}")
        for name, dsp, mean in VARIANTS:
            grid, vals = asos1min.qc_series(valid, temps, do_despike=dsp,
                                            do_mean=mean)
            pred = asos1min.daily_extremes(cfg, grid, vals, days, extremum)
            n, gaps, bias, tail = score(pred, cli)
            if not n:
                print(f"    {name:<22}{'—':>6}"); continue
            exact = gaps.get(0, 0) / n
            near = sum(c for g, c in gaps.items() if abs(g) <= 1) / n
            print(f"    {name:<22}{n:>6}{exact:>8.1%}{near:>10.1%}"
                  f"{bias:>+11.2f}{tail/n:>10.1%}")
        # the winner's full distribution, so nothing hides behind a summary stat
        grid, vals = asos1min.qc_series(valid, temps)
        pred = asos1min.daily_extremes(cfg, grid, vals, days, extremum)
        n, gaps, bias, tail = score(pred, cli)
        if n:
            print(f"    distribution, despike + 5-min mean (n={n}):")
            for g in sorted(gaps):
                if gaps[g] / n >= 0.002:
                    print(f"       {g:+d} F : {gaps[g]:5d} ({gaps[g]/n:5.1%})"
                          f" {'#' * int(40 * gaps[g] / n)}")
            rare = {g: c for g, c in gaps.items() if c / n < 0.002}
            if rare:
                print(f"       rarer than 0.2%: {dict(sorted(rare.items()))}")

print("\nREAD THE ATTRIBUTION, not just the best row: a variant that does not")
print("move its target column is code with no measured reason to exist.")
