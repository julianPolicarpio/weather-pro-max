"""WHICH KALSHI CITY IS THE MOST PREDICTABLE?

Kalshi lists daily high/low temperature markets for ~12 US cities, and every one
of them resolves against the **NWS Climatological Report (Daily)** -- verified
from each series' `rules_primary`, same source this project already builds on. So
the architecture transfers; the open question is which city is worth the effort.

This surveys them without porting the whole app. Two questions, cheapest first:

  1. HOW HARD IS THE CITY?  From settled CLI values alone: day-to-day variability
     and the persistence error (|today - yesterday|). A city whose temperature
     barely moves is easy for everyone -- including the market.
  2. HOW ACCURATE CAN WE BE?  From MOS guidance vs settled truth: bias, MAE, and
     modal-bucket hit rate. MOS is station-calibrated and available for any US
     ASOS from the same IEM archive, so this is the closest cheap proxy for the
     accuracy a full model would reach.

⚠️ EASY IS NOT THE SAME AS PROFITABLE. A city where guidance nails the high every
day is a city where the market also nails it. This ranks PREDICTABILITY, which is
what was asked for; it says nothing about edge. The two can point opposite ways --
see HANDOFF's BIAS DISCIPLINE section.

STATIONS ARE TAKEN FROM KALSHI'S RULES, NOT GUESSED. Two would have been wrong:
New York is **Central Park (KNYC)**, not JFK/LGA, and Chicago is **Midway
(KMDW)**, not O'Hare.

Usage:
  .venv/bin/python scripts/city_survey.py fetch-cli      # fast, all cities
  .venv/bin/python scripts/city_survey.py fetch-mos      # slow, all cities
  .venv/bin/python scripts/city_survey.py report
"""
import dataclasses
import sys
from datetime import date, datetime, timezone

import numpy as np

from highcast import db
from highcast.config import load_config
from highcast.fetchers import cli_reports, forecasts
from highcast.metric import get_metric

# (kalshi city code, ICAO, label, has_high_market, has_low_market)
ROSTER = [
    ("LAX",  "KLAX", "Los Angeles Intl",     True,  True),
    ("NY",   "KNYC", "New York Central Park", True,  False),
    ("CHI",  "KMDW", "Chicago Midway",        True,  True),
    ("MIA",  "KMIA", "Miami Intl",            True,  True),
    ("AUS",  "KAUS", "Austin Bergstrom",      True,  True),
    ("DEN",  "KDEN", "Denver",                True,  True),
    ("PHIL", "KPHL", "Philadelphia Intl",     True,  True),
    ("PHX",  "KPHX", "Phoenix",               False, True),
    ("SEA",  "KSEA", "Seattle",               False, True),
    ("ATL",  "KATL", "Atlanta",               False, True),
    ("BOS",  "KBOS", "Boston",                False, True),
    ("HOU",  "KHOU", "Houston Hobby",         False, True),
]
YEARS = range(2021, 2027)


def _cfg_for(icao: str):
    """The base KLAX config with the station swapped. Only `icao` is read by the
    CLI and MOS fetchers, so a full per-city config (lat/lon, WFO, zones) is not
    needed for the survey -- that is why this is cheap."""
    base = load_config("config/klax.yaml")
    return dataclasses.replace(base, station=dataclasses.replace(
        base.station, icao=icao))


def fetch_cli():
    conn = db.connect(load_config("config/klax.yaml").db_path)
    for code, icao, label, _, _ in ROSTER:
        cfg = _cfg_for(icao)
        total = 0
        for y in YEARS:
            try:
                total += cli_reports.store(conn, cli_reports.fetch_year(cfg, y))
            except Exception as e:  # noqa: BLE001 - one bad year must not stop the survey
                print(f"  [warn] {icao} {y}: {type(e).__name__}")
        print(f"{code:5} {icao} {label:24} CLI rows {total}")


def fetch_mos():
    conn = db.connect(load_config("config/klax.yaml").db_path)
    end = datetime.now(timezone.utc).date()
    for code, icao, label, hi, lo in ROSTER:
        cfg = _cfg_for(icao)
        for mk, want in (("high", hi), ("low", lo)):
            if not want:
                continue
            try:
                rows = forecasts.fetch_mos_archive(
                    cfg, date(2021, 1, 1), end, metric=get_metric(cfg, mk))
                print(f"{code:5} {icao} {mk:5} MOS rows {forecasts.store(conn, rows)}")
            except Exception as e:  # noqa: BLE001
                print(f"{code:5} {icao} {mk:5} FAILED {type(e).__name__}: {e}")


def _kalshi_lo(t):
    return t if t % 2 == 0 else t - 1


def report():
    conn = db.connect(load_config("config/klax.yaml").db_path)
    print(f"{'city':6}{'station':8}{'metric':7}{'n':>6}{'sd':>7}{'persist':>9}"
          f"{'MOS bias':>10}{'MOS MAE':>9}{'MOS hit':>9}{'corr MAE':>10}{'corr hit':>10}")
    rows_out = []
    for code, icao, label, hi, lo in ROSTER:
        for mk, col, want in (("high", "high_f", hi), ("low", "low_f", lo)):
            if not want:
                continue
            truth = {r["date"]: r[col] for r in conn.execute(
                f"SELECT date, {col} FROM cli_reports WHERE station=? AND {col} IS NOT NULL",
                (icao,))}
            if len(truth) < 300:
                print(f"{code:6}{icao:8}{mk:7}{len(truth):>6}  (too little CLI history)")
                continue
            ds = sorted(truth)
            vals = np.array([truth[d] for d in ds], dtype=float)
            sd = float(vals.std())
            # persistence: |today - yesterday| over consecutive calendar days
            import datetime as _dt
            idx = {d: i for i, d in enumerate(ds)}
            pers = [abs(truth[d] - truth[p]) for d in ds
                    if (p := (_dt.date.fromisoformat(d) - _dt.timedelta(days=1)
                              ).isoformat()) in idx]
            # MOS, day-ahead-legal, latest run per target
            mos = {}
            for r in conn.execute(
                    """SELECT target_date, max_temp_f FROM forecasts
                       WHERE station=? AND metric=? AND model='mos_mex'
                         AND run_date < target_date
                       ORDER BY target_date, run_date DESC""", (icao, mk)):
                mos.setdefault(r["target_date"], r["max_temp_f"])
            pairs = [(truth[d], mos[d]) for d in ds if d in mos and mos[d] is not None]
            if len(pairs) < 200:
                print(f"{code:6}{icao:8}{mk:7}{len(truth):>6}{sd:>7.2f}"
                      f"{np.mean(pers):>9.2f}   (no MOS yet)")
                continue
            y = np.array([p[0] for p in pairs], dtype=float)
            f = np.array([p[1] for p in pairs], dtype=float)
            err = f - y
            bias = float(err.mean())
            hit = float(np.mean([_kalshi_lo(int(round(a))) == _kalshi_lo(int(b))
                                 for a, b in zip(f, y)]))
            # bias-corrected: the single cheapest real improvement, as measured at
            # KLAX (a trailing correction cut ensemble MAE ~46%)
            corr = f - bias
            chit = float(np.mean([_kalshi_lo(int(round(a))) == _kalshi_lo(int(b))
                                  for a, b in zip(corr, y)]))
            print(f"{code:6}{icao:8}{mk:7}{len(pairs):>6}{sd:>7.2f}{np.mean(pers):>9.2f}"
                  f"{bias:>+10.2f}{np.abs(err).mean():>9.2f}{hit:>8.1%}"
                  f"{np.abs(corr - y).mean():>10.2f}{chit:>9.1%}")
            rows_out.append((code, mk, float(np.abs(corr - y).mean()), chit))
    if rows_out:
        print("\nRANKED by bias-corrected MOS bucket hit (higher = more predictable):")
        for code, mk, mae, chit in sorted(rows_out, key=lambda r: -r[3])[:12]:
            print(f"  {code:6}{mk:6}{chit:>7.1%}  MAE {mae:.2f}")
        print("\nReminder: predictable != profitable. An easy city is easy for the "
              "market too.")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    {"fetch-cli": fetch_cli, "fetch-mos": fetch_mos, "report": report}[cmd]()
