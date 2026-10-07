"""Every Kalshi daily temperature market, and how predictable each one is.

Kalshi lists daily high/low temperature markets for 12 US cities. **All of them
resolve on the NWS Climatological Report (Daily)** -- verified from each series'
own `rules_primary`, the same source this project already builds on -- so the
architecture transfers unchanged and the only real question is which city is
worth the effort.

STATIONS ARE FROM KALSHI'S RULES, NOT GUESSED. Two would have been wrong:
New York is **Central Park (KNYC)**, not JFK/LGA, and Chicago is **Midway
(KMDW)**, not O'Hare. Getting either wrong would silently score a model against
a station the market does not settle on.

⚠️ BUCKET HIT IS PARITY-AVERAGED. Kalshi's 2F ladder parity is a coin flip per
day, so a fixed-even scorer (what this used before 2026-07-28) is wrong about
half the time. Every `hit`/`corr_hit` here averages both ladders.

The ranking here is computed from MOS guidance rather than from a trained model
per city, because MOS needs only an ICAO (`fetch_mos_archive`) while a model
needs a full per-city config, a METAR backfill, and its own measured decision
hour. MOS is station-calibrated, so it is a fair floor: at KLAX the shipped model
reaches 47.6% day-ahead on the LOW where MOS alone reaches 41.2%, i.e. roughly
+6 points of headroom.

⚠️ PREDICTABLE IS NOT PROFITABLE. A city guidance nails every day is a city the
market also nails. This answers "where is the model most accurate", which is what
was asked; it says nothing about edge, and HANDOFF's §11 no-edge finding was never
city-specific.
"""
import datetime as _dt

import numpy as np

from highcast.buckets import ladder_lo

# (kalshi code, ICAO, label, has_high_market, has_low_market)
ROSTER: list[tuple[str, str, str, bool, bool]] = [
    ("LAX",  "KLAX", "Los Angeles Intl",      True,  True),
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

MIN_CLI_DAYS = 300
MIN_MOS_PAIRS = 200


def _one(conn, icao: str, metric: str, col: str) -> dict | None:
    truth = {r["date"]: r[col] for r in conn.execute(
        f"SELECT date, {col} FROM cli_reports WHERE station=? AND {col} IS NOT NULL",
        (icao,))}
    if len(truth) < MIN_CLI_DAYS:
        return None
    ds = sorted(truth)
    vals = np.array([truth[d] for d in ds], dtype=float)
    idx = set(ds)
    # Day-to-day change: the naive difficulty of the city, independent of any
    # model. A city whose temperature barely moves is easy for everyone.
    persist = [abs(truth[d] - truth[p]) for d in ds
               if (p := (_dt.date.fromisoformat(d) - _dt.timedelta(days=1)
                         ).isoformat()) in idx]

    # Latest DAY-AHEAD-LEGAL MOS run per target (run_date < target_date), so this
    # is a genuine forecast rather than a same-day reading.
    mos: dict[str, float] = {}
    for r in conn.execute(
            """SELECT target_date, max_temp_f FROM forecasts
               WHERE station=? AND metric=? AND model='mos_mex'
                 AND run_date < target_date AND max_temp_f IS NOT NULL
               ORDER BY target_date, run_date DESC""", (icao, metric)):
        mos.setdefault(r["target_date"], r["max_temp_f"])
    pairs = [(truth[d], mos[d]) for d in ds if d in mos]
    out = {"n_days": len(truth), "sd": float(vals.std()),
           "persist": float(np.mean(persist)) if persist else None,
           "n_pairs": len(pairs), "bias": None, "mae": None, "hit": None,
           "corr_mae": None, "corr_hit": None}
    if len(pairs) < MIN_MOS_PAIRS:
        return out
    y = np.array([p[0] for p in pairs], dtype=float)
    f = np.array([p[1] for p in pairs], dtype=float)
    bias = float((f - y).mean())
    corr = f - bias          # the cheapest real correction; see features.BIAS_WINDOW
    # PARITY-AVERAGED, not fixed-even. Kalshi centres each day's ladder on the
    # consensus forecast, so the parity of the 2F buckets is a coin flip per day
    # (KXLOWTMIA ran 39 even / 29 odd over 68 settled days). Scoring every city
    # on a fixed EVEN ladder -- which this did -- is right about half the time.
    # A forecast and a truth that share a bucket on one ladder can easily sit in
    # different buckets on the other, so the honest figure averages the two.
    hit = lambda p: float(np.mean(  # noqa: E731
        [np.mean([ladder_lo(int(round(a)), par) == ladder_lo(int(b), par)
                  for par in (0, 1)]) for a, b in zip(p, y)]))
    out.update(bias=bias, mae=float(np.abs(f - y).mean()), hit=hit(f),
               corr_mae=float(np.abs(corr - y).mean()), corr_hit=hit(corr))
    return out


def survey(conn, station: str | None = None) -> list[dict]:
    """One row per tradeable (city, metric), ranked most-predictable first.

    `station` marks the caller's own station so a surface can highlight it.
    Rows without enough MOS history sort last rather than being dropped -- a
    missing city is a data gap worth seeing, not something to hide.
    """
    rows = []
    for code, icao, label, hi, lo in ROSTER:
        for metric, col, want in (("high", "high_f", hi), ("low", "low_f", lo)):
            if not want:
                continue
            s = _one(conn, icao, metric, col)
            if s is None:
                continue
            rows.append({"city": code, "station": icao, "label": label,
                         "metric": metric, "is_current": icao == station, **s})
    rows.sort(key=lambda r: (r["corr_hit"] is None, -(r["corr_hit"] or 0)))
    return rows
