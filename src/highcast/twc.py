"""The Weather Company observations, and the ONE number that decides whether this
app can trade the hourly market.

`KXTEMPLAXH` ("Hourly Directional Los Angeles Temperature") resolves against The
Weather Company for coordinates KLAX -- explicitly not the NWS:

    "The official, final value for this market is the temperature reported by
     The Weather Company, not any other weather service. NWS Climatological
     Reports, Google Weather, etc. may be useful references, but are not
     authoritative for resolution."

Every model in this project is trained and verified against NWS-sourced data
(CLI reports and ASOS METAR). Those are a DIFFERENT SOURCE, and Kalshi's own
rules warn TWC data "may be subject to rounding and conversion differences".

The strikes are 1 F apart (…73.99, 74.99…). So the question is not "how close is
TWC to METAR on average" -- a 0.4 F mean gap sounds harmless and is not. The
question is **how often the two sources land on OPPOSITE SIDES of a strike**,
because on those hours a model trained on METAR is confidently wrong about a
market settled on TWC. `strike_disagreement()` computes exactly that.

Until that number is known, treat `scripts/hourly_eval.py`'s accuracy figures as
an upper bound that may not transfer at all.
"""
from datetime import datetime, timedelta, timezone

import numpy as np

from highcast import db
from highcast.config import AppConfig
from highcast.timeutil import utcnow_iso


def record(conn, cfg: AppConfig, valid_utc: str, tempf: float,
           source: str = "manual") -> int:
    """Store one TWC reading. `valid_utc` is snapped to the hour it settles.

    Kept deliberately dumb and manual-friendly: weather.com/kalshi renders its
    table client-side and exposes no documented feed, so until an authorised
    source exists these are entered by hand. One row per station-hour; re-entering
    an hour corrects it.
    """
    t = datetime.fromisoformat(valid_utc)
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    snapped = (t.astimezone(timezone.utc) + timedelta(minutes=30)).replace(
        minute=0, second=0, microsecond=0)
    return db.upsert(conn, "twc_obs", [{
        "station": cfg.station.icao, "valid_utc": snapped.isoformat(),
        "tempf": float(tempf), "source": source, "fetched_at": utcnow_iso()}])


def paired(conn, cfg: AppConfig) -> list[dict]:
    """TWC readings alongside our METAR value for the same hour.

    METAR is matched by snapping each report to its nearest hour, the same
    convention `scripts/hourly_eval.py` uses -- ASOS files at :53, so the 23:53Z
    report IS the 00Z observation.
    """
    rows = conn.execute(
        """SELECT valid_utc, tmpf, tmpf_precise FROM obs
           WHERE station=? AND source='metar' AND tmpf_precise IS NOT NULL""",
        (cfg.station.icao,)).fetchall()
    by_hour = {}
    for r in rows:
        t = datetime.fromisoformat(r["valid_utc"])
        snapped = (t + timedelta(minutes=30)).replace(
            minute=0, second=0, microsecond=0)
        by_hour[snapped.isoformat()] = (float(r["tmpf_precise"]),
                                        float(r["tmpf"]) if r["tmpf"] is not None
                                        else float(r["tmpf_precise"]))
    out = []
    for r in conn.execute(
            "SELECT valid_utc, tempf FROM twc_obs WHERE station=? ORDER BY valid_utc",
            (cfg.station.icao,)):
        m = by_hour.get(r["valid_utc"])
        if m is None:
            continue
        out.append({"valid_utc": r["valid_utc"], "twc": float(r["tempf"]),
                    "metar_precise": m[0], "metar": m[1]})
    return out


def strike_disagreement(pairs: list[dict], col: str = "metar_precise") -> dict:
    """THE decision metric: how often would METAR and TWC settle a strike
    differently?

    For each paired hour, walk the ladder of 1 F strikes at X.99 that bracket the
    two readings and count the strikes where `metar > strike` differs from
    `twc > strike`. Any such strike is a market this app would call wrong no
    matter how good the model is.
    """
    if not pairs:
        return {"n": 0}
    twc = np.array([p["twc"] for p in pairs], dtype=float)
    ours = np.array([p[col] for p in pairs], dtype=float)
    diff = twc - ours
    flips = 0
    for a, b in zip(ours, twc):
        lo, hi = (a, b) if a <= b else (b, a)
        # strikes strictly between the two readings are the ones that flip
        k = np.floor(lo) + 0.99
        while k < hi:
            if k > lo:
                flips += 1
            k += 1.0
    return {
        "n": len(pairs),
        "bias": float(diff.mean()),
        "mae": float(np.abs(diff).mean()),
        "max_abs": float(np.abs(diff).max()),
        "exact": float(np.mean(np.abs(diff) < 1e-9)),
        "hours_with_a_flipped_strike": float(
            np.mean([1.0 if _flips_one(a, b) else 0.0 for a, b in zip(ours, twc)])),
        "total_flipped_strikes": flips,
    }


def _flips_one(a: float, b: float) -> bool:
    lo, hi = (a, b) if a <= b else (b, a)
    k = np.floor(lo) + 0.99
    while k < hi:
        if k > lo:
            return True
        k += 1.0
    return False
