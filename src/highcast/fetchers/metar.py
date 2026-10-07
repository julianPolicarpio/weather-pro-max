import re
from datetime import date, timedelta

from highcast import db
from highcast.config import AppConfig
from highcast.http import get_json, get_text
from highcast.timeutil import c_to_f, norm_valid, utcnow_iso

AVWX = "https://aviationweather.gov/api/data/metar"
IEM_ASOS = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

_T_GROUP = re.compile(r"\bT([01])(\d{3})([01])(\d{3})\b")
# The 6-hourly groups are standalone 5-character tokens. The trailing (?!/)
# is load-bearing: the peak-wind group `PK WND 11032/2256` otherwise matches
# _SIX_MAX as sign=1 (negative), digits=032 -> -3.2C. Seen live at KLAX on
# 2022-09-09T23:53Z, which reported a real `10389 20322` alongside
# `PK WND 11032/2256` and parsed six_max_c as -3.2 against a 91F observation.
# A spurious LOW max happens to be harmless (the running max takes a max, and
# the plausibility guard only catches spuriously HIGH ones), but the mirrored
# _SIX_MIN has no such luck -- there a spurious low value would drag the LOW's
# ratchet bound down, which is exactly the bound the whole LOW model rests on.
_SIX_MAX = re.compile(r"\b1([01])(\d{3})\b(?!/)")
_SIX_MIN = re.compile(r"\b2([01])(\d{3})\b(?!/)")
_TT_DD = re.compile(r"\b(M?\d{2})/(M?\d{2})\b")
_WIND = re.compile(r"\b(\d{3}|VRB)(\d{2,3})(?:G\d{2,3})?KT\b")
_SKY = re.compile(r"\b(FEW|SCT|BKN|OVC|CLR|SKC)\d*\b")
# ── environment fields, added 2026-07-30 ────────────────────────────────────
# ⚠️ `skyc` above is UNCHANGED and must stay so: it feeds `cur_sky`, a live model
# feature, and the model is frozen. Everything below writes NEW columns only.
#
# WHY THESE. The KMIA HIGH is convection-limited -- an afternoon storm sets the
# ceiling -- and the nowcast has NO precipitation input at all. These are the
# storm-adjacent signals already sitting in the raw text we store:
#
#   visibility          present on 100% of reports; collapses in rain/fog
#   ALL cloud layers    present on 82%; we keep only the LAST (highest) token,
#                       so `BKN008 OVC200` is indistinguishable from `FEW200`.
#                       A low deck is what actually caps heating.
#   present weather     17%; literally spells TS / RA / BR
#
# Layer heights are in HUNDREDS of feet: BKN008 = 800 ft.
#
# ⚠️ THE `CB`/`TCU` SUFFIX IS LOAD-BEARING AND NEARLY COST US THE WHOLE POINT.
# A layer may be tagged CB (cumulonimbus = thunderstorm) or TCU (towering
# cumulus = building convection). A first draft ended the pattern at `\d{3}\b`,
# and `\b` does not match between the `2` and the `C` of `BKN032CB` -- so every
# convective layer was silently DROPPED. On a real KMIA report,
# `TSRA BKN032CB BKN060 BKN110 OVC260` parsed as 3 layers with a 6000 ft ceiling
# instead of 4 layers at 3200 ft. The parser was discarding exactly the storm
# clouds it exists to capture, and reporting a ceiling 2800 ft too high.
_SKY_LAYER = re.compile(r"\b(FEW|SCT|BKN|OVC)(\d{3})(CB|TCU)?\b")
# Visibility in statute miles: "10SM", "1/2SM", "1 1/2SM", "M1/4SM", "1/16SM".
# Vertical visibility (VV) is a ceiling, not a horizontal visibility, and is
# deliberately not matched here.
#
# ⚠️ THE DENOMINATOR MUST ALLOW TWO DIGITS, AND THE BUG IS INVERTED. A first
# draft used `(\d/\d)` for the fraction, so `1/16SM` -- one SIXTEENTH of a mile,
# i.e. dense fog -- failed the fraction branch, fell through to the whole-number
# branch, and parsed as **16.0 SM, nearly unlimited visibility**. 116 real KLAX
# fog reports were stored as the clearest days in the archive. An error that
# flips a variable's meaning is far worse than one that merely loses it, and a
# marine-layer station's fog reports are exactly the rows that matter.
_VIS = re.compile(r"\b(M)?((?:\d{1,3}\s+)?\d{1,2}/\d{1,2}|\d{1,3})SM\b")
# Present weather. Intensity/proximity prefix, then descriptor+phenomenon. Only
# the codes that actually occur at these two coastal stations are listed, so a
# malformed token cannot masquerade as weather.
_WX = re.compile(
    r"(?:^|\s)([-+]|VC)?"
    r"((?:MI|BC|PR|DR|BL|SH|TS|FZ)?"
    r"(?:DZ|RA|SN|SG|PL|GR|GS|BR|FG|FU|HZ|DU|SA|PY|SQ|FC|TS)+)"
    r"(?=\s|$)")
# Sea-level pressure, remarks only: SLPppp in TENTHS of hPa with the leading 9
# or 10 implied. SLP097 -> 1009.7, SLP502 -> 950.2. `SLPNO` (not available)
# correctly fails to match. Present on ~85% of stored reports at both stations,
# i.e. essentially every hourly one -- see the internal design note
# on pick-of-day (S5).
#
# ⚠️ THE DECODE IS A TRAP. A throwaway analysis query during this work rendered
# SLP097 as "100.97 hPa" -- a silent factor of ten. Surface pressure is ~950-1050
# hPa, never ~100, so the `< 500` branch is what carries the 900/1000 choice.
_SLP = re.compile(r"\bSLP(\d{3})\b")
# 3-hour pressure tendency, remarks only: 5appp where `a` is the WMO tendency
# shape code 0-8 and ppp is the magnitude in tenths of hPa. 58013 -> code 8,
# 1.3 hPa. Filed on the synoptic reports only, so ~28% of rows.
#
# Codes 0-3 mean higher than 3h ago, 4 steady, 5-8 LOWER -- so the sign comes
# from the code, not from the digits, and reading ppp alone loses the direction
# entirely. The trailing (?!/) mirrors the 6-hourly groups' guard against
# `PK WND ddd ff/hhmm`; a peak-wind direction cannot begin with 5 (max 360), so
# this is defensive rather than load-bearing, but the cost is nil and the failure
# it prevents is the same class that produced a -3.2C six_max_c in 2022.
_PRES_TEND = re.compile(r"\b5([0-8])(\d{3})\b(?!/)")
# WMO code table 0200. Kept as text because the SHAPE matters independently of
# the magnitude for advection work: "rising then falling, net lower" is a
# different air mass story from "falling steadily" at the same -1.3 hPa.
PRES_TEND_SHAPE = {
    0: "rising then falling, net higher",
    1: "rising then steady, or rising more slowly",
    2: "rising steadily",
    3: "falling or steady then rising, or rising more rapidly",
    4: "steady",
    5: "falling then rising, net lower",
    6: "falling then steady, or falling more slowly",
    7: "falling steadily",
    8: "steady or rising then falling, net lower",
}


def _signed(sign: str, digits: str) -> float:
    v = int(digits) / 10.0
    return -v if sign == "1" else v


def _int_c(tok: str) -> float:
    return -int(tok[1:]) if tok.startswith("M") else int(tok)


def _slp_hpa(digits: str) -> float:
    """Decode SLPppp to hPa. ppp >= 500 implies the 900s, otherwise the 1000s."""
    v = int(digits)
    return (900.0 + v / 10.0) if v >= 500 else (1000.0 + v / 10.0)


def _tendency_hpa(code: str, digits: str) -> float:
    """Signed 3-hour pressure change in hPa, direction taken from the WMO code.

    Codes 0-3 are a net RISE, 4 is steady, 5-8 a net FALL. Code 4 is forced to
    exactly 0.0 rather than trusting ppp: "steady" with a nonzero magnitude is
    self-contradictory, and some stations file 4000 while others file 4 followed
    by junk.
    """
    c = int(code)
    if c == 4:
        return 0.0
    mag = int(digits) / 10.0
    return -mag if c >= 5 else mag


def parse_metar(raw: str, valid_utc: str, icao: str, fetched_at: str) -> dict:
    tmpf = tmpf_precise = dwpf = sknt = drct = six_max_c = six_min_c = None
    skyc = None
    slp_hpa = pres_tend_hpa = None
    pres_tend_code = None
    if m := _TT_DD.search(raw):
        tmpf = c_to_f(_int_c(m.group(1)))
        dwpf = c_to_f(_int_c(m.group(2)))
    if m := _T_GROUP.search(raw):
        tmpf_precise = c_to_f(_signed(m.group(1), m.group(2)))
        dwpf = c_to_f(_signed(m.group(3), m.group(4)))
    if m := _WIND.search(raw):
        drct = None if m.group(1) == "VRB" else float(m.group(1))
        sknt = float(m.group(2))
    # 6-hr max/min live in remarks only; don't match the wind group by accident
    rmk = raw.split(" RMK ", 1)
    report_tmpf = tmpf_precise if tmpf_precise is not None else tmpf
    if len(rmk) == 2 and (m := _SIX_MAX.search(rmk[1])):
        six_max_c = _signed(m.group(1), m.group(2))
        # Plausibility guard: a 6-hr max physically cannot exceed the
        # report's own temperature by more than 30F at a surface station.
        # IEM's archive has been observed to contain corrupted 6-hr-max
        # groups (e.g. 10439 claiming +43.9C/111F on a 17.2C/63F report);
        # without this guard such a group would poison downstream
        # running-max bounds. If the report has no temperature at all,
        # there's no basis to judge, so leave six_max_c as parsed.
        if report_tmpf is not None and c_to_f(six_max_c) - report_tmpf > 30.0:
            six_max_c = None
    if len(rmk) == 2 and (m := _SIX_MIN.search(rmk[1])):
        six_min_c = _signed(m.group(1), m.group(2))
        # Mirror of the 6-hr-max guard, opposite sign: a 6-hr min physically
        # cannot run more than 30F BELOW the report's own temperature.
        if report_tmpf is not None and report_tmpf - c_to_f(six_min_c) > 30.0:
            six_min_c = None
    # Pressure lives in remarks too, and for the same reason as the 6-hourly
    # groups: matching against the whole report would let the altimeter setting
    # (A2982) or a runway visual range collide with these patterns.
    if len(rmk) == 2 and (m := _SLP.search(rmk[1])):
        slp_hpa = _slp_hpa(m.group(1))
    if len(rmk) == 2 and (m := _PRES_TEND.search(rmk[1])):
        pres_tend_code = int(m.group(1))
        pres_tend_hpa = _tendency_hpa(m.group(1), m.group(2))
    sky_tokens = _SKY.findall(raw)
    if sky_tokens:
        skyc = sky_tokens[-1]
    # ── environment fields (NEW columns; `skyc` above is untouched) ─────────
    # Only the BODY is scanned for weather and visibility: the RMK section can
    # carry lookalike tokens (e.g. a `TS` in a remark narrative), and the body is
    # where the coded observation lives.
    body = raw.split(" RMK", 1)[0]
    vis_sm = None
    if m := _VIS.search(body):
        tok = m.group(2)
        parts = tok.split()
        v = 0.0
        for part in parts:
            if "/" in part:
                a, b = part.split("/")
                v += float(a) / float(b)
            else:
                v += float(part)
        # "M" means LESS THAN the stated value; record the bound itself rather
        # than inventing a smaller number.
        vis_sm = v if v > 0 else None
    layers = [(int(h) * 100, cov, sfx) for cov, h, sfx in _SKY_LAYER.findall(body)]
    layers.sort()
    sky_lowest_ft = layers[0][0] if layers else None
    # Meteorological CEILING: the lowest BROKEN or OVERCAST layer. Scattered and
    # few are not a ceiling -- they do not cap insolation the same way.
    ceiling_ft = next((ft for ft, cov, _ in layers if cov in ("BKN", "OVC")), None)
    n_layers = len(layers) or None
    # Convective cloud actually reported overhead. This is a direct observation
    # that a storm is present or building -- stronger than inferring it from
    # cover alone, and the single most storm-specific field in the report.
    sky_cb = 1 if any(sfx in ("CB", "TCU") for _, _, sfx in layers) else 0
    wx = [f"{i or ''}{p}" for i, p in _WX.findall(body)]
    wx_present = " ".join(wx) or None
    wx_thunder = 1 if any("TS" in w for w in wx) else 0
    return {
        "station": icao, "valid_utc": valid_utc, "source": "metar",
        "tmpf": tmpf, "tmpf_precise": tmpf_precise, "dwpf": dwpf,
        "sknt": sknt, "drct": drct, "skyc": skyc, "six_max_c": six_max_c,
        "six_min_c": six_min_c, "slp_hpa": slp_hpa,
        "pres_tend_hpa": pres_tend_hpa, "pres_tend_code": pres_tend_code,
        "vis_sm": vis_sm, "ceiling_ft": ceiling_ft,
        "sky_lowest_ft": sky_lowest_ft, "n_sky_layers": n_layers,
        "sky_cb": sky_cb,
        "wx_present": wx_present, "wx_thunder": wx_thunder,
        "raw": raw, "fetched_at": fetched_at,
    }


def fetch_recent(cfg: AppConfig, hours: int = 36) -> list[dict]:
    payload = get_json(AVWX, {"ids": cfg.station.icao, "format": "json",
                              "hours": hours})
    now = utcnow_iso()
    rows = []
    for ob in payload:
        raw = ob.get("rawOb") or ""
        # obsTime is the true observation time (epoch seconds); reportTime is
        # a rounded dissemination time and only used as a last-resort fallback.
        valid = ob.get("obsTime")
        if valid is None:
            valid = ob.get("reportTime")
        if raw and valid is not None:
            rows.append(parse_metar(raw, norm_valid(valid), cfg.station.icao, now))
    return rows


def fetch_archive(cfg: AppConfig, start: date, end: date) -> list[dict]:
    """IEM METAR archive, chunked by month to keep responses small."""
    now = utcnow_iso()
    rows = []
    cur = start
    while cur <= end:
        nxt = min(end, (cur.replace(day=1) + timedelta(days=32)).replace(day=1))
        text = get_text(IEM_ASOS, {
            "station": cfg.station.iem_id, "data": "metar",
            "year1": cur.year, "month1": cur.month, "day1": cur.day,
            "year2": nxt.year, "month2": nxt.month, "day2": nxt.day,
            "tz": "Etc/UTC", "format": "onlycomma", "latlon": "no",
            "report_type": [3, 4],
        })
        for line in text.splitlines()[1:]:
            parts = line.split(",", 2)
            if len(parts) == 3:
                _, valid, raw = parts
                rows.append(parse_metar(raw.strip(), norm_valid(valid.strip()),
                                        cfg.station.icao, now))
        cur = nxt if nxt > cur else end + timedelta(days=1)
    return rows


def store(conn, rows: list[dict]) -> int:
    return db.upsert(conn, "obs", rows)
