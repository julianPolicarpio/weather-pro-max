"""Coverage report and the 2a acceptance check (spec §2.5)."""
from datetime import date, datetime

from highcast.timeutil import climate_day_bounds

MIN_PCT = 0.98
REQUIRED_MODELS = ("MEX", "NBS")


def _days(a: date, b: date) -> int:
    return max((b - a).days, 0)


def _mos_pct(conn, st, today: date) -> dict:
    out = {}
    for (model,) in conn.execute("SELECT DISTINCT model FROM mos_runs WHERE station=?", (st.icao,)):
        first = conn.execute("SELECT MIN(target_date) FROM mos_runs WHERE station=? AND model=?",
                             (st.icao, model)).fetchone()[0]
        start = date.fromisoformat(first)
        ok = 0
        rows = conn.execute("SELECT target_date, MIN(runtime_utc) FROM mos_runs WHERE station=? "
                            "AND model=? AND target_date < ? GROUP BY target_date",
                            (st.icao, model, today.isoformat())).fetchall()
        for d, first_run in rows:
            day_start = climate_day_bounds(date.fromisoformat(d), st.std_utc_offset)[0]
            if datetime.fromisoformat(first_run) < day_start:
                ok += 1
        n = _days(start, today)
        out[model] = round(ok / n, 4) if n else 0.0
    return out


def _disagreements(conn, st) -> list[dict]:
    out = []
    rows = conn.execute(
        "SELECT DISTINCT k.metric, k.event_date, k.expiration_value, c.high_f, c.low_f "
        "FROM kalshi_markets k JOIN cli_daily c ON c.station=k.station AND c.date=k.event_date "
        "WHERE k.station=? AND c.final=1 AND k.expiration_value IS NOT NULL "
        "AND k.expiration_value != ''", (st.icao,)).fetchall()
    for metric, d, ev, hi, lo in rows:
        try:
            kv = round(float(ev))
        except ValueError:
            continue
        cv = hi if metric == "high" else lo
        if cv is not None and kv != cv:
            out.append({"station": st.icao, "metric": metric, "event_date": d, "kalshi": kv, "cli": cv})
    return out


def compute(conn, stations: dict, today: date, since: date = date(2021, 1, 1)) -> dict:
    report = {"stations": {}, "disagreements": []}
    for st in stations.values():
        # spec §2.5: each source is measured from max(since, its own first date for this station)
        cli_start = conn.execute("SELECT MIN(date) FROM cli_daily WHERE station=? AND final=1 "
                                 "AND date>=?", (st.icao, since.isoformat())).fetchone()[0]
        cli_final = conn.execute("SELECT COUNT(*) FROM cli_daily WHERE station=? AND final=1 "
                                 "AND date>=? AND date<?",
                                 (st.icao, since.isoformat(), today.isoformat())).fetchone()[0]
        cli_n = _days(max(since, date.fromisoformat(cli_start)), today) if cli_start else 0
        obs_days = [r[0] for r in conn.execute(
            "SELECT substr(datetime(valid_utc, ? || ' hours'), 1, 10) d FROM obs WHERE station=? "
            "GROUP BY d HAVING COUNT(*) >= 20 ORDER BY d", (str(st.std_utc_offset), st.icao))]
        obs_days = [d for d in obs_days if since.isoformat() <= d < today.isoformat()]
        obs_n = _days(date.fromisoformat(obs_days[0]), today) if obs_days else 0
        k = conn.execute(
            "SELECT COUNT(DISTINCT event_ticker), COUNT(*), "
            "SUM(NOT (lo IS NULL AND hi IS NULL)), "
            "SUM(EXISTS (SELECT 1 FROM kalshi_candles kc WHERE kc.ticker = km.ticker)) "
            "FROM kalshi_markets km WHERE station=?", (st.icao,)).fetchone()
        mos = _mos_pct(conn, st, today)
        cli_pct = round(cli_final / cli_n, 4) if cli_n else 0.0
        s = {"cli_final_pct": cli_pct, "cli_start": cli_start, "mos_pct": mos,
             "obs_pct": round(len(obs_days) / obs_n, 4) if obs_n else 0.0,
             "kalshi_events": k[0] or 0, "kalshi_markets": k[1] or 0,
             "kalshi_usable": k[2] or 0, "kalshi_with_candles": k[3] or 0}
        s["pass"] = cli_pct >= MIN_PCT and all(mos.get(m, 0.0) >= MIN_PCT for m in REQUIRED_MODELS)
        report["stations"][st.icao] = s
        report["disagreements"].extend(_disagreements(conn, st))
    report["pass"] = all(s["pass"] for s in report["stations"].values())
    return report


def render(report: dict) -> str:
    lines = [f"{'station':8} {'cli':>6} {'cli_start':>10} {'MEX':>6} {'NBS':>6} {'GFS':>6} {'NBE':>6} {'obs':>6} "
             f"{'events':>7} {'mkts':>7} {'usable':>7} {'candles':>7}  pass"]
    for icao, s in sorted(report["stations"].items()):
        m = s["mos_pct"]
        lines.append(f"{icao:8} {s['cli_final_pct']:6.1%} {s.get('cli_start') or '-':>10} {m.get('MEX', 0):6.1%} {m.get('NBS', 0):6.1%} "
                     f"{m.get('GFS', 0):6.1%} {m.get('NBE', 0):6.1%} {s['obs_pct']:6.1%} "
                     f"{s['kalshi_events']:7d} {s['kalshi_markets']:7d} {s['kalshi_usable']:7d} "
                     f"{s['kalshi_with_candles']:7d}  {'yes' if s['pass'] else 'NO'}")
    lines.append(f"disagreements (Kalshi settled value vs CLI final): {len(report['disagreements'])}")
    for d in report["disagreements"]:
        lines.append(f"  {d['station']} {d['metric']} {d['event_date']}: kalshi {d['kalshi']} cli {d['cli']}")
    lines.append(f"ACCEPTANCE: {'PASS' if report['pass'] else 'FAIL'}")
    return "\n".join(lines)
