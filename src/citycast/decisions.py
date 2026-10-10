"""Historical bet/skip rows (spec §4.1) and the one feature function shared with the live gate bot."""
import json
from datetime import date, datetime, timedelta

from citycast.evaluate import canonical_events
from highcast.buckets import pmf_mass
from highcast.fees import fee_per_contract
from highcast.timeutil import climate_day_bounds

GATE_FEATURES = ["p_model", "model_median", "model_iqr", "ask", "bid", "mid", "spread", "market_p",
                 "fee", "edge", "hours_to_close", "lead_h", "run_gap_to_lo", "run_gap_to_hi", "month"]
STALE_H = 2
BATCH = 50_000
NAN = float("nan")
COLS = ["ticker", "asof_utc", "event_ticker", "station", "metric", "event_date", "lo", "hi",
        *GATE_FEATURES, "won", "profit_if_bet"]


def _h(a_iso: str, b_iso: str) -> float:
    return (datetime.fromisoformat(b_iso) - datetime.fromisoformat(a_iso)).total_seconds() / 3600


def gate_features(*, station, metric, event_date, lo, hi, pmf, quantiles, run_ext, bid, ask,
                  asof_iso, close_iso, climate_end_iso) -> dict:
    q = {float(a): float(v) for a, v in quantiles.items()}
    p = pmf_mass(pmf, lo, hi) or 0.0
    fee = fee_per_contract(1, ask)
    mid = (bid + ask) / 2 if bid is not None else ask
    have_run = run_ext is not None and run_ext == run_ext
    return {"station": station, "metric": metric, "event_date": event_date, "lo": lo, "hi": hi,
            "p_model": p, "model_median": q[0.5], "model_iqr": q[0.8] - q[0.2],
            "ask": ask, "bid": NAN if bid is None else bid, "mid": mid,
            "spread": NAN if bid is None else ask - bid, "market_p": mid, "fee": fee,
            "edge": p - ask - fee, "hours_to_close": _h(asof_iso, close_iso),
            "lead_h": _h(asof_iso, climate_end_iso),
            "run_gap_to_lo": run_ext - lo if have_run and lo is not None else NAN,
            "run_gap_to_hi": run_ext - hi if have_run and hi is not None else NAN,
            "month": int(event_date[5:7])}


def _flush(conn, rows: list) -> int:
    conn.executemany(f"INSERT INTO decision_rows ({','.join(COLS)}) VALUES ({','.join('?' * len(COLS))})",
                     [tuple(r[c] for c in COLS) for r in rows])
    n = len(rows)
    rows.clear()
    return n


def build(conn, stations: dict, progress=print) -> int:
    choice = {(s, m): v for s, m, v in conn.execute("SELECT station, metric, variant FROM model_choice")}
    canon = canonical_events(conn)
    markets: dict[tuple, list] = {}
    # won/profit come from what Kalshi paid (spec §4.1), so only settled yes/no markets qualify
    for m in conn.execute("SELECT ticker, event_ticker, station, metric, event_date, lo, hi, open_time, "
                          "close_time, result FROM kalshi_markets WHERE NOT (lo IS NULL AND hi IS NULL) "
                          "AND result IN ('yes','no')"):
        if m[1] in canon:
            markets.setdefault((m[2], m[3], m[4]), []).append(tuple(m))
    candles = conn.cursor()          # second cursor: the prediction cursor below stays open while we read
    n = 0
    rows: list[dict] = []
    with conn:                        # DELETE + every batch in one transaction: a crash keeps the old table
        conn.execute("DELETE FROM decision_rows")
        preds = conn.execute("SELECT station, metric, event_date, asof_utc, variant, quantiles_json, pmf_json, "
                             "run_ext FROM wf_predictions WHERE variant IN ('shared','city')")
        for s, metric, d, asof, variant, qj, pj, run_ext in preds:
            if choice.get((s, metric)) != variant or s not in stations:
                continue
            todo = markets.get((s, metric, d))
            if not todo:
                continue
            pmf = {int(k): v for k, v in json.loads(pj).items()}
            q = json.loads(qj)
            end_iso = climate_day_bounds(date.fromisoformat(d), stations[s].std_utc_offset)[1].isoformat()
            stale_iso = (datetime.fromisoformat(asof) - timedelta(hours=STALE_H)).isoformat()
            for ticker, event_ticker, _, _, _, lo, hi, open_t, close_t, result in todo:
                if (open_t and open_t > asof) or close_t <= asof:
                    continue
                # a market that closed before its climate day ended (4 of 95,468 in the archive) has a
                # close_time that is not the scheduled close, so hours_to_close would leak the future
                if close_t < end_iso:
                    continue
                # newest usable quote in the window; a crossed quote (bid > ask) is not a price
                c = candles.execute("SELECT yes_bid, yes_ask FROM kalshi_candles WHERE ticker=? AND end_utc>? "
                                    "AND end_utc<=? AND yes_ask IS NOT NULL "
                                    "AND (yes_bid IS NULL OR yes_bid <= yes_ask) "
                                    "ORDER BY end_utc DESC LIMIT 1", (ticker, stale_iso, asof)).fetchone()
                if c is None or not (0.01 <= c[1] <= 0.99):
                    continue
                f = gate_features(station=s, metric=metric, event_date=d, lo=lo, hi=hi, pmf=pmf, quantiles=q,
                                  run_ext=run_ext, bid=c[0], ask=c[1], asof_iso=asof, close_iso=close_t,
                                  climate_end_iso=end_iso)
                won = int(result == "yes")
                rows.append({**f, "ticker": ticker, "asof_utc": asof, "event_ticker": event_ticker,
                             "won": won, "profit_if_bet": won - f["ask"] - f["fee"]})
                if len(rows) >= BATCH:
                    n += _flush(conn, rows)
        if rows:
            n += _flush(conn, rows)
    progress(f"decision_rows: {n}")
    return n
