"""Final models and live predictions (spec §3.9)."""
import json
import math
import pickle
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from citycast import db
from citycast.qmodel import QUANTILES, QuantileModel, pmf_from_quantiles
from citycast.walkforward import MIN_CITY_ROWS, _prev, choose_k

MODEL_DIR = Path("data/citycast/models")


def _bundle_path(metric: str) -> Path:
    return MODEL_DIR / f"{metric}.pkl"


def train_final(conn, df: pd.DataFrame, metric: str, stations: tuple[str, ...], now: date) -> Path:
    known = df[(df["y"].notna()) & (df["metric"] == metric)] if "metric" in df else df[df["y"].notna()]
    truth = {(r.station, r.event_date): int(r.y) for r in known[["station", "event_date", "y"]]
             .drop_duplicates(["station", "event_date"]).itertuples(index=False)}
    choice = {s: (v, g) for s, v, g in conn.execute(
        "SELECT station, variant, gate_pass FROM model_choice WHERE metric=?", (metric,))}
    bundle = {"metric": metric, "trained_on": now.isoformat(),
              "shared": QuantileModel("shared", stations).fit(known, now), "city": {}, "choice": {}}
    for s, (variant, gate) in choice.items():
        if variant == "city":
            tr = known[known["station"] == s]
            if len(tr) < MIN_CITY_ROWS:
                variant = "shared"
            else:
                bundle["city"][s] = QuantileModel("city", stations).fit(tr, now)
        k = choose_k(_prev(conn, s, metric, variant, now, truth), metric)
        bundle["choice"][s] = (variant, k, int(gate))
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = _bundle_path(metric)
    path.write_bytes(pickle.dumps(bundle))
    return path


def predict(conn, ctx, metric: str, now: datetime) -> int:
    bundle = pickle.loads(_bundle_path(metric).read_bytes())
    rows = []
    for icao, (variant, k, gate) in sorted(bundle["choice"].items()):
        if not gate or icao not in ctx.stations:
            continue
        st = ctx.stations[icao]
        today = now.astimezone(ZoneInfo(st.tz)).date()
        for day in (today, today + timedelta(days=1)):
            r = ctx.row(icao, metric, day, now)
            if r is None:
                continue
            frame = pd.DataFrame([r])
            model = bundle["city"][icao] if variant == "city" else bundle["shared"]
            q = dict(zip(QUANTILES, (float(v) for v in model.predict_y_quantiles(frame)[0])))
            pmf = pmf_from_quantiles(q, k, r["run_ext"], metric)
            rows.append({"station": icao, "metric": metric, "event_date": day.isoformat(),
                         "issued_utc": now.isoformat(), "variant": variant, "anchor": r["anchor"],
                         "run_ext": None if math.isnan(r["run_ext"]) else r["run_ext"],
                         "quantiles_json": json.dumps(q), "pmf_json": json.dumps(pmf), "k": k})
    with conn:
        return db.upsert(conn, "live_predictions", rows)
