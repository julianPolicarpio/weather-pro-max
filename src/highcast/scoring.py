import json

from highcast.buckets import bucket_pair, pmf_from_buckets
from highcast.metric import get_metric
from highcast.timeutil import round_half_up

TAIL_CONF = 2 / 3   # model must back "≥T"/"≤T" at 2:1 odds to call the tail


def tail_floor(pmf: dict[int, float], conf: float = TAIL_CONF) -> int:
    """Highest integer T with sum(p for k>=T) >= conf. Always defined
    (P(>= min key) == 1 >= conf)."""
    best = min(pmf)
    for T in sorted(pmf):
        if sum(p for k, p in pmf.items() if k >= T) >= conf:
            best = T
    return best


def tail_cap(pmf: dict[int, float], conf: float = TAIL_CONF) -> int:
    """Lowest integer T with sum(p for k<=T) >= conf. Always defined
    (P(<= max key) == 1 >= conf). Mirror of tail_floor for metrics whose
    tail confidence runs from the bottom up (e.g. the low: "model backs
    low <= T at 2:1")."""
    best = max(pmf)
    for T in sorted(pmf, reverse=True):
        if sum(p for k, p in pmf.items() if k <= T) >= conf:
            best = T
    return best


def score_day(pmf: dict[int, float], point_f: float, actual: int,
              tail_dir: str = "floor") -> dict:
    """Score one prediction against the actual settled value for one metric.

    - point   = round_half_up(point_f)              # whole-degree point forecast
    - mode    = argmax bucket of the per-degree pmf # most likely single 1F
    - pick_lo, pick_hi = bucket_pair(mode)           # the 2F Kalshi-style bucket
      the model most backs
    - tail_dir="floor" (default, e.g. the high): tail_floor T = highest
      integer floor at which the model is at least 2:1 confident the value
      clears T (P(actual >= T) >= 2/3); tail_hit = actual >= T;
      tail_label = f"≥{T}"
    - tail_dir="cap" (e.g. the low): tail_cap T = lowest integer cap at
      which the model is at least 2:1 confident the value stays at or under
      T (P(actual <= T) >= 2/3); tail_hit = actual <= T; tail_label = f"≤{T}"
    """
    point = round_half_up(point_f)
    mode = max(pmf, key=pmf.get)
    pick_lo, pick_hi = bucket_pair(mode)
    abs_error = abs(point_f - actual)
    if tail_dir == "cap":
        T = tail_cap(pmf)
        tail_conf = sum(p for k, p in pmf.items() if k <= T)
        tail_label = f"≤{T}"
        tail_hit = actual <= T
    else:
        T = tail_floor(pmf)
        tail_conf = sum(p for k, p in pmf.items() if k >= T)
        tail_label = f"≥{T}"
        tail_hit = actual >= T
    return {
        "point": point,
        "mode": mode,
        "pick_lo": pick_lo,
        "pick_hi": pick_hi,
        "pick_label": f"{pick_lo}-{pick_hi}",
        "point_hit": point == actual,
        "within1": abs_error <= 1.0,
        "bucket_hit": pick_lo <= actual <= pick_hi,
        "abs_error": abs_error,
        "tail_floor": T,
        "tail_label": tail_label,
        "tail_conf": tail_conf,
        "tail_hit": tail_hit,
    }


def scorecard(conn, cfg, kind: str | None = None, metric: str = "high") -> list[dict]:
    """Score every settled (target_date, kind) -- i.e. every target_date that
    has a cli_reports row for the metric's target column -- against the
    LATEST prediction of that kind and metric (MAX(issued_at_utc) among
    predictions for that metric). kind=None scores both dayahead and
    intraday. metric selects the MetricSpec (e.g. "high" vs "low"), which
    fixes the settled column read as `actual` and the tail direction passed
    to score_day.

    spec.target_col is a fixed enum value off MetricSpec (never user input),
    so interpolating it into the SQL below is safe."""
    spec = get_metric(cfg, metric)
    kinds = [kind] if kind is not None else ["dayahead", "intraday"]
    icao = cfg.station.icao
    records = []
    for k in kinds:
        rows = conn.execute(
            f"""SELECT p.target_date, p.point_f, p.buckets,
                      c.{spec.target_col} AS actual
               FROM predictions p
               JOIN cli_reports c
                 ON c.station = p.station AND c.date = p.target_date
               WHERE p.station = ? AND p.kind = ? AND p.metric = ?
                 AND p.issued_at_utc = (
                     SELECT MAX(p2.issued_at_utc) FROM predictions p2
                     WHERE p2.station = p.station AND p2.target_date = p.target_date
                       AND p2.kind = p.kind AND p2.metric = ?
                 )
               """,
            (icao, k, metric, metric)).fetchall()
        for row in rows:
            pmf = pmf_from_buckets(json.loads(row["buckets"]))
            scored = score_day(pmf, row["point_f"], row["actual"], tail_dir=spec.tail_dir)
            records.append({
                "target_date": row["target_date"],
                "kind": k,
                "metric": metric,
                "actual": row["actual"],
                **scored,
            })
    records.sort(key=lambda r: (r["target_date"], r["kind"]))
    return records


def _rates(records: list[dict]) -> dict:
    n = len(records)
    if n == 0:
        return {"n": 0, "point_hit_rate": None, "within1_rate": None,
                "bucket_hit_rate": None, "tail_hit_rate": None, "mae": None}
    return {
        "n": n,
        "point_hit_rate": sum(r["point_hit"] for r in records) / n,
        "within1_rate": sum(r["within1"] for r in records) / n,
        "bucket_hit_rate": sum(r["bucket_hit"] for r in records) / n,
        "tail_hit_rate": sum(r["tail_hit"] for r in records) / n,
        "mae": sum(r["abs_error"] for r in records) / n,
    }


def summarize(records: list[dict]) -> dict:
    """Per-kind and overall {n, point_hit_rate, within1_rate, bucket_hit_rate,
    mae} rollups over scorecard() records."""
    by_kind: dict[str, list[dict]] = {}
    for r in records:
        by_kind.setdefault(r["kind"], []).append(r)
    out = {k: _rates(v) for k, v in by_kind.items()}
    out["overall"] = _rates(records)
    return out
