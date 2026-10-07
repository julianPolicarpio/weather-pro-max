"""PROBE: is the bound-regime reshape worth building at all?

Run BEFORE Tasks 2-4, not after. The plan put the walk-forward gate at Task 5,
which means writing three tasks of machinery before learning whether the effect
survives out of sample. Julian asked the right question; this answers it with
throwaway code so a NULL result costs an hour instead of a day.

Same pre-registration as the spec (§4):
    FIT   on Jun-Sep 2020-2024
    SCORE on Jun-Sep 2025-2026   (untouched during discovery)
    primary: mean log score of the six-bucket ladder, paired
    negative control: identical pipeline with the regime key SHUFFLED

The model's pmfs are regenerated with `backtest.walk_forward_at_hour`, which is
expanding-window (day i is scored by a model fitted only on days < i). Reading
the `predictions` table instead would be useless: it holds 5 KMIA LOW dates.
"""
import datetime as dt
import math
import random
import sys

sys.path.insert(0, "src")

from highcast import backtest, boundregime as br, db  # noqa: E402
from highcast.buckets import bucket_probs, pmf_mass  # noqa: E402
from highcast.config import load_config  # noqa: E402
from highcast.metric import get_metric  # noqa: E402

HOUR = 7
FIT_YEARS = range(2020, 2025)
SCORE_YEARS = (2025, 2026)
WEIGHT_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
MIN_CELL_N = br.MIN_CELL_N
LADDER = [(81, None), (79, 80), (77, 78), (75, 76), (73, 74), (None, 72)]
EPS = 1e-6
START = "2019-01-01"          # guidance coverage starts ~2020; 1y of warmup


def settled_index(low):
    for i, (lo, hi) in enumerate(LADDER):
        if (lo is None or low >= lo) and (hi is None or low <= hi):
            return i
    return None


def log_score(pmf, low):
    i = settled_index(low)
    if i is None or pmf is None:
        return None
    p = [max(pmf_mass(pmf, lo, hi) or 0.0, EPS) for lo, hi in LADDER]
    return math.log(p[i] / sum(p))


def observations(conn, cfg, days):
    """Task 2's function, inlined — the whole point of this probe is to decide
    whether Task 2 gets written. Uses only Task 1's shipped primitives."""
    out = []
    for day in days:
        if day.month not in br.SEASON_MONTHS:
            continue
        row = conn.execute(
            "SELECT low_f FROM cli_reports WHERE station=? AND date=?",
            (cfg.station.icao, day.isoformat())).fetchone()
        if row is None or row[0] is None:
            continue
        bound = br.bound_for(conn, cfg, day, HOUR)
        gmed = br.guidance_median(conn, cfg.station.icao, day)
        if bound is None or gmed is None:
            continue
        cell = br.regime_key(bound, bound - gmed)
        if cell is None:
            continue
        out.append({"day": day, "bound": bound, "cell": cell,
                    "delta": bound - float(row[0])})
    return out


def fit_table(obs):
    t = {}
    for o in obs:
        c = t.setdefault(o["cell"], {"n": 0, "hist": {}})
        d = round(float(o["delta"]), 1)
        c["n"] += 1
        c["hist"][d] = c["hist"].get(d, 0) + 1
    return t


def reshape(hist, bound):
    if not hist or bound is None:
        return None
    tot = sum(hist.values())
    pmf = {}
    for d, k in hist.items():
        t = int(round(bound - d))
        pmf[t] = pmf.get(t, 0.0) + k / tot
    return pmf


def blend(model, layer, w):
    if model is None or layer is None or w >= 1.0:
        return model
    keys = set(model) | set(layer)
    out = {k: w * model.get(k, 0.0) + (1 - w) * layer.get(k, 0.0) for k in keys}
    return out if abs(sum(out.values()) - 1.0) < 1e-6 else model


def bootstrap_p(diffs, n=10000, seed=20260801):
    rnd = random.Random(seed)
    k = len(diffs)
    if not k:
        return 1.0
    worse = sum(1 for _ in range(n)
                if sum(diffs[rnd.randrange(k)] for _ in range(k)) / k <= 0)
    return worse / n


def run(obs, pmfs, label):
    fit = [o for o in obs if o["day"].year in FIT_YEARS]
    score = [o for o in obs if o["day"].year in SCORE_YEARS]
    table = fit_table(fit)

    weights = {}
    for cell, e in table.items():
        if e["n"] < MIN_CELL_N:
            weights[cell] = 1.0
            continue
        days = [o for o in fit if o["cell"] == cell]
        best_w, best = 1.0, None
        for w in WEIGHT_GRID:
            tot = k = 0
            for o in days:
                mp = pmfs.get(o["day"].isoformat())
                if mp is None:
                    continue
                s = log_score(blend(mp, reshape(e["hist"], o["bound"]), w),
                              round(o["bound"] - o["delta"]))
                if s is not None:
                    tot += s
                    k += 1
            if k and (best is None or tot / k > best):
                best, best_w = tot / k, w
        weights[cell] = best_w

    pairs = []
    for o in score:
        mp = pmfs.get(o["day"].isoformat())
        if mp is None:
            continue
        low = round(o["bound"] - o["delta"])
        e = table.get(o["cell"])
        w = weights.get(o["cell"], 1.0) if e else 1.0
        lp = blend(mp, reshape(e["hist"], o["bound"]) if e else None, w)
        a, b = log_score(mp, low), log_score(lp, low)
        if a is not None and b is not None:
            pairs.append((a, b, mp, lp, low))

    diffs = [b - a for a, b, *_ in pairs]
    print(f"\n===== {label} =====")
    print(f"  fit days {len(fit)}   score days {len(score)}   scored pairs {len(pairs)}")
    print(f"  cells fitted: {sum(1 for c in table if table[c]['n'] >= MIN_CELL_N)}"
          f" of {len(table)}   weights != 1.0: "
          f"{sorted(c for c, w in weights.items() if w != 1.0)}")
    if not pairs:
        print("  NO PAIRS — cannot evaluate.")
        return
    mean_d = sum(diffs) / len(diffs)
    p = bootstrap_p(diffs)

    def hit(which):
        ok = 0
        for a, b, mp, lp, low in pairs:
            pmf = mp if which == "m" else lp
            pr = [max(pmf_mass(pmf, lo, hi) or 0.0, EPS) for lo, hi in LADDER]
            ok += pr.index(max(pr)) == settled_index(low)
        return ok / len(pairs)

    print(f"  mean log-score delta {mean_d:+.4f}   bootstrap p = {p:.4f}")
    print(f"  bucket hit   model {hit('m'):.1%}   layer {hit('l'):.1%}   "
          f"delta {(hit('l')-hit('m'))*100:+.1f} pt")
    print(f"  VERDICT: {'PASS' if (mean_d > 0 and p < 0.05 and len(pairs) >= 60) else 'FAIL'}"
          + ("" if len(pairs) >= 60 else f"  (n={len(pairs)} < 60, indicative only)"))


def main():
    cfg = load_config("config/kmia.yaml")
    conn = db.connect(cfg.db_path)
    days = [dt.date.fromisoformat(r[0]) for r in conn.execute(
        "SELECT date FROM cli_reports WHERE station='KMIA' AND low_f IS NOT NULL "
        "AND date >= ? ORDER BY date", (START,))]
    print(f"cli days from {START}: {len(days)}")

    obs = observations(conn, cfg, days)
    print(f"regime observations (Jun-Sep only): {len(obs)}")

    print("regenerating walk-forward model pmfs (this takes minutes)...")
    recs, _ = backtest.walk_forward_at_hour(
        conn, cfg, get_metric(cfg, "low"), [d.isoformat() for d in days],
        HOUR, "intraday", min_train=365, refit_every=30)
    pmfs = {r["target_date"]: bucket_probs(r["quantiles"]) for r in recs}
    print(f"walk-forward days with a pmf: {len(pmfs)}")

    run(obs, pmfs, "REAL")

    # DIAGNOSTIC: is the failure the fitted WEIGHT (overfit) or the empirical
    # distribution itself (does not transfer)? Sweep a FIXED weight on held-out
    # days — no fitting at all. w=1.0 is the model; w=0.0 is the pure layer.
    print("\n===== fixed-weight sweep on held-out days (no fitting) =====")
    fit_o = [o for o in obs if o["day"].year in FIT_YEARS]
    score_o = [o for o in obs if o["day"].year in SCORE_YEARS]
    table = fit_table(fit_o)
    for w in (0.0, 0.25, 0.5, 0.75, 1.0):
        tot = k = hits = 0
        for o in score_o:
            mp = pmfs.get(o["day"].isoformat())
            if mp is None:
                continue
            e = table.get(o["cell"])
            lp = blend(mp, reshape(e["hist"], o["bound"]) if e else None, w)
            low = round(o["bound"] - o["delta"])
            s = log_score(lp, low)
            if s is None:
                continue
            tot += s; k += 1
            pr = [max(pmf_mass(lp, lo, hi) or 0.0, EPS) for lo, hi in LADDER]
            hits += pr.index(max(pr)) == settled_index(low)
        print(f"  w={w:.2f} (model weight)  mean log score {tot/k:+.4f}   "
              f"hit {hits/k:.1%}   n={k}")

    rnd = random.Random(999)
    cells = [o["cell"] for o in obs]
    rnd.shuffle(cells)
    run([{**o, "cell": c} for o, c in zip(obs, cells)], pmfs,
        "NEGATIVE CONTROL (shuffled regime key)")


if __name__ == "__main__":
    main()
