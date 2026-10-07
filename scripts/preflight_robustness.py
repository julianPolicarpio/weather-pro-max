"""Three ways the preflight result could be fake, tested before anything ships.

`preflight_eval.py` found that on KLAX LOW the bound-vs-guidance gap separates
days the model is honest about from days it is confidently wrong (Dshortfall
+8.3%, CI [+1.5%,+15.5%]). That is one pre-registered test clearing one gate. It
is not yet a reason to put a verdict on the dashboard.

CHECK 1 -- IS IT JUST "THE MODEL ALREADY SAID SO"?  THE ONE THAT MATTERS.
Flagged days carry a LOWER stated probability (60.3% vs 66.1%), so bound_gap and
p_model are correlated. If the whole effect is that low-confidence days go wrong,
then the honest advice is the trivial "don't trade below 62%" and the gap adds
NOTHING -- a preflight built on it would be an elaborate restatement of a number
already on screen. Test: bin days by stated p_model, then compare shortfall
across the gap split WITHIN each bin. The effect must survive inside bins.

CHECK 2 -- DOES IT SURVIVE OUT OF SAMPLE IN TIME?
The median threshold was computed on the same days it was scored on. Refit the
threshold on the first half of the test period and apply it, untouched, to the
second. An effect that only exists in-sample is a description of 2025, not a
rule for tomorrow.

CHECK 3 -- IS THE THRESHOLD STABLE?
If the first-half and second-half medians differ a lot, any fixed cut shipped
today is already stale. Reported so the operating point can be chosen honestly.

None of these can promote a market that failed the primary gate. They can only
demote one that passed.
"""
import numpy as np

from preflight_eval import (MARKETS, PRIMARY, RECENT, boot_diff, collect)

CACHE = ".preflight_cache"
rng = np.random.default_rng(20260801)


def arrays(res):
    recs = [r for r in res["recs"]
            if r.get(PRIMARY) is not None and not np.isnan(r[PRIMARY])]
    recs.sort(key=lambda r: r["date"])
    return (np.array([r[PRIMARY] for r in recs], dtype=float),
            np.array([r["p"] - r["hit"] for r in recs], dtype=float),
            np.array([r["p"] for r in recs], dtype=float),
            np.array([r["hit"] for r in recs], dtype=float),
            [r["date"] for r in recs])


def check_conditional(f, short, p, hit):
    print("  CHECK 1 - within stated-confidence bins (the theatre test)")
    edges = np.quantile(p, [0, 1 / 3, 2 / 3, 1.0])
    med = float(np.median(f))
    survived = 0
    tested = 0
    for i in range(3):
        m = (p >= edges[i]) & (p <= edges[i + 1]) if i == 2 else \
            (p >= edges[i]) & (p < edges[i + 1])
        if m.sum() < 60:
            print(f"      bin {i + 1} stated {edges[i]:.0%}-{edges[i+1]:.0%}: "
                  f"n={m.sum()} too few")
            continue
        hi = m & (f > med)
        lo = m & (f <= med)
        if hi.sum() < 25 or lo.sum() < 25:
            print(f"      bin {i + 1} stated {edges[i]:.0%}-{edges[i+1]:.0%}: "
                  f"split too lopsided ({hi.sum()}/{lo.sum()})")
            continue
        tested += 1
        d = short[hi].mean() - short[lo].mean()
        survived += d > 0
        print(f"      bin {i + 1} stated {edges[i]:.0%}-{edges[i+1]:.0%}  "
              f"n={m.sum():3d}   flagged hit {hit[hi].mean():.1%} "
              f"(n={hi.sum():3d})   clean hit {hit[lo].mean():.1%} "
              f"(n={lo.sum():3d})   Dshortfall {d:+.1%}")
    if tested:
        print(f"      -> effect points the right way in {survived}/{tested} "
              f"bins{'  (SURVIVES)' if survived == tested else '  (WEAKENED)'}")


def check_temporal(f, short, p, hit, dates):
    print("  CHECK 2 - threshold fit on first half, applied to second")
    half = len(f) // 2
    thr = float(np.median(f[:half]))
    f2, s2, h2 = f[half:], short[half:], hit[half:]
    hi, lo = f2 > thr, f2 <= thr
    if hi.sum() < 25 or lo.sum() < 25:
        print(f"      out-of-sample split too lopsided ({hi.sum()}/{lo.sum()})")
        return
    d = s2[hi].mean() - s2[lo].mean()
    d_lo, d_hi = boot_diff(s2[hi], s2[lo])
    print(f"      threshold {thr:.2f} from {dates[0]}..{dates[half-1]}, "
          f"tested on {dates[half]}..{dates[-1]}")
    print(f"      flagged n={hi.sum():3d} hit {h2[hi].mean():.1%}   "
          f"clean n={lo.sum():3d} hit {h2[lo].mean():.1%}   "
          f"Dshortfall {d:+.1%}  CI [{d_lo:+.1%},{d_hi:+.1%}] (95%)")
    print(f"      -> {'HOLDS out of sample' if d > 0 else 'DOES NOT HOLD'}")


def operating_points(f, short, p, hit):
    """The numbers a shipped threshold would actually promise, per quartile.

    The eval printed shortfall by quartile; a dashboard verdict also has to state
    the HIT RATE it is claiming, because that is what sets whether a price is
    worth paying. Descriptive: the significance test already ran on the
    pre-registered median, and no cut is chosen by scanning these.
    """
    print("  OPERATING POINTS - what each cut would promise")
    qs = np.quantile(f, [0.25, 0.5, 0.75])
    edges = [-np.inf, *qs, np.inf]
    print(f"      {'quartile':10}{'gap range':>16}{'n':>6}{'stated':>9}"
          f"{'hit':>8}{'shortfall':>11}")
    for i in range(4):
        m = (f > edges[i]) & (f <= edges[i + 1])
        if m.sum() < 15:
            continue
        rng_txt = (f"<= {edges[i+1]:.2f}" if i == 0 else
                   f"> {edges[i]:.2f}" if i == 3 else
                   f"{edges[i]:.2f}-{edges[i+1]:.2f}")
        print(f"      Q{i + 1:<9}{rng_txt:>16}{m.sum():>6}{p[m].mean():>9.1%}"
              f"{hit[m].mean():>8.1%}{short[m].mean():>+11.1%}")
    # The two cuts worth considering, stated as the promise each makes.
    for name, thr in (("median", qs[1]), ("Q3", qs[2])):
        keep = f <= thr
        print(f"      trade only when gap <= {thr:.2f} ({name}): "
              f"{keep.mean():.0%} of days kept, hit {hit[keep].mean():.1%}, "
              f"shortfall {short[keep].mean():+.1%}")


def check_stability(f, dates):
    half = len(f) // 2
    a, b = float(np.median(f[:half])), float(np.median(f[half:]))
    q3a, q3b = float(np.quantile(f[:half], .75)), float(np.quantile(f[half:], .75))
    print(f"  CHECK 3 - threshold stability   median {a:.2f} -> {b:.2f}   "
          f"Q3 {q3a:.2f} -> {q3b:.2f}")


if __name__ == "__main__":
    print(f"PREFLIGHT ROBUSTNESS on `{PRIMARY}`, test >= {RECENT}")
    for cfg_path, metric_key in MARKETS:
        res = collect(cfg_path, metric_key, cache_dir=CACHE)
        f, short, p, hit, dates = arrays(res)
        print(f"\n### {res['icao']} {res['metric'].upper()} h{res['hour']}  n={len(f)}")
        if len(f) < 120:
            print("  too few days")
            continue
        check_conditional(f, short, p, hit)
        check_temporal(f, short, p, hit, dates)
        check_stability(f, dates)
        operating_points(f, short, p, hit)
