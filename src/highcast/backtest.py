import numpy as np
import pandas as pd

from highcast.buckets import bucket_probs
from highcast.metrics import crps_pmf


def walk_forward(df: pd.DataFrame, model_factory, min_train: int = 365,
                 refit_every: int = 30) -> list[dict]:
    df = df.sort_values("target_date").reset_index(drop=True)
    records: list[dict] = []
    model = None
    skipped_unfittable = 0
    for i in range(min_train, len(df)):
        if model is None or (i - min_train) % refit_every == 0:
            candidate = model_factory()
            try:
                candidate.fit(df.iloc[:i])
                model = candidate
            except ValueError:
                # Not enough usable history yet for this metric/model. The LOW's
                # forecast coverage starts years after its CLI record, so
                # `BaselineBlend` cannot fit the earliest windows and used to
                # crash the whole backtest. Keep the last good model, or wait.
                pass
        if model is None:
            skipped_unfittable += 1
            continue
        row = df.iloc[[i]]
        y = row["y"].iloc[0]
        if np.isnan(y):
            continue
        qs = model.predict_quantiles(row)[0]
        pmf = bucket_probs(qs)
        records.append({
            "target_date": row["target_date"].iloc[0], "y": int(y),
            "point": qs[0.5], "quantiles": qs,
            "crps": crps_pmf(pmf, int(y)), "abs_err": abs(qs[0.5] - y),
        })
    if skipped_unfittable:
        # never silently truncate coverage — say how many days were dropped
        print(f"# walk_forward: skipped {skipped_unfittable} day(s) with no "
              f"fittable model (insufficient history)")
    return records


def _coverage(records: list[dict], level: float) -> float:
    hits = [1.0 if r["y"] <= r["quantiles"][level] else 0.0 for r in records]
    return float(np.mean(hits)) if hits else float("nan")


def summarize(records: list[dict], df: pd.DataFrame) -> dict:
    by_date = df.set_index("target_date")
    # `fc_nws` is non-null on ~0.4% of the backtest window, so this "benchmark"
    # was a mean over a handful of days printed beside real metrics. Report the
    # supporting count alongside it so it can never be read as comparable again.
    nws_errs = ([abs(by_date.loc[r["target_date"], "fc_nws"] - r["y"])
                 for r in records] if "fc_nws" in df else [])
    nws_errs = [e for e in nws_errs if not np.isnan(e)]
    n_nws = len(nws_errs)
    mae_nws = float(np.mean(nws_errs)) if n_nws else float("nan")
    return {
        "n": len(records),
        "mae": float(np.mean([r["abs_err"] for r in records])),
        "crps": float(np.mean([r["crps"] for r in records])),
        "mae_nws": float(mae_nws), "n_nws": n_nws,
        "cov_10": _coverage(records, 0.1),
        "cov_50": _coverage(records, 0.5),
        "cov_90": _coverage(records, 0.9),
    }


def slice_metrics(records: list[dict], lo: float, hi: float) -> dict:
    """Metrics over records whose predicted point falls in [lo, hi)."""
    sub = [r for r in records if lo <= r["point"] < hi]
    if not sub:
        return {"n": 0, "mae": float("nan"), "crps": float("nan"),
                "cov_50": float("nan"), "cov_90": float("nan")}
    return {"n": len(sub),
            "mae": float(np.mean([r["abs_err"] for r in sub])),
            "crps": float(np.mean([r["crps"] for r in sub])),
            "cov_50": _coverage(sub, 0.5), "cov_90": _coverage(sub, 0.9)}


def build_hour_frame(conn, cfg, spec, dates, hour, baseline=None):
    """Checkpoint frame for a SINGLE station-local hour. Thin wrapper over
    models.intraday.build_checkpoint_frame(hours=(hour,))."""
    from highcast.models.intraday import build_checkpoint_frame
    return build_checkpoint_frame(conn, cfg, dates, hours=(hour,),
                                  baseline=baseline, metric=spec)


def walk_forward_at_hour(conn, cfg, spec, dates, hour, kind,
                         residual: bool = False, min_train: int = 365,
                         refit_every: int = 30):
    """Walk-forward the model `kind` ('intraday' | 'dayahead'), evaluated on
    the days for which a checkpoint row exists at `hour`.

    Returns (records, frame) — records is the same shape walk_forward returns."""
    if kind == "intraday":
        from highcast.models.intraday import IntradayModel
        frame = build_hour_frame(conn, cfg, spec, dates, hour)
        factory = lambda: IntradayModel(extremum=spec.extremum, residual=residual)
    elif kind == "dayahead":
        from highcast import features
        from highcast.models.dayahead import DayAheadModel, HybridDayAheadModel
        frame = features.build_features(conn, cfg, spec, dates)
        factory = HybridDayAheadModel if cfg.model.hybrid_anchor else DayAheadModel
    else:
        raise ValueError(f"unknown kind: {kind!r} (expected 'intraday' or 'dayahead')")
    records = walk_forward(frame, factory, min_train=min_train, refit_every=refit_every)
    return records, frame


def compare_at_hour(conn, cfg, spec, dates, hour, arms,
                    min_train: int = 365, refit_every: int = 30,
                    n_boot: int = 4000, seed: int = 0) -> dict:
    """Run every arm, restrict to the DAYS COMMON TO ALL ARMS (matched
    comparison), and paired-bootstrap arm[i] vs arm[0].

    `arms` is a list of dicts: {"label": str, "kind": str, "residual": bool}.
    """
    per_records: dict[str, list[dict]] = {}
    per_frame: dict[str, pd.DataFrame] = {}
    for arm in arms:
        recs, frame = walk_forward_at_hour(
            conn, cfg, spec, dates, hour, arm["kind"],
            residual=arm.get("residual", False),
            min_train=min_train, refit_every=refit_every)
        per_records[arm["label"]] = recs
        per_frame[arm["label"]] = frame

    date_sets = [set(r["target_date"] for r in per_records[a["label"]]) for a in arms]
    matched = set.intersection(*date_sets) if date_sets else set()
    matched_dates = sorted(matched)
    n_matched = len(matched_dates)

    arms_out: dict[str, dict] = {}
    matched_by_label: dict[str, dict[str, dict]] = {}
    for arm in arms:
        label = arm["label"]
        recs = per_records[label]
        full_n = len(recs)
        by_date = {r["target_date"]: r for r in recs}
        matched_recs = [by_date[d] for d in matched_dates]
        matched_by_label[label] = {r["target_date"]: r for r in matched_recs}
        s = summarize(matched_recs, per_frame[label])
        # never silently truncate coverage — say how many days each arm lost
        # when restricted to the days matched across every arm.
        s["dropped"] = full_n - len(matched_recs)
        arms_out[label] = s

    baseline_label = arms[0]["label"]
    baseline_by_date = matched_by_label[baseline_label]
    rng = np.random.default_rng(seed)
    vs_baseline: dict[str, dict] = {}
    for arm in arms:
        label = arm["label"]
        by_date = matched_by_label[label]
        mae_diff = np.array([by_date[d]["abs_err"] - baseline_by_date[d]["abs_err"]
                            for d in matched_dates])
        crps_diff = np.array([by_date[d]["crps"] - baseline_by_date[d]["crps"]
                              for d in matched_dates])
        if n_matched:
            dmae = float(np.mean(mae_diff))
            dcrps = float(np.mean(crps_diff))
            idx = rng.integers(0, n_matched, size=(n_boot, n_matched))
            mae_boot = mae_diff[idx].mean(axis=1)
            crps_boot = crps_diff[idx].mean(axis=1)
            dmae_ci = [float(np.percentile(mae_boot, 2.5)),
                      float(np.percentile(mae_boot, 97.5))]
            dcrps_ci = [float(np.percentile(crps_boot, 2.5)),
                       float(np.percentile(crps_boot, 97.5))]
        else:
            dmae = dcrps = float("nan")
            dmae_ci = [float("nan"), float("nan")]
            dcrps_ci = [float("nan"), float("nan")]
        vs_baseline[label] = {"dmae": dmae, "dmae_ci": dmae_ci,
                              "dcrps": dcrps, "dcrps_ci": dcrps_ci}

    return {"hour": hour, "metric": spec.key, "n_matched": n_matched,
            "arms": arms_out, "vs_baseline": vs_baseline}


def compare_calibration_modes(df, modes=("off", "width", "center_width"),
                              min_train: int = 365, refit_every: int = 30,
                              metric: str = "high") -> dict:
    """Walk-forward each calibration mode; report overall + seam(80-84) + <80.

    `metric` selects the feature columns via `features.model_features` — the
    trailing-bias features are LOW-only, so backtesting the LOW with the HIGH's
    column set would silently measure a model production does not run.
    """
    from highcast.features import model_features
    from highcast.models.dayahead import DayAheadModel
    cols = model_features(metric)
    out = {}
    for mode in modes:
        recs = walk_forward(df,
                            lambda m=mode: DayAheadModel(cal_mode=m,
                                                         feature_cols=cols),
                            min_train=min_train, refit_every=refit_every)
        out[mode] = {
            "overall": summarize(recs, df),
            "seam_80_84": slice_metrics(recs, 80.0, 84.0),
            "below_80": slice_metrics(recs, -999.0, 80.0),
        }
    return out
