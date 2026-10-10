"""The one-time test comparison (spec §4.6–4.7)."""
import json
import re
from pathlib import Path

import numpy as np

from citycast import gate_eval as ge
from highcast.orderbot import MIN_EDGE
from highcast.timeutil import utcnow_iso

CONTENDERS = ("B0", "B1", "B2", "L0", "L1")
UNTUNABLE = f"untunable (<{ge.MIN_BETS} validation bets)"
# every difference reported; the first four are "X beats never betting", the rest decide the winner
PAIRS = (("B1", "B0"), ("B2", "B0"), ("L0", "B0"), ("L1", "B0"),
         ("L1", "B1"), ("L1", "B2"), ("L1", "L0"), ("B2", "B1"))


PINNED_REF = re.compile(r"[\w.-]+/[\w.-]+@[0-9a-f]{7,40}")


def pinned_ref(ref) -> str:
    """A Laya model reference must name one exact revision: `repo@hexrev`. An unpinned `repo` or `repo@main`
    would resolve to whatever the repo holds later, so the scores could describe a different model."""
    if not isinstance(ref, str) or not PINNED_REF.fullmatch(ref):
        raise ValueError(f"model ref must be pinned as repo@revision (hex commit, 7-40 chars), got {ref!r}")
    return ref


def l1_key(ref: str) -> str:
    """The gate_scores contender key of the fine-tuned model: scores are cached per exact model revision."""
    return f"L1@{pinned_ref(ref)}"


def ensure_not_run(conn, force: bool = False) -> None:
    """Raise if the test split was already spent. Call it before any test scoring starts."""
    if not force and conn.execute("SELECT 1 FROM meta WHERE key='final_report_run_at'").fetchone():
        raise RuntimeError("final report already run; the test split is spent (use --force deliberately)")


def _winner(margin: dict, cmp: dict, brier: dict) -> str:
    """Spec §4.7 (revised 2026-10-10). A contender with no tunable margin never wins and is never compared."""
    def beats(a, b):
        return (a, b) in cmp and cmp[(a, b)][1] > 0
    eligible = {n for n in CONTENDERS if margin[n] is not None and beats(n, "B0")} - {"B0"}
    rivals = ["B1"] + [n for n in ("B2", "L0") if margin[n] is not None]
    if ("L1" in eligible and all(beats("L1", y) for y in rivals)
            and (margin["B2"] is None or brier["L1"] <= brier["B2"])):
        return "L1"
    if "B2" in eligible and beats("B2", "B1"):
        return "B2"
    return "B1" if "B1" in eligible else "B0"


def run_final(conn, scores: dict, report_path, force: bool = False, now_iso=utcnow_iso,
              l1_ref: str | None = None) -> dict:
    ensure_not_run(conn, force)
    s = ge.split(ge.load(conn))
    val, test = s["validation"], s["test"]
    p_test = {"B0": np.zeros(len(test)), "B1": test["p_model"].to_numpy()}
    margin = {"B0": 1.0, "B1": MIN_EDGE}
    for name in ("B2", "L0", "L1"):
        _, p_v, p_t = scores[name]
        if len(p_v) != len(val) or len(p_t) != len(test):
            raise ValueError(f"{name}: expected {len(val)} validation and {len(test)} test scores, "
                             f"got {len(p_v)} and {len(p_t)}")
        margin[name] = ge.tune_margin(val, p_v)
        p_test[name] = np.asarray(p_t, dtype=float)
    ppd = {n: ge.profit_per_decision(test, p_test[n], margin[n]) if margin[n] is not None else None
           for n in CONTENDERS}
    cmp = {(a, b): ge.compare(test, p_test[a], margin[a], p_test[b], margin[b])
           for a, b in PAIRS if margin[a] is not None and margin[b] is not None}
    brier = {n: ge.brier_score(test, p_test[n]) for n in ("B1", "B2", "L0", "L1")}
    winner = _winner(margin, cmp, brier)
    now = now_iso()

    lines = ["# Decision gate — final test report", "", f"Run at {now}. Test split: event dates "
             f"{ge.VAL_END} to {ge.TEST_END} (exclusive), {len(test)} rows, {test['ticker'].nunique()} tickers.", "",
             "| contender | margin | profit/decision | bets | Brier |", "|---|---|---|---|---|"]
    for n in CONTENDERS:
        if margin[n] is None:
            lines.append(f"| {n} | {UNTUNABLE} | — | — | — |")
        else:
            b = f"{brier[n]:.4f}" if n in brier else "—"
            lines.append(f"| {n} | {margin[n]:.2f} | {ppd[n][0]:+.4f} | {ppd[n][1]} | {b} |")
    lines += ["", "| comparison | mean diff | 95% CI |", "|---|---|---|"]
    for a, b in PAIRS:
        if (a, b) in cmp:
            m, lo, hi = cmp[(a, b)]
            lines.append(f"| {a}-{b} | {m:+.4f} | [{lo:+.4f}, {hi:+.4f}] |")
        else:
            lines.append(f"| {a}-{b} | — | — |")
    lines += ["", f"**Winner: {winner}** (rule: spec §4.7)."]
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text("\n".join(lines) + "\n")

    choice = {"winner": winner, "margin": None if winner == "B0" else margin[winner],
              "model_ref": l1_ref if winner == "L1" else None, "decided_at": now}
    with conn:
        conn.execute("INSERT OR REPLACE INTO meta VALUES ('gate_choice', ?)", (json.dumps(choice),))
        conn.execute("INSERT OR REPLACE INTO meta VALUES ('final_report_run_at', ?)", (now,))
    return {**choice, "ppd": ppd, "compare": {f"{a}-{b}": v for (a, b), v in cmp.items()}, "brier": brier,
            "margins": margin}
