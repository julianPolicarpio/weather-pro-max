import sqlite3

# Correct schemas whose PRIMARY KEY includes `metric`. Older DBs were created
# before the metric column and got it via ALTER TABLE, which cannot change a
# PK — leaving a metric-less PK. That is NOT harmless: high and low share the
# same (station, target_date, model, run_date) via the lead-1 forecast archive
# (and can share prediction keys too), so a metric-less PK lets one metric
# silently overwrite the other on upsert. These tables are rebuilt when their
# PK lacks `metric`.
_REBUILD = {
    "forecasts": (
        "CREATE TABLE forecasts (station TEXT NOT NULL, target_date TEXT NOT NULL, "
        "model TEXT NOT NULL, run_date TEXT NOT NULL, max_temp_f REAL, "
        "fetched_at TEXT NOT NULL, metric TEXT NOT NULL DEFAULT 'high', "
        "PRIMARY KEY (station, target_date, model, run_date, metric))",
        "station,target_date,model,run_date,max_temp_f,fetched_at,metric",
        "CREATE INDEX IF NOT EXISTS idx_fc_target ON forecasts (station, target_date);"),
    "predictions": (
        "CREATE TABLE predictions (station TEXT NOT NULL, target_date TEXT NOT NULL, "
        "issued_at_utc TEXT NOT NULL, kind TEXT NOT NULL, point_f REAL, "
        "quantiles TEXT, buckets TEXT, metric TEXT NOT NULL DEFAULT 'high', "
        "PRIMARY KEY (station, target_date, issued_at_utc, kind, metric))",
        "station,target_date,issued_at_utc,kind,point_f,quantiles,buckets,metric",
        ""),
}


def ensure_six_min_column(conn: sqlite3.Connection) -> None:
    """Idempotent migration adding obs.six_min_c (the ASOS 6-hour MINIMUM
    remark group, 2sTTT). The parser long read only the 6-hour MAXIMUM
    (1sTTT), so the LOW's ratchet saw hourly spot readings only and missed
    everything between them. Mirrors six_max_c exactly; NULL on old rows
    until `highcast reparse-obs` re-derives them from the stored raw METARs.
    Safe to call on every connect."""
    cols = {r[1] for r in conn.execute("PRAGMA table_info(obs)")}
    if "six_min_c" not in cols:
        conn.execute("ALTER TABLE obs ADD COLUMN six_min_c REAL")
        conn.commit()


def ensure_pressure_columns(conn: sqlite3.Connection) -> None:
    """Idempotent migration adding obs.slp_hpa / pres_tend_hpa / pres_tend_code.

    `obs` has carried wind direction, sky cover and dewpoint since the start but
    NO pressure column at all, even though sea-level pressure sits in ~85% of the
    raw METARs already stored and the 3-hour tendency group in ~28%. Existing rows
    are NULL until `highcast reparse-obs` re-derives them from that stored text --
    no network, same pattern as the six_min_c backfill.

    ⚠️ THESE COLUMNS ARE NOT MODEL FEATURES AND MUST NOT BECOME ONE while the
    model is frozen. They exist so the Station Table can display the physical
    state, and so the advection work has a five-year pressure series ready to
    measure the moment the freeze lifts. Adding a column changes no prediction;
    adding it to `features.FEATURES` or `INTRADAY_FEATURES` would.
    """
    cols = {r[1] for r in conn.execute("PRAGMA table_info(obs)")}
    for name, decl in (("slp_hpa", "REAL"), ("pres_tend_hpa", "REAL"),
                       ("pres_tend_code", "INTEGER")):
        if name not in cols:
            conn.execute(f"ALTER TABLE obs ADD COLUMN {name} {decl}")
    conn.commit()


def ensure_environment_columns(conn: sqlite3.Connection) -> None:
    """Idempotent migration adding the storm-adjacent fields from the raw METAR.

    THE GAP THESE CLOSE. The KMIA HIGH is convection-limited -- an afternoon
    thunderstorm sets the day's ceiling -- and the nowcast has NO precipitation
    input whatsoever. `cur_sky` is the only storm-adjacent feature, and it records
    the LAST sky token, i.e. the HIGHEST layer, so `BKN008 OVC200` is stored
    identically to `FEW200`. The low deck that actually caps insolation is thrown
    away on 82% of observations.

    Every field here was already sitting in text we store; recovering them needs a
    reparse, not a fetch:

        vis_sm         100% of reports   collapses in rain/fog
        sky layers      82%              ceiling_ft = lowest BKN/OVC
        wx_present      17%              literally spells TS / RA / BR

    ⚠️ SAME STANDING RULE AS THE PRESSURE COLUMNS: these are NOT model features
    and must not become one while the model is frozen. `skyc` is deliberately left
    untouched precisely because it IS one. Adding a column changes no prediction;
    adding it to `features.FEATURES` or `INTRADAY_FEATURES` would. The convection
    arms measured on 2026-07-29 lost on a data-volume tradeoff, not on physics --
    so these belong in the queue behind a pre-registered ship gate, not in the
    model on intuition.
    """
    cols = {r[1] for r in conn.execute("PRAGMA table_info(obs)")}
    for name, decl in (("vis_sm", "REAL"), ("ceiling_ft", "INTEGER"),
                       ("sky_lowest_ft", "INTEGER"), ("n_sky_layers", "INTEGER"), ("sky_cb", "INTEGER"),
                       ("wx_present", "TEXT"), ("wx_thunder", "INTEGER")):
        if name not in cols:
            conn.execute(f"ALTER TABLE obs ADD COLUMN {name} {decl}")
    conn.commit()


def _pk_cols(conn: sqlite3.Connection, table: str) -> list[str]:
    # PRAGMA table_info: row[5] is the 1-based position in the PK (0 if not PK).
    return [r[1] for r in conn.execute(f"PRAGMA table_info({table})") if r[5] > 0]


def ensure_metric_column(conn: sqlite3.Connection) -> None:
    """Idempotent migration to the metric-aware schema. First adds the `metric`
    column to any pre-metric DB, then repairs the PRIMARY KEY: any table whose
    PK does not include `metric` is rebuilt so high and low rows cannot collide.
    Safe to call on every connect."""
    for table in ("forecasts", "predictions"):
        cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
        if "metric" not in cols:
            conn.execute(
                f"ALTER TABLE {table} ADD COLUMN metric TEXT NOT NULL DEFAULT 'high'")
    conn.commit()
    for table, (ddl, collist, index_ddl) in _REBUILD.items():
        if "metric" not in _pk_cols(conn, table):
            conn.executescript(
                "PRAGMA foreign_keys=off;"
                f"ALTER TABLE {table} RENAME TO {table}__old;"
                f"{ddl};"
                f"INSERT OR IGNORE INTO {table} ({collist}) "
                f"SELECT {collist} FROM {table}__old;"
                f"DROP TABLE {table}__old;"
                f"{index_ddl}")
            conn.commit()


def ensure_app_bet_alert_columns(conn: sqlite3.Connection) -> None:
    """Idempotent migration adding app_bets.reason and app_bets.notified_utc.

    `notified_utc` is what makes Discord alerts exactly-once: it is written only
    AFTER a successful POST, so a bet is never announced twice and a failed send
    is retried on the next wake. Safe to call on every connect."""
    cols = {r[1] for r in conn.execute("PRAGMA table_info(app_bets)")}
    for col in ("reason", "notified_utc", "settle_notified_utc",
                "diverged_utc", "price_alert_utc",
                "model_fingerprint", "features_json"):
        if col not in cols:
            conn.execute(f"ALTER TABLE app_bets ADD COLUMN {col} TEXT")
    conn.commit()


def ensure_prediction_bound_gap(conn: sqlite3.Connection) -> None:
    """Idempotent migration adding `predictions.bound_gap`.

    THE GAP IS ALREADY COMPUTED at prediction time -- `build_checkpoint_frame`
    puts `cur_bound_gap` in the live frame -- and then thrown away. Storing it
    costs one float per prediction and is what lets the preflight grade a day
    without refitting a BaselineBlend on a dashboard request, which would be
    several seconds of work per page load for a number the wake cycle already
    had in hand.

    ⚠️ DISPLAY ONLY, AND IT MUST STAY THAT WAY. `preflight` reads this column to
    say "the model is less trustworthy than usual today"; nothing in the
    prediction, staking or commit path may read it. Writing a column changes no
    prediction. Feeding it back into `INTRADAY_FEATURES` would be a model
    change, and that hypothesis was already measured and REJECTED
    (`scripts/klax_low_gap_eval.py`: GAP1 +1.0%, CI [-1.1%,+3.1%]).

    Old rows stay NULL. Unlike the obs backfills there is no stored text to
    re-derive them from, so the preflight simply has no verdict for a day
    predicted before this shipped -- which it reports rather than guessing.
    """
    cols = {r[1] for r in conn.execute("PRAGMA table_info(predictions)")}
    if "bound_gap" not in cols:
        conn.execute("ALTER TABLE predictions ADD COLUMN bound_gap REAL")
    conn.commit()


def ensure_bot_orders_edge_column(conn: sqlite3.Connection) -> None:
    """Idempotent migration adding `bot_orders.edge` (calibrated - ask).

    Logged on EVERY order-bot decision, not only the ones `gating.MAX_EDGE`
    refuses -- that is the whole point. `gating.MAX_EDGE` is provisional
    because KMIA HIGH has market price stored on only a handful of days, far
    too few to tune a ceiling from how often large edges actually resolve in
    our favour. This column is what lets that measurement happen later:
    without it, the only edges on record are the ones `signals` logs from
    every intraday snapshot (mostly reads the bot never acted on), not the
    ones seen at the bot's own decision point. Old rows stay NULL; nothing
    backfills them, the same contract as every other `ensure_*` migration
    here. Safe to call on every connect."""
    cols = {r[1] for r in conn.execute("PRAGMA table_info(bot_orders)")}
    if "edge" not in cols:
        conn.execute("ALTER TABLE bot_orders ADD COLUMN edge REAL")
    conn.commit()


def ensure_paper_account_tables(conn: sqlite3.Connection) -> None:
    """Idempotent migration adding `paper_account_txns` — the PAPER DEMO
    ACCOUNT's own ledger table (`highcast/paper_account.py`).

    A BRAND-NEW TABLE, not a change to any existing one. `app_bets` is this
    project's only evidence (every conclusion on record rests on it — the
    edge/win correlation, the PLACE-vs-SKIP split, the 14W-6L count) and must
    stay byte-identical; this table exists so a replay of it can be persisted
    ALONGSIDE it, never in place of it. Every row here is fully re-derivable
    from `app_bets` (and, for the commit-hour measurement lens, `signals`) —
    `paper_account.refresh` deletes and reinserts a mode's rows on every call,
    which is safe only because nothing outside `paper_account.py` ever writes
    to this table.

    `mode` distinguishes the two REPLAYS this table stores side by side
    (`unit_replay` — fixed 1 contract/bet, the reconciliation instrument; and
    `daily_plan` — the compounding one-market-a-day accumulation plan). The
    third lens, commit-hour variants, is a pure re-aggregation of `signals`
    and is intentionally NOT persisted here — it would just be a duplicate
    copy of evidence `signals` already stores durably.
    """
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS paper_account_txns (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          mode TEXT NOT NULL,
          station TEXT NOT NULL, metric TEXT NOT NULL, target_date TEXT NOT NULL,
          txn_type TEXT NOT NULL,
          txn_utc TEXT,
          bucket TEXT, price REAL, contracts INTEGER NOT NULL DEFAULT 0,
          fee REAL NOT NULL DEFAULT 0.0, cash_delta REAL NOT NULL DEFAULT 0.0,
          balance_after REAL NOT NULL,
          rule TEXT, won INTEGER, reason TEXT);
        CREATE INDEX IF NOT EXISTS idx_paper_account_txns_mode
          ON paper_account_txns (mode, txn_utc);
        """)
    conn.commit()
