import sqlite3
from pathlib import Path

# ── PREDICTION KINDS ────────────────────────────────────────────────────────
#
# The evening run is a DIFFERENT OBJECT from the 09:00 day-ahead and gets its
# own kind so nothing can conflate them. Lives here, beside the schema, because
# it is a stored column VALUE -- and because `web/app.py` needs it too and
# importing `cli` from there would risk a cycle.
#
# WHY, measured 2026-08-01 on stored predictions vs settled CLI:
#
#     KLAX HIGH   evening (morn_* all NaN)   n=19   MAE 4.42F   bias -4.05F
#                 morning refresh            n=14   MAE 1.45F   bias +0.19F
#
# A train/serve skew, not a bad model. The HIGH's feature cutoff is 09:00 LOCAL
# ON THE TARGET DAY; the evening run fires ~17:00 the PREVIOUS day, so every
# `morn_*` observation it reads is ~16h in the future and comes back NaN.
# `features.build_features` takes no as-of parameter, so a BACKTEST always
# reconstructs real morning values -- meaning the walk-forward behind
# `gating.preferred_prediction` measured the MORNING object exclusively and
# never saw this one.
#
# 38 of 74 stored day-ahead rows were evening rows. Sharing a kind meant a
# 4F-cold forecast was served as the dashboard headline every morning until
# 09:00, with a 69% "edge" computed from it.
#
# Design: an internal design note
KIND_DAYAHEAD = "dayahead"
KIND_INTRADAY = "intraday"
KIND_EVENING = "evening"

SCHEMA = """
CREATE TABLE IF NOT EXISTS cli_reports (
  station TEXT NOT NULL, date TEXT NOT NULL, high_f INTEGER, low_f INTEGER,
  raw TEXT, fetched_at TEXT NOT NULL,
  PRIMARY KEY (station, date));
CREATE TABLE IF NOT EXISTS obs (
  station TEXT NOT NULL, valid_utc TEXT NOT NULL, source TEXT NOT NULL,
  tmpf REAL, tmpf_precise REAL, dwpf REAL, sknt REAL, drct REAL, skyc TEXT,
  six_max_c REAL, six_min_c REAL, raw TEXT, fetched_at TEXT NOT NULL,
  PRIMARY KEY (station, valid_utc, source));
-- ⚠️ DELIBERATELY NOT `obs`. The IEM 1-minute ASOS archive belongs in its own
-- table, and putting it in `obs` would be a silent train/serve skew.
--
-- `latentmax.visible_running_extremum` does NOT filter by source -- it reads
-- `asos1` rows on purpose (latentmax.py, the `syn5`/`asos1` branch). Its output
-- feeds `run_lo`, `run_hi` and `run_width`, which ARE shipped model features.
--
-- The archive lags 24-36h, so it can only ever cover TRAINING rows. Land it in
-- `obs` and every training row gets a running-extremum interval computed from
-- 1-minute data (tight, ~0.3F wide) while every LIVE row still gets one from
-- hourly METARs (~2F wide). The model would learn to trust a precision that
-- does not exist at decision time -- over-trusting the ratchet exactly when it
-- is loosest, which is the dangerous direction.
--
-- The model's own point features are safe either way: `models/intraday.py` and
-- `features.py` all pin `source='metar'`. It is only the running extremum that
-- is exposed, and that is enough.
--
-- MOVE THIS INTO `obs` ONLY WHEN A LIVE SUB-HOURLY FEED EXISTS (QUEUE item 13 --
-- `fetchers/fivemin.py` is already wired into the wake cycle and needs only a
-- `synoptic_token`), so both eras carry comparable precision.
CREATE TABLE IF NOT EXISTS obs_1min (
  station TEXT NOT NULL, valid_utc TEXT NOT NULL, tmpf REAL NOT NULL,
  fetched_at TEXT NOT NULL,
  PRIMARY KEY (station, valid_utc));
CREATE TABLE IF NOT EXISTS forecasts (
  station TEXT NOT NULL, target_date TEXT NOT NULL, model TEXT NOT NULL,
  run_date TEXT NOT NULL, max_temp_f REAL, fetched_at TEXT NOT NULL,
  metric TEXT NOT NULL DEFAULT 'high',
  PRIMARY KEY (station, target_date, model, run_date, metric));
CREATE TABLE IF NOT EXISTS forecast_ens (
  station TEXT NOT NULL, target_date TEXT NOT NULL, run_date TEXT NOT NULL,
  metric TEXT NOT NULL DEFAULT 'high', n_members INTEGER,
  ens_mean REAL, ens_std REAL, ens_p10 REAL, ens_p50 REAL, ens_p90 REAL,
  ens_frac_ge_82 REAL, ens_bimodality REAL, fetched_at TEXT NOT NULL,
  PRIMARY KEY (station, target_date, run_date, metric));
-- Day-ahead FORECAST convection: precip probability, QPF and CAPE per hour.
-- LEAK-FREE BY CONSTRUCTION: Open-Meteo's `*_previous_day1` fields are what the
-- model predicted ONE DAY AHEAD, so they are genuinely available at the
-- decision hour -- unlike observed precipitation, which is hindsight.
--
-- Was created ad hoc by `scripts/fetch_convection.py` and never added here, so
-- a fresh database had no such table. Moved into the canonical schema
-- 2026-08-07 when the fetch was wired into the wake cycle.
CREATE TABLE IF NOT EXISTS forecast_conv (
  station TEXT NOT NULL, valid_utc TEXT NOT NULL, model TEXT NOT NULL,
  pop REAL, qpf REAL, cape REAL, fetched_at TEXT NOT NULL,
  PRIMARY KEY (station, valid_utc, model));
CREATE TABLE IF NOT EXISTS forecast_aloft (
  station TEXT NOT NULL, target_date TEXT NOT NULL, run_date TEXT NOT NULL,
  metric TEXT NOT NULL DEFAULT 'high', t850 REAL, t2m REAL, inversion REAL,
  fetched_at TEXT NOT NULL,
  PRIMARY KEY (station, target_date, run_date, metric));
-- NBM/NBS bulletin fields OTHER than temperature, aggregated over the window in
-- which each metric's extremum forms. These are the marine-layer / burn-off
-- physics for a coastal station: how much sun (`sol`), how much cloud (`sky`),
-- how low the deck sits (`cig`, `lcb`), and whether the flow is onshore or
-- offshore (`wdr`). `tsd` is NBM's own forecast-uncertainty estimate.
-- Populated from the same IEM bulletin archive as `nbm_nbs`, back to 2021.
CREATE TABLE IF NOT EXISTS forecast_nbs (
  station TEXT NOT NULL, target_date TEXT NOT NULL, run_date TEXT NOT NULL,
  metric TEXT NOT NULL DEFAULT 'high',
  nbs_sol_max REAL, nbs_sol_mean REAL, nbs_sky_mean REAL, nbs_cig_min REAL,
  nbs_lcb_min REAL, nbs_offshore REAL, nbs_wsp_mean REAL, nbs_tsd_mean REAL,
  nbs_dpt_mean REAL, nbs_n INTEGER, fetched_at TEXT NOT NULL,
  PRIMARY KEY (station, target_date, run_date, metric));
-- The Weather Company hourly observations. A SEPARATE TRUTH SOURCE, kept apart
-- from `obs` on purpose. The hourly Kalshi market (KXTEMPLAXH) resolves against
-- TWC for coordinates KLAX, explicitly NOT the NWS -- and its strikes sit 1 F
-- apart, so any systematic TWC-vs-METAR difference is not a rounding detail, it
-- is the whole bet. Never merge these into `obs`; the entire point is to be able
-- to measure the gap between them.
CREATE TABLE IF NOT EXISTS twc_obs (
  station TEXT NOT NULL, valid_utc TEXT NOT NULL, tempf REAL NOT NULL,
  source TEXT NOT NULL DEFAULT 'manual', fetched_at TEXT NOT NULL,
  PRIMARY KEY (station, valid_utc));
-- THE APP'S OWN COMMITTED BET. One row per (station, metric, target_date), and
-- it is IMMUTABLE — written with db.upsert_ignore so the first commit wins. See
-- highcast/appbet.py for the four integrity rules; the short version is that a
-- bet which can be rewritten later is not a prediction. Columns mirror Julian's
-- spreadsheet so the app's ledger and his can be compared column for column.
CREATE TABLE IF NOT EXISTS app_bets (
  station TEXT NOT NULL, metric TEXT NOT NULL, target_date TEXT NOT NULL,
  placed_utc TEXT NOT NULL, placed_manila TEXT NOT NULL,
  placed_local_hour INTEGER NOT NULL,
  bucket TEXT NOT NULL, p_model REAL, price REAL NOT NULL,
  stake REAL NOT NULL, payout_if_win REAL NOT NULL,
  actual INTEGER, won INTEGER, settled_utc TEXT,
  reason TEXT, notified_utc TEXT, settle_notified_utc TEXT,
  diverged_utc TEXT, price_alert_utc TEXT, model_fingerprint TEXT,
  features_json TEXT,
  PRIMARY KEY (station, metric, target_date));
-- EXACTLY-ONCE STAMPS FOR ALERTS THAT HAVE NO ROW TO STAMP. The bet alerts hang
-- their stamps on `app_bets` columns, which only works once a bet exists. The
-- LOW preview fires at 17:00 and 18:00 Manila — BEFORE the 19:00 commit — so
-- there is no row yet and the stamp needs its own home. `kind` is the alert
-- type, `key` identifies the single occurrence (target date + hour), and the
-- primary key is what makes the 5-minute wake fire it once instead of twelve
-- times. Written with upsert_ignore, so re-sending is a no-op rather than an
-- error.
-- INTRADAY GUIDANCE CAPTURE (2026-07-31). Forward-accumulation only.
--
-- WHAT PROBLEM THIS EXISTS TO SOLVE. At the HIGH's h11 commit the guidance the
-- model uses is 11.1 HOURS OLD: measured for target 2026-07-30, every NWP channel
-- (ecmwf, gfs, icon, hrrr, nbm, nam, aifs, nws) was last fetched at 23:55 the
-- previous night. HRRR in particular is a 3km model built for same-day forecasting
-- that refreshes HOURLY, so fresher runs almost certainly exist at commit -- and
-- nowcast.trade weights HRRR primary for exactly this reason.
--
-- WHY IT CANNOT BE TESTED WITHOUT THIS TABLE. `forecasts` DOES contain same-day
-- runs, but every one was downloaded at 06:55Z the NEXT morning, after its target
-- day ended. Using them to ask "would fresh guidance have helped" reads the answer
-- sheet. It is the same trap HANDOFF documents about the archive-api precipitation
-- data, which serves OBSERVED rather than forecast values. There is no honest
-- historical arm and none can be reconstructed: freshness is a property of WHEN a
-- row was fetched, and that moment is gone once it passes.
--
-- ⚠️ DELIBERATELY A SEPARATE TABLE, not extra rows in `forecasts`. The LOW's
-- feature cutoff is 20:00 local the PREVIOUS day, so a new midday fetch landing in
-- `forecasts` would fall before the next target's LOW cutoff and silently change
-- `base_p50` for the FROZEN markets. Writing here instead means the frozen
-- markets' inputs cannot move: nothing reads this table.
--
-- `fetched_at` IS THE POINT. It is what makes a later leak-free arm possible --
-- "what did we actually know at 11:00?" -- so it must be the true wall-clock
-- instant of the fetch, never backfilled or copied from another row.
CREATE TABLE IF NOT EXISTS forecast_intraday (
  station TEXT NOT NULL, target_date TEXT NOT NULL, model TEXT NOT NULL,
  metric TEXT NOT NULL, fetched_at TEXT NOT NULL,
  run_date TEXT NOT NULL, max_temp_f REAL,
  PRIMARY KEY (station, target_date, model, metric, fetched_at));
CREATE INDEX IF NOT EXISTS idx_fcintra_target
  ON forecast_intraday (station, target_date, metric);

-- HOURLY TEMPERATURE GUIDANCE. Capture-only, and it exists because the single
-- biggest MEASURED error source in this project has no feature behind it.
--
-- On 11.0% of KLAX days the climate-day LOW lands at h18 or later -- the SAME
-- EVENING, seventeen hours after the h4 commit -- because the climate day runs
-- 01:00->01:00 local and therefore contains two cooling periods. At h4 the model
-- sees a finished dawn minimum and has NO signal about the evening. On
-- 2026-07-30 the guidance had it nearly exact (67.52 vs a settled 67) and the
-- model overrode it, then lost.
--
-- Every existing guidance table stores a DAILY min/max, which cannot express
-- "tonight will be colder than this morning was". This stores the per-hour
-- temperature curve each model is predicting, so an evening-minimum feature can
-- eventually be built and gated.
--
-- ⚠️ LIKE `forecast_intraday`, THIS CAN NEVER BE BACKFILLED. What a model
-- predicted for 20:00, as of 04:00, is only knowable if it was recorded at
-- 04:00. Keyed by `fetched_at` so each capture is its own row rather than
-- overwriting the last -- otherwise the freshness question is unanswerable.
--
-- Nothing reads this. A feature built on it is a MODEL change and needs a
-- pre-registered gate first.
CREATE TABLE IF NOT EXISTS forecast_hourly (
  station TEXT NOT NULL, valid_utc TEXT NOT NULL, model TEXT NOT NULL,
  fetched_at TEXT NOT NULL, temp_f REAL,
  PRIMARY KEY (station, valid_utc, model, fetched_at));
CREATE INDEX IF NOT EXISTS idx_fchourly_valid
  ON forecast_hourly (station, valid_utc);

-- The order bot's daily decision -- what it did, or WOULD have done in dry-run.
--
-- PRIMARY KEY (station, metric, target_date) + INSERT OR IGNORE gives
-- idempotency for free: the wake cycle fires every 5 minutes through the whole
-- commit hour, and only the FIRST decision is kept. Same principle as
-- `app_bets`: a decision that keeps being rewritten as the model changes its
-- mind is not a decision, it is hindsight.
--
-- Refusals are recorded too, not just orders. Two weeks of "why not" is the
-- evidence the arming decision rests on.
CREATE TABLE IF NOT EXISTS bot_orders (
  station TEXT NOT NULL, metric TEXT NOT NULL, target_date TEXT NOT NULL,
  decided_utc TEXT NOT NULL, local_hour INTEGER,
  place INTEGER NOT NULL, reason TEXT,
  bucket TEXT, limit_price REAL, contracts INTEGER,
  calibrated REAL, ask REAL,
  dry_run INTEGER NOT NULL, env TEXT, placed INTEGER, response TEXT,
  PRIMARY KEY (station, metric, target_date));

CREATE TABLE IF NOT EXISTS alert_stamps (
  station TEXT NOT NULL, kind TEXT NOT NULL, key TEXT NOT NULL,
  sent_utc TEXT NOT NULL,
  PRIMARY KEY (station, kind, key));
-- PICK OF THE DAY — every verdict, including the declines.
--
-- Logging the NO PICKs is the point, not bookkeeping. A bar that only records
-- what it published can never be shown to be TOO STRICT, and on the only sample
-- that exists this one declined three bets that won. `verdict` therefore carries
-- UNSCORED / UNBUYABLE / NO PICK (price) / NO PICK / NO PICK (no price) / PICK,
-- and `reason` carries the arithmetic in words.
--
-- `score_with_stability` AND `score_without` are both stored so the provisional
-- churn term measures itself: after enough picks we can ask retroactively whether
-- it improved selection, rather than leaving it an open question forever. Julian
-- chose to score stability provisionally knowing it is unvalidated; this is what
-- makes that choice falsifiable instead of permanent.
--
-- `kind` separates the PROVISIONAL preview (fired before the commit hour, on the
-- latest available read) from the definitive VERDICT at the commit hour. They can
-- legitimately disagree, and the message says so rather than silently replacing.
--
-- Written via upsert_ignore: first write wins, the same anti-cheat contract as
-- app_bets. This table never gates a bet and never writes to app_bets.
CREATE TABLE IF NOT EXISTS picks (
  station TEXT NOT NULL, metric TEXT NOT NULL, target_date TEXT NOT NULL,
  kind TEXT NOT NULL, issued_utc TEXT NOT NULL, local_hour INTEGER,
  commit_hour INTEGER,
  verdict TEXT NOT NULL, bucket TEXT, p_model REAL, p_hat REAL, price REAL,
  contracts INTEGER, fee REAL, ev REAL,
  score_with_stability REAL, score_without REAL, churn INTEGER,
  stall_hours REAL, band_note TEXT, reason TEXT,
  actual INTEGER, won INTEGER, settled_utc TEXT,
  PRIMARY KEY (station, metric, target_date, kind));
-- EVERY ALERT SENT, WITH THE KALSHI LADDER AS IT STOOD AT THAT MOMENT.
-- `alert_stamps` answers "was this already sent" (exactly-once); this answers
-- "what did we tell him, and what was the market charging when we told him".
-- Those are different questions and conflating them loses the second one.
--
-- WHY THE MARKET SNAPSHOT IS THE POINT. A week of alerts saying "68-69, model
-- 74%" is unanalysable on its own -- being right 74% of the time earns nothing
-- if Kalshi was already at 80. `kalshi_pct` is the price on the ALERTED bucket
-- at send time, and `market_json` keeps the whole ladder so the counterfactual
-- ("what would the market's own favourite have done") stays answerable later.
--
-- Append-only, one row per alert actually delivered. Never a substitute for
-- `signals` (which samples every 5 min regardless of alerts) -- this is the
-- human-facing record: exactly what Julian was told and when.
CREATE TABLE IF NOT EXISTS alert_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  station TEXT NOT NULL, metric TEXT NOT NULL, target_date TEXT NOT NULL,
  kind TEXT NOT NULL, sent_utc TEXT NOT NULL, local_hour INTEGER,
  bucket TEXT, p_model REAL, kalshi_pct REAL, edge REAL,
  market_json TEXT);
CREATE INDEX IF NOT EXISTS idx_alert_log ON alert_log (station, target_date, kind);
CREATE TABLE IF NOT EXISTS alerts (
  station TEXT NOT NULL, event TEXT NOT NULL, onset_utc TEXT NOT NULL,
  expires_utc TEXT, fetched_at TEXT NOT NULL,
  PRIMARY KEY (station, event, onset_utc));
CREATE TABLE IF NOT EXISTS predictions (
  station TEXT NOT NULL, target_date TEXT NOT NULL, issued_at_utc TEXT NOT NULL,
  kind TEXT NOT NULL, point_f REAL, quantiles TEXT, buckets TEXT,
  metric TEXT NOT NULL DEFAULT 'high',
  PRIMARY KEY (station, target_date, issued_at_utc, kind, metric));
CREATE TABLE IF NOT EXISTS signals (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  station TEXT NOT NULL, target_date TEXT NOT NULL, metric TEXT NOT NULL,
  kind TEXT NOT NULL, created_utc TEXT NOT NULL,
  committed INTEGER NOT NULL, within_window INTEGER NOT NULL,
  place INTEGER NOT NULL, bucket TEXT, p_model REAL, price REAL, edge REAL,
  stake REAL, contracts INTEGER, market_json TEXT, reason TEXT,
  actual INTEGER, won INTEGER, pnl_unit REAL, pnl REAL, settled_utc TEXT);
CREATE INDEX IF NOT EXISTS idx_signals_td ON signals (station, target_date, metric);
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
CREATE TABLE IF NOT EXISTS notes (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  station TEXT NOT NULL, target_date TEXT NOT NULL,
  created_utc TEXT NOT NULL, sky TEXT NOT NULL, context TEXT);
CREATE INDEX IF NOT EXISTS idx_obs_valid ON obs (station, valid_utc);
CREATE INDEX IF NOT EXISTS idx_fc_target ON forecasts (station, target_date);
"""


# Rollback-journal mode: a commit must wait for every other connection's read
# to finish, and while it waits, new reads elsewhere are refused. Two wake jobs
# and the dashboard share this file, and single dashboard requests read for up
# to ~2.5 s on the 1.1 GB DB, so the old 4 s limit expired often enough to log
# "database is locked" (18 tracebacks in data/launchd*.err, reproduced
# 2026-10-04 in tests/test_db_busy_timeout.py). Every caller is a background
# job or a local page, so waiting is cheap and failing loses a cycle.
# WAL would remove reader/writer blocking entirely, but scripts here back up
# and copy the DB with plain file copies, which silently miss un-checkpointed
# WAL data — so the journal mode stays as it is.
BUSY_TIMEOUT_MS = 30_000


def connect(db_path: str | Path, *, migrate: bool = True) -> sqlite3.Connection:
    """Open the database. With `migrate=True` (the default) this also builds
    and upgrades the schema, which is what the wake cycle, the CLI and the
    tests all rely on.

    `migrate=False` OPENS AND RETURNS — no `executescript`, no `ensure_*`.

    WHY IT EXISTS, from `data/serve.err`:

        File "highcast/web/app.py", in timing
            conn = db.connect(c.db_path)
        File "highcast/db.py", in connect
            conn.executescript(SCHEMA)
        sqlite3.OperationalError: database is locked

    The dashboard opens a connection PER REQUEST (16 call sites) on a
    60-second poll, so every page load fired a burst of DDL at a file the
    5-minute wake cycle was also writing. `busy_timeout` (then 4 s) only bought
    that much patience before the read failed outright, and the read had no
    business issuing DDL in the first place — the dashboard is a reader; the
    wake cycle and the CLI own the schema.

    The web app calls this with `migrate=False` AFTER `create_app` has opened
    one migrating connection at startup, so the schema is always present
    before a request is served.
    """
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    # The web write (POST /api/note) and the 5-min launchd wake write share
    # this file; without a busy timeout a collision raises "database is
    # locked" immediately instead of waiting for the other writer to finish.
    # Set for readers too -- skipping DDL does not stop a SELECT colliding
    # with the wake cycle mid-write. See BUSY_TIMEOUT_MS for why it is 30 s.
    conn.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
    if not migrate:
        return conn
    conn.executescript(SCHEMA)
    from highcast.migrate import (ensure_app_bet_alert_columns,
                                  ensure_bot_orders_edge_column,
                                  ensure_environment_columns,
                                  ensure_metric_column,
                                  ensure_paper_account_tables,
                                  ensure_prediction_bound_gap,
                                  ensure_pressure_columns,
                                  ensure_six_min_column)
    ensure_metric_column(conn)
    ensure_six_min_column(conn)
    ensure_app_bet_alert_columns(conn)
    ensure_pressure_columns(conn)
    ensure_environment_columns(conn)
    # MUST run after ensure_metric_column: that one may REBUILD `predictions` to
    # repair a legacy primary key, and its rebuild DDL does not carry this
    # column. Running after means the rebuild is followed by a re-add.
    ensure_prediction_bound_gap(conn)
    ensure_bot_orders_edge_column(conn)
    # Schema only — the PAPER DEMO ACCOUNT ledger table. Populating it is a
    # separate, explicit step (`paper_account.refresh`, driven by `highcast
    # paper-account`), never automatic on connect: it needs a Kalshi balance
    # read and station configs this function does not have.
    ensure_paper_account_tables(conn)
    return conn


def upsert(conn: sqlite3.Connection, table: str, rows: list[dict]) -> int:
    if not rows:
        return 0
    cols = list(rows[0].keys())
    sql = (
        f"INSERT OR REPLACE INTO {table} ({','.join(cols)}) "
        f"VALUES ({','.join('?' * len(cols))})"
    )
    conn.executemany(sql, [[r[c] for c in cols] for r in rows])
    conn.commit()
    return len(rows)


def upsert_ignore(conn: sqlite3.Connection, table: str, rows: list[dict]) -> int:
    """INSERT OR IGNORE — the FIRST write for a primary key wins and later ones
    are silently dropped.

    `upsert` (INSERT OR REPLACE) is wrong for anything that must be immutable
    once written. `app_bets` is the case that needs this: the app's committed bet
    has to be locked at placement, because a bet that keeps being rewritten as the
    model changes its mind is not a prediction, it is hindsight.
    """
    if not rows:
        return 0
    cols = list(rows[0].keys())
    sql = (
        f"INSERT OR IGNORE INTO {table} ({','.join(cols)}) "
        f"VALUES ({','.join('?' * len(cols))})"
    )
    cur = conn.executemany(sql, [[r[c] for c in cols] for r in rows])
    conn.commit()
    return cur.rowcount


def get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT v FROM meta WHERE k = ?", (key,)).fetchone()
    return row["v"] if row else None


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute("INSERT OR REPLACE INTO meta (k, v) VALUES (?, ?)", (key, value))
    conn.commit()
