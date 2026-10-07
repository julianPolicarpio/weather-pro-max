# Dashboard Bankroll Editor — Design

Updated 2026-07-24. View **and edit** the staking `bankroll` from the dashboard
so the user never has to open `config/klax.yaml` to change it.

## Problem

`bankroll` is the pool the tool sizes bets from (`staking.size` → fractional
Kelly, capped at `max_stake_frac`). It lives only in `config/klax.yaml` and
defaults to `0.0` → every stake computes to $0 (paper mode). To go live, or to
adjust the pool, the user must hand-edit the yaml and (for the running
dashboard) restart `serve`. The user wants to do this from the dashboard.

The dashboard already *reads* bankroll: `GET /api/bet` returns
`cfg.staking.bankroll` and `cfg.staking.currency`. What's missing is (a) a
visible control, and (b) a write path back to the yaml that takes effect in the
running server.

## Goal

- A **Bankroll** control in the Decision Desk that shows the current value, its
  currency, and a **PAPER** (bankroll == 0) / **LIVE** (bankroll > 0) badge.
- Inline edit → **Save** writes the new value back to `config/klax.yaml`
  (Option A: single source of truth) and takes effect **immediately** in the
  running dashboard, no restart.
- The launchd `wake` process needs no changes — it already re-reads the yaml
  fresh every 300s, so scheduled sizing picks up a new bankroll on its own.

## Confirmed facts (from the codebase)

- `config.py`: `StakingConfig` and `AppConfig` are `@dataclass(frozen=True)`.
  `load_config()` does `yaml.safe_load` → constructs the frozen objects.
- `config/klax.yaml` `staking:` block has hand-written **inline comments** on
  every line (e.g. `bankroll: 0.0  # <-- set this yourself ...`). A naive
  `yaml.safe_dump` round-trip would strip comments and reorder keys — so the
  write MUST be a targeted line edit, not a dump.
- `web/app.py`: `create_app(cfg)` captures `cfg` in a closure; endpoints read it
  directly. Staking is read in exactly two places: `/api/bet` (returns bankroll,
  currency; passes `cfg` into `signals.live_decision`) and `/api/timing`
  (`cfg.staking.min_confidence`). `app.state` is currently unused.
- `cli.py cmd_serve`: `uvicorn.run(create_app(cfg), host="127.0.0.1", ...)`.
  **Localhost-only bind** → an unauthenticated write endpoint carries the same
  trust model as the existing unauthenticated `POST /api/note`.

## Key definitions

- **PAPER** — `bankroll == 0`: sizing produces $0, decisions are advisory only.
- **LIVE** — `bankroll > 0`: sizing produces real dollar amounts.
- **Comment-preserving write** — replace only the numeric token after
  `bankroll:` on its line; every other byte of the file is left identical.

## Design

### Component 1 — `config.set_bankroll(path, value) -> float`

New function in `config.py`. Persists a new bankroll to the yaml, Option A.

- **Validate** `value`: must be a real, finite number (reject `NaN`/`inf`),
  `>= 0`, and `<= MAX_BANKROLL` (a sanity cap, `1e9`). On failure raise
  `ValueError` with a clear message.
- **Locate** the `bankroll:` line inside the `staking:` block. Read the file
  text, find the line matching `^(\s*bankroll\s*:\s*)([^#\n]*)(#.*)?$` under the
  `staking:` block. If no such line exists, raise `ValueError` (do not silently
  append — the yaml is hand-maintained; a missing key means something is wrong).
- **Replace** only group 2 (the value token) with the formatted number,
  preserving leading indentation (group 1) and any trailing comment (group 3).
  Format with `repr(float(value))` for a float rendering consistent with the
  existing `0.0` style (e.g. `250` → `250.0`, `250.5` → `250.5`).
- **Atomic write**: write the full new text to a temp file in the same dir,
  `os.replace` it over the original (crash-safe; never a half-written config).
- **Verify**: re-run `load_config(path)` on the result; confirm
  `cfg.staking.bankroll == value`. If not, raise (and the temp/atomic swap means
  the original is still intact on the raise path — write to temp, verify temp by
  parsing it, only then `os.replace`).
- **Return** the parsed new bankroll (float).

### Component 2 — mutable cfg holder in `web/app.py`

`create_app` today captures a frozen `cfg`. To let a request swap it live:

- Store the config on `app.state.cfg = cfg`.
- Add a tiny accessor `current() -> AppConfig` returning `app.state.cfg`.
- Change the two staking-reading endpoints (`/api/bet`, `/api/timing`) and the
  new bankroll endpoints to read `current()` instead of the closure `cfg`.
  Endpoints that only touch `station`/`db_path` (immutable in practice) may keep
  using the closure `cfg` — but prefer `current()` anywhere `cfg` is passed into
  a callee that reads staking (notably `signals.live_decision`).

### Component 3 — `GET /api/bankroll` and `POST /api/bankroll`

New endpoints in `web/app.py`.

- **`GET /api/bankroll`** → `{"bankroll": float, "currency": str,
  "paper": bool}` where `paper = bankroll == 0`. (Lets the UI render the control
  without pulling the whole `/api/bet` payload; also handy for tests.)
- **`POST /api/bankroll`** body `{"bankroll": number}`:
  1. Pydantic model coerces/validates the field is a number.
  2. Call `config.set_bankroll(cfg_path, body.bankroll)`. `cfg_path` is captured
     at `create_app` time (thread it in from `cmd_serve` via `args.config`).
  3. On success, rebuild cfg: `new = dataclasses.replace(current(),
     staking=dataclasses.replace(current().staking, bankroll=result))` and set
     `app.state.cfg = new`.
  4. Return `{"bankroll": result, "currency": ..., "paper": result == 0}`.
  5. On `ValueError` from validation/write → HTTP **400** with the message.

`create_app` signature grows a `config_path: str` param (default
`"config/klax.yaml"`) so the write targets the same file `serve` loaded.

### Component 4 — UI (`web/templates/index.html`)

- A **Bankroll** control in the Decision Desk header row. Default (read) state:
  `Bankroll: $250.00  [LIVE]` or `Bankroll: $0.00  [PAPER]`, plus a small
  **Edit** affordance (pencil/text link).
- Edit state: a number input pre-filled with the current value + **Save** /
  **Cancel**. Save → `POST /api/bankroll` → on 200 update the display, badge,
  and re-fetch the Decision Desk (`/api/bet`) so the shown stake re-sizes live.
  On 400 → show the error message inline, keep the input open.
- Bankroll is **global** (not per-metric), so the control shows the same value
  in HIGH and LOW modes and does not send `?metric=`.
- Currency comes from the API (`currency`), formatted client-side.

### Component 5 — wiring (`cli.py`)

`cmd_serve`: pass the config path through —
`create_app(cfg, config_path=args.config)` — so writes hit the file that was
loaded. No other command changes.

## Data flow

```
User edits field → POST /api/bankroll {bankroll: 250}
  → validate (finite, >=0, <=1e9)
  → config.set_bankroll: parse yaml text, regex-replace bankroll value token,
      atomic temp-write, re-parse temp to verify, os.replace
  → dataclasses.replace → app.state.cfg swapped
  → 200 {bankroll, currency, paper:false}
UI → update badge + re-fetch /api/bet (stake now sized off new bankroll)
launchd wake (separate proc, next ≤300s) → load_config reads the new yaml → same value
```

## Error handling & edge cases

- **Non-numeric / NaN / inf / negative / > cap** → `ValueError` → HTTP 400,
  yaml untouched (validation happens before any write).
- **`bankroll:` line absent** → `ValueError`, HTTP 400, no write. (Don't append;
  surface that the config is malformed.)
- **Verify-after-write fails** → raise; because we parse the temp file *before*
  `os.replace`, the live `config/klax.yaml` is never left corrupted.
- **Concurrent edits** — single local user, not a real concern; last write wins.
- **0 is valid** (explicit return to paper mode), not an error.
- **Trailing comment preserved** so future yaml readers keep the guidance text.

## Testing

- `config.set_bankroll`:
  - updates the value; `load_config` on the result returns the new bankroll;
  - the `# <-- set this yourself ...` comment and all *other* staking keys/values
    are byte-preserved (diff only the one line);
  - rejects negative, `NaN`, `inf`, and `> MAX_BANKROLL` with `ValueError`;
  - rejects a yaml with no `bankroll:` line;
  - operates on a **temp copy** of the real yaml (fixture copies `config/klax.yaml`).
- Endpoints (FastAPI `TestClient` against a temp-copied yaml):
  - `POST /api/bankroll {bankroll: 250}` → 200, `paper:false`; a subsequent
    `GET /api/bankroll` and `GET /api/bet` both report `250` in the same process
    (proves the live `app.state` swap);
  - `POST` with `-5` → 400, and `GET` still shows the pre-edit value;
  - `POST {bankroll: 0}` → 200, `paper:true`.

## Non-goals (YAGNI)

- Editing any other staking knob (kelly_fraction, cutoffs, min_edge, spreads,
  currency) from the dashboard — same pattern extends later if wanted.
- Auth / multi-user / audit log on the write — localhost single-user tool.
- Live-reloading the running server on *external* yaml edits — only the
  dashboard's own writes swap in-memory; the wake process re-reads on its own.
- A general settings page.

## Open items for review

- Bankroll control **placement**: Decision Desk header (proposed) vs. a spot in
  the top status strip. Desk is where sizing is shown, so it's the natural home.
- Number formatting: render as `250.0` (float, matches existing `0.0`) in the
  yaml; display as `$250.00` in the UI. Confirm that's fine.
