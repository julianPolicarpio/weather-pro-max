# Dashboard Bankroll Editor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the user view and edit the staking `bankroll` from the highcast dashboard so they never have to hand-edit `config/klax.yaml`.

**Architecture:** A comment-preserving, atomic write helper persists the new value back to `config/klax.yaml` (single source of truth — Option A). The running FastAPI server holds its config in a mutable `app.state.cfg` slot; a `POST /api/bankroll` endpoint writes the file, then swaps in a rebuilt frozen config so the change takes effect live with no restart. The launchd `wake` process needs no change — it re-reads the yaml every 300s. A small Decision-Desk control renders the value with a PAPER/LIVE badge and an inline edit → Save.

**Tech Stack:** Python 3, FastAPI + Pydantic, PyYAML, SQLite; vanilla-JS dashboard template; pytest.

## Global Constraints

- **Not a git repo.** There are NO commit steps. Each task ends by running the test suite; reviewers read files directly. (`.venv/bin/python -m pytest -q`)
- **Localhost-only trust model.** The server binds `127.0.0.1`; the write endpoint is unauthenticated, matching the existing `POST /api/note`. Do not add auth.
- **Option A — the yaml stays the single source of truth.** Persist by a *targeted line edit* that preserves every comment and other key. NEVER `yaml.safe_dump` the whole file (it would strip the hand-written comments).
- **Frozen dataclasses.** `StakingConfig`/`AppConfig` are `@dataclass(frozen=True)`; update via `dataclasses.replace`, never in-place mutation.
- **Validation bounds:** bankroll must be finite, `>= 0`, `<= MAX_BANKROLL` (`1e9`). `0` is valid (paper mode).
- **Run tests with the venv:** `.venv/bin/python -m pytest ...`.

---

### Task 1: `config.set_bankroll` write helper

**Files:**
- Modify: `src/highcast/config.py` (add imports, `MAX_BANKROLL`, regex, `set_bankroll`)
- Test: `tests/test_config_bankroll.py` (create)

**Interfaces:**
- Consumes: existing `load_config(path)` (re-used to verify the written file).
- Produces:
  - `MAX_BANKROLL: float = 1e9`
  - `set_bankroll(path: str | Path, value: float) -> float` — validates, writes atomically & comment-preserving, re-parses to verify, returns the persisted float. Raises `ValueError` on bad value or if the `bankroll:` line is absent/ambiguous.

- [ ] **Step 1: Write the failing test**

Create `tests/test_config_bankroll.py`:

```python
import shutil

import pytest

from highcast.config import MAX_BANKROLL, load_config, set_bankroll


def _copy_cfg(tmp_path):
    dst = tmp_path / "klax.yaml"
    shutil.copy("config/klax.yaml", dst)
    return dst


def test_set_bankroll_updates_value(tmp_path):
    p = _copy_cfg(tmp_path)
    out = set_bankroll(p, 250)
    assert out == 250.0
    assert load_config(p).staking.bankroll == 250.0


def test_set_bankroll_preserves_comment_and_other_keys(tmp_path):
    p = _copy_cfg(tmp_path)
    before = p.read_text().splitlines()
    set_bankroll(p, 250)
    after = p.read_text().splitlines()
    diff = [(b, a) for b, a in zip(before, after) if b != a]
    assert len(diff) == 1, "exactly one line should change"
    b, a = diff[0]
    assert "bankroll" in b and "bankroll" in a
    # the trailing comment text (everything after the first '#') survives verbatim
    assert "#" in a and a.split("#", 1)[1] == b.split("#", 1)[1]
    cfg = load_config(p)
    assert cfg.staking.kelly_fraction == 0.25
    assert cfg.staking.currency == "USD"


def test_set_bankroll_zero_is_valid(tmp_path):
    p = _copy_cfg(tmp_path)
    set_bankroll(p, 100)
    assert set_bankroll(p, 0) == 0.0
    assert load_config(p).staking.bankroll == 0.0


@pytest.mark.parametrize("bad", [-1, float("nan"), float("inf"), MAX_BANKROLL * 2])
def test_set_bankroll_rejects_bad_values(tmp_path, bad):
    p = _copy_cfg(tmp_path)
    with pytest.raises(ValueError):
        set_bankroll(p, bad)
    assert load_config(p).staking.bankroll == 0.0  # file untouched on rejection


def test_set_bankroll_missing_line_raises(tmp_path):
    p = tmp_path / "nobank.yaml"
    p.write_text("staking:\n  kelly_fraction: 0.25\n")
    with pytest.raises(ValueError):
        set_bankroll(p, 100)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_config_bankroll.py -q`
Expected: FAIL — `ImportError: cannot import name 'MAX_BANKROLL'` / `set_bankroll`.

- [ ] **Step 3: Write minimal implementation**

Edit `src/highcast/config.py`. Change the top imports from:

```python
from dataclasses import dataclass, field
from pathlib import Path

import yaml
```

to:

```python
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

MAX_BANKROLL = 1e9  # sanity cap; a bankroll above this is almost certainly a typo

# Matches the single `bankroll:` KEY line (indent + key + numeric value +
# optional trailing comment). The word "bankroll" also appears inside comments
# on other lines, but only the real key sits at line-start followed by ':'.
_BANKROLL_LINE = re.compile(
    r"^([ \t]*bankroll[ \t]*:[ \t]*)([0-9]*\.?[0-9]+)([ \t]*)(#.*)?$",
    re.MULTILINE,
)
```

Then add this function at the end of the file (after `load_config`):

```python
def set_bankroll(path: str | Path, value: float) -> float:
    """Persist a new staking.bankroll to the yaml at `path`, preserving every
    comment and other line (Option A: the yaml stays the single source of
    truth). Validates the value is finite, >= 0, and <= MAX_BANKROLL. Writes
    atomically (temp file + os.replace) and re-parses the result BEFORE swapping
    it in, so a crash or a bad edit never corrupts the live config. Returns the
    persisted bankroll as a float; raises ValueError on a bad value or if the
    file has no unambiguous `bankroll:` line."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"bankroll must be a number, got {value!r}")
    if not math.isfinite(v):
        raise ValueError("bankroll must be a finite number")
    if v < 0:
        raise ValueError("bankroll must be >= 0")
    if v > MAX_BANKROLL:
        raise ValueError(f"bankroll must be <= {MAX_BANKROLL:g}")

    path = Path(path)
    text = path.read_text()
    matches = _BANKROLL_LINE.findall(text)
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one 'bankroll:' line in {path}, found {len(matches)}")

    def _repl(m: "re.Match[str]") -> str:
        pre, _old, gap, comment = m.group(1), m.group(2), m.group(3), m.group(4)
        return f"{pre}{v!r}{gap}{comment or ''}"

    new_text = _BANKROLL_LINE.sub(_repl, text, count=1)

    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(new_text)
    try:
        parsed = load_config(tmp)
        if not math.isclose(parsed.staking.bankroll, v, rel_tol=0, abs_tol=1e-9):
            raise ValueError("post-write verification failed; config unchanged")
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, path)
    return v
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_config_bankroll.py -q`
Expected: PASS (6 tests: the 4 non-parametrized + parametrized expands to 4 = 8 total; all pass).

- [ ] **Step 5: Suite checkpoint (no commit — not a git repo)**

Run: `.venv/bin/python -m pytest -q`
Expected: all previously-passing tests still pass, plus the new file. Confirm no regressions before moving on.

---

### Task 2: `GET`/`POST /api/bankroll` endpoints + live cfg swap

**Files:**
- Modify: `src/highcast/web/app.py` (imports, `BankrollIn` model, `create_app` signature, `app.state.cfg` holder, route staking reads through `current()`, two new endpoints)
- Modify: `src/highcast/cli.py:254` (`cmd_serve` passes `config_path`)
- Test: `tests/test_web_bankroll.py` (create)

**Interfaces:**
- Consumes: `config.set_bankroll` (Task 1); `dataclasses.replace`; existing `signals.live_decision`.
- Produces:
  - `create_app(cfg: AppConfig, config_path: str = "config/klax.yaml") -> FastAPI`
  - `GET /api/bankroll` → `{"bankroll": float, "currency": str, "paper": bool}`
  - `POST /api/bankroll` body `{"bankroll": number}` → same shape; `400` on invalid value.
  - In-process: `app.state.cfg` is the live config; `current()` returns it.

- [ ] **Step 1: Write the failing test**

Create `tests/test_web_bankroll.py`:

```python
import shutil

from fastapi.testclient import TestClient

from highcast import db as hdb
from highcast.config import load_config
from highcast.fetchers import kalshi
from highcast.web.app import create_app


def _client(tmp_path):
    cfg_path = tmp_path / "klax.yaml"
    shutil.copy("config/klax.yaml", cfg_path)
    cfg = load_config(cfg_path)
    object.__setattr__(cfg, "db_path", tmp_path / "t.db")
    hdb.connect(cfg.db_path)  # initialize schema
    return TestClient(create_app(cfg, config_path=str(cfg_path))), cfg_path


def test_get_bankroll_defaults_paper(tmp_path):
    client, _ = _client(tmp_path)
    r = client.get("/api/bankroll")
    assert r.status_code == 200
    b = r.json()
    assert b["bankroll"] == 0.0 and b["paper"] is True and b["currency"] == "USD"


def test_post_bankroll_updates_live_and_file(tmp_path):
    client, cfg_path = _client(tmp_path)
    r = client.post("/api/bankroll", json={"bankroll": 250})
    assert r.status_code == 200
    body = r.json()
    assert body["bankroll"] == 250.0 and body["paper"] is False
    # reflected in-process (live app.state swap)
    assert client.get("/api/bankroll").json()["bankroll"] == 250.0
    # and persisted to the file
    assert load_config(cfg_path).staking.bankroll == 250.0


def test_post_bankroll_reflected_in_bet(tmp_path, monkeypatch):
    monkeypatch.setattr(kalshi, "fetch_ranges",
                        lambda cfg, target_date, metric=None: None)
    client, _ = _client(tmp_path)
    client.post("/api/bankroll", json={"bankroll": 500})
    bet = client.get("/api/bet").json()
    assert bet["bankroll"] == 500.0


def test_post_bankroll_rejects_negative(tmp_path):
    client, cfg_path = _client(tmp_path)
    r = client.post("/api/bankroll", json={"bankroll": -5})
    assert r.status_code == 400
    assert client.get("/api/bankroll").json()["bankroll"] == 0.0
    assert load_config(cfg_path).staking.bankroll == 0.0


def test_post_bankroll_zero_returns_paper(tmp_path):
    client, _ = _client(tmp_path)
    client.post("/api/bankroll", json={"bankroll": 100})
    r = client.post("/api/bankroll", json={"bankroll": 0})
    assert r.status_code == 200 and r.json()["paper"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_web_bankroll.py -q`
Expected: FAIL — `create_app()` got an unexpected keyword argument `config_path` (and/or 404 on `/api/bankroll`).

- [ ] **Step 3a: Update imports + add `BankrollIn`**

In `src/highcast/web/app.py`, change the config import line:

```python
from highcast.config import AppConfig
```

to:

```python
from dataclasses import replace as dc_replace

from highcast.config import AppConfig, set_bankroll
```

Add the request model next to `NoteIn` (after line 236 `    sky: str`):

```python
class BankrollIn(BaseModel):
    bankroll: float
```

- [ ] **Step 3b: Add the mutable cfg holder to `create_app`**

Change the signature and the top of `create_app`:

```python
def create_app(cfg: AppConfig, config_path: str = "config/klax.yaml") -> FastAPI:
    app = FastAPI(title="highcast")
    app.state.cfg = cfg  # mutable slot: POST /api/bankroll swaps this live
    env = Environment(loader=FileSystemLoader(TEMPLATES))

    def current() -> AppConfig:
        return app.state.cfg

    def _conn():
        return db.connect(cfg.db_path)  # db_path is immutable; original cfg is fine
```

- [ ] **Step 3c: Route the staking reads through `current()`**

In `/api/bet` (`def bet`), replace the `signals.live_decision` call and the staking fields in the return dict. Change:

```python
        out = signals.live_decision(conn, cfg, spec, now)
```
to:
```python
        c = current()
        out = signals.live_decision(conn, c, spec, now)
```

and change the tail of the return dict:

```python
            "bankroll": cfg.staking.bankroll, "currency": cfg.staking.currency,
            "min_confidence": cfg.staking.min_confidence,
            "min_edge": cfg.staking.min_edge, "max_spread": cfg.staking.max_spread}
```
to:
```python
            "bankroll": c.staking.bankroll, "currency": c.staking.currency,
            "min_confidence": c.staking.min_confidence,
            "min_edge": c.staking.min_edge, "max_spread": c.staking.max_spread}
```

In `/api/timing` (`def timing`), change the one staking read:

```python
                "min_confidence": cfg.staking.min_confidence}
```
to:
```python
                "min_confidence": current().staking.min_confidence}
```

- [ ] **Step 3d: Add the two endpoints**

Insert after the `/api/bet` endpoint (before `@app.get("/api/pnl")`):

```python
    @app.get("/api/bankroll")
    def get_bankroll():
        s = current().staking
        return {"bankroll": s.bankroll, "currency": s.currency,
                "paper": s.bankroll == 0}

    @app.post("/api/bankroll")
    def post_bankroll(body: BankrollIn):
        try:
            new_val = set_bankroll(config_path, body.bankroll)
        except ValueError as e:
            raise HTTPException(400, str(e))
        c = current()
        app.state.cfg = dc_replace(
            c, staking=dc_replace(c.staking, bankroll=new_val))
        return {"bankroll": new_val, "currency": c.staking.currency,
                "paper": new_val == 0}
```

- [ ] **Step 3e: Wire `config_path` through `cmd_serve`**

In `src/highcast/cli.py`, change line 254:

```python
    uvicorn.run(create_app(cfg), host="127.0.0.1", port=args.port)
```
to:
```python
    uvicorn.run(create_app(cfg, config_path=args.config),
                host="127.0.0.1", port=args.port)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_web_bankroll.py -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Suite checkpoint**

Run: `.venv/bin/python -m pytest -q`
Expected: full suite green (baseline was 201 passed, 12 deselected; now higher). No regressions in `tests/test_web.py` (the `create_app(cfg)` calls there still work via the `config_path` default).

---

### Task 3: Decision-Desk bankroll control (UI)

**Files:**
- Modify: `src/highcast/web/templates/index.html` (markup in `#decision-desk`, small CSS, `renderBankroll` + `wireBankroll` JS, call site in `refreshBet`)
- Test: `tests/test_web_bankroll.py` (append one template-anchor assertion)

**Interfaces:**
- Consumes: `GET /api/bet` payload (already fetched in `refreshBet`; carries `bankroll` + `currency`) and `POST /api/bankroll` (Task 2).
- Produces: a `#bankroll-control` block with `#bankroll-value`, `#bankroll-badge`, an edit form (`#bankroll-edit`, `#bankroll-input`, `#bankroll-save`, `#bankroll-cancel`), and a `#bankroll-status` line.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_web_bankroll.py`:

```python
def test_index_has_bankroll_control(tmp_path):
    client, _ = _client(tmp_path)
    r = client.get("/")
    assert r.status_code == 200
    assert 'id="bankroll-control"' in r.text
    assert 'id="bankroll-input"' in r.text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_web_bankroll.py::test_index_has_bankroll_control -q`
Expected: FAIL — `assert 'id="bankroll-control"' in r.text`.

- [ ] **Step 3a: Add the markup**

In `src/highcast/web/templates/index.html`, inside `#decision-desk`, replace the target eyebrow line (line 523):

```html
    <p id="decision-target" class="eyebrow" style="margin:.15rem 0 .9rem">—</p>
```

with the target line followed by the bankroll control:

```html
    <p id="decision-target" class="eyebrow" style="margin:.15rem 0 .9rem">—</p>
    <div id="bankroll-control" class="bankroll-control">
      <div class="bankroll-view">
        <span class="eyebrow">Bankroll</span>
        <span id="bankroll-value" class="bankroll-value mono">—</span>
        <span id="bankroll-badge" class="bankroll-badge">—</span>
        <button type="button" id="bankroll-edit-btn" class="bankroll-edit-btn">Edit</button>
      </div>
      <form id="bankroll-edit" class="bankroll-edit" hidden>
        <label class="eyebrow" for="bankroll-input">New bankroll</label>
        <input id="bankroll-input" class="bankroll-input mono" type="number"
               min="0" step="1" inputmode="decimal" />
        <button type="submit" id="bankroll-save" class="bankroll-save">Save</button>
        <button type="button" id="bankroll-cancel" class="bankroll-cancel">Cancel</button>
      </form>
      <span id="bankroll-status" class="bankroll-status" aria-live="polite"></span>
    </div>
```

- [ ] **Step 3b: Add minimal CSS**

Add near the `.place-rec` rule (around line 199) — a self-contained block:

```css
.bankroll-control{margin:0 0 .9rem;display:flex;flex-direction:column;gap:.4rem}
.bankroll-view{display:flex;align-items:center;gap:.5rem;flex-wrap:wrap}
.bankroll-value{font-size:1.1rem;font-weight:600}
.bankroll-badge{font-size:.7rem;font-weight:700;letter-spacing:.04em;padding:.1rem .4rem;border-radius:.25rem}
.bankroll-badge.paper{background:var(--ink-1,#334);color:#fff}
.bankroll-badge.live{background:#137a3f;color:#fff}
.bankroll-edit-btn,.bankroll-save,.bankroll-cancel{font:inherit;cursor:pointer;border:1px solid var(--ink-3,#889);background:transparent;color:inherit;border-radius:.25rem;padding:.15rem .5rem}
.bankroll-edit{display:flex;align-items:center;gap:.4rem;flex-wrap:wrap}
.bankroll-input{width:8rem;padding:.2rem .4rem}
.bankroll-status.is-error{color:#c0392b}
.bankroll-status.is-ok{color:#137a3f}
```

- [ ] **Step 3c: Add the render + wire JS**

Add these two functions just above `async function refreshBet()` (line 1553):

```javascript
function renderBankroll(bet) {
  const val = document.getElementById("bankroll-value");
  const badge = document.getElementById("bankroll-badge");
  if (!val || !badge) return;
  const b = (bet && typeof bet.bankroll === "number") ? bet.bankroll : 0;
  const cur = (bet && bet.currency) || "USD";
  val.textContent = `${cur} ${b.toFixed(2)}`;
  const paper = b === 0;
  badge.textContent = paper ? "PAPER" : "LIVE";
  badge.className = "bankroll-badge " + (paper ? "paper" : "live");
  const input = document.getElementById("bankroll-input");
  if (input && document.getElementById("bankroll-edit").hidden) input.value = b;
}

function wireBankroll() {
  const view = document.querySelector("#bankroll-control .bankroll-view");
  const form = document.getElementById("bankroll-edit");
  const editBtn = document.getElementById("bankroll-edit-btn");
  const cancelBtn = document.getElementById("bankroll-cancel");
  const input = document.getElementById("bankroll-input");
  const status = document.getElementById("bankroll-status");
  if (!form || !editBtn) return;
  const open = () => { view.hidden = true; form.hidden = false; status.textContent = ""; input.focus(); input.select(); };
  const close = () => { view.hidden = false; form.hidden = true; };
  editBtn.addEventListener("click", open);
  cancelBtn.addEventListener("click", close);
  form.addEventListener("submit", async (evt) => {
    evt.preventDefault();
    const v = Number(input.value);
    if (!Number.isFinite(v) || v < 0) {
      status.className = "bankroll-status is-error";
      status.textContent = "Enter a number ≥ 0.";
      return;
    }
    status.className = "bankroll-status"; status.textContent = "Saving…";
    try {
      const r = await fetch("/api/bankroll", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ bankroll: v }),
      });
      const body = await r.json().catch(() => ({}));
      if (!r.ok) {
        status.className = "bankroll-status is-error";
        status.textContent = body.detail || "Could not save bankroll.";
        return;
      }
      close();
      status.className = "bankroll-status is-ok";
      status.textContent = "Saved.";
      refreshBet();  // re-size the decision desk off the new bankroll
    } catch (err) {
      status.className = "bankroll-status is-error";
      status.textContent = "Could not save — check connection.";
    }
  });
}
```

Then, inside `refreshBet()`, add the render call right after `renderDecision(bet);` (line 1559):

```javascript
    renderDecision(bet);
    renderBankroll(bet);
```

Finally, add the wiring to the init sequence — after `wireMetricToggle();` (line 1620):

```javascript
wireMetricToggle();
wireBankroll();
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_web_bankroll.py -q`
Expected: PASS (6 tests including the new anchor test).

- [ ] **Step 5: Manual browser verification**

Start the dashboard and confirm the control renders, edits, and persists:

1. `preview_start` name `highcast` (or `.venv/bin/highcast serve` on :8712) and open `http://127.0.0.1:8712`.
2. `read_page` → confirm `#bankroll-control` shows `USD 0.00` + a `PAPER` badge.
3. Click **Edit**, type `250`, click **Save**; confirm the value flips to `USD 250.00` + `LIVE`, and the Decision Desk stake sizing updates.
4. Confirm `config/klax.yaml` now has `bankroll: 250.0` with its comment intact (`read_page` is UI; check the file with Read).
5. **Reset to paper for handoff:** set it back to `0` from the UI (or the file) so the repo's default stays `0.0`. Screenshot the final state.

- [ ] **Step 6: Suite checkpoint**

Run: `.venv/bin/python -m pytest -q`
Expected: full suite green, no regressions.

---

## Self-Review

**Spec coverage:**
- `config.set_bankroll` (comment-preserving, atomic, verified) → Task 1. ✓
- Mutable `app.state.cfg` holder + live swap → Task 2 (3b, 3d). ✓
- `GET`/`POST /api/bankroll` with validation → 400 → Task 2 (3d) + tests. ✓
- Route staking reads through the holder (`/api/bet`, `/api/timing`) → Task 2 (3c). ✓
- `create_app` gains `config_path`; `cmd_serve` threads it → Task 2 (3b, 3e). ✓
- Decision-Desk control: value + PAPER/LIVE badge + inline edit/Save; refetch desk on save; metric-independent → Task 3. ✓
- Tests for helper (value/comments/other-keys/bad-input/missing-line) and endpoints (live swap, file persist, bet reflection, reject, zero) → Tasks 1 & 2. ✓
- Wake process unchanged (re-reads yaml) — no task needed. ✓

**Placeholder scan:** No TBD/TODO; every code step shows complete code; every run step shows the exact command + expected result. ✓

**Type consistency:** `set_bankroll(path, value) -> float` defined in Task 1, imported and called in Task 2 with `(config_path, body.bankroll)`. `create_app(cfg, config_path=...)` defined in Task 2, used by tests and `cmd_serve` consistently. `current()` returns `AppConfig`; `/api/bankroll` shape `{bankroll, currency, paper}` matches between GET, POST, and the JS `renderBankroll`. ✓

**Open items from the spec (user's call, do not block):**
- Placement: control sits under the Decision-Desk target line (proposed). If the user prefers the top status strip, move the `#bankroll-control` block only.
- Formatting: stored as `250.0` in yaml (`repr(float)`), displayed as `USD 250.00`. Confirmed in the design.
