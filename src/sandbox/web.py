"""Local dashboard. 127.0.0.1 only. Never places real orders."""
import logging
import sqlite3
from contextlib import closing
from pathlib import Path
from zoneinfo import ZoneInfo
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from sandbox.book import parse_book, walk
from sandbox.cycle import ACCOUNTS, ensure_accounts, execute_intent, utc_now_iso
from sandbox.db import NOT_INITIALISED, NotInitialised, connect, connect_readonly, require_initialised
from sandbox.kalshi_public import PacedGet, fetch_book_payload
from sandbox.models import Intent
from sandbox.pricing import devig
from sandbox.recorder import latest_snapshot_id, load_snapshot
from sandbox.scoreboard import (N_EVENTS_REQUIRED, N_REQUIRED, confirmation_status, report,
                                verdict_text)

log = logging.getLogger(__name__)
TEMPLATES = Path(__file__).parent / "templates"
MANILA = ZoneInfo("Asia/Manila")


class BetIn(BaseModel):
    ticker: str
    side: str
    contracts: int
    price_cap: float


def _manila(ts: str | None) -> str:
    if not ts:
        return ""
    return datetime.fromisoformat(ts).astimezone(MANILA).strftime("%b %d %H:%M MNL")


def _validate(bet: BetIn) -> None:
    if bet.side not in ("yes", "no"):
        raise HTTPException(400, "side must be yes or no")
    if bet.contracts < 1 or bet.contracts > 100:
        raise HTTPException(400, "contracts must be 1..100")
    if not 0.0 < bet.price_cap < 1.0:
        raise HTTPException(400, "price_cap must be between 0 and 1")


def _require_known(snap, ticker: str) -> None:
    """A bet on a ticker outside the latest snapshot could never settle, yet would spend cash."""
    if snap is None or not any(m.ticker == ticker for m in snap.markets):
        raise HTTPException(400, "unknown ticker")


def create_app(db_path: str, get=None) -> FastAPI:
    get = get or PacedGet()
    app = FastAPI(title="Weather Desk Sandbox")
    templates = Jinja2Templates(directory=str(TEMPLATES))
    templates.env.filters["manila"] = _manila
    labels = {name: label for name, _, label in ACCOUNTS}

    def _read_conn():
        """Read-only connection for a GET; a missing file or schema becomes NotInitialised (503)."""
        try:
            conn = connect_readonly(db_path)
        except sqlite3.OperationalError as e:   # mode=ro never creates the file
            raise NotInitialised(NOT_INITIALISED) from e
        try:
            require_initialised(conn)
        except NotInitialised:
            conn.close()
            raise
        return conn

    @app.exception_handler(NotInitialised)
    def not_initialised(request: Request, exc: NotInitialised):
        if request.url.path.startswith("/api/"):
            return JSONResponse({"error": str(exc)}, status_code=503)
        return templates.TemplateResponse(request, "notice.html", {"message": str(exc)},
                                          status_code=503)

    def _snapshot(conn):
        sid = latest_snapshot_id(conn, utc_now_iso())
        return load_snapshot(conn, sid) if sid is not None else None

    @app.get("/", response_class=HTMLResponse)
    def markets(request: Request):
        with closing(_read_conn()) as conn:
            snap = _snapshot(conn)
            events = []
            if snap:
                for ev, ms in snap.events().items():
                    probs = devig(ms)
                    events.append({"event": ev, "series": ms[0].series, "station": ms[0].station,
                                   "date": ms[0].event_date,
                                   "rows": [(m, probs.get(m.ticker)) for m in ms]})
        return templates.TemplateResponse(request, "markets.html",
                                          {"snap": snap, "events": events})

    @app.get("/bet", response_class=HTMLResponse)
    def bet_page(request: Request):
        with closing(_read_conn()) as conn:
            snap = _snapshot(conn)
        return templates.TemplateResponse(request, "bet.html",
                                          {"markets": snap.markets if snap else ()})

    @app.get("/api/preview")
    def preview(ticker: str, side: str, contracts: int, price_cap: float):
        bet = BetIn(ticker=ticker, side=side, contracts=contracts, price_cap=price_cap)
        _validate(bet)
        with closing(_read_conn()) as conn:
            _require_known(_snapshot(conn), ticker)
        try:
            payload = fetch_book_payload(get, ticker)
        except Exception as e:  # noqa: BLE001 - an unreachable book is a 502, not a crash
            log.warning("preview book fetch failed for %s: %r", ticker, e)
            return JSONResponse({"error": "order book unavailable"}, status_code=502)
        fill = walk(parse_book(payload), side, contracts, price_cap)
        return {"status": fill.status, "contracts": fill.contracts, "cost": fill.cost,
                "fee": fill.fee, "levels": [list(lv) for lv in fill.levels]}

    @app.post("/api/bet")
    def place_bet(bet: BetIn):
        _validate(bet)
        with closing(connect(db_path)) as conn:
            ensure_accounts(conn)
            snap = _snapshot(conn)
            _require_known(snap, bet.ticker)
            ev = next(ms for ms in snap.events().values() if any(m.ticker == bet.ticker for m in ms))
            intent = Intent(account="manual", ticker=bet.ticker, side=bet.side,
                            contracts=bet.contracts, price_cap=bet.price_cap,
                            snapshot_id=snap.id, p_bot=None,
                            p_market=devig(ev).get(bet.ticker))
            status = execute_intent(conn, get, intent, utc_now_iso())
        return {"status": status}

    @app.get("/scoreboard", response_class=HTMLResponse)
    def scoreboard(request: Request):
        with closing(_read_conn()) as conn:
            have = {r["name"] for r in conn.execute("SELECT name FROM accounts")}
            cards = []
            for name, _, _ in ACCOUNTS:
                if name not in have:
                    continue
                status = confirmation_status(conn, name)
                cards.append({"score": report(conn, name), "label": labels[name],
                              "confirmation": status, "verdict": verdict_text(status),
                              "segments": [(title, report(conn, name, segment=segment))
                                           for title, segment in (("Pre-day", "pre_day"),
                                                                  ("Same-day", "same_day"))]})
        return templates.TemplateResponse(request, "scoreboard.html",
                                          {"cards": cards, "n_required": N_REQUIRED,
                                           "n_events_required": N_EVENTS_REQUIRED})

    @app.get("/log", response_class=HTMLResponse)
    def order_log(request: Request):
        with closing(_read_conn()) as conn:
            orders = conn.execute("SELECT * FROM orders ORDER BY id DESC LIMIT 500").fetchall()
        return templates.TemplateResponse(request, "log.html", {"orders": orders})

    return app
