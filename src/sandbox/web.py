"""Local dashboard. 127.0.0.1 only. Never places real orders."""
import logging
import sqlite3
from contextlib import closing
from pathlib import Path
from urllib.parse import urlencode
from zoneinfo import ZoneInfo
from datetime import date, datetime, timedelta

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from sandbox import views
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


def _money(x: float) -> str:
    return ("+" if x > 0 else "−" if x < 0 else "") + f"${abs(x):.2f}"


def _cents(x: int) -> str:
    return ("+" if x > 0 else "−" if x < 0 else "") + f"{abs(x)}¢"


def _points(x: float) -> str:
    v = round(x)
    return ("+" if v > 0 else "−" if v < 0 else "") + f"{abs(v)}"


def _query(request: Request, **changes) -> str:
    """Relative query string with `changes` applied in place (None drops a key), so links keep
    the other params and their order; Starlette's include_query_params would move a changed key last."""
    params = dict(request.query_params)
    for k, v in changes.items():
        if v is None:
            params.pop(k, None)
        else:
            params[k] = str(v)
    return "?" + urlencode(params)


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
    app = FastAPI(title="Weather Pro Max")
    templates = Jinja2Templates(directory=str(TEMPLATES))
    templates.env.filters["manila"] = _manila
    templates.env.filters.update(money=_money, cents=_cents, points=_points)
    templates.env.globals["query"] = _query
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
        return templates.TemplateResponse(request, "notice.html", {"message": str(exc), "nav": ""},
                                          status_code=503)

    def _snapshot(conn):
        sid = latest_snapshot_id(conn, utc_now_iso())
        return load_snapshot(conn, sid) if sid is not None else None

    def _frame(conn, request: Request, nav: str) -> dict:
        """Context every Weather Pro Max page shares: top bar, rail and freshness (spec §3, §6)."""
        now = utc_now_iso()
        snap = _snapshot(conn)
        prev = views.previous_snapshot(conn, snap)
        show_all = request.query_params.get("all") == "1"
        everyone = views.bot_summaries(conn, show_all=True)
        names = [b["name"] for b in everyone]
        wanted = request.query_params.get("account") or views.DEFAULT_BOT
        account = wanted if wanted in names else (views.DEFAULT_BOT if views.DEFAULT_BOT in names
                                                  else (names[0] if names else None))
        today = views.current_day(snap, now)
        empty = {"live": [], "awaiting": [], "count": 0, "at_risk": 0.0}
        return {"nav": nav, "now": now, "snap": snap, "snap_id": snap.id if snap else None, "prev": prev,
                "account": account, "show_all": show_all,
                "bots": [b for b in everyone if show_all or b["name"] in views.ACTIVE_BOTS],
                "bot": next((b for b in everyone if b["name"] == account), None),
                "pos": views.open_positions(conn, account, snap, now) if account else empty,
                "moves": views.biggest_moves(snap, prev, today) if today else [],
                "today": today, "fresh": views.freshness(snap, now), "n_required": N_REQUIRED,
                "live_n": views.LIVE_RAIL_N, "await_n": views.AWAIT_RAIL_N}

    @app.get("/", response_class=HTMLResponse)
    def markets(request: Request, day: str = "today", metric: str = "both", city: str = ""):
        day = day if day in ("today", "tomorrow") else "today"
        metric = metric if metric in ("both", "high", "low") else "both"
        city = city if city in views.CITY else ""
        with closing(_read_conn()) as conn:
            ctx = _frame(conn, request, "markets")
            held = views.held_tickers(conn, ctx["account"]) if ctx["account"] else set()
        d = ctx["today"]
        if d and day == "tomorrow":
            d = (date.fromisoformat(d) + timedelta(days=1)).isoformat()
        ctx.update(cards=views.market_cards(ctx["snap"], ctx["prev"], held, d,
                                            None if metric == "both" else metric, city or None, ctx["now"])
                   if d else [],
                   f={"day": day, "metric": metric, "city": city},
                   cities=sorted(views.CITY.items(), key=lambda kv: kv[1]),
                   day_text=views.day_text(d) if d else "")
        return templates.TemplateResponse(request, "markets.html", ctx)

    @app.get("/portfolio", response_class=HTMLResponse)
    def portfolio(request: Request, tab: str = "open"):
        with closing(_read_conn()) as conn:
            ctx = _frame(conn, request, "portfolio")
            acct = ctx["account"]
            if acct:
                status = confirmation_status(conn, acct)
                ctx.update(score=report(conn, acct), verdict=verdict_text(status), label=labels.get(acct, ""),
                           settled_rows=views.settled_positions(conn, acct),
                           out_of_cash=views.out_of_cash(ctx["bot"]["cash"], ctx["snap"]))
        ctx.update(n_events_required=N_EVENTS_REQUIRED, tab="settled" if tab == "settled" else "open")
        return templates.TemplateResponse(request, "portfolio.html", ctx)

    @app.get("/api/latest-snapshot")
    def latest_snapshot():
        with closing(_read_conn()) as conn:
            return {"id": latest_snapshot_id(conn, utc_now_iso())}

    @app.get("/bet", response_class=HTMLResponse)
    def bet_page(request: Request, ticker: str = ""):
        with closing(_read_conn()) as conn:
            ctx = _frame(conn, request, "bet")
        ctx["snap_id"] = None   # no auto-reload while the form is being filled
        snap = ctx["snap"]
        ctx.update(markets=[(m, views.title(m.station, m.metric, m.lo, m.hi, m.event_date))
                            for m in (snap.markets if snap else ())],
                   selected=ticker)
        return templates.TemplateResponse(request, "bet.html", ctx)

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
            ctx = _frame(conn, request, "scoreboard")
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
        ctx.update(cards=cards, n_events_required=N_EVENTS_REQUIRED)
        return templates.TemplateResponse(request, "scoreboard.html", ctx)

    @app.get("/log", response_class=HTMLResponse)
    def order_log(request: Request):
        with closing(_read_conn()) as conn:
            ctx = _frame(conn, request, "log")
            ctx["orders"] = conn.execute("SELECT * FROM orders ORDER BY id DESC LIMIT 500").fetchall()
        return templates.TemplateResponse(request, "log.html", ctx)

    return app
