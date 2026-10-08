"""IEM CLI archive -> cli_daily. Truth is final reports only (spec §1)."""
import json
from datetime import date

from citycast.backfill import Chunk
from highcast.fetchers.cli_reports import _product_issued_utc, cli_row_is_final, parse_rows

IEM_CLI = "https://mesonet.agron.iastate.edu/json/cli.py"


def cli_rows(payload: dict, st, fetched_at: str) -> list[dict]:
    out = []
    for r in parse_rows(payload, st.icao, fetched_at):
        product = json.loads(r["raw"]).get("product")
        issued = _product_issued_utc(product)
        out.append({"station": st.icao, "date": r["date"], "high_f": r["high_f"],
                    "low_f": r["low_f"],
                    "final": int(cli_row_is_final(product, r["date"], st.std_utc_offset)),
                    "issued_utc": issued.isoformat() if issued else None,
                    "raw": r["raw"], "fetched_at": fetched_at})
    return out


def chunks(paced, stations: dict, since: date, today: date, now_iso,
           complete_before: date | None = None) -> list[Chunk]:
    out = []
    complete_before = complete_before or today
    for st in stations.values():
        for year in range(since.year, today.year + 1):
            def run(st=st, year=year):
                payload = paced.get_json(IEM_CLI, {"station": st.icao, "year": year})
                return [("cli_daily", cli_rows(payload, st, now_iso()))]
            out.append(Chunk("cli", f"{st.icao}:{year}", date(year + 1, 2, 1) > complete_before, run))
    return out
