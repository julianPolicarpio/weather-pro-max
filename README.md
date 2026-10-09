# Weather Pro Max

Weather Pro Max forecasts daily temperature-market settlement outcomes. The `highcast` command builds high and low forecasts for KLAX (Los Angeles) and KMIA (Miami), using the NWS Daily Climate Report settlement value. It also includes `sandbox`, a paper-trading environment that evaluates strategies against live Kalshi temperature-market data. The sandbox covers open high and low temperature markets it discovers across 24 US cities; the weather forecasts from `highcast` are currently configured for KLAX and KMIA.

This is an accuracy-focused forecasting and research project, not production trading software. Forecasts are uncertain and may be wrong; any forecast before the final NWS Daily Climate Report is provisional because the preliminary report can change. Paper results do not establish future performance or justify real-money trades. Nothing here is financial advice, and the sandbox cannot place real orders. Market access and eligibility are determined by Kalshi and may change. See the settlement rules in [LAXHIGH.pdf](LAXHIGH.pdf) and the [sandbox design](an internal design note).

## Setup

Python 3.12 or newer is required. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

The optional Synoptic token enables a paid, higher-frequency observation feed. The core fetch, forecast, backtest, dashboard, and sandbox workflows can run without it. The first historical backfill and live data refreshes use network services.

## Forecast workflow

Initialize the database and load five years of source data:

```bash
highcast init-db
highcast backfill --years 5
```

Fetch a current snapshot, create a forecast, or start the local dashboard:

```bash
highcast fetch
highcast predict --metric high
highcast predict --metric low --live
highcast serve
```

The dashboard is available at `http://127.0.0.1:8712`. `--live` requests an intraday forecast; without it, `predict` makes a day-ahead forecast. Use `--date YYYY-MM-DD` to target a date. `highcast --help` and `highcast <command> --help` show the available commands and options.

These examples use the default KLAX configuration. To run a command for KMIA, place the global `--config` option before the subcommand, for example `highcast --config config/kmia.yaml fetch`; use the same form with `predict` or `backfill` to build Miami data and forecasts.

The `citycast` command provides a separate research pipeline for city-level datasets and model evaluation. Its station roster is maintained in [config/citycast_stations.yaml](config/citycast_stations.yaml); it is not the two-station `highcast` forecast configuration.

## Paper-trading sandbox

The sandbox keeps its paper ledger in `data/sandbox.db`, separate from the highcast forecast database. Start it and view its local dashboard or report:

```bash
sandbox init
sandbox cycle --force-decide
sandbox serve
sandbox report
```

The dashboard uses `http://127.0.0.1:8713` by default. A cycle reads public market data and records simulated decisions; it does not submit orders. The report's unlock status is an evidence heuristic with minimum sample gates and confirmation stages, not a guarantee of skill or a signal to trade. See the [sandbox design](an internal design note) for its limits and methodology.

For model caveats, decision-hour verification, and operating details, see [HANDOFF.md](HANDOFF.md), [ACCURACY_IMPROVEMENTS.md](ACCURACY_IMPROVEMENTS.md), and the [forecasting design](an internal design note). The handoff also documents the current wake and launchd setup.

## Tests

Run the project suite with:

```bash
pytest
```

Tests marked `live` are excluded by default. They require network access when explicitly selected.
