"""The 24 settlement stations (config/citycast_stations.yaml, spec §2.2)."""
from dataclasses import dataclass
from pathlib import Path

import yaml

from highcast.config import AppConfig, StationConfig

REGISTRY_PATH = Path(__file__).resolve().parents[2] / "config" / "citycast_stations.yaml"


@dataclass(frozen=True)
class Station:
    icao: str
    iem_id: str
    iem_network: str
    wfo: str
    lat: float
    lon: float
    elevation_m: float
    tz: str
    std_utc_offset: int
    high_series: str
    low_series: str
    high_aliases: tuple[str, ...] = ()
    low_aliases: tuple[str, ...] = ()


def load_stations(path: Path = REGISTRY_PATH) -> dict[str, Station]:
    raw = yaml.safe_load(Path(path).read_text())
    out: dict[str, Station] = {}
    for entry in raw["stations"]:
        e = dict(entry)
        e["high_aliases"] = tuple(e.get("high_aliases") or ())
        e["low_aliases"] = tuple(e.get("low_aliases") or ())
        out[e["icao"]] = Station(**e)
    return out


def series_index(stations: dict[str, Station]) -> dict[str, tuple[str, str]]:
    idx: dict[str, tuple[str, str]] = {}
    for st in stations.values():
        for metric, primary, aliases in (("high", st.high_series, st.high_aliases),
                                         ("low", st.low_series, st.low_aliases)):
            for s in (primary, *aliases):
                if s in idx:
                    raise ValueError(f"series {s} mapped twice")
                idx[s] = (st.icao, metric)
    return idx


def app_config(st: Station, db_path) -> AppConfig:
    """A minimal highcast AppConfig so highcast helpers (latentmax, parsers) run per station."""
    return AppConfig(
        station=StationConfig(key=st.icao.lower(), name=st.icao, icao=st.icao, iem_id=st.iem_id,
                              iem_network=st.iem_network, wfo=st.wfo, lat=st.lat, lon=st.lon,
                              tz=st.tz, std_utc_offset=st.std_utc_offset),
        display_tz="Asia/Manila", synoptic_token="", db_path=Path(db_path))
