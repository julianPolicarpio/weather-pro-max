from dataclasses import dataclass
from highcast.config import AppConfig


@dataclass(frozen=True)
class MetricSpec:
    key: str
    target_col: str
    openmeteo_var: str
    nws_field: str
    mos_field: str | None
    kalshi_series: str
    extremum: str          # "max" | "min"
    tail_dir: str          # "floor" (>=T) | "cap" (<=T)
    cutoff_local_hour: int
    cutoff_day_offset: int  # 0 = cutoff on target day; -1 = prior day (low)
    fc_models: list[str]


def get_metric(cfg: AppConfig, key: str) -> MetricSpec:
    if key == "high":
        return MetricSpec(
            key="high", target_col="high_f",
            openmeteo_var="temperature_2m_max", nws_field="maxTemperature",
            mos_field="n_x", kalshi_series=cfg.station.kalshi_series,
            extremum="max", tail_dir="floor",
            cutoff_local_hour=9, cutoff_day_offset=0,
            fc_models=["ecmwf", "gfs", "icon", "hrrr", "nws", "mos_mex",
                       "nbm", "nam", "aifs", "nbm_nbs"])
    if key == "low":
        return MetricSpec(
            key="low", target_col="low_f",
            openmeteo_var="temperature_2m_min", nws_field="minTemperature",
            # mos_field WAS None, with the comment "IEM MOS has no min-temp field
            # for the low metric". That was wrong: the MOS N/X column carries the
            # overnight MIN on its morning-valid rows, and `_mos_row` was
            # discarding them. The LOW therefore had ZERO MOS rows while the HIGH
            # had 16,254 back to 2021 -- it was the only metric with no
            # station-calibrated guidance at all, running on raw NWP grid points
            # whose KLAX offsets are large and drifting.
            mos_field="n_x", kalshi_series=cfg.station.kalshi_low_series,
            extremum="min", tail_dir="cap",
            cutoff_local_hour=20, cutoff_day_offset=-1,
            fc_models=["ecmwf", "gfs", "icon", "hrrr", "nws", "mos_mex",
                       "nbm", "nam", "aifs", "nbm_nbs"])
    raise ValueError(f"unknown metric: {key!r}")
