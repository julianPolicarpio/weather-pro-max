"""Invert the NWS 5-minute feed's rounding pipeline.

Per trader reports (verified empirically by `highcast verify-quantization`,
Task 14): the public 5-min feed rounds the true F reading to a whole degree,
converts to whole C, then displays F re-derived from that whole C. A displayed
value therefore stands for 1-2 possible true whole-F values.
"""
from highcast.timeutil import f_to_c, round_half_up


def candidates_from_reported_c(c_whole: int) -> list[int]:
    return [f for f in range(-30, 131)
            if round_half_up(f_to_c(f)) == c_whole]


def candidates_from_displayed_f(f_disp: int) -> list[int]:
    cs = [c for c in range(-40, 60) if round_half_up(c * 9.0 / 5.0 + 32) == f_disp]
    return sorted({f for c in cs for f in candidates_from_reported_c(c)})


def interval(values: list[int]) -> tuple[int, int]:
    return (min(values), max(values))
