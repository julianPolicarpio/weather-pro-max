# citycast 2a coverage — 2026-10-08

Backfill: full run 2026-10-07 19:34 → 2026-10-08 05:06 Manila, then two gap-fill reruns. Database: data/history.db (not in git).

```
station     cli    MEX    NBS    GFS    NBE    obs  events    mkts  usable candles  pass
KATL      99.8% 100.0% 100.0% 100.0% 100.0% 100.0%     432    2592    2592    2592  yes
KAUS      99.9% 100.0% 100.0% 100.0% 100.0%  99.9%    1539    9231    9231    9231  yes
KBOS      99.4% 100.0% 100.0% 100.0% 100.0%  99.9%     431    2586    2586    2586  yes
KDCA      99.5% 100.0% 100.0% 100.0% 100.0% 100.0%     450    2700    2700    2700  yes
KDEN      99.8% 100.0% 100.0% 100.0% 100.0%  99.9%     984    5904    5904    5904  yes
KDFW      99.5% 100.0% 100.0% 100.0% 100.0% 100.0%     425    2550    2550    2550  yes
KEWR      99.8% 100.0% 100.0% 100.0% 100.0% 100.0%      84     504     504     504  yes
KHOU      99.4% 100.0% 100.0% 100.0% 100.0% 100.0%     495    2970    2970    2970  yes
KLAS      99.5% 100.0% 100.0% 100.0% 100.0% 100.0%     453    2718    2718    2718  yes
KLAX      99.7% 100.0% 100.0% 100.0% 100.0% 100.0%     938    5628    5628    5628  yes
KMDW      99.9% 100.0% 100.0% 100.0% 100.0% 100.0%    2170   11479   11479   11479  yes
KMIA      99.3% 100.0% 100.0% 100.0% 100.0% 100.0%    1542    9250    9250    9250  yes
KMSP      99.9% 100.0% 100.0% 100.0% 100.0% 100.0%     432    2592    2592    2592  yes
KMSY      69.2% 100.0% 100.0% 100.0% 100.0% 100.0%     453    2718    2718    2718  NO
KNYC      99.8% 100.0% 100.0% 100.0% 100.0% 100.0%    2182   11482   11482   11482  yes
KOKC      99.6% 100.0% 100.0% 100.0% 100.0%  99.8%     425    2550    2550    2550  yes
KPHL      99.8% 100.0% 100.0% 100.0% 100.0% 100.0%     984    5904    5904    5904  yes
KPHX      99.7% 100.0% 100.0% 100.0% 100.0% 100.0%     432    2592    2592    2592  yes
KSAN      99.4% 100.0% 100.0% 100.0% 100.0% 100.0%      98     588     588     588  yes
KSAT      99.8% 100.0% 100.0% 100.0% 100.0% 100.0%     425    2550    2550    2550  yes
KSDF      99.7% 100.0% 100.0% 100.0% 100.0% 100.0%      80     480     480     480  yes
KSEA      99.7% 100.0% 100.0% 100.0% 100.0% 100.0%     453    2718    2718    2718  yes
KSFO      98.6% 100.0% 100.0% 100.0% 100.0% 100.0%     453    2718    2718    2718  yes
KTTN      99.7% 100.0% 100.0% 100.0% 100.0%  99.0%      84     504     504     504  yes
disagreements (Kalshi settled value vs CLI final): 14
  KBOS low 2026-05-26: kalshi 54 cli 57
  KDEN high 2025-06-20: kalshi 98 cli 99
  KHOU high 2025-01-26: kalshi 77 cli 70
  KLAX high 2025-07-04: kalshi 73 cli 74
  KLAX high 2026-01-04: kalshi 63 cli 62
  KMDW high 2021-10-01: kalshi 84 cli 86
  KMDW high 2022-04-02: kalshi 47 cli 45
  KMDW high 2023-01-04: kalshi 46 cli 45
  KMDW high 2026-01-04: kalshi 33 cli 34
  KMIA high 2024-11-27: kalshi 84 cli 81
  KMIA high 2026-08-29: kalshi 90 cli 85
  KMSP low 2026-05-12: kalshi 55 cli 52
  KPHL high 2025-12-26: kalshi 32 cli 33
  KSEA low 2026-07-22: kalshi 65 cli 64
ACCEPTANCE: FAIL
```

## Known gaps

- **KMSY (New Orleans) CLI 69.2 %**: IEM's CLI archive for KMSY starts 2022-10-01 (verified: year 2021 returns 0 reports). From that date it is complete (1,458 final of 1,465 days). Source limit, not a fetch failure. Acceptance otherwise passes for all 24 stations.
- Two current-half-year MOS chunks (KLAX GFS, KPHL NBE 2026-07-01) hit IEM read errors on the last rerun; current chunks are refetched every run and both stations pass.

## Kalshi settled value vs NWS final CLI

14 events differ. Checked against data/highcast.db's independently fetched CLI for KMIA 2026-08-29 (CLI 85, Kalshi paid 90–91), KMIA 2024-11-27, KLAX 2025-07-04, KLAX 2026-01-04: CLI identical, so Kalshi settled on a different value than the NWS final report. Consequence (plan 3): decision rows score wins by Kalshi's result; city models keep CLI final as weather truth.
{'cli_daily': 49806, 'mos_runs': 8390058, 'obs': 1435480, 'kalshi_markets': 95508, 'kalshi_candles': 3019131}
