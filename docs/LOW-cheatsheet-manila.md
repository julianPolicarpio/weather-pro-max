# LOW desk — Manila cheat sheet

All times **Manila (UTC+8)**. Measured 2026-08-01 from the code and the CLI
archive, not estimated. Julian trades both LOWs by hand; the bot does not touch
them.

⚠️ **SUMMER (PDT/EDT) TIMES.** Manila has no DST, the US stations do. In US
winter every station-local time below shifts **one hour later** in Manila.
Never hardcode a Manila clock — compute from the station timezone.

---

## The day at a glance

| | **KMIA LOW** | **KLAX LOW** |
|---|---|---|
| **DECIDE (commit hour)** | **19:00** | **21:00** |
| accuracy at that hour | 66.3% | 68.2% |
| still unknown at commit | 42% | 50% |
| window shuts | 04:00 | 04:00 |
| climate day runs | 13:00 → 13:00 | 16:00 → 16:00 |
| **settles (NWS CLI issued)** | **~16:25** | **~16:39** |

Both commit hours are each station's **median minimum time** — you are deciding
at the coin-flip point between "the min already happened" and "it hasn't".

---

## When the minimum actually happens

**It is BIMODAL.** The climate day runs 01:00→01:00 local, so it contains TWO
cooling periods: the ordinary dawn minimum, and a late-evening dip near the end
of the same climate day. The evening one is the trap.

### KMIA LOW (n=2,366 days)

| Manila | local | share | |
|---|---|---|---|
| 15:00 | 03:00 | 5.8% | |
| 16:00 | 04:00 | 9.2% | |
| 17:00 | 05:00 | 16.0% | |
| **18:00** | **06:00** | **18.1%** | ← modal |
| **19:00** | **07:00** | **9.8%** | ← **YOU DECIDE HERE** |
| 10:00 (next) | 22:00 | 1.8% | |
| **11:00 (next)** | **23:00** | **15.2%** | ← evening trap |

dawn 65.1% · evening 27.5% · **~32% still to come after you commit**

### KLAX LOW (n=1,961 days)

| Manila | local | share | |
|---|---|---|---|
| 18:00 | 03:00 | 6.9% | |
| 19:00 | 04:00 | 14.6% | |
| **20:00** | **05:00** | **20.8%** | ← modal |
| **21:00** | **06:00** | **14.8%** | ← **YOU DECIDE HERE** |
| 22:00 | 07:00 | 5.2% | |
| 13:00 (next) | 22:00 | 2.7% | |
| **14:00 (next)** | **23:00** | **22.1%** | ← **biggest single hour** |

dawn 67.2% · evening 32.3% · **~35% still to come after you commit**

**Read that last KLAX row again.** The single most common hour for KLAX's daily
minimum is 23:00 local — **17 hours after you commit**, and it is a bigger
share than the dawn modal hour. There is no feature anywhere in the model for
it. This is the project's largest known unmodelled error source.

---

## The 6-hour groups — these are what settle it

The hourly `:53` observations are SNAPSHOTS and miss the true extreme. The
`2sTTT` six-hour MINIMUM groups are continuously sampled, and they are part of
the settled answer. Only groups whose whole window lies inside the climate day
count.

| **KMIA** window | files (Manila) | | **KLAX** window | files (Manila) |
|---|---|---|---|---|
| 06–12Z | **19:53** | | 12–18Z | **01:53** |
| 12–18Z | **01:53** | | 18–00Z | **07:53** |
| 18–00Z | **07:53** | | 00–06Z | **13:53** |

**KMIA's 19:53 group files 53 minutes after your 19:00 commit** — so the group
covering the dawn minimum lands just *after* you decide. **KLAX's 13:53 group
is the one that captures the 23:00-local evening dip**, and it is the last word
on the day.

---

## Operating rules

1. **Never read the daily low off raw hourly obs.** Use
   `latentmax.visible_running_extremum(conn, cfg, day, "min")`, or include
   `six_min_c` in the SQL. A snapshot query understates the extreme.
2. **NWS display rounds; settlement does not.** The METAR body is whole °C, so
   the NWS page can read 1 °F away from the `T`-group value the CLI uses. The
   precise source matches settlement 96–99% of the time.
3. **A wide model-vs-market gap is a warning, not an opportunity.** Bets near
   the market price went 6W–0L; the wide-disagreement ones did not.
4. **Ladder parity is a coin flip per day.** Never assume even or odd buckets.
5. **The evening dip is your main residual risk on both LOWs.** If the dawn
   minimum is only marginally inside your bucket at commit time, the 23:00-local
   cooling can still take it out — most often at KLAX.
