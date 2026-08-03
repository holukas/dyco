# Worked example: a 10 Hz QCL record

A complete run on real data, from opening the file to reading the output. The record is CZ-Lnz: a
year of 10 Hz eddy covariance measured with a quantum cascade laser, CH₄, N₂O and H₂O down one inlet,
already split into 30-minute files.

The settings themselves belong to this record. What transfers is how each one was arrived at — some
come straight off the file, the rest come from measuring the record before committing to a full run.

## The raw file

```
"TIMESTAMP","RECORD","u","v","w","Ts","CH4","N2O","H2O","Flowrate"
"2022-11-20 01:00:00.1",16940155,-0.9412396,-1.238613,0.078326,15.06647,2142.96,338.271,3812030,4.033986
"2022-11-20 01:00:00.2",16940156,-0.8894647,-1.249233,0.1075323,15.07443,2140.56,338.102,3751980,4.033986
"2022-11-20 01:00:00.3",16940157,-0.8111387,-1.220027,-0.03982678,15.07443,2140.4,338.402,3813650,4.033986
...
"2022-11-20 01:03:28.4",16942238,-0.4274679,-0.07168716,0.1048757,14.62816,-9999,-9999,-9999,4.033985
```

Six things in that fragment decide six settings:

| What the file shows | Setting it fixes |
|---|---|
| Column names on line 1, data on line 2 — **no units row** | `--skiprows 0 --extra-rows 0` |
| Comma-separated | `--sep ,` |
| Wind and sonic temperature are `u`, `v`, `w`, `Ts` | `--col-u u --col-v v --col-w w --col-tsonic Ts` |
| Gases are `CH4`, `N2O`, `H2O` | one `--scalar` each |
| Timestamps step by 0.1 s | `--hz 10` |
| Gaps are `-9999` | nothing — already a dyco default |

`RECORD` and `Flowrate` are carried through untouched; dyco only shifts the columns it is told about.

The header is **quoted** and the data rows are not. That mismatch is normal for logger output: the
quotes are stripped from the column names on read, and the header line is written back byte-identical,
so the output stays a drop-in replacement for the input.

:::{note}
`--extra-rows` defaults to **2**, for the units and instrument rows a typical raw EC file carries.
This file has neither. Leaving the default would consume the first two data rows and silently
misalign everything after them, so it has to be set to `0`.
:::

### The record as a whole

Each file holds exactly 18000 records — 30 minutes at 10 Hz — so `--chunk-seconds 1800` gives one
chunk per file and the chunking is a formality. The full record is 16712 files, 2022-05-16 to
2023-06-26.

About **one file in eight has all three gas columns filled with `-9999` end to end**: the analyser was
offline for that half hour. dyco writes those periods through untouched and marks them `no_data`.

## Measuring before running

The remaining settings — search window, donor gas, carry limit — cannot be read off the file. They
come from a survey of 60 half hours spread across the year and across all hours of the day.

### Where the lag actually is

Running PWB with a **wide, unconstrained** window (±30 s) gives an unusable answer. On the file above,
sonic temperature peaks at **+6.1 s** — impossible, since `w` and `Ts` come from the same instrument
and share no tube. At that scale the cross-covariance is dominated by trend, not by flux.

Confining the search to physically possible delays fixes it. Across the survey, and across every
window setting tried, the answer is the same:

| Gas | Median lag | Detections PWBOPT calls reliable (S1) |
|---|---|---|
| CH₄ | +9.00 s | 28% |
| N₂O | +8.90 s | **1.7%** |
| H₂O | +8.90 s | 30% |

One instrument, one inlet, one tube delay — as expected.

:::{important}
An S1 flag from a windowed search is a weaker claim than one from an unwindowed search: the window
did part of the work. It is still the right choice here, because the unwindowed search demonstrably
locks onto trend artefacts.
:::

### The lag is not constant

Reliable H₂O detections, by month:

| Period | Lag |
|---|---|
| May 2022 | 11.1 – 13.8 s |
| June 2022 | 8.4 – 12.0 s |
| July 2022 onward | 8.0 – 9.5 s |

There is a step change in mid-June 2022 — a pump, a tube or a flow setting changed. This is what dyco
exists for, and it drives two settings. The window has to reach past 14 s, so the RFlux default of
`[0, 10]` would clip the first month. And the carry limit has to be finite, or a period in May could
inherit a lag from across the step.

Hence `--lws 4 --uws 16`: wide enough for the whole year, with a lower bound that discards the
near-zero and negative peaks no tube delay can produce.

### Which gas donates

N₂O detects its own lag in under 2% of periods — the flux is too small for the cross-correlation to
locate a peak. It needs a donor, and the donor must be **CH₄, not H₂O**, even though H₂O is the most
reliable detector in the record.

H₂O adsorbs onto and desorbs from the tube wall, so its lag runs longer than the flow-through delay
and moves with humidity and the state of the tube. CH₄ and N₂O ride the flow. Donating H₂O's lag to
N₂O would transplant a wall-interaction delay onto a gas that has none. In this record the effect is
small but consistently in that direction: on the eight periods where both detect reliably, H₂O is the
longer of the two in seven, by a median of 0.25 s. A shorter or unheated inlet, or a more humid site,
would widen that.

So `--scalar "N2O:N2O@lagfrom=CH4"`, with CH₄ declared first so its lags resolve before N₂O borrows
them.

## The command

```bash
dyco detect-remove --input-dir ./split_30min --output-dir ./dyco_out --file-pattern "*_QCL.txt" --col-u u --col-v v --col-w w --col-tsonic Ts --scalar "CH4:CH4" --scalar "N2O:N2O@lagfrom=CH4" --scalar "H2O:H2O" --hz 10 --wdt 6 --lag-max 20 --lws 4 --uws 16 --block-length 40 --n-bootstrap 99 --skiprows 0 --extra-rows 0 --sep "," --chunk-seconds 1800 --start-time-regex "(\d{12})" --start-time-format "%Y%m%d%H%M" --chunk-name-template "{starttime}_QCL{suffix}" --max-carry 48 --random-state 42 --save-plots
```

The settings that are not self-explanatory:

| Flag | Why this value |
|---|---|
| `--extra-rows 0` | The file has no units row. The default of 2 would eat data. |
| `--hz 10` | From the 0.1 s timestamp step. Also why `--wdt 6` rather than the default 5. |
| `--wdt 6` | The paper's CCF smoothing width is `hz/2 + 1`, which is 6 at 10 Hz. The default of 5 follows RFlux, which assumes 20 Hz. |
| `--lws 4 --uws 16` | Covers 8–14 s, the whole year's range, and excludes the nonphysical peaks a wider window attracts. |
| `--lag-max 20` `--block-length 40` | The window must fit inside `lag_max`. Block length follows R's `2 × lag_max` coupling; the CLI default of 20 s breaks that coupling and measurably costs reliability here (CH₄ S1 fell from 27% to 18%). |
| `@lagfrom=CH4` | N₂O detects in under 2% of periods. CH₄ is inert like N₂O; H₂O is not. |
| `--max-carry 48` | One day. The default is unlimited, which would let a lag travel across the mid-June step change. `@lagfrom=` and `--max-carry` are meant to be set together. |
| `--random-state 42` | Makes the block bootstrap reproducible. |

`--file-pattern "*_QCL.txt"` matters because the input folder also holds a split summary and a log
file, which are not raw data.

Runtime is roughly 3 s per file on one core — about two hours for the full record on eight workers.

:::{note}
**This record also contains literal `-Inf` values** — four of them in one H₂O column, written by the
logger. Nothing needs to be configured for it: dyco treats any non-finite value as missing, the same
as `-9999`. Which is just as well, since `--na-values` could not express it anyway (`argparse` reads
a leading `-Inf` as an option name and refuses the command).

They are excluded from `{gas}_n_valid`, so that half hour reports 17882 valid H₂O records rather
than 17886.
:::

## The same run in the TUI

`dyco tui` takes the same settings, with three differences worth knowing.

| TUI field | Value |
|---|---|
| Input dir / Output dir | as above |
| Wind U / V / W, Sonic T | `u` `v` `w` `Ts` |
| Scalars | `CH4:CH4,N2O:N2O,H2O:H2O` |
| Lag from | `N2O:CH4` |
| Frequency | `10` |
| Lag max s | `16` |
| Win s | `CH4:[4,16],N2O:[4,16],H2O:[4,16]` |
| Chunk s | `1800` |
| Bootstraps | `99` |
| Max carry | `48` |
| Skip rows / Extra rows | `0` / `0` |
| Separator | `,` |
| File glob | `*_QCL.txt` |
| Start regex / Start format | `(\d{12})` / `%Y%m%d%H%M` |
| Name tmpl | `{starttime}_QCL{suffix}` |
| Random seed | `42` |
| Save plots | on |

The donor goes in its own **Lag from** field, not into **Scalars**. Type **Win s** directly rather
than pressing ⟳, which re-seeds it symmetrically from **Lag max** and would discard the lower bound
of 4.

The TUI has no field for `--wdt` or `--block-length`. It leaves `wdt` at 5 and derives the block
length from the window, so `[4,16]` gives `lag_max 16, block 32` rather than `20 / 40` — which is why
**Lag max s** reads 16 above. Measured against the CLI settings on the same 60 half hours, the largest
disagreement among the detections that actually reach the data is **0.20 s**, two records at 10 Hz,
well inside the 0.5 s reliability threshold.

## What the output looks like

`2_lag_removed/` holds one lag-corrected file per period, named for its own start time
(`202211200100_QCL.txt`), with the header and every untouched column exactly as they came in.

`1_lag_detection/detect_and_remove_tlag_summary.csv` carries the numbers, and
`detect_and_remove_tlag_decisions.txt` says why each lag was applied, in words. From a run over
17–20 May 2022:

```
202205170000_QCL.txt
  CH4   +13.90 s (+139 rec)  detected here and reliable (S1): HDI 0.30 s < 0.50 s
  N2O   +11.40 s (+114 rec)  nothing had been detected yet at this point in the run, so the lag was
                             filled 48 period(s) back from 202205180000_QCL.txt, the first period
                             that did
  H2O   +11.50 s (+115 rec)  median of this gas's raw detections - PWBOPT rejected every one of them,
                             so there was nothing better anywhere in the run

202205170100_QCL.txt
  CH4   +13.90 s (+139 rec)  detected here, accepted for continuity (S2): HDI 0.50 s is wide, but the
                             lag is within 0.50 s of the preceding optimal one

202205191600_QCL.txt
  CH4           not applied  every record of this gas is missing in this period - no lag needed, and
                             none applied
```

That covers most of what PWBOPT does: a reliable detection (S1), one accepted for continuity (S2), a
back-fill for periods before the first detection, the median last resort, and a period the analyser
sat out entirely. Note the lags are 11–14 s — this is May 2022, before the step change.

Over those four days (191 periods), `{gas}_lag_source` came out as:

| Source | CH₄ | N₂O | H₂O |
|---|---|---|---|
| `own` (detected or carried) | 127 | 127 | 113 |
| `no_data` | 64 | 64 | 64 |
| `median` | — | — | 14 |
| `from:CH4` | — | **0** | — |

The donor never fired. With `--max-carry 48`, N₂O always had one of its own detections within a day,
so `@lagfrom=` acted purely as a safety net. That is the intended outcome, not a sign the setting is
unnecessary — shorten the carry limit and the donor starts carrying the load.

`{gas}_n_valid` counts the records present per period. It is what marks the `no_data` periods, and
doubles as a QC signal, since it also catches periods that are only *partly* missing — those are
detected normally.

:::{important}
Downstream flux processing must run with time-lag maximization **disabled**. The lag has already been
removed.
:::
