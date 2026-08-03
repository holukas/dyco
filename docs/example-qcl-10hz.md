# Worked example: a 10 Hz QCL record

A full run on real data, from the raw file to the output. The record is CZ-Lnz: one year of 10 Hz
eddy covariance from a quantum cascade laser, with CH₄, N₂O and H₂O sampled through one inlet, split
into 30-minute files.

Some settings can be read straight off the file. The rest have to be measured first.

## The raw file

```
"TIMESTAMP","RECORD","u","v","w","Ts","CH4","N2O","H2O","Flowrate"
"2022-11-20 01:00:00.1",16940155,-0.9412396,-1.238613,0.078326,15.06647,2142.96,338.271,3812030,4.033986
"2022-11-20 01:00:00.2",16940156,-0.8894647,-1.249233,0.1075323,15.07443,2140.56,338.102,3751980,4.033986
"2022-11-20 01:00:00.3",16940157,-0.8111387,-1.220027,-0.03982678,15.07443,2140.4,338.402,3813650,4.033986
...
"2022-11-20 01:03:28.4",16942238,-0.4274679,-0.07168716,0.1048757,14.62816,-9999,-9999,-9999,4.033985
```

Six things in that fragment fix six settings:

| What the file shows | Setting |
|---|---|
| Column names on line 1, data on line 2, no units row | `--skiprows 0 --extra-rows 0` |
| Comma-separated | `--sep ,` |
| Wind and sonic temperature are `u`, `v`, `w`, `Ts` | `--col-u u --col-v v --col-w w --col-tsonic Ts` |
| Gases are `CH4`, `N2O`, `H2O` | one `--scalar` each |
| Timestamps step by 0.1 s | `--hz 10` |
| Gaps are `-9999` | nothing, already a dyco default |

`RECORD` and `Flowrate` pass through untouched. dyco only shifts the columns you name.

The header is quoted and the data rows are not, which is normal for logger output. dyco strips the
quotes when reading and writes the header back unchanged, so the output can replace the input
directly.

:::{note}
`--extra-rows` defaults to 2, for the units and instrument rows most raw EC files carry. This file
has neither, so the default would read two data rows as header and shift everything after them. Set
it to `0`.
:::

### The record as a whole

Each file holds exactly 18000 records, 30 minutes at 10 Hz, so `--chunk-seconds 1800` makes each
file a single chunk. The full record is 16712 files, from 2022-05-16 to 2023-06-26.

In about one file in eight, all three gas columns are `-9999` throughout, because the analyser was
offline. dyco writes those periods through untouched and marks them `no_data`.

## Settings that have to be measured

The search window, the donor gas and the carry limit cannot be read off the file. The values below
come from a survey of 60 half hours spread across the year and across all hours of the day.

### Where the lag is

A wide, unconstrained window (±30 s) gives an unusable answer. On the file above, sonic temperature
peaks at +6.1 s, which is impossible: `w` and `Ts` come from the same instrument and share no tube.
Over a window that wide the cross-covariance follows the trend in the data, not the flux.

Restricting the search to physical delays fixes this. Across the survey, and across every window
setting tested, the result is the same:

| Gas | Median lag | Detections PWBOPT calls reliable (S1) |
|---|---|---|
| CH₄ | +9.00 s | 28% |
| N₂O | +8.90 s | 1.7% |
| H₂O | +8.90 s | 30% |

One instrument and one inlet, so one tube delay.

:::{important}
An S1 flag from a windowed search is weaker than one from an unwindowed search, because the window
has already excluded part of the answer. It is still the better option here, since the unwindowed
search locks onto trends.
:::

### The lag changes over the year

Reliable H₂O detections, by month:

| Period | Lag |
|---|---|
| May 2022 | 11.1 to 13.8 s |
| June 2022 | 8.4 to 12.0 s |
| July 2022 onward | 8.0 to 9.5 s |

Something changed in mid-June 2022, most likely the pump or the tubing. Two settings follow. The
window has to reach past 14 s, so the RFlux default of `[0, 10]` would cut off the first month. The
carry limit has to be finite, or a period in May can inherit a lag measured after the change.

That gives `--lws 4 --uws 16`: wide enough for the whole year, with a lower bound that drops the
near-zero and negative peaks no tube delay can produce.

### Which gas donates

N₂O finds its own lag in under 2% of periods, because the flux is too small for the
cross-correlation to locate a peak. It needs a donor, and that donor has to be CH₄ rather than H₂O,
even though H₂O is the most reliable detector here.

The reason is sorption. H₂O sticks to the tube wall and comes off again, so its lag is longer than
the travel time through the tube and shifts with humidity and tube age. CH₄ and N₂O move with the
flow. Giving N₂O the H₂O lag would add a wall effect it does not have.

The record shows this, though weakly. On the eight periods where both gases detect reliably, H₂O has
the longer lag in seven, by a median of 0.25 s. A shorter or unheated inlet, or a more humid site,
would widen the gap.

So `--scalar "N2O:N2O@lagfrom=CH4"`, with CH₄ listed first so its lags resolve before N₂O borrows
them.

## The command

```bash
dyco detect-remove --input-dir ./split_30min --output-dir ./dyco_out --file-pattern "*_QCL.txt" --col-u u --col-v v --col-w w --col-tsonic Ts --scalar "CH4:CH4" --scalar "N2O:N2O@lagfrom=CH4" --scalar "H2O:H2O" --hz 10 --wdt 6 --lag-max 20 --lws 4 --uws 16 --block-length 40 --n-bootstrap 99 --skiprows 0 --extra-rows 0 --sep "," --chunk-seconds 1800 --start-time-regex "(\d{12})" --start-time-format "%Y%m%d%H%M" --chunk-name-template "{starttime}_QCL{suffix}" --max-carry 48 --random-state 42 --save-plots
```

The settings that need explaining:

| Flag | Reason |
|---|---|
| `--extra-rows 0` | The file has no units row. The default of 2 would read data as header. |
| `--hz 10` | From the 0.1 s timestamp step. |
| `--wdt 6` | The paper sets the CCF smoothing width to `hz/2 + 1`, which is 6 at 10 Hz. The default of 5 follows RFlux, which assumes 20 Hz. |
| `--lws 4 --uws 16` | Covers 8 to 14 s, the range for the whole year, and excludes the unphysical peaks that a wider window attracts. |
| `--lag-max 20` `--block-length 40` | The window has to fit inside `lag_max`. Block length follows R's `2 × lag_max` rule. The CLI default of 20 s breaks that link and costs reliability here: CH₄ S1 drops from 27% to 18%. |
| `@lagfrom=CH4` | N₂O detects in under 2% of periods. CH₄ is inert like N₂O, H₂O is not. |
| `--max-carry 48` | One day. The default is unlimited, which would let a lag travel across the mid-June change. Set `@lagfrom=` and `--max-carry` together. |
| `--random-state 42` | Makes the block bootstrap reproducible. |

Use `--file-pattern "*_QCL.txt"` because the input folder also holds a split summary and a log file,
which are not raw data.

Runtime is about 3 s per file on one core, so roughly two hours for the full record on eight workers.

:::{note}
This record also contains literal `-Inf` values, four of them in one H₂O column, written by the
logger. Nothing needs configuring: dyco treats any non-finite value as missing, like `-9999`. Just
as well, since `--na-values` cannot express it. `argparse` reads a leading `-Inf` as an option name
and refuses the command.

Non-finite values are left out of `{gas}_n_valid`, so that half hour reports 17882 valid H₂O records
instead of 17886.
:::

## The same run in the TUI

`dyco tui` takes the same settings.

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

Three things differ from the command line. The donor goes in the **Lag from** field, not into
**Scalars**. Type **Win s** by hand rather than pressing ⟳, which re-seeds the window symmetrically
from **Lag max** and drops the lower bound of 4.

Third, the TUI has no field for `--wdt` or `--block-length`. It leaves `wdt` at 5 and derives the
block length from the window, so `[4,16]` gives `lag_max 16` and `block 32` instead of 20 and 40.
That is why **Lag max s** is 16 above. On the same 60 half hours, the largest difference from the
command-line settings, among detections that reach the data, is 0.20 s: two records at 10 Hz, well
inside the 0.5 s reliability threshold.

## The output

`2_lag_removed/` holds one lag-corrected file per period, named for its own start time
(`202211200100_QCL.txt`). The header and all untouched columns are exactly as they came in.

`1_lag_detection/detect_and_remove_tlag_summary.csv` holds the numbers.
`detect_and_remove_tlag_decisions.txt` gives the reason for each applied lag in words. From a run
over 17 to 20 May 2022:

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

That excerpt shows most of what PWBOPT does: a reliable detection (S1), one accepted for continuity
(S2), a back-fill for periods before the first detection, the median as last resort, and a period
the analyser sat out. The lags are 11 to 14 s here because this is May 2022, before the change.

Over those four days (191 periods), `{gas}_lag_source` was:

| Source | CH₄ | N₂O | H₂O |
|---|---|---|---|
| `own` (detected or carried) | 127 | 127 | 113 |
| `no_data` | 64 | 64 | 64 |
| `median` | | | 14 |
| `from:CH4` | | 0 | |

The donor was never used. With `--max-carry 48`, N₂O always had one of its own detections within a
day, so `@lagfrom=` only acted as a fallback. Shorten the carry limit and the donor starts supplying
lags.

`{gas}_n_valid` counts the records present in each period. It marks the `no_data` periods and is
useful for quality control, since it also catches periods that are only partly missing. Those are
detected as normal.

:::{important}
Downstream flux processing must run with time-lag maximization switched off. The lag has already
been removed.
:::
