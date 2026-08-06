# Worked example: a 10 Hz QCL record

A full run on one file: `CZ-Lnz_202208180700_QCL.txt`, the half hour starting 18 August 2022 at
07:00. It is 10 Hz eddy covariance from a quantum cascade laser, with CH₄, N₂O and H₂O on one inlet.
The file ships with dyco in `examples/data/`, so you can reproduce every number here.

Most settings come straight from the file. Three do not. For an easier case first, see the
[20 Hz IRGA example](example-irga-20hz.md), which uses two gases that both detect well.

## Where the half-hourly file came from

Files like this are cut from continuous logger downloads. Here the source is TOA5 output from a
CR3000, several GB per download, split into 30-minute files before dyco sees them.

:::{note}
Splitting is not part of this run, but two of its conventions decide whether your settings are right.

**The filename is the start of the period.** `CZ-Lnz_202208180700_QCL.txt` holds `07:00:00.1` to
`07:30:00.0`. Other splitters label by the end of the period instead. `--start-time-regex` and
`--start-time-format` only tell dyco how to read the timestamp out of the name, not what it means.
Get this wrong and every lag lands in the wrong half hour.

**The interval is `(start, end]`.** A TOA5 timestamp marks the end of its 0.1 s sample, so the record
stamped `07:00:00` belongs to the previous file, and `07:30:00` closes this one. This affects two
records per hour. Get it right and the file holds exactly 18000 records.

The splitter also reduced the four TOA5 header lines to the column-name row, wrote missing values as
`-9999`, and took H₂O out of scientific notation (`1.10044E+07` becomes `11004400`). The settings
below describe the file after all of that.
:::

You do not need an external splitter. dyco has `FileSplitter` and `FileSplitterMulti` as a library
API ([Splitting and rotating raw files](library.md#splitting-and-rotating-raw-files)), and
`dyco detect-remove` splits internally, so you can point it at long files. Here each file is already
one averaging period, so `--chunk-seconds 1800` makes each file one chunk.

## The raw file

```
"TIMESTAMP","RECORD","u","v","w","Ts","CH4","N2O","H2O","Flowrate"
"2022-08-18 07:00:00.1",7674070,-1.472227,0.1579756,-0.1845262,26.40051,2310.53,338.582,17098900,12.23491
"2022-08-18 07:00:00.2",7674071,-1.449659,0.2681603,-0.1539931,26.37662,2311.81,338.354,17066700,12.23491
"2022-08-18 07:00:00.3",7674072,-1.538603,0.2150593,-0.07168642,26.38458,2313.51,339.43,17110200,12.23491
...
"2022-08-18 07:00:04",7674109,-1.278408,0.3199338,-0.2840906,26.42839,-9999,-9999,-9999,12.23491
```

That fragment fixes six settings:

| What the file shows | Setting |
|---|---|
| Column names on line 1, data on line 2, no units row | `--skiprows 0 --extra-rows 0` |
| Comma-separated | `--sep ,` |
| Wind and sonic temperature are `u`, `v`, `w`, `Ts` | `--col-u u --col-v v --col-w w --col-tsonic Ts` |
| Gases are `CH4`, `N2O`, `H2O` | one `--scalar` each |
| Timestamps step by 0.1 s | `--hz 10` |
| Gaps are `-9999` | nothing, already a dyco default |

`RECORD` and `Flowrate` pass through untouched. dyco shifts only the columns you name.

The column names are quoted, and so is the timestamp on each data row. That is normal for logger
output and needs no setting, because dyco strips the quotes when reading. Without that, the third
column would be called `"u"` and every column you named would look missing.

:::{note}
`--extra-rows` defaults to 2, for the units and instrument rows most raw EC files carry. This file
has neither, so the default would read two data rows as header and shift everything after them. Set
it to `0`.
:::

The file holds 18000 records. In 21 of them all three gases are `-9999`, from brief analyser
dropouts. dyco reports `{gas}_n_valid` as 17979 and detects on the rest. If a gas is missing for a
whole period, there is nothing to detect and nothing to shift, so dyco writes the period through
untouched and marks that gas `no_data`.

## Settings you cannot read off the file

The search window, the donor gas and the carry limit. None of them shows up in a single file. The
values below come from a survey of 60 half hours across the source record, which covers one year,
2022-05-16 to 2023-06-26.

### The search window

A wide window (±30 s) can make the answer unusable. This file survives it, with all three gases still
between 8.9 and 9.1 s. `CZ-Lnz_202211200100_QCL.txt`, a November half hour, does not. There the same
command gives CH₄ +0.7 s, N₂O +5.1 s and H₂O +8.5 s, with HDI ranges of 49, 41 and 54 s. Over a
window that wide the cross-covariance follows the trend in the data instead of the flux, and the
bootstrap peaks scatter.

Limiting the search to physical delays fixes this. `--lws 4 --uws 16` is wider than this file needs,
because the tube delay drifted over the year: 11 to 14 s until mid-June 2022, then 8 to 9.5 s, most
likely a pump or tubing change. The window therefore has to reach past 14 s, and the RFlux default of
`[0, 10]` would cut off the first month. The lower bound of 4 s drops the near-zero and negative
peaks that no tube delay can produce.

:::{important}
An S1 flag from a windowed search is weaker than one from an unwindowed search, because the window
has already excluded part of the answer. It is still the better option here, since the unwindowed
search locks onto trends.
:::

### The donor gas

Across the record, N₂O finds its own lag in under 2% of periods. Its flux is too small for the
cross-correlation to find a peak, so it needs a donor. That donor has to be CH₄, not H₂O, even though
H₂O detects best here.

The reason is sorption. H₂O sticks to the tube wall and comes off again, so its lag is longer than
the travel time through the tube, and it moves with humidity and tube age. CH₄ and N₂O travel with
the flow. Giving N₂O the H₂O lag would add a wall effect it does not have.

This file shows it: 8.90 s for both inert gases, 9.10 s for H₂O. One instrument and one inlet, so the
extra 0.2 s is wall interaction, not tube length.

That gives `--scalar "N2O:N2O@lagfrom=CH4"`, with CH₄ listed first so its lags resolve before N₂O
borrows them. On this file the donor is never used, because N₂O detects on its own.

### The carry limit

`--max-carry 48` is one day. The default is unlimited, which would let a lag cross the mid-June
change and land in a period recorded weeks earlier under a different tube delay. Set `@lagfrom=` and
`--max-carry` together.

A single file has nothing to carry, so this setting does nothing here. It is in the command below
because a full run needs it.

## The command

```bash
dyco detect-remove --input-dir ./split_30min --output-dir ./dyco_out --file-pattern "*_QCL.txt" --col-u u --col-v v --col-w w --col-tsonic Ts --scalar "CH4:CH4" --scalar "N2O:N2O@lagfrom=CH4" --scalar "H2O:H2O" --hz 10 --wdt 6 --lag-max 20 --lws 4 --uws 16 --block-length 40 --n-bootstrap 99 --skiprows 0 --extra-rows 0 --sep "," --chunk-seconds 1800 --start-time-regex "(\d{12})" --start-time-format "%Y%m%d%H%M" --chunk-name-template "{starttime}_QCL{suffix}" --max-carry 48 --random-state 42 --save-plots
```

Point `--input-dir` at the folder holding the file. The settings that need explaining:

| Flag | Reason |
|---|---|
| `--extra-rows 0` | The file has no units row. The default of 2 would read data as header. |
| `--hz 10` | From the 0.1 s timestamp step. |
| `--wdt 6` | The paper sets the CCF smoothing width to `hz/2 + 1`, which is 6 at 10 Hz. The default of 5 follows RFlux, which assumes 20 Hz. |
| `--lws 4 --uws 16` | Covers 8 to 14 s and excludes the unphysical peaks a wider window attracts. |
| `--lag-max 20` `--block-length 40` | The window has to fit inside `lag_max`. Block length follows R's `2 × lag_max` rule. The CLI default of 20 s breaks that link and costs reliability here: CH₄ S1 drops from 27% to 18%. |
| `@lagfrom=CH4` | N₂O rarely detects on its own. CH₄ is inert like N₂O, H₂O is not. |
| `--max-carry 48` | One day, so no lag crosses the mid-June change. |
| `--random-state 42` | Makes the block bootstrap reproducible. |
| `--chunk-seconds 1800` | The file is already one averaging period, so this makes it one chunk. |

Use `--file-pattern "*_QCL.txt"` if the input folder also holds a split summary or a log file.

Runtime is about 3 s per file on one core.

## What the run found

```
CH4=8.90s HDI=0.10  N2O=8.90s HDI=0.30  H2O=9.10s HDI=0.10  theta=+136.1° phi=+0.7°
```

| Gas | Lag | HDI range | `lag_source` | PWBOPT |
|---|---|---|---|---|
| CH₄ | 8.90 s | 0.10 s | `own` | S1 |
| N₂O | 8.90 s | 0.30 s | `own` | S1 |
| H₂O | 9.10 s | 0.10 s | `own` | S1 |

All three HDI ranges are below the 0.5 s threshold, so all three lags are used directly. The donor
and the carry limit play no part.

This half hour is better than most, which is why it was chosen. In the source record only 33 periods
out of 8199 have all three gases reliable at the same time, and none of those fall between 09:00 and
16:00. Daytime mixing flattens the N₂O signal, so the example comes from early morning.

## What the run wrote

`1_lag_detection/detect_and_remove_tlag_decisions.txt` gives the reason for each applied lag. For
this file it is three lines:

```
202208180700_QCL.txt
  CH4     +8.90 s (+89 rec)  detected here and reliable (S1): HDI 0.10 s < 0.50 s
  N2O     +8.90 s (+89 rec)  detected here and reliable (S1): HDI 0.30 s < 0.50 s
  H2O     +9.10 s (+91 rec)  detected here and reliable (S1): HDI 0.10 s < 0.50 s
```

A longer run also produces the other cases: lags accepted for continuity (S2), carried from a
neighbouring period (S3), borrowed through `@lagfrom=`, or taken from the median as a last resort.
[What a run writes](output.md#which-lag-was-applied-and-why) covers all of them, with the summary CSV
and the diagnostic plots.

`2_lag_removed/202208180700_QCL.txt` is the corrected file, named for the start of its own period.
The header row comes back byte for byte, and the columns dyco did not shift keep their values.

:::{note}
Removing a lag costs the tail of the period. CH₄ shifts forward by 8.90 s, so its last 89 records
have nothing to draw from and are written as `-9999`. H₂O loses 91. The data is in the next file, so
this cannot be avoided. It is also why flux software should read these files with time-lag
maximization switched **off**, not just set to a narrow window.

Two formatting details change on the way through. The timestamp is written without its quotes, and a
value that happens to be a whole number gains a trailing `.0` (`0` becomes `0.0`, 75 times in the
wind columns of this file). Neither affects a numeric parser, but do not expect the output to be
byte-identical to the input.
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

Three things differ from the command line. The donor goes in the **Lag from** field, not in
**Scalars**. Type **Win s** by hand instead of pressing ⟳, which re-seeds the window symmetrically
from **Lag max** and drops the lower bound of 4.

The TUI also has no field for `--wdt` or `--block-length`. It leaves `wdt` at 5 and derives the block
length from the window, so `[4,16]` gives `lag_max 16` and `block 32` instead of 20 and 40. That is
why **Lag max s** is 16 above. On the same 60 half hours, the largest difference from the
command-line settings is 0.20 s among detections that reach the data. That is two records at 10 Hz,
well inside the 0.5 s reliability threshold.

:::{note}
This record also contains literal `-Inf` values, written by the logger into H₂O. Nothing needs
configuring, because dyco treats any non-finite value as missing, like `-9999`. Just as well, since
`--na-values` cannot express it. `argparse` reads a leading `-Inf` as an option name and refuses the
command. Non-finite values are left out of `{gas}_n_valid`.
:::
