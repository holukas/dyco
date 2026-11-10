# Command reference

Everything is reachable through one command:

```bash
dyco                    # list the workflows
dyco <command> --help   # options for one of them
```

| Command | Does | When |
|---|---|---|
| [`dyco tui`](tui.md) | `detect-remove` behind a form, with live validation and a preflight check. | **Start here.** |
| [`dyco detect-remove`](detect-remove.md) | **The main command.** Split long raw files into averaging-period chunks, rotate, detect the lag per chunk, then remove it. One pass, one output folder. | Scripted runs, or when you prefer the command line. |
| [`dyco pwb-batch`](pwb-batch.md) | Detect only, on files that are *already* split into averaging periods. Writes `tlag_results.csv` and stops. | Step 1 of the two-step route. |
| [`dyco apply-batch`](apply-batch.md) | Remove lags listed in an existing `tlag_results.csv`. Detects nothing. | Step 2 of the two-step route. |
| [`dyco fdl`](fdl.md) | Detect the lag as `detect-remove` does, then compute each period's flux detection limit and the flux at that lag. Writes `pwb_fdl_results.csv` and one figure per gas, no data files. | To find out which fluxes stand out from noise. See [Flux detection limit](../fdl.md). |

Each also exists standalone: `dyco-detect-remove`, `dyco-detect-remove-tui`, `dyco-pwb-batch`,
`dyco-apply-batch`, `dyco-fdl`.

The file splitter and the covariance-maximization estimator are library APIs with no command of their
own. The flux detection limit has both: `dyco fdl`, and the classes described in the
[Library API](../library.md).

:::{note}
The **All options** section of each page below is generated from the command's argparse parser, so
it matches what `--help` prints. The `dyco tui` page is the exception: its options are copied by
hand.
:::

```{toctree}
:maxdepth: 1

detect-remove
pwb-batch
apply-batch
fdl
tui
```

## Taking a gas's lag from another gas

A trace gas can be too noisy to locate its own lag. When that happens PWBOPT rejects the detections,
and the last resort is the *median of all the gas's raw detections in the run*, accepted or not.
Without `--max-carry` this median is reached only when PWBOPT accepted none, so it is the median of
rejected numbers, which on real data is often a negative lag no tube can produce. A reference gas travelling the same tube is a better answer.

Say so per gas, with `@lagfrom=` (**Lag from** in the TUI):

```bash
dyco detect-remove ... --scalar "CO2:CO2_DRY_[IRGA72-A]" --scalar "N2O:N2O_DRY_[QCL-C2]@lagfrom=CO2"
```

N₂O keeps every lag it *detects and PWBOPT accepts* (S1 or S2). Where it has none, it uses its own
lag carried forward from an earlier period, or filled back from a later one, as far as that lag is
allowed to travel (see [Limiting how far a lag may be carried](#limiting-how-far-a-lag-may-be-carried)).
Only beyond that does it take the CO₂ lag, **for that same period**, so a donor lag that drifts is
followed rather than flattened into a constant. The summary records the choice for every period
in `{gas}_lag_source` (`own`, `from:CO2`, `median`), so a borrowed lag is never mistaken for a
detected one.

The gas's own lag comes first in all three forms (detected, carried forward, filled back) on
purpose. Two gases down one tube still have different
delays (a systematic 0.35 s between CH₄ and N₂O is ordinary), so borrowing swaps a stale number for
a biased one. It is worth doing once the gas's own lag is old enough that staleness is the bigger
error, and that is a judgement the carry limit expresses.

**Pick a donor that behaves like the recipient.** H₂O is not interchangeable with the dry gases. It
sticks to the tube wall and comes off again, so its lag is longer than the travel time through the
tube, and it shifts with humidity and tube age. CO₂, CH₄ and N₂O move with the flow. H₂O is also, in
most runs, the gas that detects *most* reliably, which is what makes it the tempting donor and the
wrong one: giving its lag to an inert gas adds a wall effect that gas does not have, and biases the
flux. Let inert gases donate to inert gases, and keep a sticky donor for a sticky recipient.

Chains work (`CH4` from `N2O` from `CO2`); circular ones are rejected.

## Limiting how far a lag may be carried

PWBOPT's S3 rule gives a period with no usable detection the nearest earlier optimal lag, with no
limit on the distance: one good half hour can supply the rest of a week. Periods before the gas's
first detection are filled back from it, also without limit. `--max-carry N` (**Max carry** in the
TUI) caps both at N averaging periods. A forward carry beyond that expires
(`{gas}_flag_* = S3_expired`). The period then takes the gas's own lag filled back from a detection
up to N periods later, if there is one, and otherwise the donor gas's lag, or the median.

`{gas}_carry_periods` reports the distance for every period: `0` where the lag was detected in that
very period, `n` where it was carried forward `n` periods, and empty where it was filled back from a
later period, borrowed, or the median.

The default is unlimited, which is the published behaviour. It also means a donor is nearly idle:
with no limit, one accepted detection reaches every period of the run, forward and back, so the gas
borrows only if PWBOPT never accepts one of its detections. **`@lagfrom=` and `--max-carry` are meant to be set
together**, and dyco warns when a donor is named without one.

## The two-step route

`pwb-batch` to detect, then `apply-batch`, where `--scalar LABEL:column` reads the lag of one gas and
shifts the column of another. Use this route when other software has already split your files into
averaging periods, leaving nothing for `detect-remove` to chunk.

:::{important}
PWB detection needs **wind-rotation-corrected** high-frequency data. `dyco detect-remove` handles
this itself. Files fed to `dyco pwb-batch` must already be rotated (double rotation or planar fit,
e.g. EddyPro "Advanced" rotated output). In unrotated data, `W` still contains part of the
horizontal wind (its mean is not zero), and that distorts the cross-correlation.
:::

## Input file formats

These flags describe the raw files read by `detect-remove`, and the defaults given are its own.
`dyco fdl` reads files with the same flags, apart from the three for writing (`--na-rep`,
`--lineterm`, `--output-suffix`). `apply-batch` has the same flags except `--file-pattern` and `--output-suffix`, with
defaults for EddyPro rotated output (`--skiprows 9`, `--extra-rows 0`, `--sep '\s+'`), and its
`--lineterm` defaults to `\n` and has no `auto`. `pwb-batch` reads whitespace-separated columns by
position (`--usecols`, `--col-names`).

| Flag | Handles |
|---|---|
| `--sep` | Field separator. `,` by default; `\t` for TSV, `\s+` for whitespace-aligned. |
| `--skiprows` | Metadata lines **before** the column-name row. `0` for a plain CSV with names on line 1; `9` for EddyPro rotated output. |
| `--extra-rows` | Rows **after** the header but before the data, such as units and instrument tags. Default `2`. They are copied unchanged into the output. |
| `--na-values` / `--na-rep` | What counts as missing on the way in, what is written for it on the way out. |
| `--lineterm` | `auto` reproduces the input's CRLF or LF. Force it with `\r\n` or `\n`. |
| `--file-pattern` | Which files to read. Compression is transparent: `.gz`, `.bz2`, `.xz` and `.zip` are read as the text they contain. |
| `--output-suffix` | The extension the written chunks carry, dot included. Give the whole thing (`.csv`, `.csv.gz`, `.dat.zip`), just the text format (`.csv`, which writes plain text and drops any compression), or just the compression (`.zip`, which keeps the input's text format, so `file1.csv` gives `file1.csv.zip`). `auto` (default) reuses the input's own extension. The leading dot is required: `.csv`, not `csv`. |

Two limits apply to `detect-remove`. It reads **delimited text only**; Parquet is read by
`dyco.files.read_raw_data`, which serves the file splitter, not this pipeline. And it needs **no
data-timestamp column**: chunking is driven by `--hz` and record count, and the wall-clock time comes
from the filename via `--start-time-regex`.
