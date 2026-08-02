# Command reference

Everything is reachable through one command:

```bash
dyco                    # list the workflows
dyco <command> --help   # options for one of them
```

| Command | Does | When |
|---|---|---|
| [`dyco tui`](tui.md) | `detect-remove` behind a form, with live validation and a preflight check. | **Start here.** |
| [`dyco detect-remove`](detect-remove.md) | **The main command.** Split long raw files into averaging-period chunks, rotate, detect the lag per chunk, then remove it. One pass, one output folder. | Almost always. Scripting, or when you prefer a command line. |
| [`dyco pwb-batch`](pwb-batch.md) | Detect only, on files that are *already* split into averaging periods. Writes `tlag_results.csv` and stops. | Step 1 of the two-step route. |
| [`dyco apply-batch`](apply-batch.md) | Remove lags listed in an existing `tlag_results.csv`. Detects nothing. | Step 2 of the two-step route. |

Each also exists standalone: `dyco-detect-remove`, `dyco-detect-remove-tui`, `dyco-pwb-batch`,
`dyco-apply-batch`.

Everything else in `dyco` is a library API with no command of its own: the file splitter, the flux
detection limit, the covariance-maximization estimator.

:::{note}
The pages below are generated from the argparse parsers themselves, so they always match what
`--help` prints.
:::

```{toctree}
:maxdepth: 1

detect-remove
pwb-batch
apply-batch
tui
```

## Taking a gas's lag from another gas

A trace gas can be too noisy to locate its own lag. When that happens PWBOPT rejects the detections,
and the last resort is the *median of those same rejected numbers*, which on real data is often a
negative lag no tube can produce. A reference gas travelling the same tube is a better answer.

Say so per gas, with `@lagfrom=` (**Lag from** in the TUI):

```bash
dyco detect-remove ... --scalar "CO2:CO2_DRY_[IRGA72-A]" --scalar "N2O:N2O_DRY_[QCL-C2]@lagfrom=CO2"
```

N₂O keeps every lag it *can* determine for itself. Only the periods where it cannot take the CO₂ lag
instead, and they take it period by period, so a donor lag that drifts is followed rather than
flattened into a constant. The summary records the choice for every period in `{gas}_lag_source`
(`own`, `from:CO2`, `median`), so a borrowed lag is never mistaken for a detected one.

Chains work (`CH4` from `N2O` from `CO2`); circular ones are rejected.

## The two-step route

`pwb-batch` to detect, then `apply-batch`, where `--scalar LABEL:column` reads the lag of one gas and
shifts the column of another. It applies when your files are already split into averaging periods by
other software, leaving nothing for `detect-remove` to chunk.

:::{important}
PWB detection needs **wind-rotation-corrected** high-frequency data. `dyco detect-remove` handles
this itself. Files fed to `dyco pwb-batch` must already be rotated (double rotation or planar fit,
e.g. EddyPro "Advanced" rotated output). A non-zero mean `W` corrupts the cross-correlation.
:::

## Input file formats

| Flag | Handles |
|---|---|
| `--sep` | Field separator. `,` by default; `\t` for TSV, `\s+` for whitespace-aligned. |
| `--skiprows` | Metadata lines **before** the column-name row. `0` for a plain CSV with names on line 1; `9` for EddyPro rotated output. |
| `--extra-rows` | Rows **after** the header but before the data, such as units and instrument tags. Default `2`. They are preserved byte-for-byte in the output. |
| `--na-values` / `--na-rep` | What counts as missing on the way in, what is written for it on the way out. |
| `--lineterm` | `auto` reproduces the input's CRLF or LF. Force it with `\r\n` or `\n`. |
| `--file-pattern` | Which files to read. Compression is transparent: `.gz`, `.bz2`, `.xz` and `.zip` are read as the text they contain. |
| `--output-suffix` | The extension the written chunks carry. Give the whole thing (`.csv`, `.csv.gz`, `.dat.zip`), just the text format (`csv`, which writes plain text and drops any compression), or just the compression (`zip`, which keeps the input's text format, so `file1.csv` gives `file1.csv.zip`). `auto` (default) reuses the input's own extension. |

Two limits worth knowing. This path reads **delimited text only**; Parquet is read by
`dyco.files.read_raw_data`, which serves the file splitter, not this pipeline. And it needs **no
data-timestamp column**: chunking is driven by `--hz` and record count, and the wall-clock time comes
from the filename via `--start-time-regex`.
